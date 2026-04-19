"""
Evaluation metrics for flow matching MNIST models.

Supports three model types:
  - mean_flow:     MeanFlowModule (our Lightning implementation)
  - flow_matching: FlowMatchingModule (our Lightning implementation)
  - meanflow_dit:  MFDiT from the MeanFlow repo (../MeanFlow)

Usage:
    python evaluate.py --checkpoint path/to/model.ckpt --model_type mean_flow
    python evaluate.py --checkpoint path/to/model.ckpt --model_type flow_matching
    python evaluate.py --checkpoint path/to/model.pt  --model_type meanflow_dit
"""

import argparse
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.module import MeanFlowModule, FlowMatchingModule
from data_module import MNISTDataModule


# ── Sampling ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_samples_lightning(module, labels, num_steps, guidance_scale=1.0, mean_flow=True):
    """Sampling for our Lightning modules (t=0 noise, t=1 data)."""
    device = next(module.parameters()).device
    bs = len(labels)
    y_null = torch.full_like(labels, 10)

    x, _ = module.path.p_simple.sample(bs)
    x = x.to(device)

    t_sched = torch.linspace(0, 1, num_steps + 1, device=device)
    for i in range(num_steps):
        t = t_sched[i].view(1, 1, 1, 1).expand(bs, 1, 1, 1)
        r = t_sched[i + 1].view(1, 1, 1, 1).expand(bs, 1, 1, 1)
        dt = r - t

        r_input = r if mean_flow else t
        u_c = module.model(x, t, r_input, labels)
        if guidance_scale != 1.0:
            u_u = module.model(x, t, r_input, y_null)
            u = (1 - guidance_scale) * u_u + guidance_scale * u_c
        else:
            u = u_c
        x = x + dt * u

    return x


@torch.no_grad()
def generate_samples_meanflow_dit(model, meanflow, labels, num_steps, device):
    """Sampling for MeanFlow repo MFDiT (t=1 noise, t=0 data).

    Uses meanflow.sample_each_class logic but with arbitrary labels.
    """
    bs = len(labels)
    z = torch.randn(bs, meanflow.channels, meanflow.image_size, meanflow.image_size, device=device)
    t_vals = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t = torch.full((bs,), t_vals[i], device=device)
        r = torch.full((bs,), t_vals[i + 1], device=device)
        v = model(z, t, r, labels)
        z = z - (t_vals[i] - t_vals[i + 1]) * v

    z = meanflow.normer.unnorm(z)
    return z


# ── FID ──────────────────────────────────────────────────────────────────────


def _to_inception_format(images):
    """Convert [0,1] float (1,32,32) to [0,255] uint8 (3,299,299)."""
    images = images.clamp(0, 1)
    images = images.repeat(1, 3, 1, 1)
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    return (images * 255).to(torch.uint8)


def _to_inception_format_from_model_space(images):
    """Convert [-1,1] float (1,32,32) to [0,255] uint8 (3,299,299)."""
    images = ((images + 1) / 2).clamp(0, 1)
    return _to_inception_format(images)


def evaluate_fid(sample_fn, dataloader, num_gen=10000, device="cpu"):
    """Compute FID between real test images and generated samples.

    Args:
        sample_fn: callable(batch_size) -> images in [0, 1] range, (bs, 1, 32, 32)
        dataloader: test dataloader
        num_gen: number of samples
        device: device
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048).to(device)

    # real images
    n_real = 0
    for x, _ in tqdm(dataloader, desc="FID (real)"):
        # dataloader images are in [-1, 1] (after Normalize(0.5, 0.5))
        imgs = _to_inception_format_from_model_space(x.to(device))
        fid.update(imgs, real=True)
        n_real += x.shape[0]
        if n_real >= num_gen:
            break

    # generated images
    n_fake = 0
    pbar = tqdm(total=num_gen, desc="FID (gen)")
    while n_fake < num_gen:
        bs = min(250, num_gen - n_fake)
        samples = sample_fn(bs)
        fid.update(_to_inception_format(samples), real=False)
        n_fake += bs
        pbar.update(bs)
    pbar.close()

    return fid.compute().item()


# ── Model loading ────────────────────────────────────────────────────────────


def load_lightning(checkpoint, model_type):
    cls = MeanFlowModule if model_type == "mean_flow" else FlowMatchingModule
    module = cls.load_from_checkpoint(checkpoint)
    module.eval()
    return module


def load_meanflow_dit(checkpoint, device):
    sys.path.insert(0, "../MeanFlow")
    from models.dit import MFDiT
    from meanflow import MeanFlow

    model = MFDiT(
        input_size=32,
        patch_size=2,
        in_channels=1,
        dim=256,
        depth=6,
        num_heads=4,
        num_classes=10,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    meanflow = MeanFlow(
        channels=1,
        image_size=32,
        num_classes=10,
        flow_ratio=0.50,
        time_dist=["lognorm", -0.4, 1.0],
        cfg_ratio=0.10,
        cfg_scale=2.0,
        cfg_uncond="u",
    )

    return model, meanflow


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate flow matching MNIST model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model_type", type=str, default="mean_flow",
        choices=["mean_flow", "flow_matching", "meanflow_dit"],
    )
    parser.add_argument("--fid_steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=1.0, help="CFG scale (lightning models only)")
    parser.add_argument("--num_gen", type=int, default=10000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Loading: {args.checkpoint}")

    # load test data
    dm = MNISTDataModule(data_dir="./data", batch_size=250)
    dm.setup("test")
    test_dl = dm.test_dataloader()

    # build sample function that returns images in [0, 1]
    if args.model_type in ("mean_flow", "flow_matching"):
        module = load_lightning(args.checkpoint, args.model_type)
        module.to(device)
        mean_flow = args.model_type == "mean_flow"

        def sample_fn(bs):
            labels = torch.randint(0, 10, (bs,), device=device)
            imgs = generate_samples_lightning(
                module, labels, args.fid_steps, args.guidance, mean_flow,
            )
            return ((imgs + 1) / 2).clamp(0, 1)  # [-1,1] -> [0,1]

    elif args.model_type == "meanflow_dit":
        model, meanflow = load_meanflow_dit(args.checkpoint, device)

        def sample_fn(bs):
            labels = torch.randint(0, 10, (bs,), device=device)
            imgs = generate_samples_meanflow_dit(
                model, meanflow, labels, args.fid_steps, device,
            )
            return imgs.clamp(0, 1)  # unnorm already gives [0,1]

    # ── FID ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  FID   (steps={args.fid_steps}, n={args.num_gen})")
    if args.model_type != "meanflow_dit":
        print(f"  guidance={args.guidance}")
    print(f"{'='*50}")

    fid_score = evaluate_fid(sample_fn, test_dl, args.num_gen, device)
    print(f"  FID:  {fid_score:.2f}")


if __name__ == "__main__":
    main()
