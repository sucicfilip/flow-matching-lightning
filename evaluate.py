"""
Evaluation metrics for flow matching MNIST model.

Computes:
  - NLL (negative log-likelihood in nats) via CNF change-of-variables formula
  - BPD (bits per dimension) = NLL / (D * ln2)
  - FID (Frechet Inception Distance) between generated and real samples

Usage:
    python evaluate.py --checkpoint path/to/model.ckpt
    python evaluate.py --checkpoint path/to/model.ckpt --skip_fid
    python evaluate.py --checkpoint path/to/model.ckpt --skip_nll --guidance 3.0
"""

import argparse
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model.module import MeanFlowModule
from data_module import MNISTDataModule


# ── NLL / BPD ────────────────────────────────────────────────────────────────


def compute_nll_batch(module, x_data, y, num_steps=100):
    """
    Per-sample NLL via backward ODE integration + Hutchinson trace estimator.

    Uses the instantaneous change-of-variables formula for CNFs:
        log p_1(x_1) = log p_0(x_0) - int_0^1 div(v)(x_t, t) dt

    We integrate the ODE backward from t=1 (data) to t=0 (noise), accumulating
    the divergence along the trajectory.  The trace of the Jacobian is estimated
    with a single Rademacher probe per step (unbiased, low variance).

    Args:
        module: trained MeanFlowModule (eval mode)
        x_data: batch of images (bs, 1, 32, 32) in [-1, 1]
        y: class labels (bs,)
        num_steps: number of Euler steps for the backward ODE

    Returns:
        nll: (bs,) negative log-likelihood in nats
        bpd: (bs,) bits per dimension
    """
    device = x_data.device
    bs = x_data.shape[0]
    D = x_data[0].numel()  # 1*32*32 = 1024
    dt = 1.0 / num_steps

    x = x_data.clone()
    total_div = torch.zeros(bs, device=device)

    for k in range(num_steps):
        t_val = 1.0 - k * dt
        t = torch.full((bs, 1, 1, 1), t_val, device=device)

        x = x.detach().requires_grad_(True)

        # r=t gives instantaneous velocity (mean flow reduces to standard FM)
        v = module.model(x, t, t, y)

        # Hutchinson trace estimator: tr(dv/dx) ~ eps^T (dv/dx) eps
        eps = torch.randint(0, 2, x.shape, dtype=x.dtype, device=device) * 2 - 1
        (vjp,) = torch.autograd.grad(v, x, grad_outputs=eps)
        div_v = (eps * vjp).sum(dim=(1, 2, 3))

        v = v.detach()
        x = x.detach()

        total_div = total_div + div_v * dt
        x = x - v * dt  # Euler step backward

    # log p_0(x_0) under the base distribution N(0, I)
    log_p0 = -0.5 * (D * math.log(2 * math.pi) + x.pow(2).sum(dim=(1, 2, 3)))

    # change-of-variables: log p_1(x_1) = log p_0(x_0) - int div(v) dt
    log_p1 = log_p0 - total_div

    # model operates in [-1, 1], transform from [0, 1]: x_model = 2*x_01 - 1
    # Jacobian correction: log p_01(x) = log p_model(x) + D*log(2)
    log_p1_01 = log_p1 + D * math.log(2)

    # discrete BPD: dequantization gives lower bound on discrete log-likelihood
    # BPD = (-log_p_01 + D*log(256)) / (D*log(2))  =  -log_p_01/(D*log2) + 8
    nll = -log_p1_01
    bpd = nll / (D * math.log(2)) + 8.0
    return nll, bpd


def evaluate_nll(module, dataloader, num_steps=100, max_samples=1000, device="cpu"):
    """Evaluate NLL/BPD over the dataset (subsampled to max_samples)."""
    module.eval()
    all_nll, all_bpd = [], []
    n = 0

    for x, y in tqdm(dataloader, desc="NLL"):
        if n >= max_samples:
            break
        remaining = max_samples - n
        x, y = x[:remaining].to(device), y[:remaining].to(device)

        nll, bpd = compute_nll_batch(module, x, y, num_steps)
        all_nll.append(nll.detach())
        all_bpd.append(bpd.detach())
        n += len(x)

    all_nll = torch.cat(all_nll)
    all_bpd = torch.cat(all_bpd)

    return {
        "nll_mean": all_nll.mean().item(),
        "nll_std": all_nll.std().item(),
        "bpd_mean": all_bpd.mean().item(),
        "bpd_std": all_bpd.std().item(),
        "n_samples": n,
    }


# ── FID ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_samples(module, labels, num_steps, guidance_scale=1.0):
    """Generate images via mean flow ODE integration with optional CFG."""
    device = next(module.parameters()).device
    bs = len(labels)
    y_null = torch.full_like(labels, 10)

    x, _ = module.path.p_simple.sample(bs)
    x = x.to(device)

    t_sched = torch.linspace(0, 1, num_steps + 1, device=device)
    for i in range(num_steps):
        t = t_sched[i].view(1, 1, 1, 1).expand(bs, 1, 1, 1)
        r = t_sched[i + 1].view(1, 1, 1, 1).expand(bs, 1, 1, 1)

        u_c = module.model(x, t, r, labels)
        if guidance_scale != 1.0:
            u_u = module.model(x, t, r, y_null)
            u = (1 - guidance_scale) * u_u + guidance_scale * u_c
        else:
            u = u_c
        x = x + (r - t) * u

    return x


def _to_inception_format(images):
    """Convert from [-1,1] float (1,32,32) to [0,255] uint8 (3,299,299)."""
    images = ((images + 1) / 2).clamp(0, 1)
    images = images.repeat(1, 3, 1, 1)
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    return (images * 255).to(torch.uint8)


def evaluate_fid(
    module, dataloader, num_gen=10000, num_steps=50, guidance_scale=1.0, device="cpu"
):
    """Compute FID between real test images and generated samples."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    module.eval()
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # real images
    n_real = 0
    for x, _ in tqdm(dataloader, desc="FID (real)"):
        x = _to_inception_format(x.to(device))
        fid.update(x, real=True)
        n_real += x.shape[0]
        if n_real >= num_gen:
            break

    # generated images
    n_fake = 0
    batch_size = 250
    pbar = tqdm(total=num_gen, desc="FID (gen)")
    while n_fake < num_gen:
        bs = min(batch_size, num_gen - n_fake)
        labels = torch.randint(0, 10, (bs,), device=device)
        samples = generate_samples(module, labels, num_steps, guidance_scale)
        fid.update(_to_inception_format(samples), real=False)
        n_fake += bs
        pbar.update(bs)
    pbar.close()

    return fid.compute().item()


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate flow matching MNIST model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--nll_steps", type=int, default=100, help="ODE steps for NLL")
    parser.add_argument("--fid_steps", type=int, default=50, help="Sampling steps for FID")
    parser.add_argument("--guidance", type=float, default=1.0, help="CFG scale (FID only)")
    parser.add_argument("--num_gen", type=int, default=10000, help="Samples for FID")
    parser.add_argument("--max_nll", type=int, default=1000, help="Max samples for NLL")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--skip_nll", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # load model
    print(f"Loading: {args.checkpoint}")
    module = MeanFlowModule.load_from_checkpoint(args.checkpoint)
    module.eval().to(device)

    # load test data
    dm = MNISTDataModule(data_dir="./data", batch_size=250)
    dm.setup("test")
    test_dl = dm.test_dataloader()

    # ── NLL / BPD ────────────────────────────────────────────────────────
    if not args.skip_nll:
        print(f"\n{'='*50}")
        print(f"  NLL / BPD   (steps={args.nll_steps}, n<={args.max_nll})")
        print(f"{'='*50}")

        res = evaluate_nll(module, test_dl, args.nll_steps, args.max_nll, device)

        print(f"  NLL  (nats):  {res['nll_mean']:.2f} +/- {res['nll_std']:.2f}")
        print(f"  BPD:          {res['bpd_mean']:.4f} +/- {res['bpd_std']:.4f}")
        print(f"  Samples:      {res['n_samples']}")
        print()
        print("  BPD accounts for [-1,1]->[0,1] Jacobian + 256-level discretization.")
        print("  Comparable to paper values (e.g. ~1.0 for good MNIST models).")

    # ── FID ───────────────────────────────────────────────────────────────
    if not args.skip_fid:
        print(f"\n{'='*50}")
        print(f"  FID   (steps={args.fid_steps}, guidance={args.guidance}, n={args.num_gen})")
        print(f"{'='*50}")

        fid_score = evaluate_fid(
            module, test_dl, args.num_gen, args.fid_steps, args.guidance, device
        )
        print(f"  FID:  {fid_score:.2f}")


if __name__ == "__main__":
    main()
