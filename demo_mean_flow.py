import os
import torch
import matplotlib.pyplot as plt
from model.module import MeanFlowModule
from torchvision.utils import make_grid

model_path = "path/to/model.ckpt"  # <-- promijeni ovo

# ── parameters ───────────────────────────────────────────────────────────────
num_steps_list   = [1, 5, 20]   # usporedi različit broj koraka
guidance_scales  = [1.0, 3.0]
samples_per_class = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────


def load_module(path: str) -> MeanFlowModule:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    module = MeanFlowModule.load_from_checkpoint(checkpoint_path=path)
    module.eval()
    return module.to(device)


@torch.no_grad()
def sample_mean_flow(module, y, num_steps: int, guidance_scale: float):
    """
    Multi-step mean flow sampling.

    Svaki korak ide od t do r:  x_r = x_t + (r - t) * u(x_t, t, r)
    Model je treniran da predvidi *prosječnu* brzinu za interval [t, r],
    pa su veći koraci točniji nego kod standardnog FM-a.
    """
    bs = len(y)
    y_null = torch.full_like(y, 10)  # null token (unconditional)

    # Uzorkuj šum (t=0 je šum, t=1 su podaci)
    x, _ = module.path.p_simple.sample(bs)
    x = x.to(device)

    # Raspored koraka: od t=0 (šum) do t=1 (podaci)
    t_schedule = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t = t_schedule[i].view(1, 1, 1, 1).expand(bs, 1, 1, 1)
        r = t_schedule[i + 1].view(1, 1, 1, 1).expand(bs, 1, 1, 1)

        # Classifier-free guidance: blend kondicionirani i nekondicionirani
        u_cond   = module.model(x, t, r, y)
        u_uncond = module.model(x, t, r, y_null)
        u = (1.0 - guidance_scale) * u_uncond + guidance_scale * u_cond

        # Korak: x_r = x_t + (r - t) * u
        x = x + (r - t) * u

    return x


# ── generacija i plot ─────────────────────────────────────────────────────────
mf_module = load_module(model_path)

y = torch.arange(10, dtype=torch.int64, device=device).repeat_interleave(samples_per_class)

fig, axes = plt.subplots(
    len(guidance_scales), len(num_steps_list),
    figsize=(6 * len(num_steps_list), 6 * len(guidance_scales))
)

for row, w in enumerate(guidance_scales):
    for col, n_steps in enumerate(num_steps_list):
        images = sample_mean_flow(mf_module, y, num_steps=n_steps, guidance_scale=w)
        grid = make_grid(images, nrow=samples_per_class, normalize=True, value_range=(-1, 1))

        ax = axes[row][col] if len(guidance_scales) > 1 else axes[col]
        ax.imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"steps={n_steps}, w={w:.1f}", fontsize=16)

plt.tight_layout()
plt.savefig("demo_mean_flow.png", dpi=150)
print("Saved: demo_mean_flow.png")
