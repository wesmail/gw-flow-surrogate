"""Flow-matching patch head (MAR-style 'Diffusion Loss', rectified flow).

Models the JOINT distribution over one full token-feature vector
[log_a, delta_phi, residual(P)] x channels — no factorization across the
residual samples and no parametric family assumptions, replacing the
Student-t / Gaussian / von Mises heads.

    x_tau = (1 - tau) x0 + tau x1,   x0 ~ N(0, I),  x1 = target token features
    target velocity v* = x1 - x0
    sampling: Euler-integrate dx/dtau = v(x_tau, tau | cond) from 0 to 1.

The head is deliberately tiny (default width 256, depth 3) and is applied to
FLATTENED (M, D) rows, so its training cost is negligible next to the
backbone; at generation it adds `n_steps` small MLP evaluations per patch
while the backbone runs once per patch (with KV cache).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(tau: torch.Tensor, dim: int) -> torch.Tensor:
  """Sinusoidal embedding of flow time tau in [0, 1].  tau: (M,) -> (M, dim)."""
  half = dim // 2
  freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=tau.device) / half)
  args = tau[:, None].float() * freqs[None] * 1000.0
  return torch.cat([args.cos(), args.sin()], dim=-1).to(tau.dtype)


class AdaLNBlock(nn.Module):
  """Residual MLP block with adaptive LayerNorm conditioning (DiT-style)."""

  def __init__(self, width: int, cond_dim: int):
    super().__init__()
    self.norm = nn.LayerNorm(width, elementwise_affine=False)
    self.mlp = nn.Sequential(
      nn.Linear(width, 4 * width), nn.SiLU(), nn.Linear(4 * width, width)
    )
    self.ada = nn.Linear(cond_dim, 3 * width)
    nn.init.zeros_(self.ada.weight)
    nn.init.zeros_(self.ada.bias)

  def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    shift, scale, gate = self.ada(cond).chunk(3, dim=-1)
    h = self.norm(x) * (1 + scale) + shift
    return x + gate * self.mlp(h)


class FlowMatchingHead(nn.Module):
  """Velocity field v(x_tau, tau | cond) over one token-feature vector."""

  def __init__(self, patch_dim: int, cond_dim: int, width: int = 256, depth: int = 3):
    super().__init__()
    self.in_proj = nn.Linear(patch_dim, width)
    self.time_mlp = nn.Sequential(
      nn.Linear(width, width), nn.SiLU(), nn.Linear(width, cond_dim)
    )
    self.t_emb_dim = width
    self.blocks = nn.ModuleList(AdaLNBlock(width, cond_dim) for _ in range(depth))
    self.out_norm = nn.LayerNorm(width, elementwise_affine=False)
    self.out_proj = nn.Linear(width, patch_dim)
    nn.init.zeros_(self.out_proj.weight)  # v ~ 0 at init
    nn.init.zeros_(self.out_proj.bias)

  def forward(self, x_tau: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    # x_tau: (M, patch_dim), tau: (M,), cond: (M, cond_dim)
    c = cond + self.time_mlp(timestep_embedding(tau, self.t_emb_dim))
    h = self.in_proj(x_tau)
    for blk in self.blocks:
      h = blk(h, c)
    return self.out_proj(self.out_norm(h))

  @torch.no_grad()
  def sample(self, cond: torch.Tensor, patch_dim: int, n_steps: int = 20,
             temperature: float = 1.0) -> torch.Tensor:
    """Euler integration noise -> data.  The state is accumulated in fp32
    (20 bf16 additions on O(1) values lose real precision); the network is
    still evaluated in the ambient autocast dtype."""
    x = temperature * torch.randn(cond.shape[0], patch_dim, device=cond.device, dtype=torch.float32)
    dt = 1.0 / n_steps
    for i in range(n_steps):
      tau = torch.full((cond.shape[0],), i * dt, device=cond.device, dtype=cond.dtype)
      v = self(x.to(cond.dtype), tau, cond)
      x = x + dt * v.float()
    return x.to(cond.dtype)