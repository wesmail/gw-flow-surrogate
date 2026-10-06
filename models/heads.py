"""Probabilistic output heads: Student-t (log A), Gaussian (Δφ), von Mises (residual)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def log_i0(kappa: torch.Tensor) -> torch.Tensor:
  """Stable log I_0(kappa) via scaled Bessel i0e."""
  kappa = kappa.clamp(min=0.0)
  return torch.special.i0e(kappa).log() + kappa


log_i0_approx = log_i0  # backward-compatible alias


class AmplitudeHead(nn.Module):
  """Student-t head for normalized log-amplitude."""

  def __init__(
    self,
    dim: int,
    n_channels: int,
    horizon: int = 4,
    log_s_min: float = -9.21,
    log_s_max: float = 2.30,
    nu_min: float = 2.1,
  ):
    super().__init__()
    self.n_channels = n_channels
    self.horizon = horizon
    self.log_s_min = log_s_min
    self.log_s_max = log_s_max
    self.nu_min = nu_min
    self.proj = nn.Linear(dim, horizon * n_channels * 3)

  def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    b, n, _ = x.shape
    out = self.proj(x).view(b, n, self.horizon, self.n_channels, 3)
    mu = out[..., 0]
    log_s = out[..., 1].clamp(self.log_s_min, self.log_s_max)
    nu = F.softplus(out[..., 2]) + self.nu_min
    return {"mu_a": mu, "log_s_a": log_s, "nu_a": nu}


class PhaseHead(nn.Module):
  """Linear-Gaussian head for Δφ (winding-safe) + von Mises head for the
  bounded de-ramped residual.

  Δφ is the cumulative per-patch phase increment and can exceed π by several
  multiples near merger, so it is modelled as a real-valued Gaussian on the
  *normalized* increment — NOT on the circle.  Only the residual (bounded,
  gauge-invariant) keeps a circular von Mises treatment.
  """

  def __init__(
    self,
    dim: int,
    n_channels: int,
    patch_len: int,
    horizon: int = 4,
    kappa_min: float = 1e-2,
    kappa_max: float = 1e3,
    log_s_min: float = -9.21,
    log_s_max: float = 2.30,
    couple_amplitude: bool = True,
  ):
    super().__init__()
    self.n_channels = n_channels
    self.patch_len = patch_len
    self.horizon = horizon
    self.kappa_min = kappa_min
    self.kappa_max = kappa_max
    self.log_s_min = log_s_min
    self.log_s_max = log_s_max
    self.couple_amplitude = couple_amplitude
    # per channel: Δφ -> (mu, log_s) = 2 ; residual -> (kappa, cos, sin) * P = 3P
    out_dim = horizon * n_channels * (2 + 3 * patch_len)
    self.proj = nn.Linear(dim, out_dim)

  def forward(
    self,
    x: torch.Tensor,
    log_a_mu: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    b, n, _ = x.shape
    per_ch = 2 + 3 * self.patch_len
    raw = self.proj(x).view(b, n, self.horizon, self.n_channels, per_ch)

    mu_dphi = raw[..., 0]
    log_s_dphi = raw[..., 1].clamp(self.log_s_min, self.log_s_max)

    res_raw = raw[..., 2:].reshape(b, n, self.horizon, self.n_channels, self.patch_len, 3)
    mu_res = torch.atan2(res_raw[..., 2], res_raw[..., 1])
    kappa_res = F.softplus(res_raw[..., 0]).clamp(self.kappa_min, self.kappa_max)

    if self.couple_amplitude and log_a_mu is not None:
      amp_factor = torch.sigmoid(log_a_mu.detach())          # (0,1): high at large amplitude
      # higher amplitude -> tighter Δφ (smaller sigma) and more concentrated residual
      log_s_dphi = (log_s_dphi - amp_factor).clamp(self.log_s_min, self.log_s_max)
      kappa_res = (kappa_res * (1 + amp_factor.unsqueeze(-1))).clamp(self.kappa_min, self.kappa_max)

    return {
      "mu_dphi": mu_dphi,
      "log_s_dphi": log_s_dphi,
      "mu_res": mu_res,
      "kappa_res": kappa_res,
    }


class DeterministicHead(nn.Module):
  """Stage-A point estimates (§6.3).  Δφ is a linear scalar (no wrapping)."""

  def __init__(self, dim: int, n_channels: int, patch_len: int, horizon: int = 4):
    super().__init__()
    self.horizon = horizon
    self.n_channels = n_channels
    self.patch_len = patch_len
    out_dim = horizon * n_channels * (1 + 1 + patch_len)
    self.proj = nn.Linear(dim, out_dim)

  def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
    b, n, _ = x.shape
    out = self.proj(x).view(b, n, self.horizon, self.n_channels, 1 + 1 + self.patch_len)
    return {
      "mu_a": out[..., 0],
      "mu_dphi": out[..., 1],
      "mu_res": out[..., 2:],
    }