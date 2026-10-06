"""Training losses: Student-t, Gaussian, von Mises, coherence (§6.1)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from models.heads import log_i0


def student_t_nll(
  target: torch.Tensor,
  mu: torch.Tensor,
  log_s: torch.Tensor,
  nu: torch.Tensor,
) -> torch.Tensor:
  """Element-wise negative log-likelihood of Student-t."""
  s = torch.exp(log_s)
  z = (target - mu) / s
  log_prob = (
    torch.lgamma((nu + 1) / 2)
    - torch.lgamma(nu / 2)
    - 0.5 * torch.log(nu * math.pi)
    - log_s
    - (nu + 1) / 2 * torch.log1p(z.pow(2) / nu)
  )
  return -log_prob


def gaussian_nll(target: torch.Tensor, mu: torch.Tensor, log_s: torch.Tensor) -> torch.Tensor:
  """Element-wise Gaussian NLL for a *linear* (non-circular) target such as
  the per-patch phase increment Δφ.  Δφ is the cumulative phase swept across a
  patch and can exceed π by several multiples near merger, so it must NOT be
  modelled on the circle (that throws away the winding number)."""
  s = torch.exp(log_s)
  z = (target - mu) / s
  return 0.5 * z.pow(2) + log_s + 0.5 * math.log(2 * math.pi)


def von_mises_nll(target: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
  """Element-wise circular negative log-likelihood (for bounded residuals)."""
  return log_i0(kappa) - kappa * torch.cos(target - mu)


def coherence_loss(
  delta_phi_pred: torch.Tensor,
  delta_phi_true: torch.Tensor,
) -> torch.Tensor:
  """Penalize discontinuity of phase increment across patch boundaries (§6.1)."""
  return F.mse_loss(
    delta_phi_pred[:, 1:] - delta_phi_pred[:, :-1],
    delta_phi_true[:, 1:] - delta_phi_true[:, :-1],
  )