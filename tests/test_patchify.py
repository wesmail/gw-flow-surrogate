"""Tests for patch tokenization (C4, H1)."""

import numpy as np
import torch

from datasets.patchify import build_token_features, patchify_tracks, patches_to_token_features


def test_build_token_features_shape():
  c, p = 1, 8
  n = 5
  patches = {
    "log_a": torch.randn(n, c),
    "delta_phi": torch.randn(n, c),
    "phi_residual": torch.randn(n, c, p),
  }
  feat = build_token_features(patches)
  assert feat.shape == (n, c * (3 + p))


def test_telescoping_delta_phi():
  t = 65
  phi = torch.cumsum(torch.ones(1, t) * 0.1, dim=-1)
  patches = patchify_tracks(torch.zeros(1, t), phi, patch_len=32)
  dphi = patches["delta_phi"][:, 0]
  recon = torch.cumsum(dphi, dim=0) + phi[0, 0]
  boundaries = phi[0, 32::32]
  assert torch.allclose(recon, boundaries, atol=1e-5)


def test_residual_starts_at_zero():
  t = 65
  phi = torch.cumsum(torch.randn(1, t) * 0.05, dim=-1)
  patches = patchify_tracks(torch.zeros(1, t), phi, patch_len=32)
  assert torch.allclose(patches["phi_residual"][:, :, 0], torch.zeros(1), atol=1e-6)


def test_residual_small_on_smooth_chirp():
  t = 129
  times = torch.linspace(0, 1, t)
  phi = (2 * np.pi * 10 * times**2).unsqueeze(0)
  patches = patchify_tracks(torch.zeros(1, t), phi, patch_len=32)
  assert patches["phi_residual"].abs().mean() < patches["delta_phi"].abs().mean()
