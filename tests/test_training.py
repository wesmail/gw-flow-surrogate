"""Tests for model forward pass and training alignment (C1, C5, H2)."""

import torch

from datasets.patchify import build_token_features, patchify_tracks
from models.lightning_module import GWSurrogateLit
from models.surrogate import GWSurrogateModel


def _synthetic_batch(b=2, n=8, c=1, p=32, ctx=1, horizon=4, d_model=64):
  feat_dim = GWSurrogateModel.feature_dim(c, p)
  return {
    "theta": torch.randn(b, 3),
    "ref_phase": torch.zeros(b),
    "token_features": torch.randn(b, n, feat_dim),
    "log_a": torch.randn(b, n, c),
    "delta_phi": torch.randn(b, n, c) * 0.1,
    "phi_residual": torch.randn(b, n, c, p) * 0.01,
    "mask": torch.ones(b, n, dtype=torch.bool),
    "context_patches": torch.tensor([ctx]),
    "target_start": torch.tensor([ctx]),
  }


def test_deterministic_head_shapes():
  m = GWSurrogateModel(feat_dim=35, d_model=64, n_layers=2, n_heads=4, n_channels=1, patch_len=32)
  batch = _synthetic_batch()
  out = m(batch["token_features"], batch["theta"], batch["mask"])
  assert out["mu_a"].shape == (2, 8, 4, 1)
  assert out["mu_dphi"].shape == (2, 8, 4, 1)
  assert out["mu_res"].shape == (2, 8, 4, 1, 32)


def test_probabilistic_couple_amplitude_forward():
  m = GWSurrogateModel(
    feat_dim=35, d_model=64, n_layers=2, n_heads=4, n_channels=1,
    patch_len=32, probabilistic=True, couple_amplitude=True,
  )
  batch = _synthetic_batch()
  out = m(batch["token_features"], batch["theta"])
  assert out["kappa_dphi"].shape == (2, 8, 4, 1)


def test_aligned_loss_indices():
  lit = GWSurrogateLit(d_model=64, n_layers=2, n_heads=4, horizon=4)
  batch = _synthetic_batch(n=10, ctx=1)
  out = lit.model(batch["token_features"], batch["theta"])
  ctx, horizon = 1, 4
  for h in range(horizon):
    lo, hi = ctx - 1, batch["mask"].shape[1] - 1 - h
    if hi <= lo:
      continue
    pred = out["mu_a"][:, lo:hi, h]
    tgt = batch["log_a"][:, ctx + h :]
    assert pred.shape == tgt.shape


def test_no_leakage_random_targets():
  """Loss cannot be driven to ~0 in a few steps with uncorrelated targets."""
  torch.manual_seed(0)
  lit = GWSurrogateLit(d_model=64, n_layers=2, n_heads=4, horizon=4, learning_rate=1e-3)
  lit.eval()
  opt = torch.optim.Adam(lit.parameters(), lr=1e-3)
  batch = _synthetic_batch(n=16, ctx=1)
  batch["log_a"] = torch.randn_like(batch["log_a"])
  batch["delta_phi"] = torch.randn_like(batch["delta_phi"]) * 0.5
  losses = []
  lit.train()
  for _ in range(5):
    opt.zero_grad()
    loss = lit._compute_loss(batch, "train")
    loss.backward()
    opt.step()
    losses.append(loss.item())
  assert losses[-1] > 0.2


def test_scheduled_sampling_rebuild():
  lit = GWSurrogateLit(d_model=64, n_layers=2, n_heads=4, scheduled_sampling_max=1.0)
  lit._scheduled_sampling_prob = lambda: 1.0  # type: ignore[method-assign]
  lit.train()
  batch = _synthetic_batch(n=6, ctx=2)
  mixed = lit._maybe_mix_predictions(batch)
  assert mixed.shape == batch["token_features"].shape


def test_full_forward_from_patchify():
  phi = torch.cumsum(torch.randn(1, 65) * 0.05, dim=-1)
  patches = patchify_tracks(torch.randn(1, 65), phi, patch_len=32)
  feat = build_token_features(patches)
  m = GWSurrogateModel(feat_dim=feat.shape[-1], d_model=64, n_layers=2, n_heads=4)
  out = m(feat.unsqueeze(0), torch.randn(1, 3))
  assert "mu_a" in out
