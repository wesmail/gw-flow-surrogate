"""Patch tokenization for log-amplitude and gauge-invariant phase (§4.2).

Δφ-representation note (fixed): the per-patch boundary increment Δφ is the
cumulative phase swept across one patch.  Near merger it routinely exceeds π by
several multiples, so it is carried as a *linear* (unbounded) quantity here —
both in the token features and as a regression target.  The earlier cos/sin
encoding wrapped Δφ into (-π, π] and silently discarded the winding number on
exactly the merger patches; that is no longer done.  Only the bounded de-ramped
residual remains a genuinely circular quantity.

Two code paths live here:

* ``patchify_tracks`` / ``build_token_features`` — the original per-sample (C, T)
  path.  Kept for backward compatibility (norm-stat tooling, model.generate's
  single-token feature assembly via ``patches_to_token_features``).
* ``patchify_tracks_batched`` / ``build_token_features_batched`` — a fully
  vectorized (B, C, T) path with NO Python loop over patches.  This is what the
  training loop now uses, run once per batch on the GPU inside
  ``on_after_batch_transfer``.  It is numerically identical to the per-sample
  path (verified elementwise) but ~2 orders of magnitude faster and keeps the
  HDF5 layer and the ~T/patch_len-iteration loop out of the CPU dataloader.
"""

from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Original per-sample path (unchanged).                                        #
# --------------------------------------------------------------------------- #
def patchify_tracks(
    log_a: np.ndarray | torch.Tensor,
    phi: np.ndarray | torch.Tensor,
    patch_len: int,
    *,
    stride: int | None = None,
) -> dict[str, torch.Tensor]:
    """Convert (C, T) tracks into non-overlapping patch tokens.

    Each patch carries:
      - log_a:       mean log-amplitude over the patch (normalized externally)
      - delta_phi:   boundary-to-boundary phase increment (linear, telescoping-exact)
      - phi_residual: de-ramped intra-patch phase residual (gauge-invariant, bounded)

    Returns tensors with shape (N_patches, C, ...).
    """
    if isinstance(log_a, np.ndarray):
        log_a = torch.from_numpy(log_a)
    if isinstance(phi, np.ndarray):
        phi = torch.from_numpy(phi)

    stride = patch_len if stride is None else stride
    c, t = log_a.shape
    n_patches = (t - 1 - patch_len) // stride + 1
    if n_patches < 1:
        raise ValueError(
            f"Track length {t} too short for patch_len={patch_len} "
            f"(need t >= patch_len + 1 for boundary increments)."
        )

    log_a_patches = []
    delta_phi_list = []
    residual_list = []
    s_idx = torch.arange(patch_len, device=phi.device, dtype=phi.dtype)

    for n in range(n_patches):
        s0 = n * stride
        # Boundary-to-boundary increment (§4.2), kept UNWRAPPED (linear).
        dphi = phi[:, s0 + patch_len] - phi[:, s0]
        log_a_patches.append(log_a[:, s0 : s0 + patch_len].mean(dim=-1))
        delta_phi_list.append(dphi)

        ramp = (s_idx / patch_len).view(1, -1) * dphi.unsqueeze(-1)
        residual = phi[:, s0 : s0 + patch_len] - phi[:, s0 : s0 + 1] - ramp
        residual_list.append(residual)

    return {
        "log_a": torch.stack(log_a_patches, dim=0),
        "delta_phi": torch.stack(delta_phi_list, dim=0),
        "phi_residual": torch.stack(residual_list, dim=0),
    }


def patches_to_token_features(
    log_a: torch.Tensor,
    delta_phi: torch.Tensor,
    phi_residual: torch.Tensor,
) -> torch.Tensor:
    """Channel-grouped features -> (N, C*(2+P)).

    log_a and delta_phi are expected already normalized; delta_phi is carried as
    a single LINEAR feature (no cos/sin) so the winding number is preserved.
    """
    parts = [
        log_a.unsqueeze(-1),
        delta_phi.unsqueeze(-1),
        phi_residual,
    ]
    feat = torch.cat(parts, dim=-1)
    return feat.reshape(feat.shape[0], -1)


def build_token_features(
    patches: dict[str, torch.Tensor],
    n_modes: int | None = None,
) -> torch.Tensor:
    """Flatten per-patch features for the token embedding."""
    return patches_to_token_features(
        patches["log_a"],
        patches["delta_phi"],
        patches["phi_residual"],
    )


# --------------------------------------------------------------------------- #
# Vectorized batched path (new). Numerically identical to the per-sample path. #
# --------------------------------------------------------------------------- #
def n_patches_for(t: int, patch_len: int, stride: int | None = None) -> int:
    """Number of non-overlapping patches with boundary increments for length t."""
    stride = patch_len if stride is None else stride
    return (t - 1 - patch_len) // stride + 1


def patchify_tracks_batched(
    log_a: torch.Tensor,   # (B, C, T)
    phi: torch.Tensor,     # (B, C, T)
    patch_len: int,
) -> dict[str, torch.Tensor]:
    """Non-overlapping patchify for a whole batch at once (stride == patch_len).

    Returns:
      log_a:        (B, N, C)     patch-mean log-amplitude
      delta_phi:    (B, N, C)     boundary-to-boundary increment (linear)
      phi_residual: (B, N, C, P)  de-ramped intra-patch residual (radians)

    Phase math is forced to fp32: unwrapped Δφ reaches many multiples of π near
    merger and must not be accumulated in bf16.
    """
    log_a = log_a.float()
    phi = phi.float()
    B, C, T = log_a.shape
    P = patch_len
    n = n_patches_for(T, P)                     # == (T - 1) // P for stride == P
    if n < 1:
        raise ValueError(
            f"Track length {T} too short for patch_len={P} "
            f"(need T >= patch_len + 1 for boundary increments)."
        )
    usable = n * P

    # Boundaries at 0, P, 2P, ..., nP  -> (n+1) values;  Δφ = successive diffs.
    bnd = phi[:, :, 0 : usable + 1 : P]                 # (B, C, n+1)
    delta_phi = bnd[:, :, 1:] - bnd[:, :, :-1]          # (B, C, n)

    # Intra-patch windows, patch-start aligned.
    la_win = log_a[:, :, :usable].reshape(B, C, n, P)   # (B, C, n, P)
    phi_win = phi[:, :, :usable].reshape(B, C, n, P)
    log_a_patch = la_win.mean(dim=-1)                   # (B, C, n)

    s = torch.arange(P, device=phi.device, dtype=phi.dtype) / P   # (P,)
    ramp = s.view(1, 1, 1, P) * delta_phi.unsqueeze(-1)           # (B, C, n, P)
    residual = phi_win - phi_win[..., :1] - ramp                  # (B, C, n, P)

    # -> channel-last patch layout expected downstream.
    return {
        "log_a": log_a_patch.permute(0, 2, 1).contiguous(),          # (B, n, C)
        "delta_phi": delta_phi.permute(0, 2, 1).contiguous(),        # (B, n, C)
        "phi_residual": residual.permute(0, 2, 1, 3).contiguous(),   # (B, n, C, P)
    }


def build_token_features_batched(
    log_a: torch.Tensor,        # (B, N, C)
    delta_phi: torch.Tensor,    # (B, N, C)
    phi_residual: torch.Tensor, # (B, N, C, P)
) -> torch.Tensor:
    """Batched analogue of ``patches_to_token_features`` -> (B, N, C*(2+P)).

    Channel-major within each token: [la_c0, dphi_c0, res_c0(0..P-1), la_c1, ...],
    matching the per-sample reshape exactly so ``feature_dim`` is unchanged.
    """
    B, N, C = log_a.shape
    parts = torch.cat(
        [log_a.unsqueeze(-1), delta_phi.unsqueeze(-1), phi_residual], dim=-1
    )                                                    # (B, N, C, 2+P)
    return parts.reshape(B, N, C * parts.shape[-1])      # (B, N, C*(2+P))