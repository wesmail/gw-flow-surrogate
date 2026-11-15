"""Amplitude interpolation modes: scipy parity, constant regression, floor order."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.interpolate import PchipInterpolator

from datasets.patchify import patchify_tracks_batched
from models.amp_interp import AMP_MODES, log_amp_per_sample
from models.lightning_module import GWFlowSurrogateLit
from paper_figs.protocol_utils import mismatches


def _ref_scipy(la: np.ndarray, P: int, mode: str) -> np.ndarray:
    B, n = la.shape
    if mode == "constant" or n < 2:
        return np.repeat(la, P, axis=1)
    xc = np.arange(n) * P + (P - 1) / 2.0
    x = np.arange(n * P, dtype=np.float64)
    xq = np.clip(x, xc[0], xc[-1])
    if mode == "linear":
        return np.stack([np.interp(xq, xc, la[b]) for b in range(B)])
    return np.asarray(PchipInterpolator(xc, la, axis=1)(xq))


@pytest.mark.parametrize("n", [2, 3, 5, 40, 320])
@pytest.mark.parametrize("mode", ["pchip", "linear"])
def test_matches_scipy(n: int, mode: str) -> None:
    P = 32
    rng = np.random.default_rng(0)
    y = rng.normal(size=(6, n))
    y[1] = np.sort(y[1])
    if n >= 5:
        y[2, 2:5] = y[2, 2]
    out = log_amp_per_sample(torch.from_numpy(y), P, mode).numpy()
    ref = _ref_scipy(y, P, mode)
    assert np.max(np.abs(out - ref)) < 1e-12


def test_constant_is_repeat_interleave() -> None:
    P, n = 32, 17
    y = torch.randn(4, n)
    out = log_amp_per_sample(y, P, "constant")
    assert torch.equal(out, y.repeat_interleave(P, dim=1))


@pytest.mark.parametrize("mode", list(AMP_MODES))
def test_n1_constant_for_every_mode(mode: str) -> None:
    y = torch.randn(3, 1)
    out = log_amp_per_sample(y, 32, mode)
    assert torch.equal(out, y.repeat_interleave(32, dim=1))


def _reconstruct_constant_reference(
    lit: GWFlowSurrogateLit,
    la_n, dphi_n, res_n, la_m, la_s, dp_m, dp_s, patch_len: int,
) -> torch.Tensor:
    """Original patch-constant amplitude body (bitwise regression target)."""
    la_n, dphi_n = la_n.float(), dphi_n.float()
    res = res_n.float() * lit.hparams.residual_scale
    la_m = lit._norm_broadcast(la_m)
    la_s = lit._norm_broadcast(la_s)
    dp_m = lit._norm_broadcast(dp_m)
    dp_s = lit._norm_broadcast(dp_s)
    la_phys = la_n * la_s + la_m
    dphi_rad = dphi_n * dp_s + dp_m
    s = torch.arange(patch_len, device=la_phys.device, dtype=la_phys.dtype) / patch_len
    bnd = torch.cumsum(dphi_rad, dim=1) - dphi_rad
    phase = bnd[:, :, None] + s[None, None, :] * dphi_rad[:, :, None] + res
    amp = torch.exp(la_phys)[:, :, None]
    return (amp * torch.exp(-1j * phase)).reshape(la_phys.shape[0], -1)


def test_constant_reconstruct_bitwise_equal() -> None:
    lit = GWFlowSurrogateLit(
        d_model=32, n_layers=1, n_heads=2, n_kv_heads=1,
        head_width=32, head_depth=1, patch_len=32, residual_scale=0.08,
        amp_mode="constant",
    )
    B, n, P = 2, 8, 32
    torch.manual_seed(0)
    la_n = torch.randn(B, n)
    dphi_n = torch.randn(B, n)
    res_n = torch.randn(B, n, P)
    stats = (torch.zeros(B), torch.ones(B), torch.zeros(B), torch.ones(B))
    h_new = lit._reconstruct_waveform(la_n, dphi_n, res_n, *stats, P)
    h_old = _reconstruct_constant_reference(lit, la_n, dphi_n, res_n, *stats, P)
    assert torch.equal(h_new, h_old)


def test_floor_ordering_synthetic_chirp() -> None:
    """pchip < linear < constant on last-25% mismatch vs original."""
    P, n_pat = 32, 40
    T = n_pat * P + 1
    t = np.linspace(-1500.0, 60.0, T)
    tc, c1 = 40.0, 0.2
    w = 0.05 * (tc / (tc - np.minimum(t, 0))) ** (3 / 8) * (1 + 0.1 * c1)
    ph = np.cumsum(w) * (t[1] - t[0])
    la = np.log((w / 0.05) ** (2 / 3) * np.exp(-np.maximum(t, 0) / 20) + 1e-6) - 3
    patches = patchify_tracks_batched(
        torch.from_numpy(la[None, None].astype(np.float32)),
        torch.from_numpy(ph[None, None].astype(np.float32)),
        P,
    )
    usable = n_pat * P
    s0 = int(round(0.75 * n_pat)) * P
    h_orig = torch.exp(torch.from_numpy(la[:usable]).double()) * torch.exp(
        -1j * torch.from_numpy(ph[:usable] - ph[0]).double()
    )
    h_orig = h_orig[None, :]

    floors = {}
    for mode in ("constant", "linear", "pchip"):
        lit = GWFlowSurrogateLit(
            d_model=32, n_layers=1, n_heads=2, n_kv_heads=1,
            head_width=32, head_depth=1, patch_len=P, residual_scale=1.0,
            amp_mode=mode,
        )
        zeros = torch.zeros(1)
        ones = torch.ones(1)
        h = lit._reconstruct_waveform(
            patches["log_a"][:, :, 0],
            patches["delta_phi"][:, :, 0],
            patches["phi_residual"][:, :, 0],
            zeros, ones, zeros, ones, P,
        ).to(torch.complex128)
        _, _, pt, _ = mismatches(h_orig[:, s0:], h[:, s0:])
        floors[mode] = float(pt[0])

    assert floors["pchip"] < floors["linear"] < floors["constant"]


def test_old_checkpoint_amp_mode_fallback(tmp_path) -> None:
    lit = GWFlowSurrogateLit(
        d_model=32, n_layers=1, n_heads=2, n_kv_heads=1,
        head_width=32, head_depth=1, patch_len=32, residual_scale=0.05,
    )
    # Simulate a pre-amp_mode checkpoint: drop the key from hyper_parameters.
    hp = dict(lit.hparams)
    hp.pop("amp_mode", None)
    path = tmp_path / "old.ckpt"
    torch.save(
        {
            "state_dict": lit.state_dict(),
            "hyper_parameters": hp,
            "pytorch-lightning_version": "2.0.0",
        },
        path,
    )
    loaded = GWFlowSurrogateLit.load_from_checkpoint(str(path), map_location="cpu")
    assert getattr(loaded.hparams, "amp_mode", "constant") == "constant"


def test_full_vs_segment_pchip_first_half_patch() -> None:
    """Amp edge rule: segment-alone lacks the left neighbour.

    * linear: only the first half-patch (before the first centre) differs.
    * pchip: endpoint derivative also changes the first cubic piece, so agreement
      resumes after the second centre (~ 1.5 patches).
    """
    P, n = 32, 12
    torch.manual_seed(1)
    B, ctx = 2, 3
    la = torch.randn(B, n)
    half = P // 2
    # linear: sharp edge at first centre
    a_full_l = log_amp_per_sample(la, P, "linear")
    a_seg_l = log_amp_per_sample(la[:, ctx:], P, "linear")
    sl = a_full_l[:, ctx * P :]
    assert torch.allclose(sl[:, half:], a_seg_l[:, half:], rtol=0, atol=1e-6)
    assert not torch.allclose(sl[:, :half], a_seg_l[:, :half], rtol=0, atol=1e-6)
    # pchip: match after the second centre
    a_full_p = log_amp_per_sample(la, P, "pchip")
    a_seg_p = log_amp_per_sample(la[:, ctx:], P, "pchip")
    sp = a_full_p[:, ctx * P :]
    second_c = P + half
    assert torch.allclose(sp[:, second_c:], a_seg_p[:, second_c:], rtol=0, atol=1e-6)
    assert not torch.allclose(sp[:, :half], a_seg_p[:, :half], rtol=0, atol=1e-6)
