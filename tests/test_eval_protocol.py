"""Correctness tests for the frozen paper evaluation protocol (Phase 3).

CPU-only, synthetic chirps via ``scripts/make_smoke_cf.py`` plumbing.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper_figs"))

from paper_figs.protocol_utils import (  # noqa: E402
    assert_grid_matches,
    assert_not_identity_stats,
    audited_training_grid,
    mismatches,
    reconstruct_fp64,
    resolve_norm_stats,
)
from models.lightning_module import GWFlowSurrogateLit  # noqa: E402


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    """Build a tiny dataset + ckpt under a temp dir (isolated from repo data/)."""
    # Re-use make_smoke_cf logic but redirect paths via cwd trick is messy;
    # inline a minimal chirp set here.
    import h5py
    import lightning
    import pandas as pd
    from datasets.waveform_dataset import compute_norm_stats

    out = tmp_path_factory.mktemp("smoke_cf")
    P, NP = 32, 24
    T = NP * P + 1
    t = np.linspace(-1500.0, 60.0, T)
    rng = np.random.default_rng(0)
    n = 40
    theta = rng.uniform([1, -0.8, -0.8], [8, 0.8, 0.8], (n, 3)).astype(np.float32)
    la = np.empty((n, 1, T), np.float32)
    ph = np.empty((n, 1, T), np.float32)
    for i, (q, c1, c2) in enumerate(theta):
        # Smooth, slowly varying chirp so the patch-constant amplitude floor
        # stays well below 1e-3 (Phase 3 round-trip requirement).
        w0 = 0.02 * (1.0 + 0.05 * q + 0.02 * c1)
        w = w0 * (1.0 + 0.0002 * np.maximum(t, -800.0))
        ph[i, 0] = np.cumsum(w) * (t[1] - t[0])
        la[i, 0] = np.log(1.0 + 0.3 * np.exp(-((t - 0.0) / 80.0) ** 2)) - 2.0
    h5 = out / "waveforms_00000.h5"
    with h5py.File(h5, "w") as f:
        f["t"] = t
        f["log_a"] = la
        f["phi"] = ph
        f["theta"] = theta
        f["ref_phase"] = np.zeros(n, np.float32)
    split = ["train"] * 20 + ["val"] * 5 + ["test"] * 15
    man = out / "manifest.parquet"
    pd.DataFrame(
        dict(
            file_path=str(h5.resolve()),
            row_index=np.arange(n),
            split=split,
            q=theta[:, 0],
            chi1z=theta[:, 1],
            chi2z=theta[:, 2],
        )
    ).to_parquet(man)
    stats = out / "frozen_norm_stats.npz"
    compute_norm_stats(man, stats, P, max_samples=0)

    torch.manual_seed(0)
    lit = GWFlowSurrogateLit(
        d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, head_width=32,
        head_depth=1, patch_len=P, residual_scale=0.05, use_abs_time=False,
    )
    run = out / "run" / "seed42" / "checkpoints"
    run.mkdir(parents=True)
    torch.save(
        {
            "state_dict": lit.state_dict(),
            "hyper_parameters": dict(lit.hparams),
            "pytorch-lightning_version": lightning.__version__,
        },
        run / "last.ckpt",
    )
    (run.parent / "hparams.yaml").write_text(
        yaml.safe_dump({"norm_stats_path": str(stats)})
    )
    return {
        "dir": out,
        "manifest": man,
        "stats": stats,
        "ckpt": run / "last.ckpt",
        "lit": lit,
        "P": P,
        "T": T,
        "t": t,
        "la": la,
        "ph": ph,
    }


def _batch_from_smoke(smoke, n=4):
    from datasets.datamodule import _collate_batch
    from datasets.waveform_dataset import WaveformPatchDataset

    ds = WaveformPatchDataset(
        smoke["manifest"], split="test", patch_len=smoke["P"],
        context_patches=1, norm_stats_path=str(smoke["stats"]), preload=True,
    )
    samples = [ds[i] for i in range(n)]
    return _collate_batch(samples), ds


def test_round_trip_reconstruct(smoke):
    lit = smoke["lit"]
    batch, ds = _batch_from_smoke(smoke)
    batch = lit._patchify_batch(batch)
    P = smoke["P"]
    feat = batch["token_features"]
    usable = feat.shape[1] * P
    h_fp64 = reconstruct_fp64(feat, batch, float(lit.hparams.residual_scale), P)
    # lit path (fp32)
    la = batch["log_a"][:, :, 0]
    dp = batch["delta_phi"][:, :, 0]
    res = batch["phi_residual"][:, :, 0]
    h_fp32 = lit._reconstruct_waveform(
        la, dp, res,
        batch["norm_log_a_mean"], batch["norm_log_a_std"],
        batch["norm_dphi_mean"], batch["norm_dphi_std"], P,
    )
    h_fp32_c = h_fp32.to(torch.complex128)
    rel = (h_fp64 - h_fp32_c).abs() / h_fp32_c.abs().clamp_min(1e-30)
    assert float(rel.max()) < 1e-5

    # vs original tracks
    la_o = batch["log_a_track"][:, 0, :usable].double()
    ph_o = batch["phi_track"][:, 0, :usable].double()
    h_orig = torch.exp(la_o) * torch.exp(-1j * (ph_o - ph_o[:, :1]))
    _, _, pt, _ = mismatches(h_orig, h_fp64)
    assert float(pt.max()) < 1e-3


def test_perfect_prediction_equals_floor(smoke):
    lit = smoke["lit"]
    batch, _ = _batch_from_smoke(smoke)
    batch = lit._patchify_batch(batch)
    P = smoke["P"]
    feat = batch["token_features"]
    usable = feat.shape[1] * P
    h_tok = reconstruct_fp64(feat, batch, float(lit.hparams.residual_scale), P)
    la_o = batch["log_a_track"][:, 0, :usable].double()
    ph_o = batch["phi_track"][:, 0, :usable].double()
    h_orig = torch.exp(la_o) * torch.exp(-1j * (ph_o - ph_o[:, :1]))
    ctx = 4
    s = ctx * P
    # "generated" = true tokens
    h_pred = h_tok
    _, _, mm_tok, _ = mismatches(h_tok[:, s:], h_pred[:, s:])
    _, _, floor, _ = mismatches(h_orig[:, s:], h_tok[:, s:])
    _, _, mm_gen, _ = mismatches(h_orig[:, s:], h_pred[:, s:])
    assert float(mm_tok.max()) < 1e-12
    assert torch.allclose(mm_gen, floor, atol=1e-12)


def test_mismatch_sanity():
    B, T = 3, 128
    t = torch.linspace(0, 1, T).double()
    h = torch.exp(1j * 20 * t).unsqueeze(0).expand(B, -1).contiguous()
    s0, p0, pt0, sh0 = mismatches(h, h)
    assert torch.allclose(s0, torch.zeros(B, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(p0, torch.zeros(B, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(pt0, torch.zeros(B, dtype=torch.float64), atol=1e-12)

    rot = h * torch.exp(torch.tensor(0.7j))
    s, p, pt, _ = mismatches(h, rot)
    assert float(s.min()) > 1e-6
    assert float(p.max()) < 1e-10
    assert float(pt.max()) < 1e-10

    k = 5
    shifted = torch.roll(h, shifts=k, dims=1)
    # circular roll is not a pure delay for FFT mismatch on finite support;
    # use zero-padded delay instead
    delayed = torch.zeros_like(h)
    delayed[:, k:] = h[:, :-k]
    # compare on the overlapping interior
    _, _, pt2, sh = mismatches(h[:, k:], delayed[:, k:])
    assert float(pt2.max()) < 1e-8
    assert int(sh[0].item()) == 0  # already aligned windows


def test_norm_stats_guard(smoke, tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_norm_stats(smoke["ckpt"], explicit=str(tmp_path / "missing.npz"))

    # hparams resolution works
    p = resolve_norm_stats(smoke["ckpt"], explicit=None)
    assert p == smoke["stats"].resolve() or p == Path(smoke["stats"]).resolve()

    # identity raises
    ident = {
        "log_a_mean": torch.zeros(1),
        "log_a_std": torch.ones(1),
        "delta_phi_mean": torch.zeros(1),
        "delta_phi_std": torch.ones(1),
    }
    with pytest.raises(RuntimeError, match="identity"):
        assert_not_identity_stats(ident, path="fake")


def test_grid_guard():
    t = audited_training_grid()
    assert_grid_matches(t, t)
    bad = t.copy()
    bad[-1] += 1.0
    with pytest.raises(RuntimeError, match="time grid mismatch"):
        assert_grid_matches(bad, t)


def test_determinism_seed(smoke):
    """Two generate calls with the same seed+batch_index agree."""
    lit = smoke["lit"].eval()
    batch, _ = _batch_from_smoke(smoke, n=2)
    batch = lit._patchify_batch(batch)
    feat = batch["token_features"]
    ctx = 2
    outs = []
    for _ in range(2):
        torch.manual_seed(1234 + 0)
        with torch.no_grad():
            g = lit.model.generate(
                feat[:, :ctx], batch["theta"], feat.shape[1] - ctx,
                n_steps=5, n_samples=1, temperature=1.0, mean_of=2,
            )
        outs.append(g.clone())
    assert torch.equal(outs[0], outs[1])


@pytest.mark.skip(reason="GPU real-data regression — run manually once (Phase 3.7)")
def test_real_data_regression_placeholder():
    assert False
