"""Frozen PROTOCOL.yaml: load/validate, CLI conflict, norm-stats guard, smoke eval."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
PROTO = _ROOT / "results" / "protocol" / "PROTOCOL.yaml"

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "paper_figs"))

from protocol_utils import (  # noqa: E402
    assert_not_identity_stats,
    enforce_protocol,
    load_protocol,
    protocol_hash,
    resolve_norm_stats,
)


def test_protocol_loads_and_validates():
    p = load_protocol(PROTO)
    assert p["amp_mode"] == "pchip"
    assert float(p["context_fraction"]) == 0.0
    assert int(p["mean_of"]) == 1
    assert int(p["n_samples"]) == 1
    assert float(p["temperature"]) == 1.0
    assert int(p["n_steps"]) == 12
    assert int(p["seed"]) == 1234
    assert int(p["batch_size"]) == 64
    assert p["test_manifest"] == "data/n1000k/manifest.parquet"
    assert int(p["n_waveforms_flagship"]) == 5000
    assert int(p["n_waveforms_scaling"]) == 1000
    assert p["headline_metric"] == "mm_pt_full"
    assert p["_protocol_sha256"] == protocol_hash(
        {k: v for k, v in p.items() if not str(k).startswith("_")}
    )


def test_protocol_rejects_bad_amp_mode(tmp_path):
    bad = yaml.safe_load(PROTO.read_text())
    bad["amp_mode"] = "quadratic"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="amp_mode"):
        load_protocol(path)


def test_protocol_rejects_missing_key(tmp_path):
    bad = yaml.safe_load(PROTO.read_text())
    del bad["n_steps"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="missing keys"):
        load_protocol(path)


def test_conflicting_cli_flag_raises():
    proto = load_protocol(PROTO)
    args = SimpleNamespace(
        mean_of=4,  # conflicts with frozen mean_of=1
        amp_mode=None,
        manifest=None,
        split=None,
        context_fraction=None,
        n_samples=None,
        temperature=None,
        n_steps=None,
        seed=None,
        batch_size=None,
    )
    with pytest.raises(ValueError, match="conflicts with protocol"):
        enforce_protocol(args, proto, override=False)


def test_override_records_conflict():
    proto = load_protocol(PROTO)
    args = SimpleNamespace(
        mean_of=4,  # conflicts with frozen mean_of=1
        amp_mode=None,
        manifest=None,
        split=None,
        context_fraction=None,
        n_samples=None,
        temperature=None,
        n_steps=None,
        seed=None,
        batch_size=None,
    )
    ov = enforce_protocol(args, proto, override=True)
    assert "mean_of" in ov
    assert args.mean_of == 4


def test_norm_stats_guard_raises_without_hparams(tmp_path):
    ckpt = tmp_path / "fake.ckpt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    with pytest.raises(FileNotFoundError):
        resolve_norm_stats(ckpt, None)


def test_norm_stats_explicit_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_norm_stats(tmp_path / "x.ckpt", tmp_path / "missing.npz")


def test_identity_norm_stats_raises():
    ident = {
        "log_a_mean": torch.zeros(1),
        "log_a_std": torch.ones(1),
        "delta_phi_mean": torch.zeros(1),
        "delta_phi_std": torch.ones(1),
    }
    with pytest.raises(RuntimeError, match="identity"):
        assert_not_identity_stats(ident, path="fake")


@pytest.fixture(scope="module")
def smoke_env(tmp_path_factory):
    """Synthetic chirps + tiny ckpt (same plumbing as scripts/make_smoke_cf.py)."""
    import h5py
    import lightning
    import numpy as np
    import pandas as pd
    from datasets.waveform_dataset import compute_norm_stats
    from models.lightning_module import GWFlowSurrogateLit

    root = tmp_path_factory.mktemp("smoke_proto")
    data = root / "data"
    data.mkdir()
    P, NP = 32, 24
    T = NP * P + 1
    t = np.linspace(-1500, 60, T)
    rng = np.random.default_rng(0)
    n = 40
    theta = rng.uniform([1, -0.8, -0.8], [8, 0.8, 0.8], (n, 3)).astype(np.float32)
    la = np.empty((n, 1, T), np.float32)
    ph = np.empty((n, 1, T), np.float32)
    for i, (q, c1, c2) in enumerate(theta):
        tc = 30 + 3 * q
        w = 0.05 * (tc / (tc - np.minimum(t, 0))) ** (3 / 8) * (1 + 0.1 * c1)
        ph[i, 0] = np.cumsum(w) * (t[1] - t[0])
        la[i, 0] = np.log(
            (w / 0.05) ** (2 / 3) * np.exp(-np.maximum(t, 0) / 20) + 1e-6
        ) - 3
    h5 = data / "waveforms_00000.h5"
    with h5py.File(h5, "w") as f:
        f["t"] = t
        f["log_a"] = la
        f["phi"] = ph
        f["theta"] = theta
        f["ref_phase"] = np.zeros(n, np.float32)
    split = ["train"] * 20 + ["val"] * 5 + ["test"] * 15
    man = data / "manifest.parquet"
    pd.DataFrame(dict(
        file_path=str(h5.resolve()), row_index=np.arange(n), split=split,
        q=theta[:, 0], chi1z=theta[:, 1], chi2z=theta[:, 2],
    )).to_parquet(man)
    stats = data / "norm_stats.npz"
    compute_norm_stats(man, stats, P, max_samples=0)

    torch.manual_seed(0)
    lit = GWFlowSurrogateLit(
        d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
        head_width=32, head_depth=1, patch_len=P, residual_scale=0.05,
        amp_mode="constant",
    )
    run = root / "run" / "checkpoints"
    run.mkdir(parents=True)
    ckpt = run / "epoch=000-val_mismatch=0.0100.ckpt"
    torch.save(
        {
            "state_dict": lit.state_dict(),
            "hyper_parameters": dict(lit.hparams),
            "pytorch-lightning_version": lightning.__version__,
        },
        ckpt,
    )
    (run.parent / "hparams.yaml").write_text(
        yaml.safe_dump({"norm_stats_path": str(stats.resolve())})
    )

    proto = yaml.safe_load(PROTO.read_text())
    proto["test_manifest"] = str(man.resolve())
    proto["n_waveforms_flagship"] = 8
    proto["n_waveforms_scaling"] = 8
    proto["grid"]["n_time"] = int(T)
    proto["grid"]["n_patches"] = NP
    proto["grid"]["t_min"] = float(t[0])
    proto["grid"]["t_max"] = float(t[-1])
    proto["grid"]["dt"] = float(t[1] - t[0])
    proto_path = root / "PROTOCOL.yaml"
    proto_path.write_text(yaml.safe_dump(proto))
    return {
        "root": root,
        "ckpt": ckpt,
        "stats": stats,
        "manifest": man,
        "protocol": proto_path,
    }


def test_eval_fm_unified_protocol_smoke(smoke_env, tmp_path):
    out = tmp_path / "smoke.csv"
    cmd = [
        sys.executable, str(_ROOT / "eval_fm_unified.py"),
        "--protocol", str(smoke_env["protocol"]),
        "--checkpoint", str(smoke_env["ckpt"]),
        "--norm-stats", str(smoke_env["stats"]),
        "--n-waveforms", "4",
        "--batch-size", "2",
        "--n-steps", "4",
        "--device", "cpu",
        "--out", str(out),
        "--override",
    ]
    subprocess.check_call(cmd, cwd=_ROOT)
    assert out.is_file()
    import pandas as pd
    df = pd.read_csv(out)
    required = {
        "mm_pt_full", "mm_strict_full", "mm_pt_window",
        "floor_pt_full", "floor_pt_window", "dphi_peak_rad", "dt_shift_M",
        "amp_mode",
    }
    assert required.issubset(df.columns)
    side = out.with_suffix(".json")
    assert side.is_file()
    meta = json.loads(side.read_text())
    assert meta.get("protocol_sha256") or (
        isinstance(meta.get("protocol"), dict)
        and meta["protocol"].get("_protocol_sha256")
    )
    assert "overrides" in meta
    assert meta["overrides"]  # n_waveforms/batch/n_steps overridden
