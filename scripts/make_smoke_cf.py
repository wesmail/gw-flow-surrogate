"""Synthetic chirp data + an untrained tiny checkpoint, to smoke-test the
context-fraction script end to end (CPU). Not physics, just plumbing."""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, ".")
from datasets.waveform_dataset import compute_norm_stats  # noqa: E402
from models.lightning_module import GWFlowSurrogateLit  # noqa: E402

out = Path("data/smoke_cf"); out.mkdir(parents=True, exist_ok=True)
P, NP = 32, 24
T = NP * P + 1
t = np.linspace(-1500, 60, T)
rng = np.random.default_rng(0)
n = 120
theta = rng.uniform([1, -0.8, -0.8], [8, 0.8, 0.8], (n, 3)).astype(np.float32)
la = np.empty((n, 1, T), np.float32); ph = np.empty((n, 1, T), np.float32)
for i, (q, c1, c2) in enumerate(theta):
    tc = 30 + 3 * q
    w = 0.05 * (tc / (tc - np.minimum(t, 0))) ** (3 / 8) * (1 + 0.1 * c1)
    ph[i, 0] = np.cumsum(w) * (t[1] - t[0])
    la[i, 0] = np.log((w / 0.05) ** (2 / 3) * np.exp(-np.maximum(t, 0) / 20) + 1e-6) - 3
h5 = out / "waveforms_00000.h5"
with h5py.File(h5, "w") as f:
    f["t"] = t; f["log_a"] = la; f["phi"] = ph
    f["theta"] = theta; f["ref_phase"] = np.zeros(n, np.float32)
split = ["train"] * 80 + ["val"] * 10 + ["test"] * 30
pd.DataFrame(dict(file_path=str(h5.resolve()), row_index=np.arange(n), split=split,
                  q=theta[:, 0], chi1z=theta[:, 1], chi2z=theta[:, 2])).to_parquet(out / "manifest.parquet")
stats = Path("data/smoke_frozen_norm_stats.npz")       # deliberately NOT next to the manifest
compute_norm_stats(out / "manifest.parquet", out / "norm_stats.npz", P, max_samples=0)
(out / "norm_stats.npz").rename(stats)

torch.manual_seed(0)
lit = GWFlowSurrogateLit(d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, head_width=32,
                         head_depth=1, patch_len=P, residual_scale=0.05)
run = Path("runs/smoke/seed42/checkpoints"); run.mkdir(parents=True, exist_ok=True)
import lightning
torch.save({"state_dict": lit.state_dict(), "hyper_parameters": dict(lit.hparams),
            "pytorch-lightning_version": lightning.__version__}, run / "last.ckpt")
(run.parent / "hparams.yaml").write_text(yaml.safe_dump({"norm_stats_path": str(stats)}))
print("ok")
