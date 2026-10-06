"""Tests for train-split normalization statistics (§3.5)."""

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_generation.config import GenerationConfig
from data_generation.generate import MANIFEST_SCHEMA, compute_norm_stats


def _write_synthetic_shard(tmp_path: Path, rows: list[dict]) -> None:
    h5_path = tmp_path / "waveforms_00000.h5"
    c, t = 1, 16
    n = len(rows)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("log_a", data=np.random.randn(n, c, t).astype(np.float32))
        f.create_dataset("phi", data=np.cumsum(np.random.randn(n, c, t).astype(np.float32) * 0.1, axis=-1))
    manifest_rows = []
    for i, row in enumerate(rows):
        manifest_rows.append(
            {
                "file_path": str(h5_path.resolve()),
                "row_index": i,
                "split": row["split"],
                "q": 2.0,
                "chi1z": 0.1,
                "chi2z": -0.1,
                "peak_index": 8,
                "ref_phase": 0.0,
                "valid": True,
                "generation_timestamp": 0.0,
            }
        )
    pq.write_table(pa.Table.from_pylist(manifest_rows, schema=MANIFEST_SCHEMA), tmp_path / "manifest.parquet")


def test_compute_norm_stats_train_only(tmp_path):
    _write_synthetic_shard(
        tmp_path,
        [{"split": "train"}, {"split": "train"}, {"split": "val"}, {"split": "test"}],
    )
    cfg = GenerationConfig(output_dir=str(tmp_path), dt=1.0, t_min=-10.0, t_max=5.0)
    out = compute_norm_stats(tmp_path, cfg)
    assert out.name == "norm_stats.json"
    stats = json.loads(out.read_text())
    assert stats["modes"] == [[2, 2]]
    assert len(stats["log_a"]["mean"]) == 1
    assert len(stats["dphi"]["std"]) == 1
    assert stats["log_a"]["std"][0] >= 1e-8
