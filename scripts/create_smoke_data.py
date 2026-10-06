#!/usr/bin/env python
"""Create a minimal HDF5 + manifest for configs/stage1_smoke.yaml."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MANIFEST_SCHEMA = pa.schema(
    [
        ("file_path", pa.string()),
        ("row_index", pa.int64()),
        ("split", pa.string()),
        ("q", pa.float64()),
        ("chi1z", pa.float64()),
        ("chi2z", pa.float64()),
        ("peak_index", pa.int64()),
        ("ref_phase", pa.float64()),
        ("valid", pa.bool_()),
        ("generation_timestamp", pa.float64()),
    ]
)

OUT = Path("data/smoke")
C, T = 1, 321  # 10 patches at P=32 with boundary increments
N = 8


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    h5_path = OUT / "waveforms_00000.h5"
    rng = np.random.default_rng(0)

    log_a = rng.standard_normal((N, C, T)).astype(np.float32) * 0.1 - 20
    phi = np.cumsum(rng.standard_normal((N, C, T)).astype(np.float32) * 0.05, axis=-1)
    h = np.exp(log_a) * np.exp(-1j * phi)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("t", data=np.linspace(-5000, 130, T, dtype=np.float32))
        f.create_dataset("log_a", data=log_a)
        f.create_dataset("phi", data=phi.astype(np.float32))
        f.create_dataset("h_re", data=h.real.astype(np.float32))
        f.create_dataset("h_im", data=h.imag.astype(np.float32))
        f.create_dataset("theta", data=rng.uniform([1, -0.8, -0.8], [8, 0.8, 0.8], (N, 3)).astype(np.float32))
        f.create_dataset("peak_index", data=np.full(N, T // 2, dtype=np.int32))
        f.create_dataset("ref_phase", data=np.zeros(N, dtype=np.float32))

    splits = ["train"] * 6 + ["val"] * 1 + ["test"] * 1
    rows = []
    for i, split in enumerate(splits):
        rows.append(
            {
                "file_path": str(h5_path.resolve()),
                "row_index": i,
                "split": split,
                "q": 2.0,
                "chi1z": 0.1,
                "chi2z": -0.1,
                "peak_index": T // 2,
                "ref_phase": 0.0,
                "valid": True,
                "generation_timestamp": 0.0,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA), OUT / "manifest.parquet")
    print(f"Wrote smoke data → {OUT}")


if __name__ == "__main__":
    main()
