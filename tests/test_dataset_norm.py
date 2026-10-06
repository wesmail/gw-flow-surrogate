"""Dataset normalization: log_a and Δφ standardized; residual stays in radians."""

import numpy as np
import torch

from datasets.datamodule import _collate_batch
from datasets.waveform_dataset import WaveformPatchDataset
from models.lightning_module import GWSurrogateLit


def test_delta_phi_not_normalized(tmp_path):
    """Δφ targets are standardized; residuals stay in radians."""
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq

    from data_generation.generate import MANIFEST_SCHEMA

    h5 = tmp_path / "w.h5"
    c, t = 1, 65
    with h5py.File(h5, "w") as f:
        f.create_dataset("log_a", data=np.zeros((2, c, t), dtype=np.float32))
        f.create_dataset("phi", data=np.cumsum(np.ones((2, c, t), dtype=np.float32) * 0.3, axis=-1))
        f.create_dataset("theta", data=np.zeros((2, 3), dtype=np.float32))
        f.create_dataset("ref_phase", data=np.zeros(2, dtype=np.float32))

    rows = [
        {"file_path": str(h5), "row_index": 0, "split": "train", "q": 2.0, "chi1z": 0.0, "chi2z": 0.0,
         "peak_index": 0, "ref_phase": 0.0, "valid": True, "generation_timestamp": 0.0},
        {"file_path": str(h5), "row_index": 1, "split": "val", "q": 2.0, "chi1z": 0.0, "chi2z": 0.0,
         "peak_index": 0, "ref_phase": 0.0, "valid": True, "generation_timestamp": 0.0},
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA), tmp_path / "manifest.parquet")
    stats_path = tmp_path / "norm_stats.npz"
    np.savez(
        stats_path,
        log_a_mean=np.zeros(c),
        log_a_std=np.ones(c),
        delta_phi_mean=np.array([0.5]),
        delta_phi_std=np.array([0.1]),
    )

    ds = WaveformPatchDataset(
        tmp_path / "manifest.parquet",
        split="train",
        patch_len=32,
        norm_stats_path=stats_path,
        preload=True,
    )
    batch = _collate_batch([ds[0]])
    lit = GWSurrogateLit(patch_len=32)
    batch = lit._patchify_batch(batch)
    assert batch["phi_residual"].abs().max() > 0.01
    assert batch["delta_phi"].abs().max() > 0.2
