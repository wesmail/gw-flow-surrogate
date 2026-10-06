"""NRHybSur3dq8 waveform evaluation and HDF5 sharded output."""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_generation.config import (
    STORAGE_FORMAT,
    GenerationConfig,
    NRHYBSUR_T_MAX_M,
    config_to_dict,
)
from data_generation.sampling import sobol_parameters
from data_generation.waveform_utils import (
    align_time_to_peak,
    gauge_fix_orbital_phase,
    modes_to_arrays,
    validate_waveform,
)

log = logging.getLogger(__name__)

_SURROGATE = None


@contextmanager
def _suppress_surrogate_domain_warnings(enabled: bool = True):
    """Hide NRHybSur extrapolation UserWarnings (intentional for OOD samples)."""
    if not enabled:
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*outside training.*",
            category=UserWarning,
        )
        yield

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


def _load_surrogate(name: str = "NRHybSur3dq8"):
    global _SURROGATE
    if _SURROGATE is None:
        import gwsurrogate

        try:
            gwsurrogate.catalog.pull(name)
        except Exception as exc:
            log.warning("catalog.pull failed (%s); trying LoadSurrogate.", exc)
        _SURROGATE = gwsurrogate.LoadSurrogate(name)
        log.info("Loaded %s in pid %d", name, os.getpid())
    return _SURROGATE


@dataclass
class WaveformRecord:
    log_a: np.ndarray       # (C, T)
    phi: np.ndarray         # (C, T) — gauge-fixed; phi_ref_mode(t_ref) ≈ 0
    h: np.ndarray           # (C, T) complex64
    theta: np.ndarray       # (3,) = q, chi1z, chi2z
    peak_index: int
    # Applied orbital-phase rotation α (§3.4), not raw φ₂₂(t_ref). §7 reconstruction
    # should cumsum increments from 0 — stored phi already starts in this gauge.
    ref_phase: float


def evaluate_waveform(
    q: float,
    chi1z: float,
    chi2z: float,
    cfg: GenerationConfig,
) -> WaveformRecord:
    """Generate one aligned, gauge-fixed waveform in amplitude–phase form."""
    sur = _load_surrogate(cfg.surrogate_name)
    chiA0 = [0.0, 0.0, float(chi1z)]
    chiB0 = [0.0, 0.0, float(chi2z)]
    mode_list = list(cfg.modes)

    t_eval_max = min(float(cfg.t_max), NRHYBSUR_T_MAX_M)
    eval_mask = (cfg.times >= cfg.t_min) & (cfg.times <= t_eval_max)
    times_eval = cfg.times[eval_mask]

    with _suppress_surrogate_domain_warnings(cfg.suppress_surrogate_warnings):
        _, h_modes_dict, _ = sur(
            float(q),
            chiA0,
            chiB0,
            times=times_eval,
            f_low=0,
            mode_list=mode_list,
        )

    # Pad to full grid.
    h_modes: dict[tuple[int, int], np.ndarray] = {}
    for lm in cfg.modes:
        h_eval = np.asarray(h_modes_dict[lm], dtype=np.complex128)
        h_full = np.zeros(cfg.t_full, dtype=np.complex128)
        h_full[eval_mask] = h_eval
        if eval_mask[0]:
            pass
        else:
            first_valid = int(np.argmax(eval_mask))
            h_full[:first_valid] = h_full[first_valid]
        if eval_mask[-1]:
            pass
        else:
            last_valid = int(len(eval_mask) - 1 - np.argmax(eval_mask[::-1]))
            h_full[last_valid + 1 :] = h_full[last_valid]
        h_modes[lm] = h_full

    times_grid, h_aligned, peak_idx = align_time_to_peak(cfg.times, h_modes)
    h_gauge, alpha = gauge_fix_orbital_phase(
        h_aligned, times_grid, cfg.t_ref, ref_mode=cfg.modes[0]
    )
    validate_waveform(h_gauge, times_grid, cfg)

    log_a, phi, h_arr = modes_to_arrays(h_gauge, cfg.modes, cfg.log_a_eps)
    theta = np.array([q, chi1z, chi2z], dtype=np.float32)
    return WaveformRecord(
        log_a=log_a,
        phi=phi,
        h=h_arr,
        theta=theta,
        peak_index=peak_idx,
        ref_phase=float(alpha),
    )


def _write_shard(
    file_index: int,
    cfg: GenerationConfig,
    params: np.ndarray,
    splits: list[str],
) -> Path:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / f"waveforms_{file_index:05d}.h5"
    manifest_path = out_dir / f"waveforms_{file_index:05d}.manifest.parquet"

    n = params.shape[0]
    c, t = cfg.n_modes, cfg.t_full
    log_a_buf = np.empty((n, c, t), dtype=np.float32)
    phi_buf = np.empty((n, c, t), dtype=np.float32)
    h_re_buf = np.empty((n, c, t), dtype=np.float32)
    h_im_buf = np.empty((n, c, t), dtype=np.float32)
    theta_buf = np.empty((n, 3), dtype=np.float32)
    peak_buf = np.empty(n, dtype=np.int32)
    ref_phase_buf = np.empty(n, dtype=np.float32)

    manifest_rows: list[dict] = []
    timestamp = time.time()
    n_stored = 0

    for row_idx in range(n):
        q, chi1z, chi2z = params[row_idx]
        try:
            rec = evaluate_waveform(q, chi1z, chi2z, cfg)
        except Exception as exc:
            log.warning(
                "Row %d failed (q=%.2f, chi1z=%.2f, chi2z=%.2f): %s",
                row_idx, q, chi1z, chi2z, exc,
            )
            continue

        log_a_buf[n_stored] = rec.log_a
        phi_buf[n_stored] = rec.phi
        h_re_buf[n_stored] = rec.h.real
        h_im_buf[n_stored] = rec.h.imag
        theta_buf[n_stored] = rec.theta
        peak_buf[n_stored] = rec.peak_index
        ref_phase_buf[n_stored] = rec.ref_phase
        manifest_rows.append(
            {
                "file_path": h5_path.name,
                "row_index": n_stored,
                "split": splits[row_idx],
                "q": float(q),
                "chi1z": float(chi1z),
                "chi2z": float(chi2z),
                "peak_index": rec.peak_index,
                "ref_phase": rec.ref_phase,
                "valid": True,
                "generation_timestamp": timestamp,
            }
        )
        n_stored += 1

    with h5py.File(h5_path, "w", libver="latest") as f:
        f.attrs["storage_format"] = STORAGE_FORMAT
        f.attrs["surrogate"] = cfg.surrogate_name
        f.attrs["dt_M"] = cfg.dt
        f.attrs["t_min"] = cfg.t_min
        f.attrs["t_max"] = cfg.t_max
        f.attrs["t_ref"] = cfg.t_ref
        f.attrs["modes"] = json.dumps([list(m) for m in cfg.modes])
        f.attrs["n_waveforms"] = n_stored
        f.attrs["generation_config"] = json.dumps(config_to_dict(cfg))
        f.create_dataset("t", data=cfg.times.astype(np.float32))
        if n_stored > 0:
            chunk = min(64, n_stored)
            f.create_dataset("log_a", data=log_a_buf[:n_stored], chunks=(chunk, c, t))
            f.create_dataset("phi", data=phi_buf[:n_stored], chunks=(chunk, c, t))
            f.create_dataset("h_re", data=h_re_buf[:n_stored], chunks=(chunk, c, t))
            f.create_dataset("h_im", data=h_im_buf[:n_stored], chunks=(chunk, c, t))
            f.create_dataset("theta", data=theta_buf[:n_stored])
            f.create_dataset("peak_index", data=peak_buf[:n_stored])
            f.create_dataset("ref_phase", data=ref_phase_buf[:n_stored])

    if manifest_rows:
        pq.write_table(
            pa.Table.from_pylist(manifest_rows, schema=MANIFEST_SCHEMA),
            manifest_path,
            compression="snappy",
        )

    log.info("Shard %05d: %d/%d stored → %s", file_index, n_stored, n, h5_path.name)
    return manifest_path


def combine_manifests(output_dir: str | Path, combined_name: str = "manifest.parquet") -> Path:
    output_dir = Path(output_dir)
    parts = sorted(output_dir.glob("*.manifest.parquet"))
    if not parts:
        raise FileNotFoundError(f"No manifest shards in {output_dir}")
    combined = pa.concat_tables([pq.read_table(p) for p in parts])
    out = output_dir / combined_name
    pq.write_table(combined, out, compression="snappy")
    log.info("Combined manifest: %d rows → %s", combined.num_rows, out)
    return out


class _StreamingChannelStats:
    """Numerically stable per-channel mean/std over scalar streams."""

    def __init__(self, n_channels: int) -> None:
        self.count = np.zeros(n_channels, dtype=np.int64)
        self.sum = np.zeros(n_channels, dtype=np.float64)
        self.sum_sq = np.zeros(n_channels, dtype=np.float64)

    def update(self, channel: int, values: np.ndarray) -> None:
        v = np.asarray(values, dtype=np.float64).ravel()
        if v.size == 0:
            return
        self.count[channel] += v.size
        self.sum[channel] += v.sum()
        self.sum_sq[channel] += np.dot(v, v)

    def finalize(self) -> tuple[list[float], list[float]]:
        mean_out: list[float] = []
        std_out: list[float] = []
        for c in range(len(self.count)):
            if self.count[c] == 0:
                mean_out.append(0.0)
                std_out.append(1.0)
                continue
            mean = self.sum[c] / self.count[c]
            var = self.sum_sq[c] / self.count[c] - mean * mean
            std = max(float(np.sqrt(max(var, 0.0))), 1e-8)
            mean_out.append(float(mean))
            std_out.append(std)
        return mean_out, std_out


def compute_norm_stats(output_dir: str | Path, cfg: GenerationConfig) -> Path:
    """Compute train-split-only normalization stats and write norm_stats.json (§3.5)."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    table = pq.read_table(manifest_path)
    train = table.filter(pc.equal(table.column("split"), pa.scalar("train")))
    if train.num_rows == 0:
        raise ValueError("No training rows found in manifest; cannot compute norm_stats.")

    # One HDF5 open + full-dataset read per shard (avoids per-row chunk re-decompression).
    by_file: dict[str, list[int]] = defaultdict(list)
    for fp, ri in zip(
        train.column("file_path").to_pylist(),
        train.column("row_index").to_pylist(),
    ):
        by_file[fp].append(int(ri))

    n_ch = cfg.n_modes
    log_a_stats = _StreamingChannelStats(n_ch)
    dphi_stats = _StreamingChannelStats(n_ch)

    for file_path, row_indices in by_file.items():
        indices = np.asarray(row_indices, dtype=np.int64)
        shard = Path(file_path)
        if not shard.is_file():
            shard = output_dir / shard.name
        with h5py.File(shard, "r") as f:
            log_a_shard = f["log_a"][:]
            phi_shard = f["phi"][:]
        log_a_rows = log_a_shard[indices]   # (N, C, T)
        phi_rows = phi_shard[indices]
        for c in range(n_ch):
            log_a_stats.update(c, log_a_rows[:, c, :].ravel())
            dphi_stats.update(c, np.diff(phi_rows[:, c, :], axis=-1).ravel())

    log_a_mean, log_a_std = log_a_stats.finalize()
    dphi_mean, dphi_std = dphi_stats.finalize()

    payload = {
        "modes": [list(m) for m in cfg.modes],
        "log_a": {"mean": log_a_mean, "std": log_a_std},
        "dphi": {"mean": dphi_mean, "std": dphi_std},
    }
    out_path = output_dir / "norm_stats.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log.info(
        "Wrote norm_stats.json from %d train rows (%d shards, %d channels) → %s",
        train.num_rows,
        len(by_file),
        n_ch,
        out_path,
    )
    return out_path


def generate_dataset(
    cfg: GenerationConfig,
    *,
    n_train: int | None = None,
    val_fraction: float = 0.1,
    n_test: int = 1_000,
    n_ood: int = 500,
    jobs: int = 1,
) -> Path:
    """Generate sharded HDF5 dataset with Sobol train/val and uniform test/OOD."""
    if n_train is None:
        n_train = cfg.n_files * cfg.waveforms_per_file

    n_val = max(1, int(n_train * val_fraction))
    n_sobol = n_train + n_val

    params_sobol = sobol_parameters(cfg, n_sobol, seed=cfg.seed, skip=256)
    rng = np.random.default_rng(cfg.seed + 1)
    from data_generation.sampling import extrapolation_parameters, uniform_parameters

    params_test = uniform_parameters(cfg, n_test, rng)
    params_ood = extrapolation_parameters(n_ood, rng)
    all_params = np.vstack([params_sobol, params_test, params_ood])

    splits = (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * n_test
        + ["ood"] * n_ood
    )

    # Distribute across shards.
    total = all_params.shape[0]
    per_file = int(np.ceil(total / cfg.n_files))
    for file_idx in range(cfg.n_files):
        start = file_idx * per_file
        end = min(start + per_file, total)
        if start >= total:
            break
        _write_shard(
            file_idx,
            cfg,
            all_params[start:end],
            splits[start:end],
        )

    manifest_path = combine_manifests(cfg.output_dir)
    compute_norm_stats(cfg.output_dir, cfg)
    return manifest_path
