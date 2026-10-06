"""Configuration for NRHybSur waveform dataset generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

# NRHybSur3dq8 geometric-time upper bound (M).
NRHYBSUR_T_MAX_M = 134.0
STORAGE_FORMAT = "gw_surrogate_v1"

# Stage-1 default: dominant (2,2) mode only.
DEFAULT_MODES: tuple[tuple[int, int], ...] = ((2, 2),)


def time_grid(t_min: float, t_max: float, dt: float) -> np.ndarray:
    """Uniform geometric-time samples t/M on [t_min, t_max], inclusive."""
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    n = int(np.floor((t_max - t_min) / dt)) + 1
    times = t_min + dt * np.arange(n, dtype=np.float64)
    if times[-1] < t_max - 1e-9:
        times = np.append(times, t_max)
    return times.astype(np.float64)


@dataclass
class GenerationConfig:
    """Hyperparameters for sharded HDF5 waveform generation."""

    # Intrinsic parameter box (NRHybSur3dq8 training region, Assumption A2).
    q_min: float = 1.0
    q_max: float = 8.0
    chi_min: float = -0.8
    chi_max: float = 0.8

    # Geometric time window (§3.3): merger at t=0 after alignment.
    t_min: float = -5000.0
    t_max: float = 130.0
    dt: float = 0.5
    t_ref: float = -5000.0  # phase reference time (§3.4)

    modes: tuple[tuple[int, int], ...] = DEFAULT_MODES
    surrogate_name: str = "NRHybSur3dq8"
    log_a_eps: float = 1e-30

    # Sharding
    waveforms_per_file: int = 2_000
    n_files: int = 32
    output_dir: str = "data/generated"
    seed: int = 42
    # Silence gwsurrogate "outside training range" UserWarnings (expected for OOD).
    suppress_surrogate_warnings: bool = True

    # Derived geometry (set in __post_init__).
    times: np.ndarray = field(init=False, repr=False)
    t_full: int = field(init=False)

    def __post_init__(self) -> None:
        self.times = time_grid(self.t_min, self.t_max, self.dt)
        self.t_full = int(self.times.shape[0])
        if self.t_max > NRHYBSUR_T_MAX_M:
            raise ValueError(
                f"t_max={self.t_max}M exceeds NRHybSur3dq8 domain ({NRHYBSUR_T_MAX_M}M)."
            )

    @property
    def n_modes(self) -> int:
        return len(self.modes)


def config_to_dict(cfg: GenerationConfig) -> dict:
    d = asdict(cfg)
    d["times"] = np.asarray(d["times"]).tolist()
    d["modes"] = [list(m) for m in d["modes"]]
    return d


def config_from_dict(d: dict) -> GenerationConfig:
    skip = {"times", "t_full"}
    modes = d.get("modes")
    if modes is not None:
        modes = tuple(tuple(m) for m in modes)
    base = {k: v for k, v in d.items() if k not in skip}
    if modes is not None:
        base["modes"] = modes
    return GenerationConfig(**base)


def load_config_from_h5(h5_path: str) -> GenerationConfig | None:
    """Read generation config stored in an HDF5 shard attribute."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        raw = f.attrs.get("generation_config")
        if raw is None:
            return None
        text = raw.decode() if isinstance(raw, bytes) else str(raw)
        return config_from_dict(json.loads(text))
