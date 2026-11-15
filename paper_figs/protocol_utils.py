"""Shared helpers for the frozen paper evaluation protocol.

Used by ``eval_paper.py``, ``fig_context_fraction.py``, and (after port)
other figure/eval scripts. No silent fallbacks: missing files and identity
norm stats raise.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from models.amp_interp import AMP_MODES

# Keys that must be present in PROTOCOL.yaml (paper freeze).
REQUIRED_PROTOCOL_KEYS = (
    "test_manifest",
    "split",
    "n_waveforms_flagship",
    "n_waveforms_scaling",
    "context_fraction",
    "amp_mode",
    "mean_of",
    "n_samples",
    "temperature",
    "n_steps",
    "seed",
    "batch_size",
    "headline_metric",
    "window_start_fraction",
    "target_mismatch",
    "grid",
)

# CLI / Namespace attribute → protocol key (for conflict checks).
PROTOCOL_CLI_MAP = {
    "manifest": "test_manifest",
    "test_manifest": "test_manifest",
    "split": "split",
    "context_fraction": "context_fraction",
    "amp_mode": "amp_mode",
    "mean_of": "mean_of",
    "n_samples": "n_samples",
    "temperature": "temperature",
    "n_steps": "n_steps",
    "seed": "seed",
    "batch_size": "batch_size",
}


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def protocol_hash(protocol: dict[str, Any]) -> str:
    """Stable hash of the protocol mapping (canonical JSON)."""
    blob = json.dumps(protocol, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def load_protocol(path: Path | str) -> dict[str, Any]:
    """Load and validate ``PROTOCOL.yaml``. Raises on missing/invalid fields."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"protocol file not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"protocol must be a mapping: {path}")

    missing = [k for k in REQUIRED_PROTOCOL_KEYS if k not in data]
    if missing:
        raise ValueError(f"protocol {path} missing keys: {missing}")

    amp = data["amp_mode"]
    if amp not in AMP_MODES:
        raise ValueError(f"amp_mode must be one of {AMP_MODES}, got {amp!r}")

    if data["n_steps"] is None or int(data["n_steps"]) < 1:
        raise ValueError(f"n_steps must be a positive int, got {data['n_steps']!r}")
    if data["context_fraction"] is None:
        raise ValueError("context_fraction must be set (0 = theta-only)")
    cf = float(data["context_fraction"])
    if not (0.0 <= cf <= 1.0):
        raise ValueError(f"context_fraction must be in [0, 1], got {cf}")

    for key in ("mean_of", "n_samples", "seed", "batch_size",
                "n_waveforms_flagship", "n_waveforms_scaling"):
        if int(data[key]) < 1:
            raise ValueError(f"{key} must be >= 1, got {data[key]!r}")

    wsf = float(data["window_start_fraction"])
    if not (0.0 <= wsf < 1.0):
        raise ValueError(f"window_start_fraction must be in [0, 1), got {wsf}")

    grid = data["grid"]
    for gk in ("t_min", "t_max", "dt", "n_time", "patch_len", "n_patches"):
        if gk not in grid:
            raise ValueError(f"protocol grid missing {gk!r}")

    data["_protocol_path"] = str(path.resolve())
    data["_protocol_sha256"] = protocol_hash(
        {k: v for k, v in data.items() if not str(k).startswith("_")}
    )
    return data


def context_patches_from_fraction(n_patches: int, context_fraction: float) -> int:
    """Map protocol ``context_fraction`` → seed patch count (0 → 1 = theta-only)."""
    if float(context_fraction) <= 0.0:
        return 1
    return max(1, int(round(float(context_fraction) * int(n_patches))))


def _canon(pkey: str, value: Any) -> Any:
    if value is None:
        return None
    if pkey in ("test_manifest",):
        return str(Path(value))
    if pkey in ("context_fraction", "temperature", "window_start_fraction"):
        return float(value)
    if pkey in (
        "mean_of", "n_samples", "n_steps", "seed", "batch_size",
        "n_waveforms_flagship", "n_waveforms_scaling",
    ):
        return int(value)
    return value


def enforce_protocol(
    args: Any,
    protocol: dict[str, Any],
    *,
    override: bool = False,
    cli_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Apply protocol knobs to ``args``.

    CLI attributes that are ``None`` take the protocol value.  A non-None CLI
    value that differs from the protocol raises unless ``override`` is True
    (then the CLI value is kept and recorded).  Returns the overrides dict.
    """
    mapping = cli_map or PROTOCOL_CLI_MAP
    overrides: dict[str, Any] = {}
    for attr, pkey in mapping.items():
        if not hasattr(args, attr) or pkey not in protocol:
            continue
        want = protocol[pkey]
        have = getattr(args, attr)
        if have is None:
            # Map protocol test_manifest onto --manifest when that is the attr.
            setattr(args, attr, want)
            continue
        if _canon(pkey, have) == _canon(pkey, want):
            setattr(args, attr, want)
            continue
        if not override:
            raise ValueError(
                f"--{attr.replace('_', '-')}={have!r} conflicts with protocol "
                f"{pkey}={want!r}. Pass --override to allow (recorded in sidecar)."
            )
        overrides[attr] = {"cli": have, "protocol": want}
        # keep CLI value under override
    return overrides


def resolve_norm_stats(checkpoint: Path | str, explicit: str | Path | None = None) -> Path:
    """Explicit path > run's ``hparams.yaml`` ``norm_stats_path`` > hard failure.

    Never falls back to identity statistics or to the *test* manifest folder.
    """
    if explicit is not None:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"--norm-stats {p} does not exist")
        return p.resolve()

    ckpt = Path(checkpoint)
    for hp in (ckpt.parent.parent / "hparams.yaml", ckpt.parent / "hparams.yaml"):
        if not hp.is_file():
            continue
        d = yaml.safe_load(hp.read_text()) or {}
        cand = d.get("norm_stats_path") or (d.get("data") or {}).get("norm_stats_path")
        if cand in (None, "null", "None", ""):
            raise FileNotFoundError(
                f"{hp} has norm_stats_path={cand!r}. Pass --norm-stats explicitly "
                "(scan_1M runs computed stats next to the remote train manifest)."
            )
        p = Path(cand)
        if not p.is_file():
            raise FileNotFoundError(
                f"{hp} names norm_stats_path={cand!r}, which is not a file here. "
                "Pass --norm-stats with the recovered training npz."
            )
        return p.resolve()
    raise FileNotFoundError(
        f"no hparams.yaml next to {ckpt}; pass --norm-stats explicitly"
    )


def assert_not_identity_stats(norm_stats: dict[str, torch.Tensor], *, path: Path | str) -> None:
    """Raise if loaded stats look like the dataset's mean0/std1 fallback."""
    la_m = norm_stats["log_a_mean"]
    la_s = norm_stats["log_a_std"]
    dp_m = norm_stats["delta_phi_mean"]
    dp_s = norm_stats["delta_phi_std"]
    identity = (
        bool(torch.all(la_m == 0))
        and bool(torch.all(la_s == 1))
        and bool(torch.all(dp_m == 0))
        and bool(torch.all(dp_s == 1))
    )
    if identity:
        raise RuntimeError(
            f"norm stats at {path} are identity (mean 0 / std 1) — refusing to evaluate. "
            "This usually means the npz was missing and the dataset fell back silently."
        )


def load_checkpoint(path: Path | str, map_location: str | torch.device = "cpu"):
    """Load ``GWFlowSurrogateLit``; retry with ``use_abs_time=False`` if needed."""
    from models.lightning_module import GWFlowSurrogateLit

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        lit = GWFlowSurrogateLit.load_from_checkpoint(str(path), map_location=map_location)
    except RuntimeError as exc:
        if "abs" not in str(exc).lower() and "size" not in str(exc).lower():
            raise
        print("note: checkpoint load failed; retrying with use_abs_time=False")
        lit = GWFlowSurrogateLit.load_from_checkpoint(
            str(path), map_location=map_location, use_abs_time=False
        )
    lit.eval()
    return lit


def reconstruct_fp64(
    feat: torch.Tensor,
    batch: dict[str, torch.Tensor],
    residual_scale: float,
    patch_len: int,
    amp_mode: str = "constant",
) -> torch.Tensor:
    """Normalised token features ``(B, n, 2+P)`` → ``h (B, n*P)`` complex128.

    Same math as ``GWFlowSurrogateLit._reconstruct_waveform``, in fp64.
    Amplitude via ``models.amp_interp.log_amp_per_sample`` (single implementation).
    """
    from models.amp_interp import log_amp_per_sample

    P = patch_len
    feat = feat.double()
    la_m = batch["norm_log_a_mean"][:, :1].double()
    la_s = batch["norm_log_a_std"][:, :1].double()
    dp_m = batch["norm_dphi_mean"][:, :1].double()
    dp_s = batch["norm_dphi_std"][:, :1].double()
    la = feat[..., 0] * la_s + la_m
    dphi = feat[..., 1] * dp_s + dp_m
    res = feat[..., 2 : 2 + P] * float(residual_scale)
    s_frac = torch.arange(P, device=feat.device, dtype=torch.float64) / P
    bnd = torch.cumsum(dphi, dim=1) - dphi
    phase = bnd[..., None] + s_frac * dphi[..., None] + res
    la_t = log_amp_per_sample(la, P, amp_mode)  # (B, n*P)
    return torch.exp(la_t) * torch.exp(-1j * phase.reshape(la.shape[0], -1))


def original_waveform(batch: dict[str, torch.Tensor], n_samples: int) -> torch.Tensor:
    """HDF5 tracks → ``h = exp(log_a) * exp(-i (φ − φ[0]))``, fp64, length ``n_samples``."""
    la = batch["log_a_track"][:, 0, :n_samples].double()
    ph = batch["phi_track"][:, 0, :n_samples].double()
    return torch.exp(la) * torch.exp(-1j * (ph - ph[:, :1]))


def mismatches(
    h1: torch.Tensor, h2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row-wise ``(strict, phase_opt, phase_time_opt, best_shift_samples)``."""
    n1 = torch.linalg.vector_norm(h1, dim=1)
    n2 = torch.linalg.vector_norm(h2, dim=1)
    den = (n1 * n2).clamp_min(1e-300)
    inner = (h1.conj() * h2).sum(dim=1)
    strict = 1.0 - inner.real / den
    phase = 1.0 - inner.abs() / den
    L = h1.shape[1]
    nfft = 1 << int(np.ceil(np.log2(2 * L)))
    xc = torch.fft.ifft(torch.fft.fft(h1, nfft).conj() * torch.fft.fft(h2, nfft))
    mag = xc.abs()
    k = mag.argmax(dim=1)
    pt = 1.0 - mag.gather(1, k[:, None])[:, 0] / den
    shift = torch.where(k > nfft // 2, k - nfft, k)
    return strict, phase, pt, shift


def git_info(repo: Path | None = None) -> dict[str, Any]:
    repo = repo or Path.cwd()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "unknown", True
    return {"git_commit": commit, "git_dirty": dirty}


def write_sidecar(
    out_csv: Path | str,
    *,
    checkpoint: Path | str,
    norm_stats: Path | str,
    manifest: Path | str,
    protocol: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    wall_time_s: float | None = None,
    device: str | None = None,
) -> Path:
    """Write ``<out>.json`` next to a result CSV (ground rule 4)."""
    out_csv = Path(out_csv)
    ckpt = Path(checkpoint)
    stats = Path(norm_stats)
    man = Path(manifest)
    stats_npz = np.load(stats)
    meta: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "norm_stats": str(stats),
        "norm_stats_sha256": sha256_file(stats),
        "norm_stats_values": {
            k: np.asarray(stats_npz[k]).tolist()
            for k in ("log_a_mean", "log_a_std", "delta_phi_mean", "delta_phi_std")
        },
        "manifest": str(man),
        "protocol": protocol,
        "protocol_sha256": (
            protocol.get("_protocol_sha256")
            if isinstance(protocol, dict)
            else None
        ),
        "overrides": overrides or {},
        "torch": torch.__version__,
        "device": device,
        "gpu": (
            torch.cuda.get_device_name(0)
            if device is not None and str(device).startswith("cuda") and torch.cuda.is_available()
            else None
        ),
        "wall_time_s": wall_time_s,
        "created_unix": time.time(),
        **git_info(),
    }
    if extra:
        meta.update(extra)
    side = out_csv.with_suffix(".json")
    side.write_text(json.dumps(meta, indent=2) + "\n")
    return side


def assert_grid_matches(
    t_eval: np.ndarray,
    t_ref: np.ndarray,
    *,
    label: str = "eval",
) -> None:
    """P11: element-wise equality of geometric-time grids."""
    t_eval = np.asarray(t_eval, dtype=np.float64)
    t_ref = np.asarray(t_ref, dtype=np.float64)
    if t_eval.shape != t_ref.shape or not np.array_equal(t_eval, t_ref):
        raise RuntimeError(
            f"time grid mismatch ({label}): "
            f"eval n={t_eval.size} [{t_eval[0]}, {t_eval[-1]}] vs "
            f"ref n={t_ref.size} [{t_ref[0]}, {t_ref[-1]}]"
        )


# Audited training grid (Phase 0); used when the training HDF5 is unavailable.
TRAIN_T_MIN = -5000.0
TRAIN_T_MAX = 130.0
TRAIN_DT = 0.5


def audited_training_grid() -> np.ndarray:
    from data_generation.config import time_grid

    return time_grid(TRAIN_T_MIN, TRAIN_T_MAX, TRAIN_DT)
