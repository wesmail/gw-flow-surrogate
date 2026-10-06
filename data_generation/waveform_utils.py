"""Amplitude–phase decomposition, alignment, and gauge fixing (§3.4–3.5)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from data_generation.config import GenerationConfig


def unwrap_phase(phi: np.ndarray) -> np.ndarray:
    """Continuously unwrap 1-D phase along time."""
    return np.unwrap(phi.astype(np.float64))


def decompose_mode(
    h_lm: np.ndarray,
    eps: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray]:
    """Polar decomposition: A = |h|, phi = -arg(h), with unwrapped phase."""
    h_lm = np.asarray(h_lm, dtype=np.complex128)
    amplitude = np.abs(h_lm)
    phase = unwrap_phase(-np.angle(h_lm))
    log_a = np.log(amplitude + eps)
    return log_a.astype(np.float32), phase.astype(np.float32)


def compose_mode(log_a: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Reconstruct complex mode: h = A * exp(-i phi)."""
    return np.exp(log_a) * np.exp(-1j * phi)


def total_amplitude_peak_index(h_modes: dict[tuple[int, int], np.ndarray]) -> int:
    """Index of peak of sum_lm |h_lm|^2 — merger time origin (§3.4)."""
    total = np.zeros(next(iter(h_modes.values())).shape[0], dtype=np.float64)
    for h in h_modes.values():
        total += np.abs(h) ** 2
    return int(np.argmax(total))


def align_time_to_peak(
    times: np.ndarray,
    h_modes: dict[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray], int]:
    """Resample modes so the total-amplitude peak sits at t=0 on the fixed grid (§3.4)."""
    peak_idx = total_amplitude_peak_index(h_modes)
    t_peak = float(times[peak_idx])
    src = times - t_peak
    aligned: dict[tuple[int, int], np.ndarray] = {}
    for lm, h in h_modes.items():
        real = np.interp(times, src, h.real, left=h.real[0], right=h.real[-1])
        imag = np.interp(times, src, h.imag, left=h.imag[0], right=h.imag[-1])
        aligned[lm] = real + 1j * imag
    peak_idx = total_amplitude_peak_index(aligned)
    return times, aligned, peak_idx


def gauge_fix_orbital_phase(
    h_modes: dict[tuple[int, int], np.ndarray],
    times: np.ndarray,
    t_ref: float,
    ref_mode: tuple[int, int] = (2, 2),
) -> tuple[dict[tuple[int, int], np.ndarray], float]:
    """Rotate orbital phase so phi_22(t_ref)=0; apply phi_lm -> phi_lm + m*alpha (§3.4, §4.1).

    Returns (fixed modes, alpha) where alpha is the applied rotation
    (-phi_ref(t_ref) / m_ref), not the pre-fix reference phase.
    """
    ell_ref, m_ref = ref_mode
    h_ref = h_modes[ref_mode]
    _, phi_ref = decompose_mode(h_ref)
    ref_idx = int(np.argmin(np.abs(times - t_ref)))
    alpha = -float(phi_ref[ref_idx]) / m_ref

    fixed: dict[tuple[int, int], np.ndarray] = {}
    for (ell, m), h in h_modes.items():
        log_a, phi = decompose_mode(h)
        phi_gauge = phi + m * alpha
        fixed[(ell, m)] = compose_mode(log_a, phi_gauge)
    return fixed, alpha


def conjugate_mode(h_lm: np.ndarray, ell: int) -> np.ndarray:
    """h_{l,-m} = (-1)^l h_{lm}^*  (§4.3)."""
    return ((-1) ** ell) * np.conj(h_lm)


def validate_waveform(
    h_modes: dict[tuple[int, int], np.ndarray],
    times: np.ndarray,
    cfg: GenerationConfig,
) -> None:
    """Basic sanity checks before storing a waveform."""
    for lm, h in h_modes.items():
        band = (times >= cfg.t_min) & (times <= min(cfg.t_max, times[-1]))
        if not np.all(np.isfinite(h[band])):
            raise ValueError(f"Non-finite values in mode {lm}")
    peak_idx = total_amplitude_peak_index(h_modes)
    if abs(float(times[peak_idx])) > 5.0:
        raise ValueError(
            f"Peak at t={times[peak_idx]:.1f}M; expected |t_peak| <= 5M after alignment."
        )


def modes_to_arrays(
    h_modes: dict[tuple[int, int], np.ndarray],
    modes: Sequence[tuple[int, int]],
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack modes into (C, T) arrays: log_a, phi, h_complex."""
    log_a_list, phi_list, h_list = [], [], []
    for lm in modes:
        h = h_modes[lm]
        log_a, phi = decompose_mode(h, eps=eps)
        log_a_list.append(log_a)
        phi_list.append(phi)
        h_list.append(h.astype(np.complex64))
    return (
        np.stack(log_a_list, axis=0),
        np.stack(phi_list, axis=0),
        np.stack(h_list, axis=0),
    )
