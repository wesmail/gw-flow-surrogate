"""Unit tests for waveform decomposition and gauge symmetry (§9.1, §9.2)."""

import numpy as np
import pytest

from data_generation.waveform_utils import (
  align_time_to_peak,
  compose_mode,
  conjugate_mode,
  decompose_mode,
  gauge_fix_orbital_phase,
)


def test_decomposition_roundtrip():
  t = np.linspace(0, 50, 500)
  h = np.exp(-t / 20) * np.exp(-1j * (2 * np.pi * 0.01 * t**2))
  log_a, phi = decompose_mode(h)
  h2 = compose_mode(log_a, phi)
  assert np.max(np.abs(h - h2)) < 1e-5


def test_gauge_invariance_of_delta_phi():
  t = np.linspace(-100, 100, 256)
  h = np.exp(1j * 0.05 * t)
  alpha = 0.7
  h_rot = h * np.exp(-1j * 2 * alpha)  # m=2 mode orbital rotation

  _, phi0 = decompose_mode(h)
  _, phi1 = decompose_mode(h_rot)
  dphi0 = phi0[-1] - phi0[0]
  dphi1 = phi1[-1] - phi1[0]
  assert abs(dphi0 - dphi1) < 1e-5


def test_mode_conjugation():
  h = np.exp(1j * np.linspace(0, 3, 64))
  # ell=2: (-1)^l = +1  →  h_{2,-m} = h_{2m}^*
  assert np.allclose(conjugate_mode(h, ell=2), np.conj(h))
  # ell=3: (-1)^l = -1  →  h_{3,-m} = -h_{3m}^*
  assert np.allclose(conjugate_mode(h, ell=3), -np.conj(h))


def test_gauge_fix_zeros_reference_phase():
  t = np.linspace(-100, 100, 256)
  ref_mode = (2, 2)
  h22 = np.exp(1j * (0.1 * t + 0.5))
  h_modes = {ref_mode: h22}
  t_ref = -50.0
  fixed, _ = gauge_fix_orbital_phase(h_modes, t, t_ref, ref_mode=ref_mode)
  ref_idx = int(np.argmin(np.abs(t - t_ref)))
  _, phi_after = decompose_mode(fixed[ref_mode])
  wrapped = (phi_after[ref_idx] + np.pi) % (2 * np.pi) - np.pi
  assert abs(wrapped) < 1e-6


def test_gauge_fix_preserves_relative_phase():
  t = np.linspace(-50, 50, 200)
  m_a, m_b = 2, 3
  mode_a = (2, 2)
  mode_b = (3, 3)
  h_a = np.exp(1j * 0.2 * t)
  h_b = np.exp(1j * 0.15 * t)
  h_modes = {mode_a: h_a, mode_b: h_b}

  _, phi_before_a = decompose_mode(h_a)
  _, phi_before_b = decompose_mode(h_b)
  rel_before = phi_before_a - (m_a / m_b) * phi_before_b

  fixed, _ = gauge_fix_orbital_phase(h_modes, t, t_ref=-50.0, ref_mode=mode_a)
  _, phi_after_a = decompose_mode(fixed[mode_a])
  _, phi_after_b = decompose_mode(fixed[mode_b])
  rel_after = phi_after_a - (m_a / m_b) * phi_after_b

  # Gauge-invariant up to independent unwrap offsets: difference is constant in time.
  assert np.allclose(np.diff(rel_before), np.diff(rel_after), atol=1e-5)
  assert np.std((rel_after - rel_before) - (rel_after[0] - rel_before[0])) < 1e-5


def test_align_time_to_peak_places_peak_at_zero():
  dt = 1.0
  times = np.arange(-100, 101, dtype=np.float64)
  t_peak_true = 17.0
  h = np.exp(-0.5 * ((times - t_peak_true) / 5) ** 2) * np.exp(-1j * times * 0.01)
  h_modes = {(2, 2): h}

  returned_times, aligned, peak_idx = align_time_to_peak(times, h_modes)
  total = np.abs(aligned[(2, 2)]) ** 2
  peak_idx_check = int(np.argmax(total))
  assert abs(returned_times[peak_idx_check]) <= dt
  assert peak_idx == peak_idx_check
