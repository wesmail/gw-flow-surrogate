#!/usr/bin/env python
"""Checks before any PE run (minutes).

  A  the wrapper is right: wrapper(NRHybSur3dq8) vs bilby's LAL NRHybSur3dq8
     (ET-weighted mismatch, optimised over time and phase; expect ~0)
  B  the prediction: wrapper(our model) vs wrapper(NRHybSur3dq8) at the test binary,
     weighted with the ET noise curve -> mm_ET and rho* = 1/sqrt(2 mm_ET)
  C  determinism: our model's waveform for theta is the same alone and inside a batch
  D  timing: theta -> frequency-domain h+, hx for a batch (both sources)

Usage (repo root, GPU, environment with gwsurrogate + bilby + lalsuite)::

    python pe/check_wrapper.py --checkpoint "$FT" \
        --manifest data/n1000k/manifest.parquet --norm-stats data/n1000k/norm_stats.npz \
        --asd asd/ET_15km_HFLF_cryo.txt --out results/pe_check.json

--asd: the 2L (15 km) curve of the ET design study; without it, bilby's ET-D curve.

LAL check A (bilby ``NRHybSur3dq8``) needs the *lalsuite* surrogate data file on
``$LAL_DATA_PATH``.  LALSuite >= 7.25 looks for ``NRHybSur3dq8_lal_v1.0.h5`` from
https://git.ligo.org/waveforms/software/lalsuite-waveform-data
(or Zenodo https://zenodo.org/records/14999310).  That is **not** the same file as
gwsurrogate's ``NRHybSur3dq8.h5``.  If unavailable, run with ``--skip-lal``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pe.fm_waveform import FMSource, NRSource, to_frequency_domain  # noqa: E402

MTSUN_SI = 4.925490947641267e-06


def _lal_surrogate_available() -> tuple[bool, str]:
    """Return whether lalsimulation can resolve NRHybSur3dq8 data."""
    names = ("NRHybSur3dq8_lal_v1.0.h5", "NRHybSur3dq8.h5")
    roots = []
    if os.environ.get("LAL_DATA_PATH"):
        roots.extend(Path(p) for p in os.environ["LAL_DATA_PATH"].split(os.pathsep) if p)
    for env in (os.environ.get("CONDA_PREFIX"), os.environ.get("VIRTUAL_ENV")):
        if env:
            roots.append(Path(env) / "share" / "lalsuite")
    for r in roots:
        for name in names:
            if (r / name).is_file():
                return True, str(r / name)
    return False, (
        "NRHybSur3dq8_lal_v1.0.h5 not found on $LAL_DATA_PATH "
        "(see https://git.ligo.org/waveforms/software/lalsuite-waveform-data). "
        "Pass --skip-lal to skip check A."
    )


def psd_on(f, asd_file):
    import bilby
    psd = (bilby.gw.detector.PowerSpectralDensity(asd_file=asd_file) if asd_file
           else bilby.gw.detector.PowerSpectralDensity(psd_file="ET_D_psd.txt"))
    return psd.power_spectral_density_interpolated(f)


def mismatch_tp(a, b, f, S, f_min, pad=8):
    """ET-weighted mismatch of two FD signals, optimised over time and phase."""
    m = (f >= f_min) & np.isfinite(S) & (S > 0)
    df = f[1] - f[0]
    w = np.zeros_like(f); w[m] = 1.0 / S[m]
    na = np.sqrt(4 * df * np.sum(np.abs(a) ** 2 * w)); nb = np.sqrt(4 * df * np.sum(np.abs(b) ** 2 * w))
    z = np.fft.ifft(np.conj(a) * b * w, n=pad * f.size) * pad * f.size * 4 * df
    return float(1 - np.abs(z).max() / (na * nb))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--asd", default=None, help="ET 15 km (2L) ASD file; default: ET-D")
    ap.add_argument("--m1", type=float, default=200.0); ap.add_argument("--m2", type=float, default=100.0)
    ap.add_argument("--chi1", type=float, default=0.3); ap.add_argument("--chi2", type=float, default=-0.2)
    ap.add_argument("--theta-jn", type=float, default=0.4); ap.add_argument("--phase", type=float, default=1.3)
    ap.add_argument("--distance", type=float, default=10000.0, help="Mpc (only scales the SNR)")
    ap.add_argument("--duration", type=float, default=16.0); ap.add_argument("--fs", type=float, default=1024.0)
    ap.add_argument("--f-min", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=1024, help="batch size for the timing check")
    ap.add_argument("--skip-lal", action="store_true")
    ap.add_argument("--out", default="results/pe_check.json")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")

    f = np.fft.rfftfreq(int(a.duration * a.fs), 1 / a.fs)
    S = psd_on(f, a.asd)
    q = a.m1 / a.m2; M = a.m1 + a.m2
    res = dict(binary=dict(m1=a.m1, m2=a.m2, chi1=a.chi1, chi2=a.chi2, theta_jn=a.theta_jn),
               asd=a.asd or "ET-D (bilby)", f_min=a.f_min)

    fm = FMSource(a.checkpoint, a.manifest, a.norm_stats)
    nr = fm.nr
    f_start = abs(np.gradient(np.unwrap(-np.angle(nr.h22(q, a.chi1, a.chi2)[0])), nr.t_M)[0]) / (2 * np.pi) / (M * MTSUN_SI)
    res["f_start_Hz"] = f_start
    print(f"binary {a.m1:g}+{a.m2:g} Msun, chi=({a.chi1},{a.chi2}); our window starts at {f_start:.2f} Hz "
          f"(must be < {a.f_min} Hz)")
    if f_start >= a.f_min:
        print("  WARNING: window too short for this mass; use a heavier binary or raise --f-min")

    args = dict(total_mass=M, luminosity_distance=a.distance, theta_jn=a.theta_jn, phase=a.phase,
                frequency_array=f)
    h_nr = to_frequency_domain(nr.t_M, nr.h22(q, a.chi1, a.chi2), **args)
    h_fm = to_frequency_domain(fm.t_M, fm.h22(q, a.chi1, a.chi2), **args)
    snr = np.sqrt(4 * (f[1] - f[0]) * np.sum((np.abs(h_nr["plus"][0]) ** 2 / S)[f >= a.f_min]))
    res["snr_plus_one_ifo_unit_response"] = float(snr)

    # ---- A: wrapper vs LAL -------------------------------------------------
    if not a.skip_lal:
        ok_lal, lal_msg = _lal_surrogate_available()
        if not ok_lal:
            res["A_mismatch_wrapperNR_vs_LAL"] = f"skipped: {lal_msg}"
            print(f"A  skipped ({lal_msg})")
        else:
            try:
                import bilby
                wfg = bilby.gw.WaveformGenerator(
                    duration=a.duration, sampling_frequency=a.fs,
                    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                    waveform_arguments=dict(waveform_approximant="NRHybSur3dq8", mode_array=[[2, 2], [2, -2]],
                                            reference_frequency=a.f_min, minimum_frequency=a.f_min * 0.8))
                p = dict(mass_1=a.m1, mass_2=a.m2, chi_1=a.chi1, chi_2=a.chi2, luminosity_distance=a.distance,
                         theta_jn=a.theta_jn, phase=a.phase, psi=0.0, ra=0.0, dec=0.0, geocent_time=0.0)
                h_lal = wfg.frequency_domain_strain(p)
                mmA = mismatch_tp(h_lal["plus"], h_nr["plus"][0], f, S, a.f_min)
                res["A_mismatch_wrapperNR_vs_LAL"] = mmA
                print(f"A  wrapper(NRHybSur3dq8) vs LAL NRHybSur3dq8: ET-weighted mismatch {mmA:.2e} "
                      f"({'OK' if mmA < 1e-4 else 'CHECK: conventions or tapering differ'})")
            except Exception as e:
                res["A_mismatch_wrapperNR_vs_LAL"] = f"skipped: {e!r}"[:300]
                print(f"A  skipped ({e!r:.200})")
    else:
        res["A_mismatch_wrapperNR_vs_LAL"] = "skipped: --skip-lal"
        print("A  skipped (--skip-lal)")

    # ---- B: our model vs reference ------------------------------------------
    mmB = mismatch_tp(h_nr["plus"][0], h_fm["plus"][0], f, S, a.f_min)
    rho_star = 1 / np.sqrt(2 * mmB)
    w = (nr.t_M >= nr.t_M[0])
    hn, hf = nr.h22(q, a.chi1, a.chi2)[0], fm.h22(q, a.chi1, a.chi2)[0]
    mm_white = float(1 - np.abs(np.vdot(hn, hf)) / (np.linalg.norm(hn) * np.linalg.norm(hf)))
    res.update(B_mismatch_ET=mmB, B_rho_star=float(rho_star), B_mismatch_white_phase_only=mm_white)
    print(f"B  our model vs NRHybSur3dq8: ET-weighted mismatch {mmB:.3e}  ->  rho* = {rho_star:.0f}"
          f"   (white-noise, phase-only, M units: {mm_white:.2e})")
    print(f"   suggested PE loudnesses: {', '.join(f'{x:.0f}' for x in (rho_star / 2, rho_star, 2 * rho_star, 4 * rho_star))}")

    # ---- C: determinism -------------------------------------------------------
    rng = np.random.default_rng(0)
    qs = np.r_[q, rng.uniform(1.5, 7, 7)]; c1s = np.r_[a.chi1, rng.uniform(-.7, .7, 7)]
    c2s = np.r_[a.chi2, rng.uniform(-.7, .7, 7)]
    h_alone = fm.h22(q, a.chi1, a.chi2)[0]
    h_inbatch = fm.h22(qs[::-1], c1s[::-1], c2s[::-1])[-1]        # same theta, last of 8
    dC = float(np.max(np.abs(h_alone - h_inbatch)) / np.max(np.abs(h_alone)))
    res["C_max_rel_diff_alone_vs_batch"] = dC
    print(f"C  same theta alone vs inside a batch of 8: max relative difference {dC:.1e} "
          f"({'OK' if dC < 1e-5 else 'NOT DETERMINISTIC: check the shared-noise path'})")

    # ---- D: timing -------------------------------------------------------------
    Bt = a.batch
    th = np.column_stack([rng.uniform(1.2, 7.5, Bt), rng.uniform(-.7, .7, Bt), rng.uniform(-.7, .7, Bt)])
    t0 = time.perf_counter(); recs = nr.records(th[:, 0], th[:, 1], th[:, 2]); t_nr = time.perf_counter() - t0
    t0 = time.perf_counter(); H = fm.h22(th[:, 0], th[:, 1], th[:, 2]); t_fm = time.perf_counter() - t0
    t0 = time.perf_counter()
    to_frequency_domain(fm.t_M, H, M, a.distance, a.theta_jn, a.phase, f); t_fd = time.perf_counter() - t0
    res.update(D_batch=Bt, D_s_start_tokens=t_nr, D_s_model_incl_tokens=t_fm, D_s_to_fd=t_fd,
               D_ms_per_waveform_total=1e3 * (t_fm + t_fd) / Bt, D_fast_path=bool(fm._fast_ok))
    print(f"D  batch {Bt}: start tokens (NRHybSur3dq8) {t_nr:.1f} s, model incl. tokens {t_fm:.1f} s, "
          f"to frequency domain {t_fd:.1f} s  ->  {1e3 * (t_fm + t_fd) / Bt:.1f} ms per waveform"
          f"   (fast path: {fm._fast_ok})")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"-> {out}")


if __name__ == "__main__":
    main()