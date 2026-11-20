#!/usr/bin/env python
"""Parameter estimation with ET 2L: NRHybSur3dq8 (reference) vs our model, one run per call.

The experiment (zero-noise injection, so any shift is the waveform model's, not the noise's):

  data        the TRUE waveform (NRHybSur3dq8, through the same wrapper as the recovery)
              projected on the 2L ET (15 km, 45 deg, CoBA HFLF-cryo ASD), no noise;
              the distance is chosen so the network SNR equals --snr
  recovery    --model nr : NRHybSur3dq8 through the wrapper  (the best possible result)
              --model fm : our flow-matching model             (what we test)
  sampled     chirp_mass, mass_ratio (m2/m1 <= 1), chi_1, chi_2, luminosity_distance,
              theta_jn, psi, geocent_time                      (8 parameters)
  fixed       sky position (ra, dec); orbital phase marginalised analytically
  sampler     nessai (normalising-flow nested sampling), vectorised likelihood: every
              batch of proposed points goes to the GPU together, padded to a FIXED size so
              torch.compile compiles once

Likelihood (phase-marginalised, (2,2)-mode only so the phase is an overall factor):

    z  = sum_det <d, h(phase=0)>  (complex),   hh = sum_det <h, h>
    ln L = ln I0(|z|) - hh/2                    (the constant -<d,d>/2 is dropped)
    max over phase:  |z| - hh/2   ( = rho^2/2 at the truth for the exact model)

Usage (repo root, GPU)::

    # 1) quick test, no sampling (~1 min + compilation): SNR, likelihood at the truth, timing
    python pe/run_pe.py --model fm --snr 30 --test  --checkpoint "$FT" \
        --manifest data/n1000k/manifest.parquet --norm-stats data/n1000k/norm_stats.npz \
        --asd asd/ET_HFLF_cryo_15km.txt

    # 2) a run
    python pe/run_pe.py --model fm --snr 30 --checkpoint "$FT" ... --outdir results/pe

Outputs in <outdir>/<model>_snr<snr>/:  nessai's files, posterior.csv (with derived q, M,
chi_eff), summary.json (truth, SNR, distance, medians, 90% intervals, max ln L, timing).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MTSUN_SI = 4.925490947641267e-06
NAMES = ["chirp_mass", "mass_ratio", "chi_1", "chi_2", "luminosity_distance", "theta_jn", "psi",
         "geocent_time"]


# ----------------------------------------------------------------------------------- #
# masses                                                                              #
# ----------------------------------------------------------------------------------- #
def eta_of(mass_ratio):
    return mass_ratio / (1.0 + mass_ratio) ** 2


def total_mass(chirp_mass, mass_ratio):
    return chirp_mass * eta_of(mass_ratio) ** -0.6


def chirp_mass_of(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


# ----------------------------------------------------------------------------------- #
# start tokens in parallel (optional; only after the gwsurrogate 'pull' fix)            #
# ----------------------------------------------------------------------------------- #
def _records_worker(args):
    from data_generation.generate import evaluate_waveform
    cfg, rows = args
    return [evaluate_waveform(float(a), float(b), float(c), cfg) for a, b, c in rows]


def parallel_records(nr, n_workers):
    """Replace nr.records by a process-pool version (spawned BEFORE CUDA is initialised)."""
    import multiprocessing as mp
    pool = mp.get_context("spawn").Pool(n_workers)

    def records(q, chi1, chi2):
        rows = list(zip(np.atleast_1d(q), np.atleast_1d(chi1), np.atleast_1d(chi2)))
        chunks = [rows[i::n_workers] for i in range(n_workers)]
        parts = pool.map(_records_worker, [(nr.cfg, c) for c in chunks])
        out = [None] * len(rows)
        for i, part in enumerate(parts):
            out[i::n_workers] = part
        return out
    nr.records = records
    return pool


# ----------------------------------------------------------------------------------- #
# the detector network: antenna patterns, time delays, noise weights                  #
# ----------------------------------------------------------------------------------- #
class Network:
    """Fixed sky position. h_det = (F+(psi) h+ + Fx(psi) hx) exp(-2 pi i f (t_c - t0 + dt_det))."""

    def __init__(self, ifos, frequency_array, f_min, ra, dec, t_gps, start_time):
        self.f_full = frequency_array
        f_max = min(ifo.maximum_frequency for ifo in ifos)
        self.mask = (frequency_array >= f_min) & (frequency_array <= f_max)
        self.f = frequency_array[self.mask]
        df = frequency_array[1] - frequency_array[0]
        self.start_time = start_time
        self.names, self.w, self.Fp0, self.Fc0, self.dt = [], [], [], [], []
        for ifo in ifos:
            S = ifo.power_spectral_density.power_spectral_density_interpolated(self.f)
            if not np.all(np.isfinite(S) & (S > 0)):
                raise ValueError(f"{ifo.name}: PSD not finite/positive on [{f_min}, {f_max}] Hz")
            self.names.append(ifo.name)
            self.w.append(4.0 * df / S)                                   # <a,b> = sum conj(a) b w
            self.Fp0.append(ifo.antenna_response(ra, dec, t_gps, 0.0, "plus"))
            self.Fc0.append(ifo.antenna_response(ra, dec, t_gps, 0.0, "cross"))
            self.dt.append(ifo.time_delay_from_geocenter(ra, dec, t_gps))

    def project(self, hp, hc, psi, tc):
        """hp, hc: (B, n_f) on self.f; psi, tc: (B,). Returns a list (one per detector) of (B, n_f)."""
        c2, s2 = np.cos(2 * psi)[:, None], np.sin(2 * psi)[:, None]
        out = []
        for Fp0, Fc0, dt in zip(self.Fp0, self.Fc0, self.dt):
            Fp = Fp0 * c2 + Fc0 * s2                       # rotation of the polarisation frame
            Fc = -Fp0 * s2 + Fc0 * c2
            shift = np.exp(-2j * np.pi * self.f[None] * ((tc - self.start_time + dt)[:, None]))
            out.append((Fp * hp + Fc * hc) * shift)
        return out

    def inner(self, a, b, k):
        return np.sum(np.conj(a) * b * self.w[k], axis=-1)


# ----------------------------------------------------------------------------------- #
# waveforms for many parameter sets, padded to a fixed batch                            #
# ----------------------------------------------------------------------------------- #
class Waveforms:
    def __init__(self, source, net, batch, pad):
        self.source, self.net, self.batch, self.pad = source, net, batch, pad
        self.n_calls = 0
        self.t_model = 0.0
        self.t_fd = 0.0

    def __call__(self, q, chi1, chi2, M, dL, inc, phase):
        from pe.fm_waveform import to_frequency_domain
        P = len(q)
        hp = np.empty((P, self.net.f.size), complex); hc = np.empty_like(hp)
        for s in range(0, P, self.batch):
            idx = np.arange(s, min(s + self.batch, P))
            n = idx.size
            take = np.r_[idx, np.full(self.batch - n, idx[-1])] if (self.pad and n < self.batch) else idx
            t0 = time.perf_counter()
            h22 = self.source.h22(q[take], chi1[take], chi2[take])[:n]
            t1 = time.perf_counter()
            fd = to_frequency_domain(self.source.t_M, h22, M[idx], dL[idx], inc[idx], phase[idx],
                                     self.net.f_full)
            self.t_model += t1 - t0; self.t_fd += time.perf_counter() - t1
            hp[idx] = fd["plus"][:, self.net.mask]; hc[idx] = fd["cross"][:, self.net.mask]
        self.n_calls += P
        return hp, hc


# ----------------------------------------------------------------------------------- #
# the nessai model                                                                    #
# ----------------------------------------------------------------------------------- #
def make_model(wf, net, data, bounds, m_lo_truth, eta_truth, m_hi):
    from nessai.model import Model

    class PEModel(Model):
        def __init__(self):
            self.names = list(NAMES)
            self.bounds = {k: list(v) for k, v in bounds.items()}
            self.vectorised_likelihood = True     # nessai's own check is too strict (1e-15)
            self.vectorised_prior = True

        # ---- the constraint on the total mass (same for both models) ------------- #
        def allowed(self, x):
            M = total_mass(x["chirp_mass"], x["mass_ratio"])
            # our 5000 M window must start below f_min (+ taper): heavier is fine. The start
            # frequency at fixed time before merger scales like eta^(-3/8) at leading order.
            m_lo = m_lo_truth * (eta_truth / eta_of(x["mass_ratio"])) ** 0.375
            return (M >= m_lo) & (M <= m_hi)

        def log_prior(self, x):
            with np.errstate(divide="ignore"):
                lp = np.log(self.in_bounds(x) & self.allowed(x), dtype=float)
            lp = lp + 2 * np.log(x["luminosity_distance"])                 # uniform in volume
            lp = lp + np.log(np.abs(np.sin(x["theta_jn"])))                # isotropic orientation
            return lp                                                      # others: uniform

        # ---- the same prior as a map from the unit cube (for i-nessai) ---------- #
        def from_unit_hypercube(self, u):
            x = u.copy()
            for k in NAMES:
                lo, hi = self.bounds[k]
                if k == "luminosity_distance":                             # p(d) ~ d^2
                    x[k] = np.cbrt(lo ** 3 + u[k] * (hi ** 3 - lo ** 3))
                elif k == "theta_jn":                                      # p ~ sin, [0, pi]
                    x[k] = np.arccos(np.clip(1.0 - 2.0 * u[k], -1.0, 1.0))
                else:
                    x[k] = lo + u[k] * (hi - lo)
            return x

        def to_unit_hypercube(self, x):
            u = x.copy()
            for k in NAMES:
                lo, hi = self.bounds[k]
                if k == "luminosity_distance":
                    u[k] = (x[k] ** 3 - lo ** 3) / (hi ** 3 - lo ** 3)
                elif k == "theta_jn":
                    u[k] = 0.5 * (1.0 - np.cos(x[k]))
                else:
                    u[k] = (x[k] - lo) / (hi - lo)
            return u

        def log_prior_unit_hypercube(self, u):
            inside = np.all([(u[k] >= 0) & (u[k] < 1) for k in NAMES], axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.log(inside & self.allowed(self.from_unit_hypercube(u)), dtype=float)

        def log_likelihood(self, x):
            x = np.atleast_1d(x)
            out = np.full(x.size, -np.inf)
            ok = self.allowed(x)
            if ok.any():
                out[ok] = log_likelihood_arrays(wf, net, data, {k: x[k][ok] for k in NAMES})[0]
            return out

    return PEModel()


def log_likelihood_arrays(wf, net, data, p):
    """Returns (phase-marginalised ln L, ln L maximised over phase), arrays."""
    mr = np.asarray(p["mass_ratio"], float)
    q = 1.0 / mr
    M = total_mass(np.asarray(p["chirp_mass"], float), mr)
    zeros = np.zeros_like(q)
    hp, hc = wf(q, np.asarray(p["chi_1"], float), np.asarray(p["chi_2"], float), M,
                np.asarray(p["luminosity_distance"], float), np.asarray(p["theta_jn"], float), zeros)
    hs = net.project(hp, hc, np.asarray(p["psi"], float), np.asarray(p["geocent_time"], float))
    z = sum(net.inner(d, h, k) for k, (d, h) in enumerate(zip(data, hs)))
    hh = sum(net.inner(h, h, k).real for k, h in enumerate(hs))
    from scipy.special import i0e
    az = np.abs(z)
    return np.log(i0e(az)) + az - 0.5 * hh, az - 0.5 * hh


# ----------------------------------------------------------------------------------- #
# main                                                                                #
# ----------------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["nr", "fm"], required=True)
    ap.add_argument("--snr", type=float, required=True, help="network optimal SNR of the injection")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--asd", required=True, help="ET 15 km HFLF-cryo ASD (one interferometer)")
    ap.add_argument("--alpha", type=float, default=45.0, help="2L relative orientation [deg]")
    ap.add_argument("--outdir", default="results/pe")
    # the injected binary (detector frame)
    ap.add_argument("--m1", type=float, default=200.0); ap.add_argument("--m2", type=float, default=100.0)
    ap.add_argument("--chi1", type=float, default=0.3); ap.add_argument("--chi2", type=float, default=-0.2)
    ap.add_argument("--theta-jn", type=float, default=0.4); ap.add_argument("--phase", type=float, default=1.3)
    ap.add_argument("--psi", type=float, default=0.3)
    # sky: both detectors at their median response F+^2+Fx^2 (0.33 each) at --gps; not a lucky spot
    ap.add_argument("--ra", type=float, default=6.084); ap.add_argument("--dec", type=float, default=0.127)
    ap.add_argument("--gps", type=float, default=1_400_000_000.0)
    # data
    ap.add_argument("--duration", type=float, default=32.0)
    ap.add_argument("--fs", type=float, default=512.0)
    ap.add_argument("--f-min", type=float, default=5.0)
    # prior box (ADJUST to the training box)
    ap.add_argument("--q-max", type=float, default=8.0)
    ap.add_argument("--chi-max", type=float, default=0.8)
    ap.add_argument("--mc-range", type=float, nargs=2, default=[0.8, 1.25], help="x true chirp mass")
    ap.add_argument("--dl-range", type=float, nargs=2, default=[0.1, 4.0], help="x true distance")
    ap.add_argument("--dt", type=float, default=0.1, help="+- seconds around the true merger time")
    # sampler / speed
    ap.add_argument("--sampler", choices=["nessai", "inessai"], default="nessai",
                    help="nessai: standard nested sampling; inessai: importance nested sampling")
    ap.add_argument("--nlive", type=int, default=None,
                    help="live points (nessai, default 1000) / samples per iteration (inessai, default 2048)")
    ap.add_argument("--batch", type=int, default=1024, help="fixed GPU batch (likelihood chunk)")
    ap.add_argument("--token-workers", type=int, default=0,
                    help="processes for NRHybSur3dq8 (start tokens / reference); 0 = serial")
    ap.add_argument("--seed", type=int, default=150914)
    ap.add_argument("--test", action="store_true", help="no sampling: checks and timing only")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")

    out = Path(a.outdir) / f"{a.model}_snr{a.snr:g}"
    out.mkdir(parents=True, exist_ok=True)

    # ---- sources (token pool first: spawned before CUDA starts) ------------------------- #
    pool = None
    from pe.fm_waveform import FMSource, NRSource
    if a.model == "fm":
        fm = FMSource(a.checkpoint, a.manifest, a.norm_stats)
        nr = fm.nr
        source, pad = fm, True
    else:
        # same grid as the model (n_use from the model's patching) without loading it on the GPU
        fm_probe = FMSource(a.checkpoint, a.manifest, a.norm_stats, device="cpu")
        nr = NRSource(n_use=fm_probe.n_use); del fm_probe
        source, pad = nr, False
    if a.token_workers > 0:
        pool = parallel_records(nr, a.token_workers)

    # ---- detectors, data grid -------------------------------------------------------- #
    from pe.et_detectors import et_2l
    from pe.fm_waveform import to_frequency_domain
    ifos = et_2l(asd_file=a.asd, length_km=15.0, alpha_deg=a.alpha, minimum_frequency=a.f_min,
                 maximum_frequency=a.fs / 2)
    f = np.fft.rfftfreq(int(a.duration * a.fs), 1.0 / a.fs)
    start_time = a.gps - (a.duration - 2.0)          # merger 2 s before the end of the segment
    net = Network(ifos, f, a.f_min, a.ra, a.dec, a.gps, start_time)

    # ---- the injection: true waveform, true phase, distance for the target SNR ------- #
    q_true = a.m1 / a.m2; M_true = a.m1 + a.m2
    h22_true = nr.h22(q_true, a.chi1, a.chi2)

    def inject(dL):
        fd = to_frequency_domain(nr.t_M, h22_true, M_true, dL, a.theta_jn, a.phase, f)
        return net.project(fd["plus"][:, net.mask], fd["cross"][:, net.mask],
                           np.array([a.psi]), np.array([a.gps]))
    d0 = inject(1000.0)
    rho0 = math.sqrt(sum(net.inner(d, d, k).real[0] for k, d in enumerate(d0)))
    dL_true = 1000.0 * rho0 / a.snr
    data = [d[0] for d in inject(dL_true)]
    rho_det = [math.sqrt(net.inner(d, d, k).real) for k, d in enumerate(data)]
    rho = math.sqrt(sum(r * r for r in rho_det))

    # ---- prior box and the mass constraint ------------------------------------------ #
    mc_true = chirp_mass_of(a.m1, a.m2)
    t_M = nr.t_M
    i_tap = np.searchsorted(t_M, t_M[0] + 300.0)                      # end of the start taper
    omega = np.abs(np.gradient(np.unwrap(-np.angle(h22_true[0])), t_M))
    f_tap_M = omega[i_tap] / (2 * np.pi)                              # in units of 1/M
    m_lo_truth = 1.02 * f_tap_M / (a.f_min * MTSUN_SI)                # f(t0 + taper) <= f_min
    m_hi = 0.95 * (a.duration / 2) / (-t_M[0] * MTSUN_SI)             # fits in the segment
    eta_truth = eta_of(a.m2 / a.m1)
    bounds = dict(chirp_mass=(a.mc_range[0] * mc_true, a.mc_range[1] * mc_true),
                  mass_ratio=(1.0 / a.q_max, 1.0),
                  chi_1=(-a.chi_max, a.chi_max), chi_2=(-a.chi_max, a.chi_max),
                  luminosity_distance=(a.dl_range[0] * dL_true, a.dl_range[1] * dL_true),
                  theta_jn=(0.0, math.pi), psi=(0.0, math.pi),
                  geocent_time=(a.gps - a.dt, a.gps + a.dt))
    truth = dict(chirp_mass=mc_true, mass_ratio=a.m2 / a.m1, chi_1=a.chi1, chi_2=a.chi2,
                 luminosity_distance=dL_true, theta_jn=a.theta_jn, psi=a.psi, geocent_time=a.gps)

    wf = Waveforms(source, net, a.batch, pad)
    model = make_model(wf, net, data, bounds, m_lo_truth, eta_truth, m_hi)

    print(f"model={a.model}  SNR {rho:.2f} (" + ", ".join(f"{n} {r:.1f}" for n, r in zip(net.names, rho_det))
          + f")  at d_L = {dL_true:.0f} Mpc")
    print(f"total-mass constraint at the true q: {m_lo_truth:.0f} <= M <= {m_hi:.0f} Msun (truth {M_true:g})")

    # ---- checks at the truth ---------------------------------------------------------- #
    tr = {k: np.array([v]) for k, v in truth.items()}
    lnl_marg, lnl_max = log_likelihood_arrays(wf, net, data, tr)
    print(f"ln L at the truth: max over phase {lnl_max[0]:.3f}  (exact model: rho^2/2 = {rho**2 / 2:.3f}), "
          f"phase-marginalised {lnl_marg[0]:.3f}")
    summary = dict(model=a.model, snr_target=a.snr, snr=rho, snr_per_detector=dict(zip(net.names, rho_det)),
                   distance_Mpc=dL_true, truth=truth, bounds=bounds, m_lo_at_true_q=m_lo_truth, m_hi=m_hi,
                   lnL_maxphase_truth=float(lnl_max[0]), lnL_marg_truth=float(lnl_marg[0]),
                   rho2_half=rho ** 2 / 2, alpha=a.alpha, ra=a.ra, dec=a.dec, asd=a.asd,
                   duration=a.duration, fs=a.fs, f_min=a.f_min)

    # timing: one full batch of prior draws (after a warm-up that may compile)
    rng = np.random.default_rng(a.seed)
    draw = {k: rng.uniform(*bounds[k], a.batch) for k in NAMES}
    draw["chirp_mass"] = np.full(a.batch, mc_true) * rng.uniform(0.95, 1.05, a.batch)   # inside the constraint
    t0 = time.perf_counter(); log_likelihood_arrays(wf, net, data, draw); t_first = time.perf_counter() - t0
    wf.t_model = wf.t_fd = 0.0
    t0 = time.perf_counter(); log_likelihood_arrays(wf, net, data, draw); t_warm = time.perf_counter() - t0
    ms = 1e3 * t_warm / a.batch
    print(f"timing, batch {a.batch}: first call {t_first:.1f} s, warm {t_warm:.1f} s "
          f"(waveforms {wf.t_model:.1f} s, frequency domain {wf.t_fd:.1f} s)  ->  {ms:.1f} ms per likelihood")
    print(f"  expected run time: 5e5 / 1e6 / 2e6 likelihood calls -> "
          f"{5e5 * ms / 3.6e6:.1f} / {1e6 * ms / 3.6e6:.1f} / {2e6 * ms / 3.6e6:.1f} h")
    summary["ms_per_likelihood"] = ms

    if a.test:
        (out / "test.json").write_text(json.dumps(summary, indent=2, default=float))
        print(f"-> {out / 'test.json'}")
        if pool: pool.close()
        return

    # ---- sampling ---------------------------------------------------------------------- #
    from nessai.flowsampler import FlowSampler
    from nessai.gw.proposal import GWFlowProposal
    try:                                                   # progress -> the log (and a file in out/)
        from nessai.utils.logging import configure_logger as _log
    except ImportError:                                    # nessai < 0.14
        from nessai.utils import setup_logger as _log
    _log(output=str(out), label="nessai", log_level="INFO")
    reparam = {"luminosity_distance": {"reparameterisation": "distance", "prior": "power-law",
                                       "converter_kwargs": {"power": 2}}}
    t0 = time.perf_counter()
    if a.sampler == "nessai":
        fs = FlowSampler(model, output=str(out), nlive=a.nlive or 1000, seed=a.seed, resume=True,
                         flow_proposal_class=GWFlowProposal, reparameterisations=reparam,
                         likelihood_chunksize=a.batch, signal_handling=True)
    else:
        fs = FlowSampler(model, output=str(out), importance_nested_sampler=True,
                         nlive=a.nlive or 2048, seed=a.seed, resume=True,
                         likelihood_chunksize=a.batch, signal_handling=True)
    fs.run(plot=True)
    t_run = time.perf_counter() - t0
    post = fs.posterior_samples

    # ---- post-processing: derived quantities, intervals, best fit ---------------------- #
    cols = {k: np.asarray(post[k], float) for k in NAMES}
    cols["q"] = 1.0 / cols["mass_ratio"]
    cols["total_mass"] = total_mass(cols["chirp_mass"], cols["mass_ratio"])
    cols["chi_eff"] = (cols["chi_1"] + cols["mass_ratio"] * cols["chi_2"]) / (1.0 + cols["mass_ratio"])
    cols["logL"] = np.asarray(post["logL"], float)
    header = ",".join(cols)
    np.savetxt(out / "posterior.csv", np.column_stack(list(cols.values())), delimiter=",", header=header,
               comments="")
    truth_d = dict(truth, q=q_true, total_mass=M_true, chi_eff=(a.chi1 + a.m2 / a.m1 * a.chi2) / (1 + a.m2 / a.m1))
    stats = {}
    for k, v in cols.items():
        if k == "logL":
            continue
        lo, med, hi = np.quantile(v, [0.05, 0.5, 0.95])
        sd = float(np.std(v))
        stats[k] = dict(median=float(med), q05=float(lo), q95=float(hi), std=sd, truth=float(truth_d[k]),
                        bias_over_std=float((med - truth_d[k]) / sd) if sd > 0 else None)
    # best fit over phase on the posterior samples (up to 4 x batch of the highest logL)
    top = np.argsort(cols["logL"])[::-1][: 4 * a.batch]
    _, lmax = log_likelihood_arrays(wf, net, data, {k: cols[k][top] for k in NAMES})
    summary.update(sampler=a.sampler, nlive=a.nlive, stats=stats, lnL_maxphase_best=float(lmax.max()),
                   delta_lnL_max=float(rho ** 2 / 2 - lmax.max()),
                   log_evidence=float(fs.log_evidence), log_evidence_error=float(fs.log_evidence_error),
                   n_likelihood=int(wf.n_calls), run_hours=t_run / 3600, n_posterior=int(post.size))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"done in {t_run / 3600:.2f} h, {wf.n_calls} likelihood calls; "
          f"Delta lnL_max = rho^2/2 - max lnL = {summary['delta_lnL_max']:.3f}; ln Z = {fs.log_evidence:.2f}")
    for k in ("chirp_mass", "q", "chi_eff", "luminosity_distance", "theta_jn"):
        s = stats[k]
        print(f"  {k:20s} {s['median']:.4g} [{s['q05']:.4g}, {s['q95']:.4g}]  truth {s['truth']:.4g}  "
              f"bias/std {s['bias_over_std']:+.2f}")
    print(f"-> {out}")
    if pool:
        pool.close()


if __name__ == "__main__":
    main()