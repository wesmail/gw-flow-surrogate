"""Turn the trained flow-matching surrogate (and NRHybSur3dq8) into bilby waveforms.

One pipeline, two sources of h22(t) in geometric units on the TRAINING grid
(t = -5000 ... +119.5 M, dt = 0.5 M, peak at t = 0, phase fixed at t = -5000 M):

  NRSource  NRHybSur3dq8 through the project's own data_generation.evaluate_waveform
            (identical conventions to the training data)
  FMSource  our model, theta-only, frozen protocol (12 steps, 1 sample, PCHIP amplitude);
            start token = the first true patch, from NRSource (the paper states this)

Both then go through the SAME conversion (to_frequency_domain):
  geometric units -> seconds and strain at distance d_L -> (2,+-2) modes combined into
  h+ and hx at inclination theta_jn and phase -> smooth taper at both ends ->
  interpolation onto bilby's time grid (amplitude and phase separately, which is far
  more accurate than interpolating the oscillating signal) -> FFT with bilby's
  normalisation. Merger at t = 0, earlier times wrapped to the end of the segment
  (bilby then moves the signal to geocent_time itself).

DETERMINISM (important for a likelihood): the model is random. Every waveform uses
the SAME fixed noise draw, independent of theta and of its position in a batch:
the draw that `generate` makes for a single waveform (batch 1) after
torch.manual_seed(seed). So FMSource(theta) is a fixed, smooth function of theta.
Batched evaluation uses `generate_fast(..., noise=[...])`.
"""
from __future__ import annotations

import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

MTSUN_SI = 4.925490947641267e-06     # G Msun / c^3  [s]
MRSUN_SI = 1476.6250385063112        # G Msun / c^2  [m]
MPC_SI = 3.085677581491367e22        # 1 Mpc [m]

# Module-level cfg for ProcessPool workers (picklable via global).
_POOL_CFG = None


def _eval_one(args):
    from data_generation.generate import evaluate_waveform
    q, chi1, chi2 = args
    return evaluate_waveform(float(q), float(chi1), float(chi2), _POOL_CFG)


def _pool_init(cfg):
    global _POOL_CFG
    _POOL_CFG = cfg


def generation_config_from_pool(manifest: str | Path | None = None):
    """Load GenerationConfig matching data/n1000k (from a shard's generation_config)."""
    from data_generation.config import GenerationConfig, load_config_from_h5

    if manifest is not None:
        import pandas as pd
        man = pd.read_parquet(manifest)
        shard = Path(man["file_path"].iloc[0])
        if not shard.is_file():
            shard = Path(manifest).parent / shard.name
        cfg = load_config_from_h5(str(shard))
        if cfg is not None:
            return cfg
    # Defaults match the audited n1000k pool (t_min=-5000, t_max=130, dt=0.5, (2,2)).
    return GenerationConfig(output_dir="unused")


# --------------------------------------------------------------------------- #
# Source 1: NRHybSur3dq8 exactly as in the training data                       #
# --------------------------------------------------------------------------- #
class NRSource:
    """h22 on the training grid via the project's evaluate_waveform."""

    def __init__(self, n_use: int | None = None, cfg=None, manifest: str | None = None):
        self.cfg = cfg or generation_config_from_pool(manifest)
        self.t_M = np.asarray(self.cfg.times, dtype=np.float64)
        self.n_use = n_use or self.t_M.size
        self.t_M = self.t_M[: self.n_use]
        self._pool_workers = max(1, int(os.cpu_count() or 1))

    def records(self, q, chi1, chi2, *, parallel: bool | None = None):
        """Evaluate waveforms.  Parallel process pool when B > 64 (default)."""
        from data_generation.generate import evaluate_waveform
        q = np.atleast_1d(np.asarray(q, float))
        chi1 = np.atleast_1d(np.asarray(chi1, float))
        chi2 = np.atleast_1d(np.asarray(chi2, float))
        B = int(q.size)
        use_pool = (B > 64) if parallel is None else bool(parallel) and B > 1
        if not use_pool:
            return [evaluate_waveform(float(a), float(b), float(c), self.cfg)
                    for a, b, c in zip(q, chi1, chi2)]
        args = list(zip(q.tolist(), chi1.tolist(), chi2.tolist()))
        n_workers = min(self._pool_workers, B)
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_pool_init, initargs=(self.cfg,),
        ) as ex:
            return list(ex.map(_eval_one, args, chunksize=max(1, B // (n_workers * 4))))

    def h22(self, q, chi1, chi2) -> np.ndarray:
        """(B, T) complex128, geometric units (r h / M)."""
        recs = self.records(q, chi1, chi2)
        return np.stack([np.asarray(r.h[0], np.complex128)[: self.n_use] for r in recs])


# --------------------------------------------------------------------------- #
# Source 2: our trained model                                                  #
# --------------------------------------------------------------------------- #
class FMSource:
    """h22 from the flow-matching surrogate, theta-only, frozen protocol."""

    def __init__(self, checkpoint: str, manifest: str, norm_stats: str, device: str | None = None,
                 n_steps: int = 12, seed: int = 1234, amp_mode: str = "pchip"):
        import torch
        from datasets.patchify import n_patches_for
        from datasets.waveform_dataset import WaveformPatchDataset
        from models.lightning_module import GWFlowSurrogateLit
        self.torch = torch
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lit = GWFlowSurrogateLit.load_from_checkpoint(checkpoint, map_location=self.dev).eval().to(self.dev)
        self.P = int(self.lit.hparams.patch_len)
        self.rscale = float(self.lit.hparams.residual_scale)
        self.n_steps, self.seed, self.amp_mode = n_steps, seed, amp_mode
        # norm stats exactly as the dataset loads them (same keys, same dtype)
        ds = WaveformPatchDataset(manifest, split="test", patch_len=self.P, context_patches=1,
                                  norm_stats_path=norm_stats, preload=False)
        # npz keys: log_a_*, delta_phi_*; batch keys for _patchify_batch: norm_*
        need = ("log_a_mean", "log_a_std", "delta_phi_mean", "delta_phi_std")
        for k in need:
            if k not in ds.norm_stats:
                raise KeyError(f"norm_stats missing {k!r}; have {sorted(ds.norm_stats)}")
        self.norm = {
            "norm_log_a_mean": ds.norm_stats["log_a_mean"],
            "norm_log_a_std": ds.norm_stats["log_a_std"],
            "norm_dphi_mean": ds.norm_stats["delta_phi_mean"],
            "norm_dphi_std": ds.norm_stats["delta_phi_std"],
        }
        self.N = n_patches_for(ds.n_time, self.P)
        self.n_use = self.N * self.P
        self.nr = NRSource(n_use=self.n_use, manifest=manifest)
        self.t_M = self.nr.t_M
        self.D = self.lit.model.feat_dim
        # the fixed noise: what generate() draws for ONE waveform after manual_seed(seed)
        torch.manual_seed(seed)
        self.base_noise = [torch.randn(1, self.D, device=self.dev, dtype=torch.float32)
                           for _ in range(self.N - 1)]
        self._fast_ok = None

    # -- the first true patch, as normalised token features ------------------ #
    def _batch_from_records(self, recs, theta):
        torch = self.torch
        B = len(recs)
        # records store (C, T); stack -> (B, C, T) as DataLoader collate does
        batch = {
            "log_a_track": torch.from_numpy(np.stack([r.log_a for r in recs]).astype(np.float32)),
            "phi_track": torch.from_numpy(np.stack([r.phi for r in recs]).astype(np.float32)),
            "theta": torch.as_tensor(theta, dtype=torch.float32),
        }
        for k, v in self.norm.items():
            batch[k] = torch.as_tensor(np.asarray(v), dtype=torch.float32)[None].expand(B, -1).clone()
        return self.lit._patchify_batch({k: v.to(self.dev) for k, v in batch.items()})

    # -- tokens -> h22, identical to the evaluation scripts ------------------- #
    def _reconstruct(self, feat, b):
        torch = self.torch
        from models.amp_interp import log_amp_per_sample
        P = self.P
        s_frac = torch.arange(P, device=self.dev, dtype=torch.float64) / P
        feat = feat.double()
        la = log_amp_per_sample(feat[..., 0] * b["norm_log_a_std"][:, :1].double()
                                + b["norm_log_a_mean"][:, :1].double(), P, self.amp_mode)
        dphi = feat[..., 1] * b["norm_dphi_std"][:, :1].double() + b["norm_dphi_mean"][:, :1].double()
        res = feat[..., 2:2 + P] * self.rscale
        bnd = torch.cumsum(dphi, dim=1) - dphi
        phase = (bnd[..., None] + s_frac * dphi[..., None] + res).reshape(feat.shape[0], -1)
        return torch.exp(la) * torch.exp(-1j * phase)

    def _generate(self, first, theta):
        """(B, N-1, D) generated tokens with the shared fixed noise."""
        torch = self.torch
        m = self.lit.model
        B = first.shape[0]
        if self._fast_ok is not False:
            try:
                noise = [z.expand(B, -1).contiguous() for z in self.base_noise]
                out = m.generate_fast(first, theta, self.N - 1, n_steps=self.n_steps, n_samples=1,
                                      temperature=1.0, mean_of=1, noise=noise)
                self._fast_ok = True
                return out[:, 0]
            except (TypeError, AttributeError) as e:
                self._fast_ok = False
                warnings.warn(
                    f"generate_fast(noise=...) not available ({e!r}): falling back to one "
                    "waveform at a time (correct, but slow)."
                )
        outs = []
        for i in range(B):                       # reference semantics: batch 1, fixed seed
            torch.manual_seed(self.seed)
            outs.append(m.generate(first[i:i + 1], theta[i:i + 1], self.N - 1, n_steps=self.n_steps,
                                   n_samples=1, temperature=1.0, mean_of=1)[:, 0])
        return torch.cat(outs, 0)

    def h22(self, q, chi1, chi2) -> np.ndarray:
        """(B, T) complex128, geometric units, on self.t_M."""
        torch = self.torch
        q, chi1, chi2 = (np.atleast_1d(np.asarray(x, np.float64)) for x in (q, chi1, chi2))
        recs = self.nr.records(q, chi1, chi2)
        theta = np.stack([q, chi1, chi2], 1)
        with torch.inference_mode():
            b = self._batch_from_records(recs, theta)
            first = b["token_features"][:, :1]
            gen = self._generate(first, b["theta"])
            h = self._reconstruct(torch.cat([first, gen], dim=1), b)
        return h.cpu().numpy()


# --------------------------------------------------------------------------- #
# The shared conversion: geometric h22 -> bilby frequency-domain h+, hx        #
# --------------------------------------------------------------------------- #
def _cos_window(x: np.ndarray) -> np.ndarray:
    """0 -> 1 smoothly for x in [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


def to_frequency_domain(t_M: np.ndarray, h22: np.ndarray, total_mass: np.ndarray,
                        luminosity_distance: np.ndarray, theta_jn: np.ndarray, phase: np.ndarray,
                        frequency_array: np.ndarray, taper_start_M: float = 300.0,
                        taper_end_M: float = 20.0) -> dict:
    """h22 (B, T) in geometric units on t_M -> {"plus","cross"}: (B, n_f) bilby FD strain."""
    h22 = np.atleast_2d(h22)
    B = h22.shape[0]
    M = np.broadcast_to(np.asarray(total_mass, float), (B,))
    dL = np.broadcast_to(np.asarray(luminosity_distance, float), (B,))
    inc = np.broadcast_to(np.asarray(theta_jn, float), (B,))
    ph = np.broadcast_to(np.asarray(phase, float), (B,))

    df = frequency_array[1] - frequency_array[0]
    duration = 1.0 / df
    fs = 2.0 * frequency_array[-1]
    n = int(round(duration * fs))
    tt = np.arange(n) / fs
    tt = np.where(tt < duration / 2, tt, tt - duration)          # merger at 0, past wrapped
    order = np.argsort(tt); tt_sorted = tt[order]

    # taper in M units: on at the start, off at the very end of the ringdown
    w = _cos_window((t_M - t_M[0]) / taper_start_M) * _cos_window((t_M[-1] - t_M) / taper_end_M)
    amp = np.abs(h22) * w[None]
    phi = np.unwrap(-np.angle(h22), axis=1)                        # h22 = A exp(-i phi)

    c = math.sqrt(5.0 / (64.0 * math.pi))
    hp_f = np.empty((B, frequency_array.size), complex)
    hc_f = np.empty_like(hp_f)
    for i in range(B):
        Ms = M[i] * MTSUN_SI
        if -t_M[0] * Ms > duration / 2:
            raise ValueError(f"waveform ({-t_M[0] * Ms:.1f} s) longer than half the segment "
                             f"({duration / 2:.1f} s): use a longer duration")
        tm = tt_sorted / Ms                                        # target times in M units
        inside = (tm >= t_M[0]) & (tm <= t_M[-1])
        A = np.zeros_like(tm); P = np.zeros_like(tm)
        A[inside] = np.interp(tm[inside], t_M, amp[i])
        P[inside] = np.interp(tm[inside], t_M, phi[i])
        h = A * np.exp(-1j * P) * (M[i] * MRSUN_SI / (dL[i] * MPC_SI))    # h22 in strain
        y22 = c * (1 + np.cos(inc[i])) ** 2 * np.exp(2j * ph[i])
        y2m2 = c * (1 - np.cos(inc[i])) ** 2 * np.exp(-2j * ph[i])
        hpc = h * y22 + np.conj(h) * y2m2                          # h+ - i hx ; h_{2,-2} = conj(h22)
        hp = np.empty(n); hc = np.empty(n)
        hp[order] = hpc.real; hc[order] = -hpc.imag
        hp_f[i] = np.fft.rfft(hp) / fs
        hc_f[i] = np.fft.rfft(hc) / fs
    return {"plus": hp_f, "cross": hc_f}


# --------------------------------------------------------------------------- #
# bilby source-model functions                                                 #
# --------------------------------------------------------------------------- #
def make_bilby_source(source):
    """A bilby frequency_domain_source_model for one parameter set at a time.

    Use with parameter_conversion=convert_to_lal_binary_black_hole_parameters
    (detector-frame mass_1 >= mass_2, aligned spins chi_1, chi_2)."""
    def model(frequency_array, mass_1, mass_2, chi_1, chi_2, luminosity_distance,
              theta_jn, phase, **kwargs):
        m1, m2 = max(mass_1, mass_2), min(mass_1, mass_2)
        c1, c2 = (chi_1, chi_2) if mass_1 >= mass_2 else (chi_2, chi_1)
        h22 = source.h22(m1 / m2, c1, c2)
        out = to_frequency_domain(source.t_M, h22, m1 + m2, luminosity_distance, theta_jn, phase,
                                  frequency_array)
        return {"plus": out["plus"][0], "cross": out["cross"][0]}
    return model


def batch_frequency_domain(source, frequency_array, mass_1, mass_2, chi_1, chi_2,
                           luminosity_distance, theta_jn, phase) -> dict:
    """Many parameter sets at once (arrays): for the vectorised likelihood."""
    m1 = np.maximum(mass_1, mass_2); m2 = np.minimum(mass_1, mass_2)
    swap = np.asarray(mass_1) < np.asarray(mass_2)
    c1 = np.where(swap, chi_2, chi_1); c2 = np.where(swap, chi_1, chi_2)
    h22 = source.h22(m1 / m2, c1, c2)
    return to_frequency_domain(source.t_M, h22, m1 + m2, luminosity_distance, theta_jn, phase,
                               frequency_array)
