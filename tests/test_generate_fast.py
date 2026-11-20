"""generate vs generate_fast; shared-noise PE path; FMSource start-token check."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from datasets.datamodule import _collate_batch
from datasets.patchify import n_patches_for
from datasets.waveform_dataset import WaveformPatchDataset
from models.lightning_module import GWFlowSurrogateLit


CKPT = _ROOT / "last.ckpt"
MANIFEST = _ROOT / "data" / "n1000k" / "manifest.parquet"
NORM = _ROOT / "data" / "n1000k" / "norm_stats.npz"
N_WAVEFORMS = 64
SEED = 1234
N_STEPS = 12
MEAN_OF = 1


@pytest.mark.skipif(not CKPT.is_file(), reason="last.ckpt missing")
@pytest.mark.skipif(not MANIFEST.is_file(), reason="manifest missing")
def test_generate_fast_matches_generate():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Gate A capture-route check")
    dev = torch.device("cuda")
    lit = GWFlowSurrogateLit.load_from_checkpoint(str(CKPT), map_location=dev).eval().to(dev)
    P = int(lit.hparams.patch_len)
    ds = WaveformPatchDataset(
        str(MANIFEST), split="test", patch_len=P, context_patches=1,
        norm_stats_path=str(NORM), preload=False,
    )
    N = n_patches_for(ds.n_time, P)
    sub = Subset(ds, range(min(N_WAVEFORMS, len(ds))))
    batch = next(iter(DataLoader(sub, batch_size=len(sub), collate_fn=_collate_batch, shuffle=False)))
    b = lit._patchify_batch({k: v.to(dev) for k, v in batch.items()})
    seed = b["token_features"][:, :1]
    theta = b["theta"]

    torch.manual_seed(SEED)
    with torch.no_grad():
        slow = lit.model.generate(
            seed, theta, N - 1, n_steps=N_STEPS, n_samples=1,
            temperature=1.0, mean_of=MEAN_OF,
        )
    torch.manual_seed(SEED)
    with torch.no_grad():
        fast = lit.model.generate_fast(
            seed, theta, N - 1, n_steps=N_STEPS, n_samples=1,
            temperature=1.0, mean_of=MEAN_OF,
        )

    diff = (slow - fast).abs()
    max_diff = float(diff.max())
    med_diff = float(diff.median())
    bit_identical = bool(torch.equal(slow, fast))
    route = getattr(lit.model, "_fast_capture_route", "?")
    print(f"\nGate A: max|diff|={max_diff:.3e}  median|diff|={med_diff:.3e}  "
          f"bit_identical={bit_identical}  route={route}")
    if hasattr(lit.model, "_fast_compile_error"):
        print(f"  compile_error={lit.model._fast_compile_error}")
    if hasattr(lit.model, "_fast_cudagraph_error"):
        print(f"  cudagraph_error={lit.model._fast_cudagraph_error}")

    assert slow.shape == fast.shape
    # Eager (sliced KV): expect bit-identity / 1e-4.  Compile/graph use full
    # buffer + mask → different kernels; allow up to 1e-3 over 320 AR steps.
    tol = 1e-4 if str(route).startswith("eager") else 1e-3
    assert max_diff <= tol, f"max|diff|={max_diff} exceeds {tol} (route={route})"
    if str(route).startswith("eager"):
        raise AssertionError(
            f"both capture routes fell back to eager on CUDA; "
            f"compile={getattr(lit.model, '_fast_compile_error', None)}; "
            f"cudagraph={getattr(lit.model, '_fast_cudagraph_error', None)}"
        )


@pytest.mark.skipif(not CKPT.is_file(), reason="last.ckpt missing")
@pytest.mark.skipif(not MANIFEST.is_file(), reason="manifest missing")
def test_shared_noise_batched_matches_serial_generate():
    """Shared noise: batched generate_fast(noise=) ≡ loop of generate(B=1)+seed."""
    from pe.fm_waveform import FMSource

    # Build FMSource for norms/reconstruct helpers; avoid NR evaluate in this test
    src = FMSource(str(CKPT), str(MANIFEST), str(NORM), n_steps=N_STEPS, seed=SEED)
    ds = WaveformPatchDataset(
        str(MANIFEST), split="test", patch_len=src.P, context_patches=1,
        norm_stats_path=str(NORM), preload=False,
    )
    sub = Subset(ds, range(8))
    batch = next(iter(DataLoader(sub, batch_size=8, collate_fn=_collate_batch, shuffle=False)))
    b = src.lit._patchify_batch({k: v.to(src.dev) for k, v in batch.items()})
    first = b["token_features"][:, :1]
    theta = b["theta"]
    B, D, Ngen = first.shape[0], src.D, src.N - 1
    torch = src.torch

    torch.manual_seed(SEED)
    base_noise = [torch.randn(1, D, device=src.dev, dtype=torch.float32) for _ in range(Ngen)]
    noise = [z.expand(B, -1).contiguous() for z in base_noise]

    with torch.inference_mode():
        fast = src.lit.model.generate_fast(
            first, theta, Ngen, n_steps=N_STEPS, n_samples=1,
            temperature=1.0, mean_of=1, noise=noise,
        )[:, 0]
        h_batch = src._reconstruct(torch.cat([first, fast], 1), b).cpu().numpy()

        outs = []
        for i in range(B):
            torch.manual_seed(SEED)
            gen = src.lit.model.generate(
                first[i:i + 1], theta[i:i + 1], Ngen, n_steps=N_STEPS, n_samples=1,
                temperature=1.0, mean_of=1,
            )[:, 0]
            outs.append(src._reconstruct(
                torch.cat([first[i:i + 1], gen], 1),
                {k: (v[i:i + 1] if torch.is_tensor(v) and v.shape[0] == B else v) for k, v in b.items()},
            ).cpu().numpy()[0])
    h_ref = np.stack(outs)
    route = getattr(src.lit.model, "_fast_capture_route", "eager")
    scale = np.max(np.abs(h_ref))
    rel = np.max(np.abs(h_batch - h_ref)) / max(scale, 1e-30)
    print(f"\nshared-noise: max relative |diff|={rel:.3e}  route={route}")
    assert rel <= 1e-5, f"max relative |diff|={rel} exceeds 1e-5"


@pytest.mark.skipif(not CKPT.is_file(), reason="last.ckpt missing")
@pytest.mark.skipif(not MANIFEST.is_file(), reason="manifest missing")
def test_fmsource_start_token_and_true_reconstruction():
    """FMSource._batch_from_records matches dataset patchify (start token + true recon)."""
    from types import SimpleNamespace
    from pe.fm_waveform import FMSource

    src = FMSource(str(CKPT), str(MANIFEST), str(NORM), n_steps=N_STEPS, seed=SEED)
    ds = WaveformPatchDataset(
        str(MANIFEST), split="test", patch_len=src.P, context_patches=1,
        norm_stats_path=str(NORM), preload=False,
    )
    sub = Subset(ds, range(4))
    batch = next(iter(DataLoader(sub, batch_size=4, collate_fn=_collate_batch, shuffle=False)))
    b_ds = src.lit._patchify_batch({k: v.to(src.dev) for k, v in batch.items()})

    # Feed the same stored tracks through FMSource._batch_from_records
    recs = [
        SimpleNamespace(
            log_a=batch["log_a_track"][i].numpy(),
            phi=batch["phi_track"][i].numpy(),
            h=np.zeros((1, batch["log_a_track"].shape[-1]), np.complex64),
        )
        for i in range(4)
    ]
    th = batch["theta"].numpy()
    b_fm = src._batch_from_records(recs, th)

    d0 = (b_ds["token_features"][:, 0] - b_fm["token_features"][:, 0]).abs()
    print(f"\nstart-token max|diff|={float(d0.max()):.3e}")
    assert float(d0.max()) < 1e-5

    with src.torch.inference_mode():
        h_ds = src._reconstruct(b_ds["token_features"], b_ds)
        h_fm = src._reconstruct(b_fm["token_features"], b_fm)
    dh = (h_ds - h_fm).abs()
    print(f"true-token recon max|diff|={float(dh.max()):.3e}")
    assert float(dh.max()) < 1e-5

    # GenerationConfig matches the pool shard
    cfg = src.nr.cfg
    assert cfg.t_min == -5000.0 and cfg.t_max == 130.0 and cfg.dt == 0.5
    assert cfg.t_ref == -5000.0 and list(cfg.modes) == [(2, 2)]
    print(f"NRSource cfg: t_min={cfg.t_min} t_max={cfg.t_max} dt={cfg.dt} modes={cfg.modes}")
