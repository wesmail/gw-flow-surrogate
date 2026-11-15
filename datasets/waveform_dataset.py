"""PyTorch Dataset over HDF5 waveform shards.

Design (rewritten):

* Each split is read ONCE, shard-by-shard, with a full ``[:]`` read (never
  per-row indexing into an LZF-compressed dataset).  The tracks are kept in
  process RAM as shared-memory tensors, so DataLoader workers and, within a
  rank, forked processes map the same pages instead of each holding a copy.
* ``__getitem__`` is then just a slice: it returns the RAW (C, T) log_a / phi
  tracks plus metadata.  No HDF5, no Python patch loop, no normalization on the
  CPU worker.  Patchify + normalize + feature-build happen once per batch on the
  GPU (see ``on_after_batch_transfer`` in the LightningModule).
* Generalizable to any N: ``preload=False`` keeps an LRU of *decoded* shards
  (``max_cached_shards`` caps RAM).  Pair with ``ShardGroupedSampler`` so the
  loader finishes one shard before opening the next — random shuffle across
  shards thrash-loads and is unusable on large pools.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from datasets.patchify import patchify_tracks_batched


class ShardGroupedSampler(Sampler[int]):
    """Epoch order: shuffle shards, then shuffle rows within each shard.

    Keeps consecutive ``__getitem__`` calls on one HDF5 file so a 1–2 slot
    full-shard cache stays hot.  Publishes the shard visit plan so the dataset
    can prefetch the next file while the GPU trains on the current one.
    """

    def __init__(
        self,
        file_paths: list[str],
        *,
        seed: int = 0,
        dataset: "WaveformPatchDataset | None" = None,
    ) -> None:
        by_shard: dict[str, list[int]] = defaultdict(list)
        for i, fp in enumerate(file_paths):
            by_shard[fp].append(i)
        self._groups = [
            (fp, np.asarray(ix, dtype=np.int64)) for fp, ix in by_shard.items()
        ]
        self.seed = int(seed)
        self.epoch = 0
        self._n = sum(len(g) for _, g in self._groups)
        self.dataset = dataset

    def __len__(self) -> int:
        return self._n

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self._groups))
        plan: list[str] = []
        seq: list[int] = []
        for gi in order:
            fp, g0 = self._groups[int(gi)]
            plan.append(fp)
            g = g0.copy()
            rng.shuffle(g)
            seq.extend(int(i) for i in g)
        if self.dataset is not None:
            self.dataset.set_shard_plan(plan)
        yield from seq


def resolve_shard_path(path: str | Path, data_dir: str | Path) -> Path:
    """Resolve an HDF5 shard path against the manifest directory.

    Manifests may store absolute paths from an old output location, or bare
    filenames when the dataset folder was moved — try manifest parent / basename.
    """
    p = Path(path)
    if p.is_file():
        return p
    base = Path(data_dir)
    for candidate in (base / p.name, base / p):
        if candidate.is_file():
            return candidate
    return p


class WaveformPatchDataset(Dataset):
    """Load precomputed log_a / phi tracks and emit RAW patchless training samples.

    Patchification is deferred to a batched GPU step; this class only handles
    storage and fast indexing.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        patch_len: int = 32,
        context_patches: int = 0,
        norm_stats_path: str | Path | None = None,
        *,
        preload: bool = True,
        max_cached_shards: int | None = None,
        share_memory: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.data_dir = self.manifest_path.parent
        self.split = split
        self.patch_len = patch_len
        self.context_patches = context_patches
        self.preload = preload
        # Decoded-shard LRU; 2 slots = current + background prefetch.
        if not preload and max_cached_shards is None:
            max_cached_shards = 2
        self.max_cached_shards = max_cached_shards
        self.share_memory = share_memory

        table = pd.read_parquet(self.manifest_path)
        rows = table[table["split"] == split].reset_index(drop=True)
        if rows.empty:
            raise ValueError(f"No rows for split={split!r} in {manifest_path}")
        # Cache the two columns we index by as plain lists (fast, fork-friendly).
        self._file_path = rows["file_path"].tolist()
        self._row_index = rows["row_index"].astype(int).tolist()
        self._n = len(rows)

        # Channel / time geometry from the first shard.
        first = resolve_shard_path(self._file_path[0], self.data_dir)
        with h5py.File(first, "r") as f:
            _, self.n_channels, self.n_time = f["log_a"].shape

        self.norm_stats = self._load_norm_stats(norm_stats_path)

        # Storage backend.
        self._log_a = self._phi = self._theta = self._ref_phase = None  # preload
        self._shard_cache: "OrderedDict[str, dict]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._path_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._shard_plan: list[str] = []
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_fp: str | None = None
        if self.preload:
            self._preload_split()

    # ------------------------------------------------------------------ #
    # Pickle (Python 3.14 DataLoader uses forkserver → must pickle dataset)
    # ------------------------------------------------------------------ #
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Locks / live threads are not picklable; workers rebuild empty caches.
        state["_cache_lock"] = None
        state["_path_locks"] = None
        state["_prefetch_thread"] = None
        state["_prefetch_fp"] = None
        state["_shard_cache"] = OrderedDict()
        state["_shard_plan"] = []
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._cache_lock = threading.Lock()
        self._path_locks = defaultdict(threading.Lock)
        self._prefetch_thread = None
        self._prefetch_fp = None
        if not isinstance(self._shard_cache, OrderedDict):
            self._shard_cache = OrderedDict()
        if self._shard_plan is None:
            self._shard_plan = []

    # ------------------------------------------------------------------ #
    # Norm stats                                                          #
    # ------------------------------------------------------------------ #
    def _load_norm_stats(self, path: str | Path | None) -> dict[str, torch.Tensor]:
        c = self.n_channels
        if path is None or not Path(path).is_file():
            # Identity stats (C,) so features pass through untouched.
            return {
                "log_a_mean": torch.zeros(c),
                "log_a_std": torch.ones(c),
                "delta_phi_mean": torch.zeros(c),
                "delta_phi_std": torch.ones(c),
            }
        data = np.load(path)
        out: dict[str, torch.Tensor] = {}
        for k in ("log_a_mean", "log_a_std", "delta_phi_mean", "delta_phi_std"):
            v = torch.as_tensor(np.asarray(data[k]).reshape(-1), dtype=torch.float32)
            if v.numel() == 1:
                v = v.expand(c).contiguous()
            out[k] = v
        return out

    # ------------------------------------------------------------------ #
    # Preload backend                                                    #
    # ------------------------------------------------------------------ #
    def _preload_split(self) -> None:
        n, c, t = self._n, self.n_channels, self.n_time
        log_a = torch.empty(n, c, t, dtype=torch.float32)
        phi = torch.empty(n, c, t, dtype=torch.float32)
        theta = torch.empty(n, 3, dtype=torch.float32)
        ref_phase = torch.empty(n, dtype=torch.float32)

        # Group destination rows by shard so each shard is opened + read once.
        by_shard: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for dst, (fp, ri) in enumerate(zip(self._file_path, self._row_index)):
            by_shard[fp].append((dst, ri))

        for fp, items in by_shard.items():
            shard = resolve_shard_path(fp, self.data_dir)
            with h5py.File(shard, "r") as f:
                la = f["log_a"][:]        # full shard read (all splits) — one decompress
                ph = f["phi"][:]
                th = f["theta"][:]
                rp = f["ref_phase"][:]
            src = np.fromiter((ri for _, ri in items), dtype=np.int64, count=len(items))
            dst = np.fromiter((d for d, _ in items), dtype=np.int64, count=len(items))
            log_a[dst] = torch.from_numpy(la[src].astype(np.float32, copy=False))
            phi[dst] = torch.from_numpy(ph[src].astype(np.float32, copy=False))
            theta[dst] = torch.from_numpy(th[src].astype(np.float32, copy=False))
            ref_phase[dst] = torch.from_numpy(rp[src].astype(np.float32, copy=False))

        if self.share_memory:
            for tsr in (log_a, phi, theta, ref_phase):
                tsr.share_memory_()
        self._log_a, self._phi, self._theta, self._ref_phase = log_a, phi, theta, ref_phase

    # ------------------------------------------------------------------ #
    # Decoded-shard LRU + background prefetch                             #
    # ------------------------------------------------------------------ #
    def set_shard_plan(self, plan: list[str]) -> None:
        """Visit order for this epoch; prefetch shard 1 while shard 0 loads."""
        self._join_prefetch()
        self._shard_plan = list(plan)
        # Overlap first two shard reads (different files → safe concurrency).
        if len(plan) >= 2 and self.max_cached_shards and self.max_cached_shards >= 2:
            self._start_prefetch(plan[1], keep_fp=plan[0])

    def _join_prefetch(self) -> None:
        t = self._prefetch_thread
        if t is not None and t.is_alive():
            t.join()
        self._prefetch_thread = None
        self._prefetch_fp = None

    def _decode_shard_file(self, shard: Path) -> dict:
        # Per-path lock: same file must not be read from two threads (h5py hang).
        key = str(shard)
        with self._path_locks[key]:
            with h5py.File(shard, "r") as f:
                return {
                    "log_a": np.asarray(f["log_a"][:], dtype=np.float32),
                    "phi": np.asarray(f["phi"][:], dtype=np.float32),
                    "theta": np.asarray(f["theta"][:], dtype=np.float32),
                    "ref_phase": np.asarray(f["ref_phase"][:], dtype=np.float32),
                }

    def _insert_shard(self, fp: str, entry: dict, *, keep_fp: str | None) -> None:
        cache = self._shard_cache
        cache[fp] = entry
        cache.move_to_end(fp)
        while self.max_cached_shards and len(cache) > self.max_cached_shards:
            victim = next((k for k in cache if k != keep_fp and k != fp), None)
            if victim is None:
                break
            del cache[victim]

    def _next_in_plan(self, fp: str) -> str | None:
        try:
            i = self._shard_plan.index(fp)
        except ValueError:
            return None
        if i + 1 < len(self._shard_plan):
            return self._shard_plan[i + 1]
        return None

    def _start_prefetch(self, fp: str, *, keep_fp: str | None) -> None:
        if not self.max_cached_shards or self.max_cached_shards < 2:
            return
        with self._cache_lock:
            if fp in self._shard_cache:
                return
            if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
                return
        shard = resolve_shard_path(fp, self.data_dir)

        def work() -> None:
            entry = self._decode_shard_file(shard)
            with self._cache_lock:
                self._insert_shard(fp, entry, keep_fp=keep_fp)

        self._prefetch_fp = fp
        self._prefetch_thread = threading.Thread(target=work, daemon=True)
        self._prefetch_thread.start()

    def _shard(self, fp: str) -> dict:
        with self._cache_lock:
            hit = self._shard_cache.get(fp)
            if hit is not None:
                self._shard_cache.move_to_end(fp)
        if hit is not None:
            nxt = self._next_in_plan(fp)
            if nxt is not None:
                self._start_prefetch(nxt, keep_fp=fp)
            return hit

        # Finish background load if it's for this shard (or any — I/O is serial).
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._join_prefetch()
            with self._cache_lock:
                hit = self._shard_cache.get(fp)
                if hit is not None:
                    self._shard_cache.move_to_end(fp)
            if hit is not None:
                nxt = self._next_in_plan(fp)
                if nxt is not None:
                    self._start_prefetch(nxt, keep_fp=fp)
                return hit

        entry = self._decode_shard_file(resolve_shard_path(fp, self.data_dir))
        with self._cache_lock:
            self._insert_shard(fp, entry, keep_fp=fp)
        nxt = self._next_in_plan(fp)
        if nxt is not None:
            self._start_prefetch(nxt, keep_fp=fp)
        return entry

    # ------------------------------------------------------------------ #
    # Dataset protocol                                                   #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._n

    def _raw(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.preload:
            return (
                self._log_a[idx],
                self._phi[idx],
                self._theta[idx],
                float(self._ref_phase[idx]),
            )
        fp = self._file_path[idx]
        ri = int(self._row_index[idx])
        e = self._shard(fp)
        return (
            torch.from_numpy(e["log_a"][ri]),
            torch.from_numpy(e["phi"][ri]),
            torch.from_numpy(e["theta"][ri]),
            float(e["ref_phase"][ri]),
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        log_a, phi, theta, ref_phase = self._raw(idx)
        ctx = self.context_patches
        return {
            # RAW tracks — patchified on the GPU, once per batch.
            "log_a_track": log_a,                      # (C, T)
            "phi_track": phi,                          # (C, T)
            "theta": theta,                            # (3,)
            "ref_phase": torch.tensor(ref_phase, dtype=torch.float32),
            "context_patches": torch.tensor(ctx, dtype=torch.long),
            "target_start": torch.tensor(max(ctx, 1), dtype=torch.long),
            # Per-channel (C,) normalization stats, carried so the GPU step and
            # the physical rollout reconstruction have them on-device.
            "norm_log_a_mean": self.norm_stats["log_a_mean"],
            "norm_log_a_std": self.norm_stats["log_a_std"],
            "norm_dphi_mean": self.norm_stats["delta_phi_mean"],
            "norm_dphi_std": self.norm_stats["delta_phi_std"],
        }


def compute_norm_stats(
    manifest_path: str | Path,
    output_path: str | Path,
    patch_len: int = 32,
    max_samples: int = 1000,
    seed: int = 0,
) -> Path:
    """Compute training-set normalization statistics (patch-level log_a and Δφ).

    Streams train rows (no full-split preload) so 1M-scale pools fit in RAM.
    Subsamples up to ``max_samples`` train rows; pass ``max_samples=0`` for all.
    """
    import logging

    log = logging.getLogger(__name__)
    # preload=False: never materialise the full train split in RAM.
    ds = WaveformPatchDataset(
        manifest_path,
        split="train",
        patch_len=patch_len,
        preload=False,
        max_cached_shards=2,
    )
    n = len(ds)
    if max_samples > 0 and n > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(n, size=max_samples, replace=False))
        log.info("Computing norm_stats from %d / %d train samples", len(indices), n)
    else:
        indices = np.arange(n)
        log.info("Computing norm_stats from all %d train samples (streaming)", n)

    # Group selected rows by shard; decompress one shard at a time (LZF-safe).
    by_shard: dict[str, list[int]] = defaultdict(list)
    for src_i in indices.tolist():
        by_shard[ds._file_path[src_i]].append(int(ds._row_index[src_i]))

    c = ds.n_channels
    count = 0
    la_sum = torch.zeros(c, dtype=torch.float64)
    la_sq = torch.zeros(c, dtype=torch.float64)
    dp_sum = torch.zeros(c, dtype=torch.float64)
    dp_sq = torch.zeros(c, dtype=torch.float64)
    chunk = 256
    data_dir = ds.data_dir

    for fp, row_indices in by_shard.items():
        shard = resolve_shard_path(fp, data_dir)
        src = np.sort(np.asarray(row_indices, dtype=np.int64))
        with h5py.File(shard, "r") as f:
            n_rows = int(f["log_a"].shape[0])
            # Sparse subsample: fancy-index selected rows only.
            # Dense / full-shard: one [:] decompress (LZF-safe), then select.
            if src.size < max(64, n_rows // 8):
                log_a_np = np.asarray(f["log_a"][src], dtype=np.float32)
                phi_np = np.asarray(f["phi"][src], dtype=np.float32)
            else:
                log_a_np = np.asarray(f["log_a"][:], dtype=np.float32)[src]
                phi_np = np.asarray(f["phi"][:], dtype=np.float32)[src]
        log_a = torch.from_numpy(log_a_np)
        phi = torch.from_numpy(phi_np)
        for s in range(0, log_a.shape[0], chunk):
            p = patchify_tracks_batched(log_a[s : s + chunk], phi[s : s + chunk], patch_len)
            la = p["log_a"].reshape(-1, c).double()
            dp = p["delta_phi"].reshape(-1, c).double()
            count += la.shape[0]
            la_sum += la.sum(0)
            la_sq += (la * la).sum(0)
            dp_sum += dp.sum(0)
            dp_sq += (dp * dp).sum(0)
        del log_a, phi, log_a_np, phi_np

    la_mean = la_sum / count
    la_std = (la_sq / count - la_mean * la_mean).clamp_min(0).sqrt().clamp_min(1e-8)
    dp_mean = dp_sum / count
    dp_std = (dp_sq / count - dp_mean * dp_mean).clamp_min(0).sqrt().clamp_min(1e-8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        log_a_mean=la_mean.float().numpy(),
        log_a_std=la_std.float().numpy(),
        delta_phi_mean=dp_mean.float().numpy(),
        delta_phi_std=dp_std.float().numpy(),
    )
    return output_path