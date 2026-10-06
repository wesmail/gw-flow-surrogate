"""PyTorch Lightning DataModule for patchified waveforms.

The collate function now stacks RAW (C, T) tracks — since every waveform shares
the same fixed time grid, all samples have identical length and no padding is
needed.  Patchify + normalize + feature-build run once per batch on the GPU in
the LightningModule's ``on_after_batch_transfer`` hook.
"""

from __future__ import annotations

from pathlib import Path

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from datasets.waveform_dataset import WaveformPatchDataset, compute_norm_stats


_NORM_KEYS = ("norm_log_a_mean", "norm_log_a_std", "norm_dphi_mean", "norm_dphi_std")


def _collate_batch(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Stack raw tracks + metadata. All tracks share one fixed grid → plain stack."""
    batch = {
        "log_a_track": torch.stack([s["log_a_track"] for s in samples]),   # (B, C, T)
        "phi_track": torch.stack([s["phi_track"] for s in samples]),       # (B, C, T)
        "theta": torch.stack([s["theta"] for s in samples]),               # (B, 3)
        "ref_phase": torch.stack([s["ref_phase"] for s in samples]),       # (B,)
        "context_patches": torch.stack([s["context_patches"] for s in samples]),
        "target_start": torch.stack([s["target_start"] for s in samples]),
    }
    # Per-channel norm stats (identical across the batch) -> (B, C).
    for k in _NORM_KEYS:
        batch[k] = torch.stack([s[k] for s in samples])
    return batch


class WaveformDataModule(LightningDataModule):
    """Train / val / test loaders over HDF5 waveform manifests."""

    def __init__(
        self,
        manifest_path: str = "data/generated/manifest.parquet",
        norm_stats_path: str | None = None,
        batch_size: int = 8,
        num_workers: int = 2,
        patch_len: int = 32,
        context_patches: int = 0,
        compute_stats_if_missing: bool = True,
        norm_stats_max_samples: int = 1000,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
        preload: bool = True,
        max_cached_shards: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def _make_dataloader(self, dataset: WaveformPatchDataset, *, shuffle: bool) -> DataLoader:
        nw = int(self.hparams.num_workers)
        kwargs: dict = dict(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=nw,
            collate_fn=_collate_batch,
            pin_memory=torch.cuda.is_available(),
        )
        if nw > 0:
            kwargs["persistent_workers"] = bool(self.hparams.persistent_workers)
            kwargs["prefetch_factor"] = int(self.hparams.prefetch_factor)
        return DataLoader(**kwargs)

    def setup(self, stage: str | None = None) -> None:
        manifest = Path(self.hparams.manifest_path)
        stats_path = self.hparams.norm_stats_path
        if stats_path is None:
            # Patch-level stats from this module; data_generation/norm_stats.json is redundant.
            stats_path = str(manifest.parent / "norm_stats.npz")
        if self.hparams.compute_stats_if_missing and not Path(stats_path).is_file():
            compute_norm_stats(
                manifest,
                stats_path,
                self.hparams.patch_len,
                max_samples=self.hparams.norm_stats_max_samples,
            )

        kwargs = dict(
            manifest_path=manifest,
            patch_len=self.hparams.patch_len,
            context_patches=self.hparams.context_patches,
            norm_stats_path=stats_path,
            preload=bool(self.hparams.preload),
            max_cached_shards=self.hparams.max_cached_shards,
        )
        if stage in ("fit", None):
            self.train_ds = WaveformPatchDataset(split="train", **kwargs)
            self.val_ds = WaveformPatchDataset(split="val", **kwargs)
        if stage in ("test", None):
            self.test_ds = WaveformPatchDataset(split="test", **kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.test_ds, shuffle=False)