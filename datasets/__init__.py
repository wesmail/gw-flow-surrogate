from datasets.patchify import build_token_features, patchify_tracks, patches_to_token_features
from datasets.waveform_dataset import WaveformPatchDataset, compute_norm_stats

__all__ = [
    "WaveformDataModule",
    "WaveformPatchDataset",
    "compute_norm_stats",
    "build_token_features",
    "patchify_tracks",
    "patches_to_token_features",
]


def __getattr__(name: str):
    if name == "WaveformDataModule":
        from datasets.datamodule import WaveformDataModule
        return WaveformDataModule
    raise AttributeError(name)
