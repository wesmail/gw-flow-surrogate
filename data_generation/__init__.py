"""NRHybSur waveform generation with amplitude–phase decomposition."""

from data_generation.config import GenerationConfig

__all__ = ["GenerationConfig"]


def __getattr__(name: str):
  if name in {"generate_dataset", "combine_manifests"}:
    from data_generation.generate import combine_manifests, generate_dataset
    return {"generate_dataset": generate_dataset, "combine_manifests": combine_manifests}[name]
  raise AttributeError(name)
