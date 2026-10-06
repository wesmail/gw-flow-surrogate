# PyTorch imports
import torch

# PyTorch Lightning imports
from lightning.pytorch.cli import LightningCLI

from models.lightning_module import GWFlowSurrogateLit
from datasets.datamodule import WaveformDataModule


def cli_main() -> None:
  """Entry point: python main.py fit -c configs/stage1.yaml"""
  LightningCLI(
    GWFlowSurrogateLit,
    WaveformDataModule,
    save_config_callback=None,
  )


if __name__ == "__main__":
  torch.set_float32_matmul_precision("medium")
  cli_main()