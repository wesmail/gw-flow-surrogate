"""Shared publication style for all FM-surrogate figures (talk + paper).

Import in every figure script (run from repo root)::

    from style import apply_style, save, C

Colors are colorblind-safe (Okabe-Ito derived); fonts embed in PDF (Type 42)
so figures survive journal pipelines and Illustrator edits.

Outputs go to ``paper_figs/out/`` by default (PNG for slides + PDF for paper).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Default output directory for regenerated publication figures.
OUT_DIR = Path(__file__).resolve().parent / "out"

C = dict(
    fm="#0072B2",        # flow matching (blue)
    fm_light="#8FC3E4",
    accent="#D55E00",    # highlights / targets (vermillion)
    gray="#7F7F7F",
    good="#009E73",      # green (e.g., threshold-passed regions)
    band="#B7D7EC",
)


def apply_style(talk: bool = False) -> None:
  """talk=True -> larger fonts for slides; False -> paper (single-column)."""
  base = 14 if talk else 10
  plt.rcParams.update({
    "font.size": base,
    "axes.labelsize": base,
    "axes.titlesize": base + 1,
    "legend.fontsize": base - 2,
    "xtick.labelsize": base - 1,
    "ytick.labelsize": base - 1,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
  })


def save(fig, name: str, outdir: str | Path | None = None) -> None:
  """Write PNG (slides/preview) + PDF (paper) with one call."""
  out = Path(outdir) if outdir is not None else OUT_DIR
  out.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / f"{name}.png")
  fig.savefig(out / f"{name}.pdf")
  print(f"-> {out}/{name}.png|.pdf")
