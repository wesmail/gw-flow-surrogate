"""Quasi-random parameter sampling (§3.2)."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from data_generation.config import GenerationConfig


def sobol_parameters(
    cfg: GenerationConfig,
    n: int,
    seed: int,
    *,
    skip: int = 0,
) -> np.ndarray:
    """Draw n points from a scrambled Sobol sequence over [q, chi1z, chi2z].

    Returns array of shape (n, 3).
    """
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    # Burn-in skip avoids low-discrepancy artifacts at the sequence start.
    if skip > 0:
        _ = sampler.random(skip)
    unit = sampler.random(n)
    q = cfg.q_min + unit[:, 0] * (cfg.q_max - cfg.q_min)
    chi1z = cfg.chi_min + unit[:, 1] * (cfg.chi_max - cfg.chi_min)
    chi2z = cfg.chi_min + unit[:, 2] * (cfg.chi_max - cfg.chi_min)
    return np.column_stack([q, chi1z, chi2z]).astype(np.float64)


def uniform_parameters(
    cfg: GenerationConfig,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform random parameters for the held-out test set (§3.2)."""
    q = rng.uniform(cfg.q_min, cfg.q_max, size=n)
    chi1z = rng.uniform(cfg.chi_min, cfg.chi_max, size=n)
    chi2z = rng.uniform(cfg.chi_min, cfg.chi_max, size=n)
    return np.column_stack([q, chi1z, chi2z]).astype(np.float64)


def extrapolation_parameters(
    n: int,
    rng: np.random.Generator,
    *,
    q_range: tuple[float, float] = (8.0, 10.0),
    chi_range: tuple[float, float] = (0.8, 0.9),
) -> np.ndarray:
    """OOD extrapolation test points just outside the training domain."""
    q = rng.uniform(*q_range, size=n)
    chi1z = rng.uniform(*chi_range, size=n) * rng.choice([-1.0, 1.0], size=n)
    chi2z = rng.uniform(*chi_range, size=n) * rng.choice([-1.0, 1.0], size=n)
    return np.column_stack([q, chi1z, chi2z]).astype(np.float64)


def make_splits(
    n_train: int,
    n_val: int,
    n_test: int,
    n_ood: int,
    seed: int,
) -> dict[str, slice]:
    """Index slices for train / val / test / ood within a single parameter table."""
    total = n_train + n_val + n_test + n_ood
    return {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n_train + n_val + n_test),
        "ood": slice(n_train + n_val + n_test, total),
    }
