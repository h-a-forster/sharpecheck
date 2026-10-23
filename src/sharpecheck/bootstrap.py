"""Circular block bootstrap with studentized statistics (milestone M3).

The bootstrap standard error uses the block-sum covariance of Goetze and Kuensch (1996),
as in Ledoit and Wolf (2008) section 3.2.2. Formulas: DESIGN 4.4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from sharpecheck._validation import FloatArray


def default_block(nobs: int) -> int:
    """Default block size ``max(1, round(nobs ** (1/3)))`` (DESIGN 4.4)."""
    raise NotImplementedError("M3")


def circular_block_indices(
    nobs: int, block: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_boot x nobs) integer index matrix of circular block bootstrap resamples."""
    raise NotImplementedError("M3")


def studentized_sr_boot(
    x: ArrayLike, block: int, n_boot: int, seed: int | None = None
) -> tuple[float, float, FloatArray]:
    """Return ``(sr_hat, se_hat, t_star)`` for one series.

    ``t_star[m] = (sr*_m - sr_hat) / se*_m`` (signed; take ``abs`` for the symmetric
    interval). ``se_hat`` is the prewhitened-QS HAC standard error of the original data.
    """
    raise NotImplementedError("M3")


def studentized_diff_boot(
    x: ArrayLike, block: int, n_boot: int, seed: int | None = None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Joint studentized bootstrap of SR differences against column 0 of a (T x k) matrix.

    Returns ``(delta_hat, se_hat, t_star)`` with shapes ``(k-1,)``, ``(k-1,)`` and
    ``(n_boot, k-1)``, where ``delta_j = SR_j - SR_0``. All columns share the resampled
    time indices, so ``t_star`` feeds the Romano-Wolf stepdown directly.
    """
    raise NotImplementedError("M3")


def calibrate_block(
    x: ArrayLike,
    candidates: tuple[int, ...] = (1, 2, 4, 6, 8, 10),
    *,
    level: float = 0.95,
    n_pseudo: int = 1000,
    n_boot: int = 499,
    seed: int | None = None,
) -> tuple[int, dict[int, float]]:
    """Ledoit-Wolf (2008) Algorithm 3.1 block-size calibration for a (T x 2) pair.

    Pseudo data: VAR(1) fit with stationary-bootstrap residuals (mean block 5). Returns
    the chosen block and the estimated coverage g(b) per candidate. Expensive: for
    empirical applications only, never inside the Monte Carlo (DESIGN 2, item 4).
    """
    raise NotImplementedError("M7 (should)")
