"""Allocation rules (milestone M7). Formulas: DESIGN 5.3.

Each rule maps an estimation window of excess returns (M x N, rows are months) to a weight
vector summing to one. No rule sees data after its window.
"""

from __future__ import annotations

from typing import Literal

from numpy.typing import ArrayLike

from sharpecheck._validation import FloatArray

Rule = Literal["equal", "inverse_vol", "gmv_sample", "gmv_lw", "hrp"]


def equal_weight(window: ArrayLike) -> FloatArray:
    """1/N."""
    raise NotImplementedError("M7")


def inverse_vol(window: ArrayLike) -> FloatArray:
    """Weights proportional to 1 / sigma_i (sample standard deviations)."""
    raise NotImplementedError("M7")


def ledoit_wolf_cov(window: ArrayLike) -> tuple[FloatArray, float]:
    """Ledoit-Wolf (2004) shrinkage to the scaled identity; returns (Sigma, delta)."""
    raise NotImplementedError("M7")


def gmv(cov: ArrayLike) -> FloatArray:
    """Unconstrained global minimum variance: w = Sigma^-1 1 / (1' Sigma^-1 1)."""
    raise NotImplementedError("M7")


def hrp(cov: ArrayLike) -> FloatArray:
    """Hierarchical risk parity (Lopez de Prado 2016): single linkage on
    d_ij = sqrt((1 - rho_ij) / 2), quasi-diagonalisation, recursive bisection with
    inverse-variance cluster weights."""
    raise NotImplementedError("M7")


def weights(rule: Rule, window: ArrayLike) -> FloatArray:
    """Dispatch to one rule."""
    raise NotImplementedError("M7")
