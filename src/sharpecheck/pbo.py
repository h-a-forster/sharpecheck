"""Probability of backtest overfitting by combinatorially symmetric cross-validation (CSCV).

Bailey, Borwein, Lopez de Prado and Zhu (2017), Algorithm 2.3, with these conventions
(docs/pbo.md lists where other implementations differ):

- T must be divisible by S (S even): S contiguous blocks of T/S rows, every one of the
  C(S, S/2) combinations is used, each in lexicographic order of its IS blocks.
- Metric: per-period Sharpe ratio, mean/std with ddof 0 (DESIGN 3), on the joined blocks.
- IS winner n*: first index of the maximum IS metric.
- OOS rank of n*: average rank, 1 = worst, N = best; omega = r / (N + 1),
  lambda = ln(omega / (1 - omega)) is always finite.
- PBO = share of combinations with lambda < 0 (``rule="lt"``, the paper's integral up to 0)
  or lambda <= 0 (``rule="le"``, the rule pypbo, fynance and sharpebench use). The two differ
  only for odd N, at r = (N + 1) / 2.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sharpecheck._validation import (
    FloatArray,
    as_float_1d,
    as_float_2d,
    corrected_mean,
    zero_variance,
)

PboRule = Literal["lt", "le"]


@dataclass(frozen=True)
class CscvResult:
    """CSCV output. Arrays are indexed by combination (lexicographic order of IS blocks).

    ``sr_is`` and ``sr_oos`` are the IS winner's SRs, ``sr_oos_all`` (C x N) the OOS SRs of
    every column. ``degradation_slope`` and ``degradation_intercept`` are NaN when the IS
    winner's SR is the same in every combination, because the regression of OOS on IS SR is
    then undefined.
    """

    pbo: float
    n_overfit: int
    n_combinations: int
    rule: PboRule
    best_is: NDArray[np.int64]
    rank_oos: FloatArray
    logits: FloatArray
    sr_is: FloatArray
    sr_oos: FloatArray
    sr_oos_all: FloatArray
    prob_loss: float
    degradation_slope: float
    degradation_intercept: float


def cscv_combinations(n_blocks: int) -> NDArray[np.bool_]:
    """Boolean (C, S) matrix: row c marks the IS blocks of combination c, lexicographic."""
    return _design(n_blocks)[0].copy()


_Halves = tuple[tuple[FloatArray, FloatArray], tuple[FloatArray, FloatArray]]


@lru_cache(maxsize=4)
def _design(n_blocks: int) -> tuple[NDArray[np.bool_], NDArray[np.intp], NDArray[np.intp], _Halves]:
    """Combination mask, block pairs (b < b') and, for the IS and the OOS half, the (C, S)
    block indicators and (C, S(S-1)/2) indicators of pairs inside the half. Read-only, cached:
    they depend on S only (C(16, 8) = 12,870 rows)."""
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError(f"n_blocks must be even and >= 2, got {n_blocks}")
    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    mask = np.zeros((len(combos), n_blocks), dtype=bool)
    for c, blocks in enumerate(combos):
        mask[c, list(blocks)] = True
    pairs = list(itertools.combinations(range(n_blocks), 2))
    first, second = (np.array(p, dtype=np.intp) for p in zip(*pairs, strict=True))
    halves = tuple(
        (sel.astype(np.float64), (sel[:, first] & sel[:, second]).astype(np.float64))
        for sel in (mask, ~mask)
    )
    for a in (mask, first, second, *halves[0], *halves[1]):
        a.flags.writeable = False
    return mask, first, second, (halves[0], halves[1])


def logit_rank(rank: ArrayLike, n: int) -> FloatArray:
    """lambda = ln(omega / (1 - omega)) with omega = rank / (n + 1); rank in [1, n]."""
    r = np.asarray(rank, dtype=np.float64)
    if np.any(r < 1) or np.any(r > n):
        raise ValueError(f"ranks must lie in [1, {n}]")
    return np.asarray(np.log(r / (n + 1 - r)), dtype=np.float64)


def noise_pbo(n: int, rule: PboRule = "lt") -> Fraction:
    """Exact expected PBO on pure noise (any S and T).

    Requires independent rows (so the IS and OOS metric vectors are independent),
    exchangeable columns and a tie-free metric; exchangeable columns alone are not enough
    (iid per-column drifts are exchangeable, yet the IS winner then tends to win OOS too).
    Under these conditions the OOS rank of the IS winner is uniform on {1, ..., n} in every
    combination, so the expectation is the share of ranks classified as overfit: 1/2 for even n, and
    1/2 - 1/(2n) (``"lt"``) or 1/2 + 1/(2n) (``"le"``) for odd n.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    _check_rule(rule)
    # lambda < 0 <=> r < (n + 1) / 2 <=> 2 r < n + 1
    if rule == "lt":
        count = sum(1 for r in range(1, n + 1) if 2 * r < n + 1)
    else:
        count = sum(1 for r in range(1, n + 1) if 2 * r <= n + 1)
    return Fraction(count, n)


def _check_rule(rule: str) -> None:
    if rule not in ("lt", "le"):
        raise ValueError(f"rule must be 'lt' or 'le', got {rule!r}")


def cscv(returns: ArrayLike, n_blocks: int = 16, rule: PboRule = "lt") -> CscvResult:
    """PBO of selecting the best in-sample Sharpe ratio among the N columns of ``returns``.

    ``returns`` is T x N with T divisible by ``n_blocks``. Raises ``ValueError`` for
    NaN/inf, fewer than two columns, uneven blocks, or a column with zero variance in any
    IS or OOS half (its Sharpe ratio is undefined; it is never silently ranked).
    """
    _check_rule(rule)
    x = as_float_2d(returns, "returns", min_rows=2)
    t, n = x.shape
    if n < 2:
        raise ValueError(f"returns needs at least 2 columns (strategies), got {n}")
    mask, first, second, (is_half, oos_half) = _design(n_blocks)
    if t % n_blocks:
        raise ValueError(
            f"T = {t} is not divisible by n_blocks = {n_blocks}; trim or pad the sample "
            "explicitly (uneven blocks make IS and OOS halves unequal)"
        )
    k = n_blocks // 2
    blocks = x.reshape(n_blocks, t // n_blocks, n)
    # Half variance = (within-block sums of squares + between-block term) / half length, with
    # the between term as a sum of squared pairwise block-mean differences:
    # sum_b L (m_b - m)^2 = (L / k) sum_{b < b'} (m_b - m_b')^2. Every term is non-negative,
    # so a half that is constant gets a variance of order (eps * mean)^2, as zero_variance
    # expects; E[x^2] - E[x]^2 around a common centre would leave eps * E[x^2] of noise.
    means = corrected_mean(blocks, axis=1)  # (S, N)
    within = ((blocks - means[:, None, :]) ** 2).sum(axis=1)
    pair_sq = (means[first] - means[second]) ** 2  # (S(S-1)/2, N)
    half = k * (t // n_blocks)

    def sharpe(w: FloatArray, both: FloatArray) -> FloatArray:
        mean = w @ means / k
        var = w @ within / half + both @ pair_sq / k**2
        if np.any(zero_variance(var, mean)):
            raise ValueError("a column has zero variance in a CSCV half; its Sharpe is undefined")
        return np.asarray(mean / np.sqrt(var), dtype=np.float64)

    sr_is_all = sharpe(*is_half)
    sr_oos_all = sharpe(*oos_half)
    c = mask.shape[0]
    rows = np.arange(c)
    best = np.argmax(sr_is_all, axis=1).astype(np.int64)
    oos_best = sr_oos_all[rows, best]
    below = (sr_oos_all < oos_best[:, None]).sum(axis=1)
    ties = (sr_oos_all == oos_best[:, None]).sum(axis=1) - 1
    rank = 1.0 + below + 0.5 * ties
    logits = logit_rank(rank, n)
    overfit = logits < 0 if rule == "lt" else logits <= 0
    n_overfit = int(overfit.sum())
    sr_is = sr_is_all[rows, best]
    if np.ptp(sr_is) > 0:  # OLS of OOS (response) on IS
        dx = sr_is - sr_is.mean()
        slope = float(dx @ (oos_best - oos_best.mean()) / (dx @ dx))
        intercept = float(oos_best.mean() - slope * sr_is.mean())
    else:
        slope, intercept = math.nan, math.nan
    return CscvResult(
        pbo=n_overfit / c,
        n_overfit=n_overfit,
        n_combinations=c,
        rule=rule,
        best_is=best,
        rank_oos=rank,
        logits=logits,
        sr_is=sr_is,
        sr_oos=oos_best,
        sr_oos_all=sr_oos_all,
        prob_loss=float(np.mean(oos_best < 0)),
        degradation_slope=float(slope),
        degradation_intercept=float(intercept),
    )


@dataclass(frozen=True)
class StochasticDominance:
    """Empirical stochastic dominance of ``selected`` over ``benchmark`` (p. 18).

    ``grid`` holds the pooled distinct values, ``cdf_*`` the empirical CDFs there and
    ``sd2[k]`` the integral of ``F_benchmark - F_selected`` from -inf to ``grid[k]``. The
    CDFs are step functions, so ``sd2`` is piecewise linear between grid points and constant
    after the last one: its sign at the grid points decides second-order dominance.
    """

    grid: FloatArray
    cdf_selected: FloatArray
    cdf_benchmark: FloatArray
    sd2: FloatArray
    first_order: bool
    second_order: bool


def stochastic_dominance(selected: ArrayLike, benchmark: ArrayLike) -> StochasticDominance:
    """Does the OOS performance of the selected strategy dominate that of all strategies?

    First order: ``F_selected <= F_benchmark`` everywhere (compared exactly, as counts).
    Second order: ``SD2(x) = integral_{-inf}^{x} (F_benchmark - F_selected) dt >= 0``
    everywhere; SD2 is an integral, not the CDF difference itself. Typical use:
    ``stochastic_dominance(res.sr_oos, res.sr_oos_all.ravel())``.
    """
    a = as_float_1d(selected, "selected", min_len=1)
    b = as_float_1d(benchmark, "benchmark", min_len=1)
    grid = np.unique(np.concatenate([a, b]))
    count_a = np.searchsorted(np.sort(a), grid, side="right")
    count_b = np.searchsorted(np.sort(b), grid, side="right")
    first = bool(np.all(count_a * b.size <= count_b * a.size))
    cdf_a = count_a / a.size
    cdf_b = count_b / b.size
    sd2 = np.concatenate([[0.0], np.cumsum((cdf_b - cdf_a)[:-1] * np.diff(grid))])
    # Allow for the rounding of the cumulative sum, relative to the span of the values.
    tol = 1e-12 * float(grid[-1] - grid[0])
    return StochasticDominance(
        grid=grid,
        cdf_selected=cdf_a,
        cdf_benchmark=cdf_b,
        sd2=sd2,
        first_order=first,
        second_order=bool(sd2.min() >= -tol),
    )
