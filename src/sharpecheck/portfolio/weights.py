"""Allocation rules (milestone M7). Formulas: DESIGN 5.3.

Each rule maps an estimation window of excess returns (M x N, rows are months) to a weight
vector summing to one. No rule sees data after its window.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from sharpecheck._validation import FloatArray, as_float_2d

Rule = Literal["equal", "inverse_vol", "gmv_sample", "gmv_lw", "hrp"]
RULES: tuple[Rule, ...] = ("equal", "inverse_vol", "gmv_sample", "gmv_lw", "hrp")


def equal_weight(window: ArrayLike) -> FloatArray:
    """1/N."""
    n = as_float_2d(window, "window").shape[1]
    return np.full(n, 1.0 / n)


def inverse_vol(window: ArrayLike) -> FloatArray:
    """Weights proportional to 1 / sigma_i (sample standard deviations, ddof 1)."""
    x = as_float_2d(window, "window")
    sd = x.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        raise ValueError("inverse_vol needs positive standard deviations")
    w = 1.0 / sd
    return w / w.sum()


def ledoit_wolf_cov(window: ArrayLike) -> tuple[FloatArray, float]:
    """Ledoit-Wolf (2004) shrinkage to the scaled identity; returns (Sigma, delta).

    ``X`` demeaned (M x N), ``S = X'X/M``, ``m = tr(S)/N``, ``d^2 = ||S - m I||_F^2 / N``,
    ``b_bar^2 = (1/M^2) sum_t ||x_t x_t' - S||_F^2 / N``, ``b^2 = min(b_bar^2, d^2)``,
    ``delta = b^2 / d^2`` and ``Sigma = delta m I + (1 - delta) S``.
    """
    x = as_float_2d(window, "window")
    m_obs, n = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / m_obs
    mu = np.trace(s) / n
    d2 = float(np.sum((s - mu * np.eye(n)) ** 2) / n)
    # sum_t ||x_t x_t' - S||_F^2 = sum_t ||x_t||^4 - M ||S||_F^2
    sq = np.sum(xc * xc, axis=1)
    b_bar2 = float((np.sum(sq * sq) - m_obs * np.sum(s * s)) / (m_obs**2) / n)
    b2 = min(b_bar2, d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    return delta * mu * np.eye(n) + (1 - delta) * s, float(delta)


def gmv(cov: ArrayLike) -> FloatArray:
    """Unconstrained global minimum variance: w = Sigma^-1 1 / (1' Sigma^-1 1)."""
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square")
    ones = np.ones(c.shape[0])
    z = np.linalg.solve(c, ones)
    return z / z.sum()


def _ivp(cov: FloatArray) -> FloatArray:
    w = 1.0 / np.diag(cov)
    return w / w.sum()


def hrp(cov: ArrayLike) -> FloatArray:
    """Hierarchical risk parity (Lopez de Prado 2016): single linkage on
    d_ij = sqrt((1 - rho_ij) / 2), quasi-diagonalisation, recursive bisection with
    inverse-variance cluster weights.

    Following the paper's code, the linkage uses the Euclidean distance between columns
    of ``d`` (PyPortfolioOpt clusters on ``d`` directly, so results can differ).
    """
    c = np.asarray(cov, dtype=np.float64)
    n = c.shape[0]
    if n == 1:
        return np.ones(1)
    sd = np.sqrt(np.diag(c))
    corr = np.clip(c / np.outer(sd, sd), -1.0, 1.0)
    d = np.sqrt((1 - corr) / 2)
    link = linkage(pdist(d), method="single")
    order = list(leaves_list(link))
    w = np.ones(n)
    clusters = [order]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]
            vl = _cluster_var(c, left)
            vr = _cluster_var(c, right)
            alpha = 1 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1 - alpha
            nxt += [left, right]
        clusters = nxt
    return w / w.sum()


def _cluster_var(cov: FloatArray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    w = _ivp(sub)
    return float(w @ sub @ w)


def weights(rule: Rule, window: ArrayLike) -> FloatArray:
    """Dispatch to one rule."""
    x = as_float_2d(window, "window")
    if rule == "equal":
        return equal_weight(x)
    if rule == "inverse_vol":
        return inverse_vol(x)
    if rule == "gmv_sample":
        return gmv(np.cov(x, rowvar=False, ddof=1))
    if rule == "gmv_lw":
        return gmv(ledoit_wolf_cov(x)[0])
    if rule == "hrp":
        return hrp(np.cov(x, rowvar=False, ddof=1))
    raise ValueError(f"unknown rule {rule!r}; choose from {RULES}")
