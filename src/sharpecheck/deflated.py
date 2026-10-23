"""Selection-adjusted Sharpe-ratio inference (milestone M4).

- expected maximum of N trial SRs: the Bailey-Lopez de Prado (2014) Gumbel approximation
  and the exact expectation of the maximum of N iid normals (quadrature), with the
  equicorrelated extension E = sqrt(1 - rho) E[max_N], Var = rho + (1 - rho) Var[max_N]
- deflated Sharpe ratio (BLdP 2014, eq. 2)
- implied number of independent trials N_hat = rho_bar + (1 - rho_bar) M (BLdP 2014, eq. 9)
- pFDR and oFDR (Lopez de Prado, Lipton and Zoonekynd 2025)

Formulas: DESIGN 4.6.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate, stats

from sharpecheck._validation import as_float_2d, check_level
from sharpecheck.results import TestResult
from sharpecheck.single import psr

EmaxMethod = Literal["gumbel", "exact"]

EULER_GAMMA = 0.5772156649015329


def _check_rho(rho: float) -> None:
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho must be in [0, 1), got {rho}")


@lru_cache(maxsize=4096)
def max_normal_moments(n: float) -> tuple[float, float]:
    """``(E[M_n], Var[M_n])`` of the maximum of ``n`` iid N(0, 1) by adaptive quadrature.

    ``n`` may be non-integer (``n phi(z) Phi(z)^(n-1)`` is still a density).
    """
    if not n >= 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n == 1:
        return 0.0, 1.0

    def dens(z: float) -> float:
        return n * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) * stats.norm.cdf(z) ** (n - 1)

    lo, hi = -9.0, 9.0 + math.sqrt(2 * math.log(n))
    pts = [float(stats.norm.ppf(1 - 1 / (n + 1)))]
    kw: dict[str, Any] = {"points": pts, "limit": 200, "epsabs": 1e-12}
    m1 = integrate.quad(lambda z: z * dens(z), lo, hi, **kw)[0]
    m2 = integrate.quad(lambda z: z * z * dens(z), lo, hi, **kw)[0]
    return m1, m2 - m1 * m1


def expected_max_sr(
    n_trials: float, var_sr: float, *, rho: float = 0.0, method: EmaxMethod = "gumbel"
) -> float:
    """Expected maximum of ``n_trials`` zero-mean trial SRs with variance ``var_sr``.

    ``method="gumbel"``: BLdP (2014) eq. 1,
    ``sqrt(V) ((1 - g) z(1 - 1/N) + g z(1 - 1/(N e)))``; ``"exact"``: ``sqrt(V) E[M_N]``.
    With equicorrelation ``rho`` both are multiplied by ``sqrt(1 - rho)`` (one-factor
    construction). ``n_trials = 1`` gives 0.
    """
    _check_rho(rho)
    if not n_trials >= 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if not var_sr >= 0:
        raise ValueError(f"var_sr must be non-negative, got {var_sr}")
    if method not in ("gumbel", "exact"):
        raise ValueError(f"method must be 'gumbel' or 'exact', got {method!r}")
    if n_trials == 1:
        return 0.0
    if method == "gumbel":
        z1 = stats.norm.ppf(1 - 1 / n_trials)
        z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
        e = float((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    else:
        e = max_normal_moments(float(n_trials))[0]
    return math.sqrt(var_sr) * math.sqrt(1 - rho) * e


def var_max_sr(n_trials: float, var_sr: float, *, rho: float = 0.0) -> float:
    """Variance of the maximum of ``n_trials`` trial SRs (exact, equicorrelated normal)."""
    _check_rho(rho)
    return float(var_sr * (rho + (1 - rho) * max_normal_moments(float(n_trials))[1]))


def implied_trials(n_trials: int, avg_corr: float) -> float:
    """Implied number of independent trials, BLdP (2014) eq. 9.

    ``avg_corr`` is clipped to [0, 1]; the result is at least 1.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    r = min(max(float(avg_corr), 0.0), 1.0)
    return max(1.0, r + (1 - r) * n_trials)


def deflated_sharpe_ratio(
    sr: float,
    nobs: int,
    n_trials: float,
    var_sr: float,
    g3: float = 0.0,
    g4: float = 3.0,
    *,
    emax_method: EmaxMethod = "gumbel",
) -> float:
    """DSR = PSR(sr0 = expected_max_sr(n_trials, var_sr)) with the observed-variance PSR."""
    sr0 = expected_max_sr(n_trials, var_sr, method=emax_method)
    return psr(sr, sr0, nobs, g3, g4, variance_at="observed")


def dsr_test(
    trial_returns: ArrayLike,
    *,
    n_trials: float | None = None,
    var_sr: float | None = None,
    emax_method: EmaxMethod = "gumbel",
) -> TestResult:
    """DSR test on the best column of a (T x N) matrix of trial returns.

    ``n_trials`` defaults to the number of columns and ``var_sr`` to the ddof-1
    cross-sectional variance of the column SRs, as practitioners compute it.
    ``pvalue = 1 - DSR``; the statistic is the DSR.
    """
    x = as_float_2d(trial_returns, "trial_returns", min_rows=3)
    t, n = x.shape
    mu = x.mean(axis=0)
    d = x - mu
    m2 = np.mean(d * d, axis=0)
    if np.any(m2 <= 0):
        return TestResult("dsr", math.nan, math.nan, t, "greater", {"reason": "zero variance"})
    srs = mu / np.sqrt(m2)
    k = int(np.argmax(srs))
    nt = float(n if n_trials is None else n_trials)
    if var_sr is None:
        if n < 2:
            raise ValueError("var_sr is required with a single trial")
        vs = float(np.var(srs, ddof=1))
    else:
        vs = float(var_sr)
    g3 = float(np.mean(d[:, k] ** 3) / m2[k] ** 1.5)
    g4 = float(np.mean(d[:, k] ** 4) / m2[k] ** 2)
    sr0 = expected_max_sr(nt, vs, method=emax_method)
    p = psr(float(srs[k]), sr0, t, g3, g4, variance_at="observed")
    det: dict[str, Any] = {
        "best": k,
        "sr_best": float(srs[k]),
        "sr0": sr0,
        "n_trials": nt,
        "var_sr": vs,
    }
    if math.isnan(p):
        det["reason"] = "non-positive variance term"
        return TestResult("dsr", math.nan, math.nan, t, "greater", det)
    return TestResult("dsr", p, 1.0 - p, t, "greater", det)


def pfdr(p_h1: float, alpha: float, beta: float) -> float:
    """Posterior probability of H0 given a rejection at level ``alpha`` (LZL 2025)."""
    check_level(p_h1, "p_h1")
    check_level(alpha, "alpha")
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    return 1.0 / (1.0 + (1.0 - beta) * p_h1 / (alpha * (1.0 - p_h1)))


def ofdr(
    sr: float,
    sr0: float,
    sr1: float,
    nobs: float,
    p_h1: float,
    g3: float = 0.0,
    g4: float = 3.0,
    *,
    rho: float = 0.0,
) -> float:
    """Posterior probability of H0 given the observed ``sr`` (LZL 2025).

    ``p_k = 1 - PSR(sr; sr_k)`` with the null-variance convention;
    ``oFDR = p0 (1 - p_H1) / (p0 (1 - p_H1) + p1 p_H1)``.
    """
    check_level(p_h1, "p_h1")
    p0 = 1.0 - psr(sr, sr0, nobs, g3, g4, rho=rho, variance_at="null")
    p1 = 1.0 - psr(sr, sr1, nobs, g3, g4, rho=rho, variance_at="null")
    return p0 * (1 - p_h1) / (p0 * (1 - p_h1) + p1 * p_h1)
