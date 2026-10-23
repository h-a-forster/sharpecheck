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

from typing import Literal

from numpy.typing import ArrayLike

from sharpecheck.results import TestResult

EmaxMethod = Literal["gumbel", "exact"]


def expected_max_sr(
    n_trials: float, var_sr: float, *, rho: float = 0.0, method: EmaxMethod = "gumbel"
) -> float:
    """Expected maximum of ``n_trials`` zero-mean trial SRs with variance ``var_sr``."""
    raise NotImplementedError("M4")


def var_max_sr(n_trials: int, var_sr: float, *, rho: float = 0.0) -> float:
    """Variance of the maximum of ``n_trials`` trial SRs (exact, equicorrelated normal)."""
    raise NotImplementedError("M4")


def implied_trials(n_trials: int, avg_corr: float) -> float:
    """Implied number of independent trials, BLdP (2014) eq. 9."""
    raise NotImplementedError("M4")


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
    raise NotImplementedError("M4")


def dsr_test(
    trial_returns: ArrayLike,
    *,
    n_trials: float | None = None,
    var_sr: float | None = None,
    emax_method: EmaxMethod = "gumbel",
) -> TestResult:
    """DSR test on the best column of a (T x N) matrix of trial returns.

    ``n_trials`` defaults to the number of columns and ``var_sr`` to the cross-sectional
    variance of the column SRs, as practitioners compute it. ``pvalue = 1 - DSR``.
    """
    raise NotImplementedError("M4")


def pfdr(p_h1: float, alpha: float, beta: float) -> float:
    """Posterior probability of H0 given a rejection at level ``alpha`` (LZL 2025)."""
    raise NotImplementedError("M4")


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
    """Posterior probability of H0 given the observed ``sr`` (LZL 2025)."""
    raise NotImplementedError("M4")
