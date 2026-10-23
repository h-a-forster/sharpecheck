"""Inference for one Sharpe ratio (milestones M1, M3).

Methods (formulas in DESIGN 4.2):

- ``iid_normal``   Lo (2002) iid / Jobson-Korkie: Var = (1 + SR^2/2) / T
- ``exact_nct``    exact under iid normal: sqrt(T) * SR_hat ~ noncentral t(T-1, sqrt(T) SR)
- ``mertens``      iid non-normal, Mertens (2002) / Opdyke (2007):
                   Var = (1 - g3 SR + (g4 - 1)/4 SR^2) / T
- ``ar1``          AR(1)-adjusted variance of Lopez de Prado, Lipton and Zoonekynd (2025)
- ``hac``          delta method with a kernel long-run covariance of (r_t, r_t^2): Lo (2002)
                   non-iid, Ledoit and Wolf (2008) adapted to one ratio; QS kernel
- ``hac_pw``       as ``hac`` with Andrews-Monahan VAR(1) prewhitening
- ``boot_ts``      studentized circular block bootstrap, symmetric interval (LW 2008 eq. 7)

Plus PSR (both published variance conventions), MinTRL, critical SR and power.
"""

from __future__ import annotations

from typing import Literal

from numpy.typing import ArrayLike

from sharpecheck.results import SharpeInterval, TestResult

Method = Literal["iid_normal", "exact_nct", "mertens", "ar1", "hac", "hac_pw", "boot_ts"]
Side = Literal["two-sided", "lower", "upper"]
VarianceAt = Literal["observed", "null"]

METHODS: tuple[Method, ...] = (
    "iid_normal",
    "exact_nct",
    "mertens",
    "ar1",
    "hac",
    "hac_pw",
    "boot_ts",
)


def sr_variance(
    sr: float, nobs: float, g3: float = 0.0, g4: float = 3.0, rho: float = 0.0
) -> float:
    """Asymptotic variance of the SR estimator: the LZL (2025) AR(1) formula.

    ``rho = 0`` gives Mertens/Opdyke; additionally ``g3 = 0, g4 = 3`` gives the iid-normal
    formula. ``g4`` is the non-excess kurtosis. Requires ``-1 < rho < 1``.
    """
    raise NotImplementedError("M1")


def sharpe_ci(
    x: ArrayLike,
    method: Method = "mertens",
    level: float = 0.95,
    side: Side = "two-sided",
    *,
    block: int | None = None,
    n_boot: int = 999,
    seed: int | None = None,
) -> SharpeInterval:
    """Confidence interval for the per-period Sharpe ratio of ``x`` by ``method``.

    ``block``, ``n_boot`` and ``seed`` apply to ``boot_ts`` only (default block size
    ``round(T ** (1/3))``, DESIGN 4.4).
    """
    raise NotImplementedError("M1 (analytic methods), M3 (hac, hac_pw, boot_ts)")


def psr(
    sr: float,
    sr0: float,
    nobs: float,
    g3: float = 0.0,
    g4: float = 3.0,
    *,
    rho: float = 0.0,
    variance_at: VarianceAt = "observed",
) -> float:
    """Probabilistic Sharpe ratio: one minus the p-value of the test of H0: SR <= sr0.

    ``variance_at="observed"``: Bailey and Lopez de Prado (2012), variance at the observed
    ``sr`` with divisor ``nobs - 1``. ``variance_at="null"``: Lopez de Prado, Lipton and
    Zoonekynd (2025), variance at ``sr0`` with divisor ``nobs`` (DESIGN 4.2.3).
    """
    raise NotImplementedError("M1")


def psr_test(x: ArrayLike, sr0: float = 0.0, *, variance_at: VarianceAt = "observed") -> TestResult:
    """One-sided test of H0: SR <= sr0 based on PSR; ``pvalue = 1 - PSR``."""
    raise NotImplementedError("M1")


def min_track_record_length(
    sr: float,
    sr0: float,
    g3: float = 0.0,
    g4: float = 3.0,
    *,
    rho: float = 0.0,
    alpha: float = 0.05,
    variance_at: VarianceAt = "observed",
) -> float:
    """MinTRL: observations needed for PSR(sr0) to reach ``1 - alpha`` (DESIGN 4.2.3)."""
    raise NotImplementedError("M1")


def critical_sr(
    sr0: float,
    nobs: float,
    g3: float = 0.0,
    g4: float = 3.0,
    *,
    rho: float = 0.0,
    alpha: float = 0.05,
) -> float:
    """Critical SR of the one-sided level-``alpha`` test of H0: SR = sr0 (LZL 2025)."""
    raise NotImplementedError("M1")


def power(
    sr0: float,
    sr1: float,
    nobs: float,
    g3: float = 0.0,
    g4: float = 3.0,
    *,
    rho: float = 0.0,
    alpha: float = 0.05,
) -> float:
    """Power of the one-sided test at the alternative ``sr1`` (LZL 2025)."""
    raise NotImplementedError("M1")


def annualize(sr: float, periods: int, rho: float = 0.0) -> float:
    """Annualize a per-period SR with Lo's (2002) AR(1) factor eta(q); rho=0 gives sqrt(q)."""
    raise NotImplementedError("M1")
