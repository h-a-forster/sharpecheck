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

import math
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, stats

from sharpecheck._validation import FloatArray, as_float_1d, check_level
from sharpecheck.moments import sample_moments
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

#: The AR(1) method clips the estimated lag-1 autocorrelation to [-RHO_CLIP, RHO_CLIP].
RHO_CLIP = 0.95


def _check_side(side: str) -> None:
    if side not in ("two-sided", "lower", "upper"):
        raise ValueError(f"side must be 'two-sided', 'lower' or 'upper', got {side!r}")


def _check_rho(rho: float) -> None:
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")


def variance_terms(rho: float) -> tuple[float, float, float]:
    """The LZL (2025) AR(1) coefficients ``(a, b, c)`` of ``V = a - b g3 SR + c (g4-1)/4 SR^2``."""
    _check_rho(rho)
    a = 1 + 2 * rho / (1 - rho)
    b = 1 + rho / (1 - rho) + rho**2 / (1 - rho**2)
    c = 1 + 2 * rho**2 / (1 - rho**2)
    return a, b, c


def sr_variance(
    sr: float, nobs: float, g3: float = 0.0, g4: float = 3.0, rho: float = 0.0
) -> float:
    """Asymptotic variance of the SR estimator: the LZL (2025) AR(1) formula.

    ``rho = 0`` gives Mertens/Opdyke; additionally ``g3 = 0, g4 = 3`` gives the iid-normal
    formula. ``g4`` is the non-excess kurtosis. Requires ``-1 < rho < 1``. Returns
    ``V(SR, g3, g4, rho) / nobs``.
    """
    if not nobs > 0:
        raise ValueError(f"nobs must be positive, got {nobs}")
    a, b, c = variance_terms(rho)
    return (a - b * g3 * sr + c * (g4 - 1) / 4 * sr**2) / nobs


def z_quantile(level: float, side: str) -> float:
    """Normal critical value: ``z_{1 - alpha/2}`` two-sided, ``z_{level}`` one-sided."""
    return float(stats.norm.ppf(1 - (1 - level) / 2 if side == "two-sided" else level))


def _normal_interval(
    method: str,
    est: float,
    se: float,
    nobs: int,
    level: float,
    side: str,
    details: dict[str, Any] | None = None,
) -> SharpeInterval:
    det = dict(details or {})
    if not (math.isfinite(se) and se > 0) or not math.isfinite(est):
        det.setdefault("reason", "standard error undefined (non-positive or non-finite)")
        return SharpeInterval(method, est, math.nan, math.nan, level, math.nan, nobs, det)
    z = z_quantile(level, side)
    lo = est - z * se if side != "upper" else -math.inf
    hi = est + z * se if side != "lower" else math.inf
    return SharpeInterval(method, est, lo, hi, level, se, nobs, det)


def nct_cdf(tstat: ArrayLike, df: int, delta: ArrayLike) -> FloatArray:
    """Noncentral t CDF ``P(T <= t)`` that is finite where SciPy returns NaN.

    SciPy (checked on 1.18.1) returns NaN for ``stats.nct.cdf`` far in the tails, e.g.
    ``cdf(-1.37, 249, 11.95)``; the NaN band reaches into the brackets used to invert the
    CDF. Such entries are recomputed by the reflection ``cdf(t, df, d) = sf(-t, df, -d)``,
    and any entry still not finite is set to its limit (1 where ``d < t``, else 0).
    """
    t = np.asarray(tstat, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    t, d = np.broadcast_arrays(t, d)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        c = np.asarray(stats.nct.cdf(t, df, d), dtype=np.float64)
        bad = ~np.isfinite(c)
        if bad.any():
            c = c.copy()
            c[bad] = stats.nct.sf(-t[bad], df, -d[bad])
            bad = ~np.isfinite(c)
            if bad.any():
                c[bad] = np.where(d[bad] < t[bad], 1.0, 0.0)
    return np.clip(c, 0.0, 1.0)


def nct_delta_roots(
    tstat: ArrayLike, df: int, probs: ArrayLike, *, iterations: int = 60
) -> tuple[FloatArray, FloatArray]:
    """Solve ``nct.cdf(t, df, delta) = p`` for ``delta`` by vectorised bisection.

    ``tstat`` and ``probs`` broadcast. The CDF is decreasing in ``delta``. Returns
    ``(delta, ok)``; ``ok`` is False where the bracket did not enclose a root (callers then
    fall back to the normal approximation). Used by the Monte Carlo for whole batches.
    """
    t = np.asarray(tstat, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    t, p = np.broadcast_arrays(t, p)
    half = 12.0 * np.sqrt(1.0 + t * t / (2.0 * df)) + 2.0
    lo = t - half
    hi = t + half
    f_lo = nct_cdf(t, df, lo) - p
    f_hi = nct_cdf(t, df, hi) - p
    ok = np.isfinite(f_lo) & np.isfinite(f_hi) & (f_lo >= 0) & (f_hi <= 0)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        right = (nct_cdf(t, df, mid) - p) > 0
        lo = np.where(right, mid, lo)
        hi = np.where(right, hi, mid)
    return 0.5 * (lo + hi), ok


def nct_interval(
    tstat: float, nobs: int, level: float = 0.95, side: Side = "two-sided"
) -> tuple[float, float, bool]:
    """Exact interval for SR from the ddof-1 t statistic ``sqrt(T) mean / s``.

    Inverts the noncentral t CDF in the noncentrality with brentq and divides by
    ``sqrt(T)``. The CDF is evaluated by :func:`nct_cdf`. Returns ``(lower, upper, exact)``;
    ``exact`` is False when a root could not be bracketed or found and the normal
    approximation ``t/sqrt(T) -/+ z sqrt((1 + SR^2/2)/T)`` was
    used instead.
    """
    check_level(level)
    _check_side(side)
    df = nobs - 1
    rt = math.sqrt(nobs)
    alpha = 1 - level
    q_lo = 1 - alpha / 2 if side == "two-sided" else level
    q_hi = alpha / 2 if side == "two-sided" else 1 - level
    half = 12.0 * math.sqrt(1.0 + tstat * tstat / (2.0 * df)) + 2.0

    def root(q: float) -> float | None:
        def f(d: float) -> float:
            return float(nct_cdf(tstat, df, d)) - q

        a, b = tstat - half, tstat + half
        fa, fb = f(a), f(b)
        if not (math.isfinite(fa) and math.isfinite(fb)) or fa < 0 or fb > 0:
            return None
        try:
            return float(optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12)) / rt
        except (ValueError, RuntimeError):
            return None

    lo = root(q_lo) if side != "upper" else -math.inf
    hi = root(q_hi) if side != "lower" else math.inf
    if lo is None or hi is None:
        sr = tstat / rt
        se = math.sqrt((1 + sr * sr / 2) / nobs)
        z = z_quantile(level, side)
        lo = sr - z * se if side != "upper" else -math.inf
        hi = sr + z * se if side != "lower" else math.inf
        return lo, hi, False
    return lo, hi, True


def hac_se(x: ArrayLike, *, prewhiten: bool = False) -> tuple[float, float, dict[str, Any]]:
    """Delta-method HAC standard error of the ddof-0 SR (DESIGN 4.2.5).

    QS kernel, Andrews (1991) bandwidth, ``T/(T-2)`` adjustment, optional Andrews-Monahan
    prewhitening. Returns ``(sr_hat, se, details)``; ``se`` is ``nan`` with
    ``details["reason"]`` when the long-run variance is not positive or the variance is zero.
    """
    from sharpecheck.hac import long_run_cov

    a = as_float_1d(x, "x", min_len=4)
    t = a.size
    mu = float(a.mean())
    gam = float(np.mean(a * a))
    var = gam - mu * mu
    if not var > 0:
        return math.nan, math.nan, {"reason": "zero variance"}
    sr = mu / math.sqrt(var)
    grad = np.array([gam / var**1.5, -mu / (2 * var**1.5)])
    y = np.column_stack([a - mu, a * a - gam])
    try:
        psi = long_run_cov(y, "qs", prewhiten=prewhiten, dof_adjust=2)
    except np.linalg.LinAlgError as exc:
        return sr, math.nan, {"reason": f"long-run covariance failed: {exc}"}
    v = float(grad @ psi @ grad)
    if not (math.isfinite(v) and v > 0):
        return sr, math.nan, {"reason": "non-positive HAC variance"}
    return sr, math.sqrt(v / t), {}


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
    ``round(T ** (1/3))``, DESIGN 4.4). ``side="lower"`` gives a one-sided lower bound
    (``upper = inf``), ``"upper"`` a one-sided upper bound. The estimate is always the
    ddof-0 sample SR.
    """
    check_level(level)
    _check_side(side)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    a = as_float_1d(x, "x", min_len=4 if method in ("hac", "hac_pw", "boot_ts") else 2)
    m = sample_moments(a)
    t = m.nobs
    if not math.isfinite(m.sr):
        return SharpeInterval(
            method, math.nan, math.nan, math.nan, level, math.nan, t, {"reason": "zero variance"}
        )
    if method == "iid_normal":
        return _normal_interval(method, m.sr, math.sqrt(sr_variance(m.sr, t)), t, level, side)
    if method == "mertens":
        v = sr_variance(m.sr, t, m.g3, m.g4)
        return _normal_interval(method, m.sr, math.sqrt(v) if v > 0 else math.nan, t, level, side)
    if method == "ar1":
        rho = float(np.clip(m.rho1, -RHO_CLIP, RHO_CLIP))
        v = sr_variance(m.sr, t, m.g3, m.g4, rho)
        se = math.sqrt(v) if v > 0 else math.nan
        return _normal_interval(method, m.sr, se, t, level, side, {"rho": rho})
    if method == "exact_nct":
        tstat = m.sr * math.sqrt(t - 1)  # sqrt(T) mean / s with s the ddof-1 sd
        lo, hi, exact = nct_interval(tstat, t, level, side)
        det: dict[str, Any] = {"tstat": tstat}
        if not exact:
            det["approximation"] = "normal (noncentral t root not found)"
        return SharpeInterval(method, m.sr, lo, hi, level, math.nan, t, det)
    if method in ("hac", "hac_pw"):
        _, se, det = hac_se(a, prewhiten=method == "hac_pw")
        return _normal_interval(method, m.sr, se, t, level, side, det)
    from sharpecheck.bootstrap import default_block, studentized_sr_boot

    b = default_block(t) if block is None else int(block)
    _, se, tstar = studentized_sr_boot(a, b, n_boot, seed=seed)
    det = {"block": b, "n_boot": n_boot}
    tstar = tstar[np.isfinite(tstar)]
    if not (math.isfinite(se) and se > 0) or tstar.size == 0:
        det["reason"] = "standard error undefined"
        return SharpeInterval(method, m.sr, math.nan, math.nan, level, math.nan, t, det)
    det["n_valid"] = int(tstar.size)
    return boot_interval(method, m.sr, se, tstar, t, level, side, det)


def boot_interval(
    method: str,
    sr: float,
    se: float,
    tstar: FloatArray,
    nobs: int,
    level: float,
    side: str,
    details: dict[str, Any] | None = None,
) -> SharpeInterval:
    """Studentized bootstrap interval from signed ``t* = (SR* - SR_hat)/se*``.

    Two-sided: symmetric ``SR_hat -/+ q_level(|t*|) se``. One-sided lower:
    ``SR_hat - q_level(t*) se`` (percentile-t); upper: ``SR_hat - q_{1-level}(t*) se``.
    """
    det = dict(details or {})
    if side == "two-sided":
        q = float(np.quantile(np.abs(tstar), level))
        return SharpeInterval(method, sr, sr - q * se, sr + q * se, level, se, nobs, det)
    if side == "lower":
        q = float(np.quantile(tstar, level))
        return SharpeInterval(method, sr, sr - q * se, math.inf, level, se, nobs, det)
    q = float(np.quantile(tstar, 1 - level))
    return SharpeInterval(method, sr, -math.inf, sr - q * se, level, se, nobs, det)


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
    Zoonekynd (2025), variance at ``sr0`` with divisor ``nobs`` (DESIGN 4.2.3). Returns
    ``nan`` when the variance term is not positive.
    """
    if variance_at == "observed":
        if not nobs > 1:
            raise ValueError(f"nobs must exceed 1, got {nobs}")
        v = sr_variance(sr, nobs - 1, g3, g4, rho)
    elif variance_at == "null":
        v = sr_variance(sr0, nobs, g3, g4, rho)
    else:
        raise ValueError(f"variance_at must be 'observed' or 'null', got {variance_at!r}")
    if not v > 0:
        return math.nan
    return float(stats.norm.cdf((sr - sr0) / math.sqrt(v)))


def psr_test(
    x: ArrayLike,
    sr0: float = 0.0,
    *,
    variance_at: VarianceAt = "observed",
    rho: float = 0.0,
) -> TestResult:
    """One-sided test of H0: SR <= sr0 based on PSR; ``pvalue = 1 - PSR``.

    ``rho`` is the AR(1) coefficient used in the variance (0: iid, as in BLdP 2012). The
    statistic is the PSR itself.
    """
    m = sample_moments(x)
    if not math.isfinite(m.sr):
        return TestResult("psr", math.nan, math.nan, m.nobs, "greater", {"reason": "zero variance"})
    p = psr(m.sr, sr0, m.nobs, m.g3, m.g4, rho=rho, variance_at=variance_at)
    det: dict[str, Any] = {
        "sr": m.sr,
        "sr0": sr0,
        "variance_at": variance_at,
        "g3": m.g3,
        "g4": m.g4,
    }
    if math.isnan(p):
        det["reason"] = "non-positive variance term"
        return TestResult("psr", math.nan, math.nan, m.nobs, "greater", det)
    return TestResult("psr", p, 1.0 - p, m.nobs, "greater", det)


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
    """MinTRL: observations needed for PSR(sr0) to reach ``1 - alpha`` (DESIGN 4.2.3).

    Observed convention (BLdP 2012): ``1 + V(sr) (z / (sr - sr0))^2``; null convention
    (LZL 2025): ``V(sr0) (z / (sr - sr0))^2``. Returns ``inf`` when ``sr <= sr0``.
    """
    check_level(alpha, "alpha")
    if sr <= sr0:
        return math.inf
    z = float(stats.norm.ppf(1 - alpha))
    if variance_at == "observed":
        return 1.0 + sr_variance(sr, 1.0, g3, g4, rho) * (z / (sr - sr0)) ** 2
    if variance_at == "null":
        return sr_variance(sr0, 1.0, g3, g4, rho) * (z / (sr - sr0)) ** 2
    raise ValueError(f"variance_at must be 'observed' or 'null', got {variance_at!r}")


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
    check_level(alpha, "alpha")
    v = sr_variance(sr0, nobs, g3, g4, rho)
    return sr0 + float(stats.norm.ppf(1 - alpha)) * math.sqrt(v)


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
    src = critical_sr(sr0, nobs, g3, g4, rho=rho, alpha=alpha)
    v1 = sr_variance(sr1, nobs, g3, g4, rho)
    return float(1.0 - stats.norm.cdf((src - sr1) / math.sqrt(v1)))


def annualize(sr: float, periods: int, rho: float = 0.0) -> float:
    """Annualize a per-period SR with Lo's (2002) AR(1) factor eta(q); rho=0 gives sqrt(q)."""
    q = int(periods)
    if q < 1:
        raise ValueError(f"periods must be >= 1, got {periods}")
    _check_rho(rho)
    k = np.arange(1, q, dtype=np.float64)
    s = float(np.sum((q - k) * rho**k))
    return sr * q / math.sqrt(q + 2 * s)
