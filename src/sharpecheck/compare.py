"""Tests for the difference of two Sharpe ratios (milestone M3).

- ``jkm_test``      Jobson and Korkie (1981) with the Memmel (2003) correction (iid normal)
- ``lw_hac_test``   Ledoit and Wolf (2008) section 3.1: delta method, QS kernel, optional
                    prewhitening, T/(T-4) adjustment
- ``lw_boot_test``  Ledoit and Wolf (2008) section 3.2: studentized circular block bootstrap,
                    p-value by eq. (9); ``block=1`` is their Boot-IID variant
                    (original-data SE from the sample covariance, LW section 3.2.1)

Formulas: DESIGN 4.5. All tests are two-sided tests of H0: SR_1 = SR_2 on paired series.
SRs use the ddof-0 moment estimators throughout.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from sharpecheck._validation import FloatArray, as_float_1d, check_level
from sharpecheck.results import TestResult


def _pair(x1: ArrayLike, x2: ArrayLike, min_len: int = 4) -> tuple[FloatArray, FloatArray]:
    a = as_float_1d(x1, "x1", min_len=min_len)
    b = as_float_1d(x2, "x2", min_len=min_len)
    if a.size != b.size:
        raise ValueError(f"x1 and x2 must have equal length, got {a.size} and {b.size}")
    return a, b


def jkm_variance(sr1: float, sr2: float, rho: float, nobs: float) -> float:
    """Jobson-Korkie variance of ``SR_1 - SR_2`` with the Memmel (2003) correction."""
    return (2 - 2 * rho + 0.5 * (sr1**2 + sr2**2 - 2 * sr1 * sr2 * rho**2)) / nobs


def jkm_test(x1: ArrayLike, x2: ArrayLike) -> TestResult:
    """Jobson-Korkie-Memmel test of equal Sharpe ratios.

    ``Var = (2 - 2 rho + (SR1^2 + SR2^2 - 2 SR1 SR2 rho^2) / 2) / T`` with ``rho`` the
    sample correlation (Memmel 2003); two-sided normal p-value.
    """
    a, b = _pair(x1, x2, 2)
    t = a.size
    sa, sb = a.std(), b.std()
    if not (sa > 0 and sb > 0):
        return TestResult("jkm", math.nan, math.nan, t, details={"reason": "zero variance"})
    sr1, sr2 = a.mean() / sa, b.mean() / sb
    rho = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
    var = jkm_variance(float(sr1), float(sr2), rho, t)
    delta = float(sr1 - sr2)
    det = {"delta": delta, "sr1": float(sr1), "sr2": float(sr2), "rho": rho}
    if not var > 0:
        det["reason"] = "non-positive variance (perfectly correlated series)"
        return TestResult("jkm", math.nan, math.nan, t, details=det)
    se = math.sqrt(var)
    z = delta / se
    det["se"] = se
    return TestResult("jkm", z, float(2 * stats.norm.sf(abs(z))), t, details=det)


def lw_hac_se(x1: ArrayLike, x2: ArrayLike, *, prewhiten: bool = True) -> tuple[float, float]:
    """``(Delta_hat, se)`` of ``SR_1 - SR_2`` by the Ledoit-Wolf (2008) HAC delta method.

    ``se`` is ``nan`` when a variance is zero or the quadratic form is not positive.
    """
    from sharpecheck.hac import long_run_cov

    a, b = _pair(x1, x2)
    t = a.size
    m1, m2 = a.mean(), b.mean()
    g1, g2 = np.mean(a * a), np.mean(b * b)
    v1, v2 = g1 - m1 * m1, g2 - m2 * m2
    if not (v1 > 0 and v2 > 0):
        return math.nan, math.nan
    delta = float(m1 / math.sqrt(v1) - m2 / math.sqrt(v2))
    grad = np.array([g1 / v1**1.5, -g2 / v2**1.5, -m1 / (2 * v1**1.5), m2 / (2 * v2**1.5)])
    y = np.column_stack([a - m1, b - m2, a * a - g1, b * b - g2])
    try:
        psi = long_run_cov(y, "qs", prewhiten=prewhiten, dof_adjust=4)
    except np.linalg.LinAlgError:
        return delta, math.nan
    q = float(grad @ psi @ grad)
    if not (math.isfinite(q) and q > 0):
        return delta, math.nan
    return delta, math.sqrt(q / t)


def lw_hac_test(x1: ArrayLike, x2: ArrayLike, *, prewhiten: bool = True) -> TestResult:
    """Ledoit-Wolf HAC test of equal Sharpe ratios (QS kernel, Andrews bandwidth)."""
    a, b = _pair(x1, x2)
    name = "lw_hac_pw" if prewhiten else "lw_hac"
    delta, se = lw_hac_se(a, b, prewhiten=prewhiten)
    det: dict[str, object] = {"delta": delta, "se": se, "prewhiten": prewhiten}
    if not (math.isfinite(se) and se > 0):
        det["reason"] = "standard error undefined"
        return TestResult(name, math.nan, math.nan, a.size, details=det)
    z = delta / se
    return TestResult(name, z, float(2 * stats.norm.sf(abs(z))), a.size, details=det)


def lw_boot_test(
    x1: ArrayLike,
    x2: ArrayLike,
    *,
    block: int | None = None,
    n_boot: int = 4999,
    seed: int | np.random.Generator | None = None,
    level: float = 0.95,
) -> TestResult:
    """Ledoit-Wolf studentized bootstrap test of equal Sharpe ratios.

    ``block=None`` uses :func:`sharpecheck.bootstrap.default_block`. ``d = |Delta|/se``,
    ``d*_m = |Delta*_m - Delta| / se*_m``, ``p = (1 + #{d* >= d}) / (M + 1)`` (LW eq. 9).
    ``details`` carries the symmetric bootstrap interval ``Delta -/+ q_level(d*) se``.
    ``se`` is the prewhitened LW HAC SE, or the iid sample-covariance SE when ``block=1``
    (Boot-IID, LW 2008 section 3.2.1).
    """
    from sharpecheck.bootstrap import default_block, studentized_diff_boot

    check_level(level)
    a, b = _pair(x1, x2)
    t = a.size
    blk = default_block(t) if block is None else int(block)
    # studentized_diff_boot computes SR_j - SR_0; put x2 in column 0.
    delta, se, tstar = studentized_diff_boot(np.column_stack([b, a]), blk, n_boot, seed)
    d0, s0 = float(delta[0]), float(se[0])
    ts = np.abs(tstar[:, 0])
    ts = ts[np.isfinite(ts)]
    det: dict[str, object] = {"delta": d0, "se": s0, "block": blk, "n_boot": n_boot}
    if not (math.isfinite(s0) and s0 > 0) or ts.size == 0:
        det["reason"] = "standard error undefined"
        return TestResult("lw_boot", math.nan, math.nan, t, details=det)
    d = abs(d0) / s0
    p = (1 + int(np.sum(ts >= d))) / (ts.size + 1)
    q = float(np.quantile(ts, level))
    det.update({"ci_lower": d0 - q * s0, "ci_upper": d0 + q * s0, "level": level})
    det["n_valid"] = int(ts.size)
    return TestResult("lw_boot", d0 / s0, p, t, details=det)
