"""Single-SR methods, PSR, MinTRL, power and annualization against DESIGN 7.1 values."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from sharpecheck import single
from sharpecheck.moments import batch_moments, sample_moments
from sharpecheck.single import (
    annualize,
    critical_sr,
    min_track_record_length,
    nct_cdf,
    nct_delta_roots,
    nct_interval,
    power,
    psr,
    psr_test,
    sharpe_ci,
    sr_variance,
)

# LZL (2025) numeric example.
SR, G3, G4, T = 0.036 / 0.079, -2.448, 10.164, 24


# --- moments -------------------------------------------------------------------------


def test_sample_moments_hand_example() -> None:
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    m = sample_moments(x)
    d = x - x.mean()
    m2 = np.mean(d**2)
    assert m.nobs == 5
    assert m.mean == pytest.approx(x.mean())
    assert m.sd == pytest.approx(math.sqrt(m2))
    assert m.sr == pytest.approx(x.mean() / math.sqrt(m2))
    assert m.g3 == pytest.approx(stats.skew(x))
    assert m.g4 == pytest.approx(stats.kurtosis(x, fisher=False))
    assert m.rho1 == pytest.approx(np.sum(d[1:] * d[:-1]) / np.sum(d**2))


def test_sample_moments_edge_cases() -> None:
    with pytest.raises(ValueError):
        sample_moments([1.0])
    with pytest.raises(ValueError):
        sample_moments([1.0, np.nan, 2.0])
    with pytest.raises(ValueError):
        sample_moments(np.ones((3, 2)))
    m = sample_moments([0.1, 0.1, 0.1])
    assert math.isnan(m.sr) and m.sd == 0.0


def test_batch_moments_matches_scalar() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_t(5, size=(7, 50))
    b = batch_moments(x)
    for i in range(7):
        m = sample_moments(x[i])
        for k in ("mean", "sd", "sr", "g3", "g4", "rho1"):
            assert b[k][i] == pytest.approx(getattr(m, k), rel=1e-12, abs=1e-15)


# --- variance formulas and the LZL example -------------------------------------------


def test_lzl_numeric_example() -> None:
    assert round(math.sqrt(sr_variance(SR, T, G3, G4)), 3) == 0.329
    assert math.sqrt(sr_variance(SR, T, G3, G4)) == pytest.approx(0.328589, abs=1e-6)
    assert round(math.sqrt(sr_variance(SR, T)), 3) == 0.214
    assert math.sqrt(sr_variance(SR, T)) == pytest.approx(0.214460, abs=1e-6)
    assert round(psr(SR, 0.0, T, G3, G4, variance_at="null"), 3) == 0.987
    assert round(psr(SR, 0.1, T, G3, G4, variance_at="null"), 3) == 0.939
    assert round(min_track_record_length(SR, 0.0, G3, G4, variance_at="null"), 3) == 13.029
    beta = 1 - power(0.0, 0.5, T, G3, G4)
    assert round(beta, 3) == 0.315


def test_psr_observed_convention() -> None:
    assert psr(SR, 0.0, T, G3, G4) == pytest.approx(0.912709, abs=1e-6)
    assert psr(SR, 0.0, T) == pytest.approx(0.981243, abs=1e-6)


def test_bldp_2012_mintrl() -> None:
    sr, sr_ref = 2.0 / 12**0.5, 1 / 12**0.5
    assert round(min_track_record_length(sr, sr_ref, -0.72, 5.78), 3) == 59.895
    assert psr(sr, sr_ref, 59.895, -0.72, 5.78) == pytest.approx(0.95, abs=1e-5)


def test_mintrl_infinite_and_validation() -> None:
    assert min_track_record_length(0.1, 0.2) == math.inf
    assert min_track_record_length(0.1, 0.1, variance_at="null") == math.inf
    with pytest.raises(ValueError):
        min_track_record_length(0.3, 0.1, variance_at="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        psr(0.1, 0.0, 10, variance_at="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        psr(0.1, 0.0, 1.0)


def test_psr_at_mintrl_is_one_minus_alpha() -> None:
    for va in ("observed", "null"):
        n = min_track_record_length(0.15, 0.05, -0.5, 6.0, alpha=0.05, variance_at=va)
        assert psr(0.15, 0.05, n, -0.5, 6.0, variance_at=va) == pytest.approx(0.95, abs=1e-10)


def test_ar1_variance_is_delta_method_for_gaussian_ar1() -> None:
    assert sr_variance(0.1, 1, 0.0, 3.0, 0.2) == pytest.approx(1.505417, abs=1e-6)
    with pytest.raises(ValueError):
        sr_variance(0.1, 1, rho=1.0)
    with pytest.raises(ValueError):
        sr_variance(0.1, 0)


def test_critical_sr_and_power_consistency() -> None:
    c = critical_sr(0.0, 100)
    assert c == pytest.approx(stats.norm.ppf(0.95) * 0.1)
    assert power(0.0, 0.0, 100) == pytest.approx(0.05, abs=1e-12)
    assert power(0.0, 0.3, 100) > power(0.0, 0.2, 100) > 0.05


def test_lo_eta_values() -> None:
    assert annualize(1.0, 12, 0.1) == pytest.approx(3.160111, abs=1e-6)
    assert annualize(1.0, 12, 0.2) == pytest.approx(2.878849, abs=1e-6)
    assert annualize(1.0, 12, 0.5) == pytest.approx(2.121288, abs=1e-6)
    assert annualize(0.1, 12) == pytest.approx(0.1 * math.sqrt(12))
    assert annualize(0.1, 1) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        annualize(0.1, 0)


# --- exact noncentral t ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("nobs", "sr_hat", "lo", "hi"),
    [(60, 0.2, -0.056409, 0.454750), (250, 0.05, -0.074087, 0.173987)],
)
def test_nct_interval_reference(nobs: int, sr_hat: float, lo: float, hi: float) -> None:
    a, b, exact = nct_interval(math.sqrt(nobs) * sr_hat, nobs)
    assert exact
    assert a == pytest.approx(lo, abs=1e-6)
    assert b == pytest.approx(hi, abs=1e-6)


def test_nct_vectorised_matches_brentq() -> None:
    tst = np.array([math.sqrt(60) * 0.2, math.sqrt(250) * 0.05, -1.3, 0.0])
    nobs = np.array([60, 250, 60, 60])
    for i in range(4):
        d, ok = nct_delta_roots(tst[i], int(nobs[i]) - 1, [0.975, 0.025])
        lo, hi, _ = nct_interval(float(tst[i]), int(nobs[i]))
        assert ok.all()
        assert d[0] / math.sqrt(nobs[i]) == pytest.approx(lo, abs=1e-9)
        assert d[1] / math.sqrt(nobs[i]) == pytest.approx(hi, abs=1e-9)


def test_nct_large_noncentrality_t2520() -> None:
    # SR = 0.3 at T = 2520: delta ~ 15; the root must still be found exactly.
    lo, hi, exact = nct_interval(math.sqrt(2520) * 0.3, 2520)
    assert exact
    se = math.sqrt((1 + 0.045) / 2520)
    assert lo == pytest.approx(0.3 - 1.96 * se, abs=2e-3)
    assert hi == pytest.approx(0.3 + 1.96 * se, abs=2e-3)


def test_nct_cdf_finite_where_scipy_nan() -> None:
    # SciPy returns NaN here; the true value is ~0 (t far below the noncentrality).
    assert nct_cdf(-1.37, 249, 11.95) == pytest.approx(0.0, abs=1e-12)
    assert nct_cdf(1.37, 249, -11.95) == pytest.approx(1.0, abs=1e-12)
    d = np.linspace(-40, 40, 2001)
    for t in (-8.0, -1.37, 0.0, 1.37, 8.0):
        c = nct_cdf(t, 249, d)
        assert np.all(np.isfinite(c)) and np.all(np.diff(c) <= 1e-12)  # decreasing in d
        ok = np.isfinite(stats.nct.cdf(t, 249, d))
        assert np.allclose(c[ok], stats.nct.cdf(t, 249, d[ok]), atol=1e-10)


@pytest.mark.parametrize("nobs", [24, 36, 60, 120, 250, 1000, 2520])
def test_nct_negative_t_mirrors_positive(nobs: int) -> None:
    # Realistic negative SRs used to raise in brentq (NaN inside the bracket).
    tgrid = np.linspace(0.05, 9.95, 34)
    for tval in tgrid:
        lo_p, hi_p, ex_p = nct_interval(float(tval), nobs)
        lo_n, hi_n, ex_n = nct_interval(float(-tval), nobs)
        assert ex_p and ex_n
        assert lo_n == pytest.approx(-hi_p, abs=1e-9)
        assert hi_n == pytest.approx(-lo_p, abs=1e-9)
        for side in ("lower", "upper"):
            nct_interval(float(-tval), nobs, 0.95, side)  # must not raise
    d, ok = nct_delta_roots(-tgrid[:, None], nobs - 1, np.array([[0.975, 0.025, 0.95]]))
    assert ok.all()


def test_sharpe_ci_exact_nct_negative_sr() -> None:
    rng = np.random.default_rng(8)
    x = -0.087 + rng.standard_normal(250)
    ci = sharpe_ci(x, "exact_nct")
    assert ci.lower < ci.estimate < ci.upper
    assert "approximation" not in ci.details


def test_nct_one_sided() -> None:
    lo, hi, _ = nct_interval(1.5, 60, 0.95, "lower")
    lo2, _, _ = nct_interval(1.5, 60, 0.90, "two-sided")
    assert hi == math.inf and lo == pytest.approx(lo2, abs=1e-9)
    lo3, hi3, _ = nct_interval(1.5, 60, 0.95, "upper")
    assert lo3 == -math.inf and hi3 > 1.5 / math.sqrt(60)


# --- sharpe_ci dispatch ---------------------------------------------------------------


@pytest.fixture
def returns() -> np.ndarray:
    rng = np.random.default_rng(42)
    return 0.1 + rng.standard_normal(250)


@pytest.mark.parametrize("method", ["iid_normal", "mertens", "ar1", "exact_nct"])
def test_analytic_intervals_contain_estimate(returns: np.ndarray, method: str) -> None:
    ci = sharpe_ci(returns, method)  # type: ignore[arg-type]
    m = sample_moments(returns)
    assert ci.estimate == pytest.approx(m.sr)
    assert ci.lower < m.sr < ci.upper
    assert ci.nobs == 250 and ci.level == 0.95
    # Close to iid normal for Gaussian data.
    ref = sharpe_ci(returns, "iid_normal")
    assert ci.upper - ci.lower == pytest.approx(ref.upper - ref.lower, rel=0.2)


def test_iid_normal_formula(returns: np.ndarray) -> None:
    m = sample_moments(returns)
    ci = sharpe_ci(returns, "iid_normal", 0.9)
    se = math.sqrt((1 + m.sr**2 / 2) / m.nobs)
    assert ci.se == pytest.approx(se)
    assert ci.lower == pytest.approx(m.sr - stats.norm.ppf(0.95) * se)


def test_mertens_formula(returns: np.ndarray) -> None:
    m = sample_moments(returns)
    ci = sharpe_ci(returns, "mertens")
    v = (1 - m.g3 * m.sr + (m.g4 - 1) / 4 * m.sr**2) / m.nobs
    assert ci.se == pytest.approx(math.sqrt(v))


def test_ar1_uses_clipped_rho() -> None:
    x = np.cumsum(np.ones(30)) + 0.01 * np.arange(30) ** 2  # trending: rho1 near 1
    ci = sharpe_ci(x, "ar1")
    assert abs(ci.details["rho"]) <= single.RHO_CLIP


def test_one_sided_lower_equals_two_sided_90(returns: np.ndarray) -> None:
    for method in ("iid_normal", "mertens", "ar1", "exact_nct"):
        a = sharpe_ci(returns, method, 0.95, "lower")  # type: ignore[arg-type]
        b = sharpe_ci(returns, method, 0.90)  # type: ignore[arg-type]
        assert a.upper == math.inf
        assert a.lower == pytest.approx(b.lower, abs=1e-9)


def test_sharpe_ci_invalid_and_undefined(returns: np.ndarray) -> None:
    with pytest.raises(ValueError):
        sharpe_ci(returns, "bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sharpe_ci(returns, level=1.2)
    with pytest.raises(ValueError):
        sharpe_ci(returns, side="both")  # type: ignore[arg-type]
    ci = sharpe_ci(np.full(10, 0.01), "mertens")
    assert math.isnan(ci.lower) and "reason" in ci.details
    assert not ci.covers(0.0)


def test_psr_test(returns: np.ndarray) -> None:
    m = sample_moments(returns)
    r = psr_test(returns, 0.0)
    assert r.alternative == "greater"
    assert r.pvalue == pytest.approx(1 - psr(m.sr, 0.0, m.nobs, m.g3, m.g4))
    r2 = psr_test(returns, 0.0, variance_at="null")
    assert r2.pvalue == pytest.approx(1 - psr(m.sr, 0.0, m.nobs, m.g3, m.g4, variance_at="null"))
    assert math.isnan(psr_test(np.zeros(5)).pvalue)


@pytest.mark.slow
def test_exact_nct_coverage_normal_t60() -> None:
    rng = np.random.default_rng(20261024)
    reps, nobs, sr = 2000, 60, 0.3
    hits = 0
    for _ in range(reps):
        x = sr + rng.standard_normal(nobs)
        hits += sharpe_ci(x, "exact_nct").covers(sr)
    assert abs(hits / reps - 0.95) <= 0.01
