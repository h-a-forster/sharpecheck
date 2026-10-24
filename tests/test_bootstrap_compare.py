"""Circular block bootstrap, studentized statistics, difference tests, HAC intervals."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from sharpecheck import bootstrap
from sharpecheck.bootstrap import (
    block_se_sr,
    circular_block_indices,
    default_block,
    iid_se_diff,
    studentized_diff_boot,
    studentized_sr_boot,
)
from sharpecheck.compare import jkm_test, jkm_variance, lw_boot_test, lw_hac_se, lw_hac_test
from sharpecheck.single import hac_se, sharpe_ci


def test_default_block() -> None:
    assert default_block(1) == 1
    assert default_block(60) == 4
    assert default_block(120) == 5
    assert default_block(250) == 6
    assert default_block(2520) == 14
    with pytest.raises(ValueError):
        default_block(0)


def test_indices_shape_range_and_contiguity() -> None:
    rng = np.random.default_rng(0)
    idx = circular_block_indices(23, 5, 40, rng)
    assert idx.shape == (40, 23)
    assert idx.min() >= 0 and idx.max() < 23
    # Within each block, consecutive indices modulo T.
    for row in idx:
        for start in range(0, 20, 5):
            blk = row[start : start + 5]
            assert np.all((blk[1:] - blk[:-1]) % 23 == 1)
    with pytest.raises(ValueError):
        circular_block_indices(10, 11, 5, rng)
    with pytest.raises(ValueError):
        circular_block_indices(10, 2, 0, rng)


def test_indices_deterministic_with_seed() -> None:
    a = circular_block_indices(50, 4, 10, np.random.default_rng(7))
    b = circular_block_indices(50, 4, 10, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_block_one_is_sample_covariance_delta_method() -> None:
    rng = np.random.default_rng(1)
    x = 0.1 + rng.standard_t(6, 200)
    mu, gam = x.mean(), np.mean(x * x)
    var = gam - mu * mu
    grad = np.array([gam / var**1.5, -mu / (2 * var**1.5)])
    y = np.column_stack([x - mu, x * x - gam])
    cov = y.T @ y / len(x)
    se = math.sqrt(grad @ cov @ grad / len(x))
    assert block_se_sr(x, 1) == pytest.approx(se, rel=1e-12)
    # And equals the Mertens variance with sample moments (same delta method).
    from sharpecheck.moments import sample_moments
    from sharpecheck.single import sr_variance

    m = sample_moments(x)
    assert se == pytest.approx(math.sqrt(sr_variance(m.sr, m.nobs, m.g3, m.g4)), rel=1e-10)


def test_studentized_sr_boot_deterministic_and_centred() -> None:
    rng = np.random.default_rng(2)
    x = 0.1 + rng.standard_normal(300)
    a = studentized_sr_boot(x, 6, 499, seed=11)
    b = studentized_sr_boot(x, 6, 499, seed=11)
    assert a[0] == b[0] and a[1] == b[1] and np.array_equal(a[2], b[2])
    assert a[2].shape == (499,)
    assert abs(np.nanmean(a[2])) < 0.2
    assert 0.8 < np.nanstd(a[2]) < 1.25
    sr, se, _ = hac_se(x, prewhiten=True)
    assert a[0] == pytest.approx(sr) and a[1] == pytest.approx(se)


def test_diff_boot_shapes_and_consistency() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((150, 4)) + 0.05
    delta, se, ts = studentized_diff_boot(x, 5, 199, seed=1)
    assert delta.shape == (3,) and se.shape == (3,) and ts.shape == (199, 3)
    srs = x.mean(axis=0) / x.std(axis=0)
    assert np.allclose(delta, srs[1:] - srs[0])
    for j in range(1, 4):
        d, s = lw_hac_se(x[:, j], x[:, 0])
        assert se[j - 1] == pytest.approx(s)
    with pytest.raises(ValueError):
        studentized_diff_boot(x[:, :1], 5, 10)


def test_jkm_variance_reference() -> None:
    assert jkm_variance(0.2, 0.1, 0.5, 120) == pytest.approx(0.0085, abs=1e-12)
    rng = np.random.default_rng(4)
    z = rng.standard_normal((120, 2))
    r = jkm_test(z[:, 0] + 0.2, z[:, 1] + 0.1)
    d = r.details
    assert d["se"] == pytest.approx(math.sqrt(jkm_variance(d["sr1"], d["sr2"], d["rho"], 120)))
    assert r.pvalue == pytest.approx(2 * stats.norm.sf(abs(d["delta"] / d["se"])))


def test_jkm_undefined() -> None:
    x = np.arange(10.0)
    assert math.isnan(jkm_test(np.ones(10), x).pvalue)
    with pytest.raises(ValueError):
        jkm_test(x, x[:5])


def test_lw_hac_identical_series_and_symmetry() -> None:
    rng = np.random.default_rng(5)
    a = 0.1 + rng.standard_normal(200)
    b = 0.05 + 0.5 * a + rng.standard_normal(200)
    r1 = lw_hac_test(a, b)
    r2 = lw_hac_test(b, a)
    assert r1.statistic == pytest.approx(-r2.statistic)
    assert r1.pvalue == pytest.approx(r2.pvalue)
    same = lw_hac_test(a, a, prewhiten=False)  # zero difference: never rejects
    assert same.details["delta"] == 0.0 and not same.reject(0.5)
    assert lw_hac_test(a, b, prewhiten=False).test == "lw_hac"


def test_lw_boot_pvalue_eq9() -> None:
    rng = np.random.default_rng(6)
    a = 0.1 + rng.standard_normal(120)
    b = 0.0 + rng.standard_normal(120)
    r = lw_boot_test(a, b, block=4, n_boot=499, seed=3)
    delta, se, ts = studentized_diff_boot(np.column_stack([b, a]), 4, 499, 3)
    d = abs(delta[0]) / se[0]
    assert r.pvalue == pytest.approx((1 + np.sum(np.abs(ts[:, 0]) >= d)) / 500)
    assert r.details["ci_lower"] < r.details["delta"] < r.details["ci_upper"]
    # Same seed, same answer.
    assert lw_boot_test(a, b, block=4, n_boot=499, seed=3).pvalue == r.pvalue
    assert lw_boot_test(a, b, block=1, n_boot=99, seed=3).details["block"] == 1


@pytest.mark.parametrize("method", ["hac", "hac_pw", "boot_ts"])
def test_robust_intervals(method: str) -> None:
    rng = np.random.default_rng(9)
    x = 0.1 + rng.standard_normal(500)
    ci = sharpe_ci(x, method, n_boot=499, seed=1)  # type: ignore[arg-type]
    ref = sharpe_ci(x, "iid_normal")
    assert ci.lower < ci.estimate < ci.upper
    assert ci.upper - ci.lower == pytest.approx(ref.upper - ref.lower, rel=0.25)
    lo = sharpe_ci(x, method, side="lower", n_boot=499, seed=1)  # type: ignore[arg-type]
    assert lo.upper == math.inf and lo.lower < ci.estimate
    up = sharpe_ci(x, method, side="upper", n_boot=499, seed=1)  # type: ignore[arg-type]
    assert up.lower == -math.inf and up.upper > ci.estimate


def test_hac_zero_variance() -> None:
    ci = sharpe_ci(np.full(20, 0.3), "hac")
    assert math.isnan(ci.lower)


def test_hac_se_gaussian_ar1_large_t() -> None:
    rng = np.random.default_rng(10)
    t, phi, sr = 100_000, 0.2, 0.1
    e = rng.standard_normal(t) * math.sqrt(1 - phi**2)
    y = np.empty(t)
    y[0] = rng.standard_normal()
    for i in range(1, t):
        y[i] = phi * y[i - 1] + e[i]
    x = sr + y
    target = math.sqrt(1.505417 / t)
    for pw in (False, True):
        _, se, _ = hac_se(x, prewhiten=pw)
        assert se == pytest.approx(target, rel=0.03)


def test_calibrate_block_small() -> None:
    rng = np.random.default_rng(12)
    x = rng.standard_normal((60, 2)) + 0.1
    best, g = bootstrap.calibrate_block(x, (1, 4), n_pseudo=8, n_boot=49, seed=1)
    assert best in (1, 4)
    assert set(g) == {1, 4} and all(0 <= v <= 1 for v in g.values())


def test_hac_se_gradient_matters_large_sr_fat_tails() -> None:
    # iid skewed fat-tailed data at SR = 0.5: the r^2 gradient term carries a large share
    # of the variance, so a wrong gamma-gradient cannot pass. HAC SE ~ Mertens SE from the
    # same sample (iid, so the long-run covariance is the ordinary covariance).
    rng = np.random.default_rng(21)
    t = 20_000
    z = rng.standard_t(9, t) + 0.5 * (rng.chisquare(4, t) - 4) / math.sqrt(8)
    x = 0.5 + (z - z.mean()) / z.std()
    ref = sharpe_ci(x, "mertens")
    for pw in (False, True):
        _, se, _ = hac_se(x, prewhiten=pw)
        assert se == pytest.approx(ref.se, rel=0.03)
    # Wrong gradient (gamma term dropped) must be far off.
    mu, gam = x.mean(), np.mean(x * x)
    var = gam - mu * mu
    se_bad = math.sqrt(gam**2 / var**3 * var / t)
    assert abs(se_bad / ref.se - 1) > 0.1


def test_block1_se_equals_sample_covariance_delta_method() -> None:
    rng = np.random.default_rng(22)
    x = 0.3 + rng.standard_t(5, 500)
    mu, gam = x.mean(), np.mean(x * x)
    var = gam - mu * mu
    grad = np.array([gam / var**1.5, -mu / (2 * var**1.5)])
    m = np.column_stack([x - mu, x * x - gam])
    cov = m.T @ m / x.size
    assert block_se_sr(x, 1) == pytest.approx(math.sqrt(grad @ cov @ grad / x.size), rel=1e-10)


@pytest.mark.parametrize("method", ["hac", "hac_pw", "boot_ts"])
def test_intervals_scale_invariant(method: str) -> None:
    # The SR is scale-free; so must be the HAC bandwidth and every interval built on it.
    rng = np.random.default_rng(23)
    t = 1000
    state = np.cumsum(rng.random(t) < 0.02) % 2
    x = np.where(state == 0, 0.08 + rng.standard_normal(t), -0.05 + 2.5 * rng.standard_normal(t))
    base = sharpe_ci(x, method, n_boot=199, seed=3)  # type: ignore[arg-type]
    for c in (0.01, 100.0, 1e-4):
        ci = sharpe_ci(c * x, method, n_boot=199, seed=3)  # type: ignore[arg-type]
        assert ci.lower == pytest.approx(base.lower, rel=1e-8, abs=1e-12)
        assert ci.upper == pytest.approx(base.upper, rel=1e-8, abs=1e-12)


def test_lw_tests_scale_invariant() -> None:
    rng = np.random.default_rng(24)
    t = 600
    common = rng.standard_normal(t) * np.repeat(rng.uniform(0.5, 2.0, t // 50), 50)
    a = 0.1 + common + 0.5 * rng.standard_normal(t)
    b = 0.05 + 0.8 * common + 0.7 * rng.standard_normal(t)
    for pw in (False, True):
        p0 = lw_hac_test(a, b, prewhiten=pw).pvalue
        assert lw_hac_test(0.01 * a, 0.01 * b, prewhiten=pw).pvalue == pytest.approx(p0, rel=1e-8)
    p0 = lw_boot_test(a, b, n_boot=199, seed=2).pvalue
    assert lw_boot_test(100 * a, 100 * b, n_boot=199, seed=2).pvalue == pytest.approx(p0)


def test_boot_iid_uses_iid_standard_error() -> None:
    # LW 2008 section 3.2.1: Boot-IID studentizes the original statistic with the sample
    # covariance of (r1, r2, r1^2, r2^2), not the HAC estimate.
    rng = np.random.default_rng(25)
    t = 120
    a = 1.0 + rng.standard_normal(t)
    b = 1.0 + 0.5 * (a - 1.0) + math.sqrt(0.75) * rng.standard_normal(t)
    m1, m2, g1, g2 = a.mean(), b.mean(), np.mean(a * a), np.mean(b * b)
    v1, v2 = g1 - m1 * m1, g2 - m2 * m2
    grad = np.array([g1 / v1**1.5, -g2 / v2**1.5, -m1 / (2 * v1**1.5), m2 / (2 * v2**1.5)])
    y = np.column_stack([a - m1, b - m2, a * a - g1, b * b - g2])
    se_iid = math.sqrt(grad @ (y.T @ y / t) @ grad / t)
    assert iid_se_diff(np.column_stack([b, a]))[0] == pytest.approx(se_iid, rel=1e-10)
    res = lw_boot_test(a, b, block=1, n_boot=199, seed=1)
    assert res.details["se"] == pytest.approx(se_iid, rel=1e-10)
    res4 = lw_boot_test(a, b, block=4, n_boot=199, seed=1)
    assert res4.details["se"] == pytest.approx(lw_hac_se(a, b)[1], rel=1e-12)
