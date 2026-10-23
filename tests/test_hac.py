"""Kernel long-run covariance: weights, arch oracle, Andrews bandwidth, prewhitening."""

from __future__ import annotations

import math

import numpy as np
import pytest
from arch.covariance.kernel import Bartlett, Parzen, QuadraticSpectral

from sharpecheck.hac import andrews_bandwidth, autocovariances, kernel_weights, long_run_cov


def test_kernel_weights_values() -> None:
    lags = np.array([0, 1, 2, 3, 4])
    assert np.allclose(kernel_weights(lags, 2.0, "bartlett"), [1, 0.5, 0, 0, 0])
    # Parzen at x = 0.25, 0.5, 0.75, 1: 1 - 6/16 + 6/64; 0.25; 2 * 0.25^3; 0.
    pz = kernel_weights(lags, 4.0, "parzen")
    assert np.allclose(pz, [1, 1 - 6 / 16 + 6 / 64, 0.25, 2 * 0.25**3, 0.0])
    qs = kernel_weights(np.array([0.0, 1e-9, 1.0]), 1.0, "qs")
    assert qs[0] == 1.0 and qs[1] == pytest.approx(1.0)
    x = 1.0
    hand = (
        25
        / (12 * math.pi**2 * x**2)
        * (math.sin(6 * math.pi * x / 5) / (6 * math.pi * x / 5) - math.cos(6 * math.pi * x / 5))
    )
    assert qs[2] == pytest.approx(hand, rel=1e-12)


def test_kernel_validation() -> None:
    with pytest.raises(ValueError):
        kernel_weights([0, 1], 0.0)
    with pytest.raises(ValueError):
        kernel_weights([0, 1], 1.0, "gauss")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        long_run_cov(np.ones((10, 2)), dof_adjust=10)


@pytest.fixture
def series() -> np.ndarray:
    rng = np.random.default_rng(0)
    e = rng.standard_normal((400, 3))
    y = np.empty_like(e)
    y[0] = e[0]
    for t in range(1, 400):
        y[t] = 0.4 * y[t - 1] + e[t]
    y[:, 1] += 0.5 * y[:, 0]
    return y


@pytest.mark.parametrize("bw", [2.5, 6.0, 13.7])
def test_qs_equals_arch_fixed_bandwidth(series: np.ndarray, bw: float) -> None:
    ref = QuadraticSpectral(series, bandwidth=bw, df_adjust=0, center=True).cov.long_run
    assert np.max(np.abs(long_run_cov(series, "qs", bw) - ref)) < 1e-10


@pytest.mark.parametrize("bw", [2, 5, 9])
def test_bartlett_parzen_equal_arch(series: np.ndarray, bw: int) -> None:
    # arch truncates at lag `bandwidth` with weights k(j / (bandwidth + 1)).
    for cls, name in [(Bartlett, "bartlett"), (Parzen, "parzen")]:
        ref = cls(series, bandwidth=bw, df_adjust=0, center=True).cov.long_run
        assert np.max(np.abs(long_run_cov(series, name, bw + 1) - ref)) < 1e-10


def test_autocovariances_direct(series: np.ndarray) -> None:
    y = series - series.mean(axis=0)
    g = autocovariances(y)
    t = len(y)
    for j in (0, 1, 7, t - 1):
        assert np.allclose(g[j], y[j:].T @ y[: t - j] / t, atol=1e-12)


def test_andrews_bandwidth_hand_formula() -> None:
    rng = np.random.default_rng(3)
    t = 500
    y = np.empty(t)
    y[0] = 0
    e = rng.standard_normal(t)
    for i in range(1, t):
        y[i] = 0.5 * y[i - 1] + e[i]
    d = y - y.mean()
    rho = (d[:-1] @ d[1:]) / (d[:-1] @ d[:-1])
    s2 = np.mean((d[1:] - rho * d[:-1]) ** 2)
    a1 = 4 * rho**2 * s2**2 / ((1 - rho) ** 6 * (1 + rho) ** 2) / (s2**2 / (1 - rho) ** 4)
    a2 = 4 * rho**2 * s2**2 / (1 - rho) ** 8 / (s2**2 / (1 - rho) ** 4)
    assert andrews_bandwidth(y[:, None], "bartlett") == pytest.approx(1.1447 * (a1 * t) ** (1 / 3))
    assert andrews_bandwidth(y[:, None], "parzen") == pytest.approx(2.6614 * (a2 * t) ** 0.2)
    assert andrews_bandwidth(y[:, None], "qs") == pytest.approx(1.3221 * (a2 * t) ** 0.2)


def test_andrews_bandwidth_caps_rho() -> None:
    y = np.cumsum(np.ones(100)) ** 2  # deterministic trend: rho near 1
    bw = andrews_bandwidth(y[:, None])
    assert math.isfinite(bw) and bw > 0
    assert andrews_bandwidth(np.ones((20, 1))) == 1.0


def test_dof_adjust_scales() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((50, 2))
    a = long_run_cov(x, bandwidth=3.0)
    b = long_run_cov(x, bandwidth=3.0, dof_adjust=4)
    assert np.allclose(b, a * 50 / 46)


def test_one_dimensional_input() -> None:
    rng = np.random.default_rng(6)
    x = rng.standard_normal(100)
    assert long_run_cov(x).shape == (1, 1)


def _var1(t: int, amat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = amat.shape[0]
    e = rng.standard_normal((t + 500, k))
    y = np.zeros_like(e)
    for i in range(1, t + 500):
        y[i] = amat @ y[i - 1] + e[i]
    return y[500:]


def test_prewhitened_matches_true_lrv_var1() -> None:
    amat = np.array([[0.5, 0.1], [0.0, 0.3]])
    rng = np.random.default_rng(7)
    y = _var1(200_000, amat, rng)
    binv = np.linalg.inv(np.eye(2) - amat)
    truth = binv @ binv.T
    for pw in (True, False):
        est = long_run_cov(y, "qs", prewhiten=pw)
        assert np.all(np.abs(est - truth) / np.abs(truth).max() < 0.03), (pw, est, truth)


def test_prewhitening_near_unit_root_is_finite() -> None:
    rng = np.random.default_rng(8)
    y = np.cumsum(rng.standard_normal((300, 2)), axis=0)  # random walk: |A| ~ 1, capped
    est = long_run_cov(y, prewhiten=True)
    assert np.all(np.isfinite(est))
    assert np.all(np.linalg.eigvalsh(est) > 0)
