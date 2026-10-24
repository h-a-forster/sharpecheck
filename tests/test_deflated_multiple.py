"""DSR, expected maximum, FDR quantities and multiple-testing adjustments (DESIGN 7.1, 7.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sharpecheck.deflated import (
    deflated_sharpe_ratio,
    dsr_test,
    expected_max_sr,
    implied_trials,
    max_normal_moments,
    ofdr,
    pfdr,
    var_max_sr,
)
from sharpecheck.multiple import benjamini_hochberg, holm, model_confidence_set, romano_wolf
from sharpecheck.single import power

SR_D = 2.5 / math.sqrt(250)


def test_bldp_2014_dsr_example() -> None:
    sr0 = expected_max_sr(100, 0.5 / 250)
    assert round(sr0, 4) == 0.1132
    assert round(deflated_sharpe_ratio(SR_D, 1250, 100, 0.5 / 250, -3.0, 10.0), 4) == 0.9004
    assert round(deflated_sharpe_ratio(SR_D, 1250, 46, 0.5 / 250, -3.0, 10.0), 4) == 0.9505
    assert round(deflated_sharpe_ratio(SR_D, 1250, 88, 0.5 / 250), 4) == 0.9505


@pytest.mark.parametrize(
    ("n", "exact", "gumbel"),
    [(10, 1.538753, 1.574598), (100, 2.507594, 2.530603), (1000, 3.241436, 3.255122)],
)
def test_expected_max_values(n: int, exact: float, gumbel: float) -> None:
    assert expected_max_sr(n, 1.0, method="exact") == pytest.approx(exact, abs=1e-6)
    assert expected_max_sr(n, 1.0) == pytest.approx(gumbel, abs=1e-6)


def test_equicorrelated_max_reference() -> None:
    assert expected_max_sr(100, 1.0, rho=0.5, method="exact") == pytest.approx(1.773136, abs=1e-6)
    assert var_max_sr(100, 1.0, rho=0.5) == pytest.approx(0.592202, abs=1e-6)


def test_expected_max_edge_cases() -> None:
    assert expected_max_sr(1, 0.3) == 0.0
    assert max_normal_moments(1) == (0.0, 1.0)
    m1, v = max_normal_moments(2)
    assert m1 == pytest.approx(1 / math.sqrt(math.pi), abs=1e-10)
    assert v == pytest.approx(1 - 1 / math.pi, abs=1e-10)
    with pytest.raises(ValueError):
        expected_max_sr(0.5, 1.0)
    with pytest.raises(ValueError):
        expected_max_sr(10, -1.0)
    with pytest.raises(ValueError):
        expected_max_sr(10, 1.0, rho=1.0)
    with pytest.raises(ValueError):
        expected_max_sr(10, 1.0, method="bogus")  # type: ignore[arg-type]


def test_equicorrelated_max_vs_simulation() -> None:
    rng = np.random.default_rng(0)
    n, rho, draws = 50, 0.3, 100_000
    common = rng.standard_normal((draws, 1))
    z = math.sqrt(rho) * common + math.sqrt(1 - rho) * rng.standard_normal((draws, n))
    mx = z.max(axis=1)
    e, v = expected_max_sr(n, 1.0, rho=rho, method="exact"), var_max_sr(n, 1.0, rho=rho)
    assert abs(mx.mean() - e) < 4 * math.sqrt(v / draws)
    # Var of the sample variance ~ (kurtosis - 1) v^2 / draws; 4 SE with a generous kurtosis.
    assert abs(mx.var() - v) < 4 * math.sqrt(4 * v * v / draws)


def test_implied_trials() -> None:
    assert implied_trials(100, 0.0) == 100
    assert implied_trials(100, 1.0) == 1
    assert implied_trials(100, 0.5) == pytest.approx(50.5)
    assert implied_trials(10, -0.2) == 10
    with pytest.raises(ValueError):
        implied_trials(0, 0.1)


def test_lzl_fdr_values() -> None:
    assert round(pfdr(0.05, 0.05, 0.315), 3) == 0.581
    sr, g3, g4 = 0.036 / 0.079, -2.448, 10.164
    assert round(ofdr(sr, 0.0, 0.5, 24, 0.05, g3, g4), 3) == 0.306
    beta = 1 - power(0.0, 0.5, 24, g3, g4)
    assert round(pfdr(0.05, 0.05, beta), 3) == 0.581
    with pytest.raises(ValueError):
        pfdr(0.05, 0.05, 1.0)


def test_dsr_test_defaults() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((250, 20))
    x[:, 3] += 0.3
    r = dsr_test(x)
    srs = x.mean(axis=0) / x.std(axis=0)
    assert r.details["best"] == 3
    assert r.details["var_sr"] == pytest.approx(np.var(srs, ddof=1))
    assert r.details["n_trials"] == 20
    assert r.alternative == "greater" and 0 <= r.pvalue <= 1
    r2 = dsr_test(x, n_trials=5, var_sr=1 / 250)
    assert r2.details["n_trials"] == 5 and r2.details["var_sr"] == 1 / 250
    assert r2.details["sr0"] == pytest.approx(expected_max_sr(5, 1 / 250))
    with pytest.raises(ValueError):
        dsr_test(x[:, :1])
    zero = np.zeros((10, 3))
    assert math.isnan(dsr_test(zero).pvalue)


def test_holm_and_bh_hand_examples() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # Holm: sorted 0.005, 0.01, 0.03, 0.04 -> 0.02, 0.03, 0.06, 0.06
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    # BH: sorted p*k/i = 0.02, 0.02, 0.04, 0.04
    assert np.allclose(benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])
    assert holm([]).size == 0 and benjamini_hochberg([]).size == 0
    assert np.allclose(holm([0.6, 0.9]), [1.0, 1.0])
    with pytest.raises(ValueError):
        holm([0.1, 1.2])
    with pytest.raises(ValueError):
        benjamini_hochberg([np.nan])


def test_romano_wolf_single_hypothesis_equals_bootstrap_p() -> None:
    rng = np.random.default_rng(2)
    ts = rng.standard_normal((999, 1))
    t = np.array([1.7])
    p = romano_wolf(t, ts)
    assert p[0] == pytest.approx((1 + np.sum(np.abs(ts[:, 0]) >= 1.7)) / 1000)


def test_romano_wolf_stepdown_hand_example() -> None:
    # |t| = (3, 2). Step 1 uses max(|t*_1|, |t*_2|): one of four rows >= 3, p = 2/5.
    # Step 2 drops hypothesis 1 and uses |t*_2| alone: one row >= 2, p = 2/5.
    # A single-step max-T adjustment would give hypothesis 2 the value 3/5 instead.
    ts = np.array([[3.5, 0.0], [0.0, 2.5], [0.1, 0.1], [0.2, 0.2]])
    t = np.array([3.0, -2.0])
    assert romano_wolf(t, ts) == pytest.approx([0.4, 0.4])
    single = (1 + np.sum(np.abs(ts).max(axis=1)[:, None] >= np.abs(t), axis=0)) / 5
    assert single == pytest.approx([0.4, 0.6])
    # Monotonicity enforcement: a raw step-2 p-value below step 1 is raised to it.
    ts2 = np.array([[3.5, 0.0], [3.6, 0.0], [0.1, 0.1], [0.2, 2.5]])
    assert romano_wolf(t, ts2) == pytest.approx([0.6, 0.6])


def test_romano_wolf_properties() -> None:
    rng = np.random.default_rng(3)
    k, b = 6, 1999
    cov = 0.5 * np.eye(k) + 0.5
    ts = rng.multivariate_normal(np.zeros(k), cov, size=b)
    t = np.array([3.0, -0.5, 2.2, 1.0, -2.6, 0.1])
    adj = romano_wolf(t, ts)
    raw = (1 + np.sum(np.abs(ts) >= np.abs(t), axis=0)) / (b + 1)
    assert np.all(adj >= raw - 1e-12)
    order = np.argsort(-np.abs(t))
    assert np.all(np.diff(adj[order]) >= -1e-12)
    # Never larger than the single-step (max over all) adjustment.
    single = (1 + np.sum(np.abs(ts).max(axis=1)[:, None] >= np.abs(t), axis=0)) / (b + 1)
    assert np.all(adj <= single + 1e-12)
    with pytest.raises(ValueError):
        romano_wolf(t, ts[:, :3])
    nan_ts = ts.copy()
    nan_ts[0, 0] = np.nan
    assert np.all(np.isfinite(romano_wolf(t, nan_ts)))


def test_model_confidence_set_runs() -> None:
    rng = np.random.default_rng(4)
    losses = rng.standard_normal((300, 3)) + np.array([0.0, 0.0, 1.0])
    out = model_confidence_set(losses, size=0.1, reps=200, seed=1)
    assert 2 in out["excluded"]
    assert set(out["included"]) | set(out["excluded"]) == {0, 1, 2}
