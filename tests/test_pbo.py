"""CSCV / PBO reference tests (docs/pbo.md)."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import integrate, stats

from sharpecheck.deflated import EULER_GAMMA, max_normal_moments
from sharpecheck.pbo import (
    cscv,
    cscv_combinations,
    logit_rank,
    noise_pbo,
    stochastic_dominance,
)

FREQ = 365.25 * 5 / 7


def _block(values: np.ndarray) -> np.ndarray:
    """Two rows per column, a + 1 and a - 1: mean a, ddof-0 std 1, so the SR is exactly a."""
    return np.vstack([values + 1.0, values - 1.0])


def _paper_panel(rng: np.random.Generator, t: int, n: int, sr_case: float) -> np.ndarray:
    """Nov 2014 Sec. 6/7 generator: full-sample annualised SR exactly 0, last column sr_case."""
    x = rng.standard_normal((t, n))
    x = (x - x.mean(axis=0)) / x.std(axis=0) / math.sqrt(FREQ)
    x[:, -1] += sr_case / FREQ
    return x


def _evt_pbo(sr: float, t: int, n: int) -> float:
    """Nov 2014 Eq. 7.2, an approximation to PBO under the generator above (annualised SRs).

    The skilled IS SR is N(SR, sigma_c^2) with sigma_c^2 = (freq + SR^2 / 2) / T (T, not T/2:
    the full-sample constraint halves the variance), its OOS SR is 2 SR - IS SR, and the best
    of the N - 1 noise columns is Gumbel, moment-matched to the maximum of N - 1 iid
    N(0, freq / T). PBO = phi_1 + phi_2 with
    phi_1 = int_{-inf}^{2 SR} N(x; SR, sigma_c) (1 - G(max(0, x))) dx and
    phi_2 = 1 - Phi(SR / sigma_c).
    """
    sigma_c = math.sqrt((FREQ + sr * sr / 2) / t)
    sigma_0 = math.sqrt(FREQ / t)
    mean_max, var_max = max_normal_moments(n - 1)
    beta = sigma_0 * math.sqrt(var_max) * math.sqrt(6) / math.pi
    alpha = sigma_0 * mean_max - EULER_GAMMA * beta

    def integrand(x: float) -> float:
        gumbel = math.exp(-math.exp(-(max(0.0, x) - alpha) / beta))
        return float(stats.norm.pdf(x, sr, sigma_c)) * (1.0 - gumbel)

    phi1 = integrate.quad(integrand, sr - 12 * sigma_c, 2 * sr, points=[0.0], limit=200)[0]
    return float(phi1 + stats.norm.sf(sr / sigma_c))


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("rule", ["lt", "le"])
def test_noise_law_exact_by_enumeration(n: int, rule: str) -> None:
    # S = 2, exchangeable OOS: averaging over every column permutation of block 2 makes the
    # OOS rank of n* exactly uniform in both combinations, so the mean PBO is noise_pbo(n).
    a = np.arange(n, dtype=np.float64) * 0.37 + 0.1
    b = np.arange(n, dtype=np.float64)[::-1] * 0.23 - 0.4
    overfit = 0
    total = 0
    for perm in itertools.permutations(range(n)):
        res = cscv(np.vstack([_block(a), _block(b[list(perm)])]), n_blocks=2, rule=rule)  # type: ignore[arg-type]
        overfit += res.n_overfit
        total += res.n_combinations
    assert Fraction(overfit, total) == noise_pbo(n, rule)  # type: ignore[arg-type]


def test_noise_pbo_table() -> None:
    for n in range(2, 40):
        if n % 2 == 0:
            assert noise_pbo(n, "lt") == noise_pbo(n, "le") == Fraction(1, 2)
        else:
            assert noise_pbo(n, "lt") == Fraction(1, 2) - Fraction(1, 2 * n)
            assert noise_pbo(n, "le") == Fraction(1, 2) + Fraction(1, 2 * n)
    assert noise_pbo(5, "lt") == Fraction(2, 5)
    assert noise_pbo(5, "le") == Fraction(3, 5)


def test_median_rank_gives_lambda_zero_and_rule_decides() -> None:
    # N = 5, IS winner is column 4; OOS SRs put it exactly in the middle (rank 3).
    is_sr = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    oos_sr = np.array([0.5, 0.1, 0.9, 0.7, 0.6])
    x = np.vstack([_block(is_sr), _block(oos_sr)])
    lt = cscv(x, n_blocks=2, rule="lt")
    le = cscv(x, n_blocks=2, rule="le")
    assert lt.best_is[0] == 4 and lt.rank_oos[0] == 3.0 and lt.logits[0] == 0.0
    assert not (lt.logits[0] < 0) and le.logits[0] <= 0
    # Combination 2 (block 2 IS): winner column 2 ranks 3 of 5 in block 1 as well.
    assert lt.rank_oos[1] == 3.0
    assert lt.pbo == 0.0 and le.pbo == 1.0


def test_logits_finite_at_extremes() -> None:
    # omega = r/(N+1): the best rank gives ln(N), never +inf (R pbo's omega = r/N does).
    assert logit_rank([1, 10], 10).tolist() == pytest.approx([-math.log(10), math.log(10)])
    with pytest.raises(ValueError):
        logit_rank([0], 10)


def test_ties_use_average_rank() -> None:
    # n* tied with the whole OOS field: average rank (N+1)/2, lambda 0, not overfit under
    # "lt". A min-rank rule (sharpebench) would give rank 1 and PBO 1.
    is_sr = np.array([0.25, 0.5, 0.75, 1.0])  # dyadic, so the tied SRs are exactly equal
    x = np.vstack([_block(is_sr), _block(np.full(4, 0.5))])
    res = cscv(x, n_blocks=2)
    assert res.rank_oos[0] == 2.5 and res.logits[0] == 0.0
    assert res.pbo == 0.5  # combination 2: all-tied IS, first index wins, ranks 1 of 4


def test_combination_count_and_split_integrity() -> None:
    mask = cscv_combinations(16)
    assert mask.shape == (12870, 16)  # C(16, 8); the paper's "12,780" is a typo
    assert np.all(mask.sum(axis=1) == 8)
    assert len({tuple(r) for r in mask}) == 12870
    # Complements are in the set; every block is IS in exactly half the combinations.
    assert np.all(mask.sum(axis=0) == 12870 // 2)
    rows = {tuple(r) for r in mask}
    assert all(tuple(~np.array(r)) in rows for r in list(rows)[:200])


def test_refusals_are_loud() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="divisible"):
        cscv(rng.standard_normal((1000, 5)), n_blocks=16)
    with pytest.raises(ValueError, match="even"):
        cscv(rng.standard_normal((90, 5)), n_blocks=9)
    with pytest.raises(ValueError, match="2 columns"):
        cscv(rng.standard_normal((100, 1)), n_blocks=10)
    x = rng.standard_normal((100, 5))
    x[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        cscv(x, n_blocks=10)
    x = rng.standard_normal((100, 5))
    x[:, 1] = 0.01  # constant positive column: SR undefined, never ranked best
    with pytest.raises(ValueError, match="zero variance"):
        cscv(x, n_blocks=10)
    # All columns constant: every SR is undefined; a min-rank rule would report PBO 1.
    with pytest.raises(ValueError, match="zero variance"):
        cscv(np.tile([0.01, 0.02, -0.01], (100, 1)), n_blocks=10)
    with pytest.raises(ValueError, match="rule"):
        cscv(rng.standard_normal((100, 5)), n_blocks=10, rule="lte")  # type: ignore[arg-type]


def test_column_constant_in_one_half_is_refused() -> None:
    # Constant over S/2 blocks only, so the full-sample variance is positive but the SR of
    # that half is undefined. It must be refused, not ranked with an SR of order 1e5-1e14
    # (the variance of a constant half has to come out as zero up to rounding).
    rng = np.random.default_rng(0)
    for t, s in [(1000, 2), (20000, 4), (1600, 16)]:
        for _ in range(15):
            x = rng.standard_normal((t, 4)) * 0.01
            nb = t // s
            c = rng.choice([-1, 1]) * 10 ** rng.uniform(-6, 3)
            for b in rng.choice(s, s // 2, replace=False):
                x[b * nb : (b + 1) * nb, 1] = c
            with pytest.raises(ValueError, match="zero variance"):
                cscv(x, n_blocks=s)
    # A tiny but real variance is not refused.
    x = rng.standard_normal((1600, 4)) * 0.01
    x[:, 1] = 0.01 + 1e-10 * rng.standard_normal(1600)
    assert cscv(x, n_blocks=16).n_combinations == 12870


def test_matches_direct_loop() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((60, 7)) + 0.05 * rng.standard_normal(7)
    res = cscv(x, n_blocks=6)
    blocks = np.split(x, 6)
    for c, is_blocks in enumerate(itertools.combinations(range(6), 3)):
        xin = np.vstack([blocks[i] for i in is_blocks])
        xout = np.vstack([blocks[i] for i in range(6) if i not in is_blocks])
        sin = xin.mean(0) / xin.std(0)
        sout = xout.mean(0) / xout.std(0)
        k = int(np.argmax(sin))
        assert res.best_is[c] == k
        assert res.sr_is[c] == pytest.approx(sin[k], rel=1e-12)
        assert res.sr_oos[c] == pytest.approx(sout[k], rel=1e-12)
        assert res.sr_oos_all[c] == pytest.approx(sout, rel=1e-12)
        assert res.rank_oos[c] == 1 + np.sum(sout < sout[k])


def test_permutation_and_block_reversal_invariance() -> None:
    rng = np.random.default_rng(11)
    x = rng.standard_normal((80, 9))
    base = cscv(x, n_blocks=8).pbo
    assert cscv(x[:, rng.permutation(9)], n_blocks=8).pbo == base
    rev = np.vstack(np.split(x, 8)[::-1])
    assert cscv(rev, n_blocks=8).pbo == base


def test_degradation_recovers_planted_line() -> None:
    # OOS = a + b * IS on the selected pairs; OOS is the response (R pbo regresses IS on OOS).
    rng = np.random.default_rng(3)
    x = _paper_panel(rng, 1000, 50, 0.0)
    res = cscv(x, n_blocks=10)
    a, b = np.polyfit(res.sr_is, res.sr_oos, 1)
    assert (res.degradation_slope, res.degradation_intercept) == pytest.approx((a, b))
    # Full-sample constraint with SR = 0: SR_OOS ~ -SR_IS for every column, so slope ~ -1.
    # (With a skilled column the selected pairs mix columns and the slope is not -1.)
    assert -1.3 < res.degradation_slope < -0.7


def test_degradation_nan_when_is_winner_sr_is_constant() -> None:
    # S = 2 and column 0 repeats the same skilled block, so it wins IS with one SR in both
    # combinations: the OOS-on-IS regression is undefined and must be NaN, not a number.
    rng = np.random.default_rng(11)
    half = rng.standard_normal((50, 4))
    half[:, 0] += 1.0
    x = np.vstack([half, half])
    x[50:, 1:] = rng.standard_normal((50, 3))
    res = cscv(x, n_blocks=2)
    assert np.all(res.best_is == 0)
    assert res.sr_is[0] == pytest.approx(res.sr_is[1])
    assert math.isnan(res.degradation_slope)
    assert math.isnan(res.degradation_intercept)


def test_paper_generator_sr0_is_one() -> None:
    # Nov 2014 Fig. 6, SR = 0 block: re-centred noise gives PBO 1 (D3), not 1/2.
    res = cscv(_paper_panel(np.random.default_rng(5), 1000, 100, 0.0), n_blocks=10)
    assert res.pbo > 0.99


def test_raw_noise_mean_even_n() -> None:
    # iid noise, N even: E[PBO] = 1/2 exactly under every rank convention except Def 2.2's
    # r < N/2 (1/2 - 1/N = 0.4). One panel's sd is about 0.23 at N = 10, S = 16 (D2), so
    # K = 200 panels give a tolerance of 3 * 0.23 / sqrt(200) = 0.049.
    rng = np.random.default_rng(12345)
    vals = [cscv(rng.standard_normal((1008, 10)), n_blocks=16).pbo for _ in range(200)]
    assert abs(np.mean(vals) - 0.5) < 3 * 0.23 / math.sqrt(200)
    assert 0.18 < np.std(vals) < 0.3  # far above the binomial 1/(2 sqrt(12870)) = 0.0044


# Nov 2014 Fig. 6 (p. 33): (SR, T, N, Prob_EVT, paper Mean_CSCV). S = 10, K = 100 panels.
FIG6 = [
    (1, 500, 100, 0.926, 0.929),
    (1, 1000, 100, 0.713, 0.755),
    (2, 1000, 100, 0.099, 0.163),
    (3, 500, 10, 0.015, 0.028),
    (1, 2500, 10, 0.086, 0.123),
    (1, 1000, 10, 0.360, 0.399),
    (2, 500, 100, 0.405, 0.488),
    pytest.param(1, 500, 500, 0.994, 0.993, marks=pytest.mark.slow),
    (2, 2500, 50, 0.000, 0.002),
]


@pytest.mark.parametrize(("sr", "t", "n", "evt", "paper_cscv"), FIG6)
def test_figure6_prob_evt(sr: float, t: int, n: int, evt: float, paper_cscv: float) -> None:
    # Target Prob_EVT, not Mean_CSCV: the paper's Mean_CSCV is not reproduced where its own
    # CSCV - EVT gap is large (docs/pbo.md section 3).
    rng = np.random.default_rng(12345)
    k = 100
    res = [cscv(_paper_panel(rng, t, n, sr), n_blocks=10) for _ in range(k)]
    vals = np.array([r.pbo for r in res])
    loss = np.array([r.prob_loss for r in res])
    assert abs(vals.mean() - evt) < max(0.02, 3 * vals.std() / math.sqrt(k))
    # Full-sample constraint: median OOS SR ~ 0, so probability of loss ~ PBO (D3).
    assert abs(loss.mean() - vals.mean()) <= 0.02
    if paper_cscv - evt > 0.05:  # recorded divergence from the paper's CSCV column
        assert vals.mean() < paper_cscv - 3 * vals.std() / math.sqrt(k)


@pytest.mark.slow
def test_noise_sd_at_n100_s16() -> None:
    # D2: one panel's PBO on iid noise has sd ~0.16 at N = 100, S = 16, not 1/(2 sqrt(12870)).
    rng = np.random.default_rng(2026)
    vals = np.array([cscv(rng.standard_normal((1008, 100)), n_blocks=16).pbo for _ in range(200)])
    assert abs(vals.mean() - 0.5) < 3 * 0.16 / math.sqrt(200)
    assert 0.13 < vals.std() < 0.19


# Nov 2014 Fig. 6 (p. 33), Prob_EVT column: (SR, T, N, Prob_EVT).
FIG6_EVT = [
    (1, 500, 500, 0.994), (1, 1000, 500, 0.870), (1, 2500, 500, 0.476),
    (1, 500, 100, 0.926), (1, 1000, 100, 0.713), (1, 2500, 100, 0.288),
    (1, 500, 50, 0.859), (1, 1000, 50, 0.626), (1, 2500, 50, 0.220),
    (1, 500, 10, 0.608), (1, 1000, 10, 0.360), (1, 2500, 10, 0.086),
    (2, 500, 500, 0.601), (2, 1000, 500, 0.204), (2, 2500, 500, 0.002),
    (2, 500, 100, 0.405), (2, 1000, 100, 0.099), (2, 2500, 100, 0.001),
    (2, 500, 50, 0.312), (2, 1000, 50, 0.066), (2, 500, 10, 0.137), (2, 1000, 10, 0.023),
    (3, 500, 500, 0.148), (3, 1000, 500, 0.005), (3, 500, 100, 0.068), (3, 1000, 100, 0.002),
    (3, 500, 50, 0.045), (3, 1000, 50, 0.001), (3, 500, 10, 0.015), (3, 1000, 10, 0.001),
]  # fmt: skip


def test_evt_oracle_matches_paper_prob_evt() -> None:
    # The oracle reproduces every printed Prob_EVT within 0.0036 (rounding to 3 decimals is
    # 0.0005 of it). With the maximum of N draws instead of N - 1 the error reaches 0.019
    # at N = 10, so the published column uses N - 1.
    errors = [abs(_evt_pbo(sr, t, n) - evt) for sr, t, n, evt in FIG6_EVT]
    assert max(errors) < 0.004


@pytest.mark.parametrize("sr", [1, 2])
@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize("t", [500, 1000])
def test_cscv_mean_matches_evt_oracle(sr: float, n: int, t: int) -> None:
    # Independent check of the metric, split and argmax against Eq. 7.2, on a seed
    # independent of test_figure6_prob_evt. EVT is an approximation: with K = 2000 panels
    # (S = 10, seed 99) CSCV - EVT measured +0.024 at (1, 500, 10), +0.018 at (1, 1000, 10),
    # +0.006 at (2, 500, 10), -0.002 at (2, 1000, 10) and at most +0.010 for N = 100.
    # Tolerance: that model error (0.025) plus 3 Monte Carlo standard errors.
    rng = np.random.default_rng(2024)
    k = 100
    vals = np.array([cscv(_paper_panel(rng, t, n, sr), n_blocks=10).pbo for _ in range(k)])
    assert abs(vals.mean() - _evt_pbo(sr, t, n)) < 0.025 + 3 * vals.std() / math.sqrt(k)


def test_second_order_dominance_is_an_integral() -> None:
    # Selected {0, 0} against benchmark {-1, 1}: a mean-preserving spread. F_selected exceeds
    # F_benchmark on [0, 1), so there is no first-order dominance, but the integral of
    # F_benchmark - F_selected is 0.5 at 0 and 0 from 1 on, never negative: second order
    # holds. The CDF difference itself (the "SD2" of pypbo and R pbo) is negative on [0, 1).
    sd = stochastic_dominance([0.0, 0.0], [-1.0, 1.0])
    assert sd.grid.tolist() == [-1.0, 0.0, 1.0]
    assert sd.sd2.tolist() == [0.0, 0.5, 0.0]
    assert not sd.first_order and sd.second_order
    assert np.any(sd.cdf_benchmark - sd.cdf_selected < 0)
    # Reversed: the spread does not dominate the point mass at second order.
    rev = stochastic_dominance([-1.0, 1.0], [0.0, 0.0])
    assert rev.sd2.tolist() == [0.0, -0.5, 0.0]
    assert not rev.first_order and not rev.second_order
    # A shift up dominates at both orders; first order is compared exactly, as counts.
    up = stochastic_dominance([0.5, 1.5, 2.5], [0.0, 1.0, 2.0])
    assert up.first_order and up.second_order
    assert stochastic_dominance([1.0, 2.0], [1.0, 2.0]).first_order


def test_dominance_on_cscv_output() -> None:
    # Paper generator with SR = 0: the IS winner is the OOS loser, so its OOS SRs are
    # dominated by those of all strategies at both orders.
    res = cscv(_paper_panel(np.random.default_rng(5), 1000, 50, 0.0), n_blocks=10)
    sd = stochastic_dominance(res.sr_oos, res.sr_oos_all.ravel())
    assert not sd.first_order and not sd.second_order
    # A strongly skilled column (SR 3, T = 1000) is selected and stays ahead out of sample.
    res = cscv(_paper_panel(np.random.default_rng(5), 1000, 50, 3.0), n_blocks=10)
    sd = stochastic_dominance(res.sr_oos, res.sr_oos_all.ravel())
    assert sd.first_order and sd.second_order
