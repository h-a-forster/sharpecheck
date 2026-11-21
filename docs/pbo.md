# PBO / CSCV: conventions, reference targets and other implementations

`sharpecheck.pbo.cscv` implements Algorithm 2.3 of Bailey, Borwein, López de Prado and Zhu,
*The Probability of Backtest Overfitting*, J. Comput. Finance 20(4), 2017 (SSRN 2326253).
Page numbers below refer to the accepted Feb 2015 text and to the Nov 2014 version, which
still contains the Monte Carlo/EVT accuracy study (Sec. 7, Figure 6, p. 33). Tests:
`tests/test_pbo.py`.

## 1. Conventions chosen

| Choice | sharpecheck | Why |
|---|---|---|
| Blocks | S even, T divisible by S, contiguous; otherwise `ValueError` | IS and OOS halves are exactly T/2; no silent row drop or overlap |
| Combinations | all C(S, S/2), lexicographic | C(16, 8) = 12,870 (the paper's "12,780", p. 11 and p. 22, is a typo) |
| Metric | per-period SR, mean/std ddof 0 | DESIGN 3; ddof does not change ranks within a combination |
| IS winner | first index of the maximum | |
| OOS rank | average rank, 1 = worst | ties do not inflate PBO |
| ω̄, λ | r/(N+1), ln(ω̄/(1−ω̄)), always finite | Alg. 2.3 steps f-g |
| PBO | `rule="lt"`: λ < 0 (default); `rule="le"`: λ ≤ 0 | φ = ∫_{−∞}^{0} f(λ)dλ (Sec. 3.1); the text does not say whether λ = 0 counts |
| NaN, constant column, one column | `ValueError` | an undefined SR is never ranked |
| Degradation | OLS of OOS SR (response) on IS SR over the selected pairs | p. 14 |
| Probability of loss | share of combinations with OOS SR of n* < 0 | p. 14 |

Not implemented: stochastic dominance (the paper's SD2 is an integral, p. 18), the EVT
formula (Nov 2014 Eq. 7.2), other metrics.

## 2. Exact results the tests rely on

**D1, pure noise.** If rows are independent (so the IS and OOS metric vectors are
independent), columns exchangeable and the metric tie-free, the OOS rank of the IS winner is
uniform on {1, …, N} in every combination, so E[PBO] equals the single-split probability
exactly, for every S and T (`noise_pbo`). Exchangeable columns alone are not enough: iid
per-column drifts are exchangeable, but the IS winner then tends to win OOS too.

| Rule | N even | N odd |
|---|---|---|
| λ < 0 (r < (N+1)/2) | 1/2 | 1/2 − 1/(2N) |
| λ ≤ 0 with ω̄ = r/(N+1) | 1/2 | 1/2 + 1/(2N) |
| λ ≤ 0 with ω̄ = r/N (r ≤ N/2) | 1/2 | 1/2 − 1/(2N) |
| Def 2.2 literal, r < N/2 | 1/2 − 1/N | 1/2 − 1/(2N) |

`test_noise_law_exact_by_enumeration` checks this as a `Fraction`, without randomness: with
S = 2, averaging over all N! column permutations of the second block makes the OOS rank
exactly uniform in both combinations.

**D2, finite S affects the variance, not the mean.** Combinations share blocks, so one
panel's PBO has a much larger sd than the binomial 1/(2√C) of p. 22: simulated about 0.28
(N = 10, S = 4), 0.23 (N = 10, S = 16), 0.16 (N = 100, S = 16). A noise test must average
K panels with tolerance about 3·sd/√K.

**D3, the paper's own noise target is 1, not 1/2.** The Sec. 6/7 generator re-centres every
column to a full-sample SR of exactly 0, so mean_OOS = −mean_IS and the IS winner is
(almost) the OOS loser. Figure 6's SR = 0 rows all report 1.000. An implementation that
gives about 0.5 on re-centred noise is wrong, and so is one that gives about 1 on raw iid
noise; the tests cover both. With SR = 0 everywhere, SR_OOS ≈ −SR_IS and the degradation
slope is about −1. With a skilled column the selected pairs mix columns, and the slope is
not −1 (one panel at SR = 1, T = 1000, N = 50, S = 10 gave +1.06).

## 3. Figure 6 (Nov 2014, p. 33)

Generator: T×N iid N(0, 1), each column rescaled (ddof 0) to annualised SR 0 with
freq = 365.25·5/7, the last column shifted to SR_case. S is not stated; T ∈ {500, 1000,
2500} is not divisible by the recommended 16, and an independent reproduction matched the
paper's Std_CSCV best at S = 10.

That reproduction (a separate reference CSCV, seed 12345) matched Prob_MC and Prob_EVT to
about 0.02, but not Mean_CSCV where the paper's CSCV − EVT gap is large: at (SR, T, N) =
(2, 1000, 100), (1, 2500, 100), (2, 500, 100) and (3, 500, 500) the paper's Mean_CSCV is
0.06-0.09 higher, more than 10 standard errors. The paper's conclusion that CSCV "errs on the
conservative side" (p. 32) may therefore come from the authors' unpublished code. This is an
open question for the authors, not a confirmed paper error; the 2017 journal appendix
(SSRN 2568435) was not available and may differ.

`sharpecheck.pbo.cscv` reproduces the same numbers (S = 10, K = 100, seed 12345), e.g.
0.723 ± 0.037 at (1, 1000, 100), 0.106 ± 0.046 at (2, 1000, 100), 0.410 ± 0.038 at
(2, 500, 100). `test_figure6_prob_evt` checks nine rows against Prob_EVT within
max(0.02, 3·sd/√K), checks that the mean probability of loss is within 0.02 of the mean PBO,
and records the divergence from Mean_CSCV for the two rows where the paper's CSCV − EVT gap
exceeds 0.05. The N = 500 row is marked slow.

Other inconsistencies in the paper text: Def 2.2 uses r < N/2 (gives 1/2 − 1/N on noise for
even N) while Sec. 3.1 integrates λ up to 0; "four-dimensional mesh of 8,800 elements"
(p. 25) against ranges giving 9,680; Fig. 3 says both "4%" and "0.04%"; example 2 quotes
example 1's SR as 1.3 instead of 1.27.

## 4. Other implementations (read as text, not installed or run)

Read at pinned commits on 2026-11-21: pypbo (esvhd/pypbo @ 4d723f0, `pypbo/pbo.py`), fynance
(ArthurBernard/Fynance @ 1ae9d03, `fynance/research/overfit.py`), sharpebench
(general-liquidity/sharpebench @ 7b199c5, `crates/sharpebench-edge/src/pbo.rs`), R `pbo`
1.3.5 (cran/pbo @ 64b3545, `R/pbo.R`). Line numbers refer to those commits; the cited
ω̄ and PBO-rule lines were re-checked at them through the GitHub API. These are
findings from reading code; none was executed, and the behaviours marked "inferred" were
not verified.

| | pypbo @ 4d723f0 | fynance @ 1ae9d03 | sharpebench @ 7b199c5 | R `pbo` 1.3.5 @ 64b3545 |
|---|---|---|---|---|
| T % S ≠ 0 | drops the first T mod S rows (L141-144) | `np.array_split`, uneven (L300) | uneven, remainder in leading blocks (L191-202) | non-integer `t/s`, rows overlap or are skipped, no check (L43-44, L56-63; inferred) |
| S checks | even (L122) | even, S ≤ 16 (L281, L286) | even ≥ 2, T ≥ S, N ≥ 2 (L104-128) | none |
| OOS rank | average (L199, L206) | average (L140-160) | min rank (L153-156) | average (L83) |
| ω̄ | r/(N+1) (L211) | r/(N+1) (L325) | r/(N+1) (L157) | r/N (L84); λ = +Inf at r = N |
| PBO rule | λ ≤ 0 (L250) | λ ≤ 0 (L331) | λ ≤ 0 (L159) | λ ≤ 0 (L114) |
| Noise mean, odd N | 1/2 + 1/(2N) | 1/2 + 1/(2N) | 1/2 + 1/(2N), more with ties | 1/2 − 1/(2N) |
| NaN metric | ranked best OOS (inferred) | NaN wins IS and ranks best OOS (L315) | n/a | NA skipped IS, last = best OOS (L79, L83) |
| Zero std | user metric | ±1e6 (L116-137) | SR 0 | user metric |
| Degradation | `linregress(IS, OOS)` (L254-256) | OLS of OOS on IS (L163-172) | none | `lm(rn_pairs)` regresses IS on OOS and stores intercept and slope swapped (L131-133; summary.pbo.R L8-9, lattice.pbo.R L326) |
| SD2 | not integrated, a first-order comparison (L263-278) | none | none | plot only, not integrated (lattice.pbo.R L369-374) |

Consequences, and the test in `tests/test_pbo.py` that guards sharpecheck against each (these
tests would fail if sharpecheck adopted that convention; they do not test the other packages):

- λ ≤ 0 with ω̄ = r/(N+1) overstates noise PBO for odd N by 1/N relative to λ < 0 (0.6 vs
  0.4 at N = 5): `test_noise_law_exact_by_enumeration`,
  `test_median_rank_gives_lambda_zero_and_rule_decides`.
- ω̄ = r/N makes λ infinite at the best rank and the noise logits asymmetric:
  `test_logits_finite_at_extremes`.
- Uneven or truncated blocks make IS and OOS unequal or lose rows: `test_refusals_are_loud`,
  `test_combination_count_and_split_integrity`.
- Min-rank on ties inflates PBO (all-tied OOS gives PBO 1): `test_ties_use_average_rank`.
- NaN or constant columns ranked silently push PBO down: `test_refusals_are_loud`.
- Reversed regression (IS on OOS) or swapped coefficients:
  `test_degradation_recovers_planted_line`.

A cross-implementation harness that runs the four packages on identical matrices would need
those packages vetted and installed in an isolated environment; that has not been done.
