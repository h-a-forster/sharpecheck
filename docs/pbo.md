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
| NaN, one column, a column constant in any IS or OOS half | `ValueError` | an undefined SR is never ranked; half variances are summed from non-negative block terms so a constant half is recognised |
| Degradation | OLS of OOS SR (response) on IS SR over the selected pairs | p. 14 |
| Probability of loss | share of combinations with OOS SR of n* < 0 | p. 14 |
| Stochastic dominance | `stochastic_dominance(res.sr_oos, res.sr_oos_all.ravel())`: first order compared exactly as counts; second order as SD2(x) = ∫_{−∞}^{x} (F_all − F_selected) dt ≥ 0 | p. 18; SD2 is an integral, not the CDF difference |

Not implemented: the EVT formula (Nov 2014 Eq. 7.2; the tests carry it as an oracle, section
3), other metrics.

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

**The EVT column.** Eq. 7.2 (skilled IS SR ~ N(SR, (freq + SR²/2)/T), OOS SR = 2·SR − IS SR,
best noise SR Gumbel, moment-matched to the maximum of N − 1 iid N(0, freq/T)) reproduces
within 0.0036 the 30 Prob_EVT entries with SR > 0 that print above 0.000. Snippet 2 as
printed takes the maximum over N trials; that version misses the column by up to 0.019 (at
N = 10), so the published numbers correspond to N − 1. EVT is an approximation, not the CSCV
expectation: with K = 2000 panels (S = 10, seed 99) the CSCV mean exceeds it by 0.024 at (1, 500, 10) and 0.018
at (1, 1000, 10), and by at most 0.010 at N = 100. This gap is far smaller than the paper's
0.06-0.09 above.

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

Section 5 maps each of these divergences to the test that guards sharpecheck against it.

A cross-implementation harness that runs the four packages on identical matrices would need
those packages vetted and installed in an isolated environment; that has not been done.

## 5. Reference test battery

Thirteen checks, each aimed at a convention or a divergence above. They guard sharpecheck
against adopting that convention; they do not test the other packages. All are in
`tests/test_pbo.py`, use fixed seeds and state their tolerance; "slow" tests run with
`--runslow`. K is the number of simulated panels, sd the spread of one panel's PBO (D2).

| # | Test | What it checks | Guards against |
|---|---|---|---|
| 1 | `test_raw_noise_mean_even_n` | iid noise, T = 1008, N = 10, S = 16, K = 200: mean PBO within 3·0.23/√200 = 0.049 of 1/2; sd in (0.18, 0.3) | Def 2.2's strict r < N/2 (mean 0.4), off-by-one thresholds |
| 2 | `test_noise_law_exact_by_enumeration`, `test_noise_pbo_table`, `test_median_rank_gives_lambda_zero_and_rule_decides`, `test_logits_finite_at_extremes` | odd N noise law exactly (1/2 − 1/(2N) for λ < 0, 1/2 + 1/(2N) for λ ≤ 0); λ = 0 at r = (N+1)/2 classified by `rule`; λ finite at r = 1 and r = N | λ ≤ 0 with ω̄ = r/(N+1) (pypbo, fynance, sharpebench); ω̄ = r/N (R `pbo`, λ = +∞) |
| 3 | `test_paper_generator_sr0_is_one` | paper generator, SR = 0, T = 1000, N = 100: PBO > 0.99 (D3) | ignoring the full-sample constraint (gives about 1/2); a flipped rank direction (about 0) |
| 4 | `test_figure6_prob_evt` | nine Figure 6 rows, S = 10, K = 100, seed 12345: mean PBO within max(0.02, 3·sd/√K) of Prob_EVT; Mean_CSCV divergence recorded (section 3) | wrong metric, split or argmax |
| 5 | `test_evt_oracle_matches_paper_prob_evt`, `test_cscv_mean_matches_evt_oracle` | Eq. 7.2 reproduces the 30 non-zero SR > 0 Prob_EVT entries within 0.004; CSCV mean within 0.025 + 3·sd/√K of it on SR ∈ {1, 2} × N ∈ {10, 100} × T ∈ {500, 1000}, K = 100, another seed | the same errors, independently of the paper's table and of one seed |
| 6 | `test_figure6_prob_evt` | mean probability of loss within 0.02 of mean PBO under the paper generator (D3) | sign or threshold errors in the probability of loss |
| 7 | `test_combination_count_and_split_integrity`, `test_matches_direct_loop`, `test_refusals_are_loud` | 12,870 combinations at S = 16, each block IS in half of them; IS and OOS SRs equal a direct loop over `np.split` blocks; T % S ≠ 0 refused | dropped leading rows (pypbo), overlapping blocks (R `pbo`), uneven halves (fynance, sharpebench), the "12,780" count |
| 8 | `test_ties_use_average_rank`, `test_refusals_are_loud` | n* tied with the whole OOS field gets rank (N+1)/2 and λ = 0; an all-constant panel is refused | min rank on ties (sharpebench; all-tied OOS gives PBO 1) |
| 9 | `test_refusals_are_loud`, `test_column_constant_in_one_half_is_refused` | NaN, a constant column, or a column constant over one CSCV half raise `ValueError`; a tiny real variance does not | NaN ranked best (pypbo, fynance, R `pbo`), ±1e6 or 0 for zero std (fynance, sharpebench), a near-zero half variance ranked as a huge SR |
| 10 | `test_degradation_recovers_planted_line`, `test_degradation_nan_when_is_winner_sr_is_constant` | slope and intercept equal the OLS fit of OOS (response) on IS; slope about −1 under the SR = 0 generator; NaN when undefined | the reversed regression and swapped coefficients of R `pbo` |
| 11 | `test_second_order_dominance_is_an_integral`, `test_dominance_on_cscv_output` | a mean-preserving spread: F_selected > F_all on [0, 1) yet SD2 ≥ 0, so second order holds and first order fails; the reverse fails both | SD2 as the CDF difference, not integrated (pypbo, R `pbo` plots) |
| 12 | `test_permutation_and_block_reversal_invariance` | permuting columns or reversing the block order leaves PBO unchanged | order-dependent combinations or tie-breaking |
| 13 | `test_noise_sd_at_n100_s16` (slow) | iid noise, N = 100, S = 16, K = 200: sd in (0.13, 0.19), mean within 3·0.16/√200 of 1/2 | quoting the binomial 1/(2√C) = 0.0044 as the error of one PBO estimate |
