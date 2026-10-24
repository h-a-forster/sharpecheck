# sharpecheck report

Results of the quick Monte Carlo profile and the 1/N portfolio study, written by hand from
the committed files under `results/`. Every number below can be traced to one of those
files. The design, the formulas and the expectations that were written down before any
run are in `docs/DESIGN.md` (sections 5, 6 and 6.6). Deviations from that design are
listed in `docs/DESIGN.md` section 13.

**Revision.** A pre-release review found six code defects (DESIGN section 13, item 14):
`exact_nct` raised on many negative t statistics, Boot-IID used the HAC instead of the
iid standard error, the HAC bandwidth depended on the units of the returns, the t6-GARCH
pair DGP drew independent t6 components, `sharpecheck compare` could misalign columns
with missing values, and a version bump blocked reports. After the fixes, the coverage and
difference experiments and the portfolio report were re-run. All numbers below are from
the re-run. Coverage moved by at most 0.4 points in any cell. The LW Table 1 agreement
went from 32 to 34 of 36, because Boot-IID now follows LW.

**Summary.**

- Under iid returns (normal, t5) every method except the bootstrap keeps 95% coverage
  within Monte Carlo error from T = 60 upward. The studentized block bootstrap
  over-covers at small T: 98.0% at T = 60 under normal returns.
- With serial dependence the iid formulas fail as theory predicts: about 89% coverage
  under AR(1) with phi = 0.2. The AR(1) correction, HAC with prewhitening and the
  bootstrap repair this.
- At SR > 0, no method reaches nominal coverage under the two processes whose dependence
  sits in the variance or the regime rather than at lag 1: GARCH(1,1)-t7 and 2-state
  regime switching. At SR = 0.3 and T = 2520, coverage is 81-83% under GARCH and 75-85%
  under regime switching. At SR = 0 the GARCH cells are near nominal for the analytic and
  HAC methods (94.2-95.3% at T >= 120), because the variance-of-variance term is then
  zero, whereas regime switching stays at 89-93%. The cause is measured directly. The true variance of the SR estimator is about twice what the iid
  and AR(1) formulas give (`sr_variance.csv`), and the HAC methods' AR(1)-plug-in
  bandwidth does not capture that persistence.
- Our Ledoit-Wolf (2008) Table 1 replication agrees in 34 of 36 entries, including all 24
  JKM, HAC, HACPW and Boot-IID entries. The two misses are our fixed-block Boot-TS with
  b = 6, which is conservative (LW calibrate the block).
- On correlated trials the DSR rules with the true number of trials are very
  conservative. PSR without deflation is badly oversized, up to 100% family-wise error.
  Holm over the per-trial PSR p-values exceeds 5% in three fat-tailed cells with many
  trials, and the exact-moment rule exceeds it in 11 of 36 cells.
- Portfolio study: no rule's Sharpe ratio differs from 1/N at 5% after the pre-registered
  Romano-Wolf adjustment. The closest is Ledoit-Wolf minimum variance on the 25
  size/book-to-market portfolios: adjusted p = 0.072, raw p = 0.014.

## 1. What was run

| Item | Value |
|---|---|
| Config | `configs/mc_quick.toml` (config hash `4b4c09b295deca28`), seed 20261024 |
| Coverage grid | 6 DGPs x SR {0, 0.1, 0.3} x T {60, 120, 250, 1000, 2520} = 90 cells, 7 methods |
| Replications | 2000 per cell for the analytic and HAC methods; `boot_ts` on the first 500 (B = 499, block `round(T^(1/3))`) |
| Difference tests | LW (2008) DGPs, T = 120, 5000 replications, M = 499 |
| DSR experiment | N {10, 100, 1000} x rho {0, 0.3, 0.7} x T {250, 1250} x {normal, t5}, 1000 replications per cell |
| Sizes after pilot | the timing pilot projected 39 wall minutes (> 35 budget), so `boot_reps` went 1000 -> 500 and `dsr.reps` 2000 -> 1000 (`results/mc-quick/calibration.json`) |
| Run | 984 checkpointed units, 3 worker processes. First run: 27.7 min wall, no failures. After the corrections, the 840 coverage and difference units were re-run (34.6 min wall, no failures, no retries); the 144 DSR units use no changed code and were kept. 124 CPU-minutes inside the current units |
| Versions | Python 3.13.1, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, arch 8.0.0 (`results/mc-quick/summary.json`, from the first run's manifest; the re-run used the same environment with the package at 0.1.0) |

Every rate has a 95% Wilson interval in the CSV files. A coverage cell is flagged as
under-covering when a one-sided binomial test of coverage < 0.95 stays significant after
Holm's correction across all 630 (cell, method) rows **and** the coverage is below 0.93.
`nan` intervals count as misses. There were none in this run.

Monte Carlo error: with 2000 replications the standard error of a coverage near 0.95 is
0.49 points; with 500 (`boot_ts`) it is 0.97 points.

## 2. Coverage of single-SR intervals

![coverage heatmap](results/mc-quick/coverage_heatmap.png)

Files: `results/mc-quick/coverage.csv`, `onesided.csv`, `sr_variance.csv`.

Flagged cells per method (of 90): iid_normal 46, exact_nct 46, mertens 46, ar1 34, hac 35,
hac_pw 33, boot_ts 16 (256 in all). For the 2000-replication methods every cell below 0.93
is flagged. boot_ts has only 500 replications per cell, so the Holm test has less power:
25 of its cells are below 0.93 but only 16 are flagged. Compare methods by the raw count
(25 against 33-46) rather than by flags. Mean absolute deviation from 0.95 over all 90 cells: boot_ts 0.024,
hac 0.030, hac_pw 0.030, ar1 0.033, mertens 0.041, exact_nct 0.044, iid_normal 0.045. No
cell was flagged as over-covering. The largest over-coverage, boot_ts under normal
returns at T = 60, SR = 0 (0.980, Wilson [0.964, 0.989]), misses the Holm threshold.

By process:

- **normal, t5.** No method is flagged under normal returns. Under t5, a few small-T cells
  of `ar1`, `hac`, `hac_pw` and `mertens` (6 in all) are flagged, all between 0.92 and
  0.93, all at T = 60. The bootstrap is wide at small T: under normal returns, SR = 0, coverage is
  0.980 at T = 60 and 0.960 at T = 120, with mean widths 13% and 7% above iid_normal.
- **ar1-0.2.** iid_normal, exact_nct and mertens cover 88-90% at every T, against the 89%
  predicted by the variance factor 1.5. ar1, hac_pw and boot_ts are at 92.6-97.0%; plain
  hac is at 92-94% at T <= 250.
- **crash.** At SR = 0.3, iid_normal and exact_nct cover about 83% at every T. The
  skewness and kurtosis terms are missing, and the empirical T Var(SR_hat) is 2.15
  against 1.045. mertens, ar1 and HAC recover with T: 82-85% at T <= 120, 89-90% at
  T = 250, 94-95% at T >= 1000.
- **garch-t7.** At SR = 0.3, coverage falls with T for every method, from about 88-92%
  at T = 60 to 81-83% at T = 2520. The empirical T Var(SR_hat) at T = 2520 is 2.27,
  while the iid formula with the population kurtosis gives 1.13. The volatility
  clustering makes r_t^2 strongly persistent (alpha + beta = 0.98), so the variance of
  the sample variance is much larger than its iid value. The (g4 - 1)/4 SR^2 term misses
  this, and so does a kernel estimator whose bandwidth comes from the lag-1
  autocorrelation of r_t^2.
- **regime.** Even at SR = 0 every method covers only 89-93% at all T. The lag-1
  autocorrelation is 0.025, but it decays only as 0.88^k. The empirical T Var(SR_hat)
  is 1.43, while the AR(1) formula gives 1.05. At SR = 0.3 coverage is 75-86% at
  T = 2520. HAC is best there (0.85) and still far from nominal.

**One-sided size** (`onesided.csv`). One-sided size is the rejection rate of
`H0: SR <= SR_true` at nominal 5%, i.e. how often a one-sided 95% lower bound lies above
the truth. Averaged over all cells of a process, boot_ts is closest: 0.048 under normal
returns, 0.047 under ar1-0.2, 0.077 under crash, 0.076 under garch-t7 and 0.105 under
regime. Both PSR conventions are oversized wherever the two-sided intervals under-cover.
At small T the LZL 2025 "null" convention is more oversized than the BLdP 2012
"observed" convention. For crash at SR = 0.3 the two give 0.220 vs 0.203 at T = 60 and
0.128 vs 0.112 at T = 250. Under normal returns both stay at 4.2-7.0% (the extremes
come from different cells).

### Pre-registered expectations (DESIGN 6.6)

| # | Expectation | Verdict | Evidence |
|---|---|---|---|
| 1 | iid_normal under-covers for crash at SR > 0 and for ar1-0.2 everywhere | confirmed | crash SR = 0.3: 82-83%; SR = 0.1: 90-91%; ar1-0.2: 88-90% in all 15 cells |
| 2 | mertens fixes t5 and crash at large T, not ar1-0.2; erratic for t5 at small T | mostly confirmed | crash SR = 0.3: 94.4% (T = 1000), 95.4% (2520); ar1-0.2 88-90%; t5 small T is mildly low (0.926-0.942 at T = 60), not erratic |
| 3 | ar1 fixes ar1-0.2 but not garch-t7 at SR > 0 | confirmed | ar1-0.2: 92.6-95.2%; garch-t7 SR = 0.3: 83-88% |
| 4 | hac/hac_pw liberal at T <= 120, near nominal at T >= 1000; prewhitening helps under ar1-0.2 | mixed | under normal and t5, hac is near 95% already at T = 60; under garch-t7 and regime it is far from nominal even at T = 2520; prewhitening helps under ar1-0.2 (hac_pw 93-95% vs hac 92-95%) |
| 5 | boot_ts closest to nominal overall, not exact at T = 60 | confirmed | lowest mean deviation (0.024) and fewest cells below 0.93 (25, against 33-46; its 16 Holm flags are not comparable because its cells have 500 replications); conservative at T = 60 (98.0% under normal, SR = 0) |
| 6 | exact_nct exact for normal, inherits iid_normal's failures elsewhere | confirmed | normal: 94.0-95.4% in all 15 cells; elsewhere within about 1 point of iid_normal; no bound fell back to the normal approximation (`fallback_bounds` in `coverage.csv`) |
| 7 | DSR rules 1-2 conservative under independent trials; rule 5 badly oversized; rule 3 below 5% for moderate undercounting; rule 6 controls FWER; rule 7 near 5% | mixed | see section 4 |
| 8 | our difference-test sizes agree with LW Table 1 for JKM, HAC, HACPW and Boot-IID | confirmed | all 24 such entries agree (section 3) |

The surprise relative to the expectations is that HAC does not rescue the
GARCH and regime cells at large T. The design anticipated small-sample problems for HAC,
not a failure that persists at T = 2520.

## 3. Ledoit-Wolf (2008) Table 1 replication

File: `results/mc-quick/difference.csv`. Nominal 5%, T = 120, 5000 replications. The
agreement criterion is `|ours - LW| <= 3 sqrt(p (1 - p) (1/5000 + 1/5000))`, with `p` the
pooled rate. Boot-TS is compared with fixed blocks b = 4 and b = 6, whereas LW calibrate
the block size.

| DGP | JKM ours / LW | HAC | HACPW | Boot-IID | Boot-TS b=4 | Boot-TS b=6 |
|---|---|---|---|---|---|---|
| Normal-IID | 5.0 / 5.0 | 5.4 / 5.3 | 5.7 / 5.4 | 4.9 / 4.9 | 4.1 / 4.8 | **3.5** / 4.8 |
| t6-IID | 9.8 / 10.7 | 7.0 / 6.7 | 7.2 / 6.9 | 5.6 / 5.2 | 4.8 / 5.0 | 4.1 / 5.0 |
| Normal-GARCH | 6.6 / 7.2 | 6.7 / 7.1 | 6.9 / 7.2 | 6.0 / 6.0 | 5.1 / 5.5 | 4.5 / 5.5 |
| t6-GARCH | 7.2 / 7.4 | 7.4 / 7.7 | 7.5 / 7.5 | 6.9 / 6.9 | 5.6 / 5.7 | 4.8 / 5.7 |
| Normal-VAR | 9.1 / 9.5 | 7.0 / 6.9 | 6.1 / 6.1 | 8.6 / 8.5 | 4.3 / 5.0 | 3.8 / 5.0 |
| t6-VAR | 12.5 / 14.5 | 7.6 / 7.9 | 6.4 / 7.3 | 8.0 / 7.3 | 4.2 / 5.1 | **3.7** / 5.1 |

Bold entries fail the criterion (34 of 36 agree). The qualitative findings of LW
reproduce:

- JKM is liberal under fat tails and serial correlation: 12.5% for t6-VAR.
- HAC is mildly liberal at T = 120.
- The studentized block bootstrap is the only test close to or below nominal in every
  DGP.

Notes on the comparison:

- **Boot-IID** follows LW section 3.2.1: the original statistic is studentized with the
  iid standard error (sample covariance of `(r1, r2, r1^2, r2^2)`). It reproduces LW's
  liberal size under the VAR DGPs: 8.6% against 8.5% and 8.0% against 7.3%. The first
  quick run studentized Boot-IID with the prewhitened HAC standard error. That was a
  deviation from LW's stated procedure, not a convention, and it gave 5.5% and 5.1%
  (two misses). `scripts/lw_bootiid_convention.py` runs both variants on the same draws
  (2000 replications per DGP, `results/mc-quick/bootiid_convention.json`): iid SE 9.2%
  and 9.0%, HAC SE 5.8% and 5.9% for Normal-VAR and t6-VAR; the IID and GARCH rates
  differ by at most 0.5 points between the variants.
- **Boot-TS with b = 6** is conservative (3.5-4.8%). LW's calibrated block would differ
  per data set; we did not calibrate inside the Monte Carlo (DESIGN 2, item 4). With
  b = 4 all six entries agree.
- **t6-GARCH** now uses bivariate t6 innovations with one shared mixing variable, as LW
  specify. The first run drew two independent t6 components; its rates agreed with LW
  too, so the effect at this sample size is small.

## 4. Deflated Sharpe ratio under correlated and miscounted trials

File: `results/mc-quick/dsr.csv`. All trials have SR = 0, so every rejection is a family-wise
error. 1000 replications per cell (Wilson half-width about 1.4 points at 5%).

- **psr_best** (best trial, no deflation): FWER 31-42% at N = 10 with rho <= 0.3
  (16.5-21% at rho = 0.7) and 0.96-1.00 at N = 1000 with rho <= 0.3. Badly oversized, as expected.
- **dsr_trueN_estV** and **dsr_trueN_oracleV** (DSR with the true N): FWER 0-1.2% for
  rho <= 0.3 and 2-4% (estV) at rho = 0.7. Very conservative: a DSR "rejection at 95%"
  is much stricter than 5%.
- **dsr_N/10** (trials undercounted tenfold): with N = 10 this is the undeflated PSR
  (16.5-42%). With N = 100 it gives 5.2-11.4%, and with N = 1000 it gives 0.7-10.1%. So
  moderate undercounting is not always absorbed by DSR's conservatism, contrary to
  expectation 7.
- **dsr_impliedN** (BLdP eq. 9 from the average correlation): close to dsr_trueN_estV for
  rho <= 0.3. At rho = 0.7 it gives 4-7.5%, above the true-N rule, because it shrinks
  N while the variance input already reflects the correlation.
- **holm_psr** (Holm over the N per-trial PSR(0) p-values): mostly at or below 5%. Its
  Wilson interval lies above 5% in three cells, all with t5 trials and rho = 0:
  N = 1000, T = 250 (10.3%, Wilson about [8.6, 12.3]); N = 100, T = 250 (8.1%); and
  N = 1000, T = 1250 (6.9%, lower limit 5.5%). The normal approximation behind PSR is not accurate
  in the far tail (alpha / N = 5e-5) for fat-tailed returns at moderate T.
- **lzl_max** (best SR against the exact mean and sd of the maximum, true rho): 4.2-9.5%.
  Its Wilson interval lies above 5% in 11 of 36 cells. The largest rates are at N = 1000,
  T = 250, rho = 0 (9.5% normal, 7.3% t5); at N = 100 with rho = 0 it is 7.6-7.7% in
  three cells, and 7.5% at rho = 0.3, T = 250, normal. The maximum's right tail is heavier
  than the normal approximation assumes.

Expectation 7 is therefore mixed:

- confirmed for rules 1-2 and 5; rule 7 is near 5% only loosely (above 5% in 11 cells,
  up to 9.5%);
- refuted for rule 3 in several cells;
- only approximately confirmed for rule 6, which fails with fat tails and many trials.

## 5. 1/N versus optimised portfolios

Files: `results/portfolio/tests.csv`, `rules.csv`, `block_sensitivity.csv`, `summary.json`.

Data: Kenneth R. French Data Library, vintage 202608 (CRSP), value-weighted monthly
returns of 10 industry and 25 size/book-to-market portfolios, minus the one-month T-bill.
SHA-256 values are recorded in `results/portfolio/summary.json`. Rules are re-estimated
monthly on the previous 120 months. Out of sample runs 1973-07 to 2026-08 (638 months).

**Primary test** (fixed in DESIGN 5.4 before the run): LW studentized circular block
bootstrap, two-sided, M = 4999, block 9, Romano-Wolf adjusted over the 8 comparisons
against 1/N, full period, alpha = 0.05. **Result: no rule is significantly different from
1/N.**

| Dataset | Rule | SR (monthly) | 95% CI (boot_ts) | SR - SR(1/N) | raw p | Romano-Wolf p | JKM p |
|---|---|---|---|---|---|---|---|
| ind10 | 1/N | 0.159 | [0.071, 0.248] | | | | |
| ind10 | inverse_vol | 0.163 | [0.074, 0.253] | 0.004 | 0.168 | 0.438 | 0.174 |
| ind10 | gmv_sample | 0.166 | [0.087, 0.245] | 0.007 | 0.831 | 0.831 | 0.820 |
| ind10 | gmv_lw | 0.172 | [0.093, 0.252] | 0.013 | 0.641 | 0.700 | 0.624 |
| ind10 | hrp | 0.166 | [0.078, 0.254] | 0.007 | 0.319 | 0.573 | 0.319 |
| ff25 | 1/N | 0.160 | [0.070, 0.249] | | | | |
| ff25 | inverse_vol | 0.164 | [0.074, 0.255] | 0.005 | 0.028 | 0.127 | 0.009 |
| ff25 | gmv_sample | 0.241 | [0.152, 0.330] | 0.082 | 0.032 | 0.127 | 0.021 |
| ff25 | gmv_lw | 0.234 | [0.142, 0.327] | 0.075 | 0.014 | 0.072 | 0.010 |
| ff25 | hrp | 0.164 | [0.073, 0.254] | 0.004 | 0.273 | 0.573 | 0.259 |

Annualized SR (x sqrt(12), iid scaling) for the ff25 minimum-variance rules: 0.84
(sample) and 0.81 (Ledoit-Wolf), against 0.55 for 1/N. Mean monthly turnover is 0.82 for
ff25 sample GMV, 0.34 for ff25 Ledoit-Wolf GMV and 0.018 for 1/N. Transaction costs are
not modelled, and they would work against the GMV rules.

Secondary evidence:

- Unadjusted, three ff25 comparisons have p < 0.05 by JKM (the test DeMiguel, Garlappi
  and Uppal used) and three by the bootstrap. The bootstrap p-value for ff25 inverse_vol
  fell from 0.091 to 0.028 with the scale-invariant bandwidth: this pair is almost
  perfectly correlated, so its standard error is sensitive to the bandwidth. This is how a 1/N comparison can look
  significant without multiplicity control.
- Holm-adjusted JKM and LW HAC p-values are all above 0.05.
- In the DGU era (1973-07 to 2004-11) and after publication (2004-12 to 2026-08) the
  smallest adjusted p-value is 0.120.
- Block-size sensitivity: for every b in {2, 4, 6, 8, 10} all adjusted p-values are
  above 0.05 (ff25 Ledoit-Wolf GMV: 0.057-0.079). With b = 1 (Boot-IID, which treats the
  months as independent and uses the iid standard error) two ff25 comparisons fall below
  0.05: inverse_vol (0.037) and Ledoit-Wolf GMV (0.047). Monthly portfolio returns are
  serially dependent, so b = 1 is not the appropriate choice, and the pre-registered test
  uses b = 9. The conclusion holds for the dependent bootstrap, but it is close to the
  threshold.

Calibrated block size (LW Algorithm 3.1) and the model confidence set were not run (both
were "should" items, `calibrate_block = false` in `configs/portfolio.toml`).

## 6. Limitations

- **Quick profile only.** Bootstrap coverage cells use 500 replications (SE about 1
  point) and DSR cells 1000. The full profile (`configs/mc_full.toml`: 10,000
  replications, B = 999, more T and SR levels) is projected at 15-25 CPU-hours and was
  not run:

  ```sh
  uv run python scripts/run_with_retry.py --log runs/mc-full/retry.log -- \
      uv run sharpecheck mc run --config configs/mc_full.toml --workers 3
  uv run sharpecheck mc report --config configs/mc_full.toml
  ```

- **Fixed bootstrap block size** (`round(T^(1/3))`) in the Monte Carlo. A larger or
  calibrated block might improve `boot_ts` under GARCH and regime switching. This was not
  tested.
- **HAC bandwidth.** Only the Andrews (1991) AR(1) plug-in rule was tested. Its failure
  under GARCH and regime switching may not carry over to other rules, such as Newey-West
  (1994) or a fixed-b asymptotic. That is a direction for follow-up work, not a result.
- **DGP choices drive the headline.** The GARCH (alpha + beta = 0.98) and regime
  (p11 = 0.98, p22 = 0.90) parameters are persistent but typical of daily equity
  returns. Less persistent processes would give milder results.
- **Truth is the population SR.** The small-sample upward bias of SR_hat counts against
  coverage, as intended. It reaches 0.060 at T = 60 (crash, SR = 0.3) and 0.018 at
  T = 250 (`sr_variance.csv`).
- **Portfolio study.** Results are tied to the data vintage, which the library revises.
  There are no transaction costs. Turnover drifts weights with excess rather than total
  returns. There is no numerical comparison with DGU's tables (different sample).
- **Not verified here.** CI has not run on GitHub; the repository has not been pushed.