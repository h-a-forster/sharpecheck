# sharpecheck design

Status: v0.1 plan, written at scaffold time (2026-10-23). The module stubs in
`src/sharpecheck/` follow this document. When code and this document disagree, fix one of
them in the same commit.

## 1. Goal and claims

sharpecheck is a typed, tested Python package for Sharpe-ratio (SR) inference, plus a
pre-registered Monte Carlo that measures whether each published interval and test keeps its
nominal coverage or size, and a small application to the "1/N versus optimised portfolios"
question with honest multiple-strategy inference.

**Headline deliverable: the coverage Monte Carlo** (section 6). A heatmap of two-sided 95%
coverage for 7 interval methods x 6 return processes x 3 SR levels x 5 sample sizes,
companion tables for one-sided size and interval width, a replication of the Ledoit-Wolf
(2008) size table for the difference tests, and a DSR experiment on correlated and
miscounted trials. The portfolio application (section 5) is secondary.

v0.1 claims, and how each is supported:

| Claim | Evidence |
|---|---|
| The functions compute the published statistics | published numeric examples, closed forms, `arch` and `scikit-learn` oracles (section 7) |
| The coverage/size of each method at T = 60...2520 under six processes is as reported | Monte Carlo with Wilson intervals and Holm-controlled flags (section 6) |
| Our difference tests behave like the published ones | replication of LW (2008) Table 1 within Monte Carlo error (section 6.3.2) |
| Portfolio findings | `REPORT.md`, generated from committed derived results, with sizes, seeds, data vintage, versions and Romano-Wolf adjustment (section 5) |

At scaffold time (2026-10-23), searches found no Python implementation of the Ledoit-Wolf
(2008) HAC or studentized-bootstrap difference test (section 12). Say "to our knowledge"
whenever this appears in user-facing text.

Out of scope for v0.1: probability of backtest overfitting (CSCV/PBO), Hansen SPA and the
Romano-Wolf StepM on mean returns (available in `arch`; our stepdown works on SR
differences), nonlinear shrinkage, RMT clipping, HERC/NCO, transaction costs, Bayesian SR
inference, the clustering-based trial counting of LZL (2025), a PyPI release.

## 2. Scope review (decisions taken at scaffold time)

Implementation budget: about 4-5 hours of agent time; each study run under 45 minutes of
wall time on 3 worker processes. What was cut or changed from the idea, and why:

1. **Coverage Monte Carlo first, portfolio second.** The original idea had both as
   headlines. The Monte Carlo is the part nobody has published as a cross-method,
   cross-process grid with a library behind it; 1/N versus optimised is crowded.
2. **Seven single-SR methods**: iid-normal, exact noncentral t, Mertens/Opdyke, AR(1)
   (LZL 2025), HAC (QS), HAC with prewhitening, studentized circular block bootstrap. PSR
   is not a separate interval: its one-sided test is the Mertens interval's lower bound
   (observed-variance convention) or the null-variance test; both are reported in the
   one-sided table.
3. **Six processes**: normal, Student t5, GARCH(1,1)-t7, Gaussian AR(1) with phi = 0.2,
   2-state Markov regime switching, and an iid crash mixture. Correlated trials appear in the
   DSR experiment, not in the coverage grid.
4. **Block size is fixed in the Monte Carlo** (`round(T^(1/3))`, plus b = 4 and 6 in the
   LW Table 1 replication). LW's calibration (Algorithm 3.1) needs K x |b| x M bootstrap
   resamples per data set (1000 x 6 x 499 = 3 million), far beyond the budget inside a
   Monte Carlo. It is implemented as a "should" item for the empirical application only.
5. **The multiple-strategy layer is the Romano-Wolf stepdown on studentized SR
   differences** from one joint block bootstrap, plus Holm and BH. `arch`'s SPA/StepM test
   mean losses, not Sharpe ratios, so they do not answer the SR question; `arch`'s MCS is
   used for out-of-sample variance only ("should").
6. **Portfolio set cut to five rules** (1/N, inverse volatility, sample GMV, Ledoit-Wolf
   2004 GMV, HRP) and **two datasets** (10 industries, 25 size/book-to-market portfolios).
   The 48-industry file has missing values (-99.99) after 1963-07 for some industries, which
   needs a policy that is not worth the budget.
7. **DSR experiment kept small**: N in {10, 100, 1000} trials, equicorrelation
   rho in {0, 0.3, 0.7}, T in {250, 1250}, normal and t5 returns, 6 decision rules.
8. **No R oracles.** Unlike the sibling project tailcheck, the reference values here come
   from published numeric examples and closed forms (section 7), and the kernel estimator
   is checked against `arch`. SharpeR (LGPL-3) could be added later through webR.
9. **PSR conventions.** The two published PSR formulas differ: Bailey and Lopez de Prado
   (2012) evaluate the SR variance at the observed SR with divisor T - 1; Lopez de Prado,
   Lipton and Zoonekynd (2025) evaluate it at SR0 with divisor T. Both are implemented
   (`variance_at="observed" | "null"`) and both are in the Monte Carlo; the default is the
   2012 form because DSR is defined with it.

## 3. Conventions

- Inputs are per-period **excess** returns (decimal, not percent). Ken French files are in
  percent and are divided by 100 on load.
- Every SR, standard error and interval is **per period**. `single.annualize` converts with
  Lo's (2002) factor; reports quote annualized values only next to the per-period ones.
- Moments: `mu = mean(r)`, `sigma^2 = mean((r - mu)^2)` (ddof 0), `SR = mu / sigma`,
  `g3 = m3 / m2^1.5`, `g4 = m4 / m2^2` (non-excess; 3 for normal), `rho1` = lag-1 sample
  autocorrelation with the full-sample mean and variance. The exact noncentral-t method
  works on the ddof-1 t statistic internally (section 4.2.2) but reports the ddof-0 SR as
  its estimate.
- Intervals: `SharpeInterval` with `side` in {"two-sided", "lower", "upper"}. A one-sided
  lower 95% bound is the lower end of a two-sided 90% interval for symmetric methods.
- Tests: `TestResult`. Statistically undefined input (e.g. zero variance, non-positive
  variance estimate) gives `nan` plus `details["reason"]`; invalid input raises
  `ValueError`.
- Bootstrap and simulation p-values use `(1 + count) / (B + 1)`. Every stochastic function
  takes `seed` (or a `numpy.random.Generator`).
- No LLM or model API calls. CPU only. At most 3 worker processes, BLAS threads set to 1 in
  each worker.
- Long jobs: each finished unit goes through `checkpoint.UnitStore` (atomic file per unit,
  skip on rerun, `progress.log`).

## 4. Module specifications

### 4.1 `moments` (M1)

`sample_moments(x) -> Moments(nobs, mean, sd, sr, g3, g4, rho1)` with the conventions of
section 3. Raises on fewer than 2 observations; `sr = nan` with zero variance.

### 4.2 `single` (M1, M3)

#### 4.2.1 Asymptotic variances

With `V(SR, g3, g4, rho)` the asymptotic variance of `sqrt(T) (SR_hat - SR)`:

- iid normal (Lo 2002; Jobson and Korkie 1981): `V = 1 + SR^2 / 2`.
- iid non-normal (Mertens 2002; Opdyke 2007): `V = 1 - g3 SR + (g4 - 1)/4 SR^2`.
  Because `g4 >= 1 + g3^2`, `V >= (1 - g3 SR / 2)^2 >= 0`.
- AR(1) (Lopez de Prado, Lipton and Zoonekynd 2025):
  `V = a - b g3 SR + c (g4 - 1)/4 SR^2` with `a = 1 + 2 rho/(1 - rho)`,
  `b = 1 + rho/(1 - rho) + rho^2/(1 - rho^2)`, `c = 1 + 2 rho^2/(1 - rho^2)`.
  For Gaussian AR(1) this equals the delta-method value `a + c SR^2 / 2`, since the long-run
  variance of the mean is `(1 + rho)/(1 - rho)` and that of the sample variance is
  `2 (1 + rho^2)/(1 - rho^2)` (checked in `tests/test_reference_values.py`). In
  `sharpe_ci(method="ar1")`, `rho = rho1` clipped to [-0.95, 0.95]; a non-positive `V`
  returns a `nan` interval.

Interval: `SR_hat -/+ z se` with `se = sqrt(V_hat / T)` evaluated at the estimates
(`z = z_{1 - alpha/2}` two-sided, `z_{1 - alpha}` one-sided).

#### 4.2.2 Exact interval under iid normal returns

`t = sqrt(T) mu_hat / s` with `s` the ddof-1 standard deviation follows the noncentral t
distribution with `T - 1` degrees of freedom and noncentrality `delta = sqrt(T) SR`. The
interval inverts `F(t; T - 1, delta)` in `delta` (monotone decreasing): the lower bound
solves `F = 1 - alpha/2`, the upper `F = alpha/2`; divide by `sqrt(T)`. Implementation:
vectorised bisection on `scipy.stats.nct.cdf` (60 iterations from `t -/+ 40`) so that a
whole Monte Carlo batch is solved at once; brentq for single calls. Reference: Johnson and
Welch (1940); SharpeR documents the same construction.

#### 4.2.3 PSR, MinTRL, critical SR, power

- `psr(..., variance_at="observed")` (Bailey and Lopez de Prado 2012):
  `Phi((SR_hat - SR0) sqrt(T - 1) / sqrt(1 - g3 SR_hat + (g4 - 1)/4 SR_hat^2))`.
- `psr(..., variance_at="null")` (LZL 2025): `Phi((SR_hat - SR0) / sqrt(V(SR0, g3, g4, rho)/T))`.
- MinTRL observed (BLdP 2012): `1 + (1 - g3 SR_hat + (g4 - 1)/4 SR_hat^2) (z_{1-alpha}/(SR_hat - SR0))^2`.
- MinTRL null (LZL 2025): `V(SR0, g3, g4, rho) (z_{1-alpha}/(SR_hat - SR0))^2`.
  Both return `inf` when `SR_hat <= SR0`.
- Critical SR (LZL 2025): `SR_c = SR0 + z_{1-alpha} sqrt(V(SR0)/T)`.
- Power (LZL 2025): `1 - Phi((SR_c - SR1) / sqrt(V(SR1)/T))`.
- `psr_test` returns `pvalue = 1 - PSR`, `alternative = "greater"`.

#### 4.2.4 Annualization (Lo 2002)

`SR(q) = eta(q) SR`, `eta(q) = q / sqrt(q + 2 sum_{k=1}^{q-1} (q - k) rho^k)`; `rho = 0`
gives `sqrt(q)`.

#### 4.2.5 HAC and bootstrap intervals (M3)

- `hac` / `hac_pw`: `v = (mu, gamma)` with `gamma = E r^2`; `f(mu, gamma) = mu / sqrt(gamma - mu^2)`;
  `grad f = (gamma / (gamma - mu^2)^1.5, -mu / (2 (gamma - mu^2)^1.5))`;
  `y_t = (r_t - mu_hat, r_t^2 - gamma_hat)`; `Psi_hat = long_run_cov(y, "qs", prewhiten=...,
  dof_adjust=2)`; `se = sqrt(grad' Psi_hat grad / T)`. This is Lo's (2002) non-iid GMM
  estimator and LW (2008) section 3.1 specialised to one ratio.
- `boot_ts`: section 4.4, single-series version. Symmetric two-sided interval
  `SR_hat -/+ q_{1-alpha}(|t*|) se_hat`; one-sided lower bound
  `SR_hat - q_{1-alpha}(t*) se_hat` (percentile-t), with `t* = (SR* - SR_hat)/se*` and
  `se_hat` from `hac_pw`.

### 4.3 `hac` (M2)

Kernels (Andrews 1991), `x = j / S_T`:

- Bartlett: `1 - |x|` for `|x| <= 1`, else 0.
- Parzen: `1 - 6x^2 + 6|x|^3` for `|x| <= 1/2`; `2 (1 - |x|)^3` for `1/2 < |x| <= 1`; else 0.
- QS: `25 / (12 pi^2 x^2) (sin(6 pi x / 5) / (6 pi x / 5) - cos(6 pi x / 5))`, `k(0) = 1`.
  This equals `3/z^2 (sin z / z - cos z)` with `z = 6 pi x / 5`, the form used by `arch`.

Estimator: `Psi_hat = T/(T - d) sum_{j=-(T-1)}^{T-1} k(j / S_T) Gamma_hat(j)`,
`Gamma_hat(j) = (1/T) sum_{t=j+1}^{T} y_t y_{t-j}'`, `Gamma_hat(-j) = Gamma_hat(j)'`,
`y` demeaned. Compute all autocovariances by FFT (O(T log T)) for the QS kernel.

Bandwidth (Andrews 1991, AR(1) plug-in, equal weights on standardised columns): for each
column `a`, divide by its sample sd, fit `y_{a,t} = rho_a y_{a,t-1} + e_t` by OLS,
`s_a^2 = var(e)` (so `s_a^2 = var(e) / var(y_a)` on the raw column, Andrews' weights
`w_a = 1 / var(y_a)^2`);
`alpha(1) = sum 4 rho^2 s^4 / ((1 - rho)^6 (1 + rho)^2) / sum s^4 / (1 - rho)^4`,
`alpha(2) = sum 4 rho^2 s^4 / (1 - rho)^8 / sum s^4 / (1 - rho)^4`;
`S_T = 1.1447 (alpha(1) T)^(1/3)` (Bartlett), `2.6614 (alpha(2) T)^(1/5)` (Parzen),
`1.3221 (alpha(2) T)^(1/5)` (QS). Clip `|rho_a| <= 0.97`. Standardising matters: with
raw columns, the SR moment vector `(r - mu, r^2 - gamma)` weights the `r^2` column by
`c^8` against `c^4` when returns are multiplied by `c`, so the bandwidth (and every HAC and
bootstrap interval) would depend on whether returns are in decimals or percent. The whole
estimator, prewhitening included, runs on standardised columns and is scaled back, so
`long_run_cov(y D) = D long_run_cov(y) D` for diagonal `D` (tested through
`sharpe_ci(c x)`; added in 0.1.0 before release, see section 13 item 14).

Prewhitening (Andrews and Monahan 1992): OLS VAR(1) `y_t = A y_{t-1} + e_t`; if a singular
value of `A` exceeds 0.97, set it to 0.97 (SVD) and recompute residuals; estimate
`Psi_e` on `e` (bandwidth from `e`); recolour `Psi = (I - A)^{-1} Psi_e (I - A)^{-T}`.

Note: `arch.covariance` uses the Newey-West (1994) nonparametric bandwidth, not Andrews'
AR(1) plug-in, so `arch` is an oracle only for fixed-bandwidth kernel sums.

### 4.4 `bootstrap` (M3)

Circular block bootstrap (Politis and Romano 1992): draw `ceil(T/b)` block starts uniformly
from `0..T-1`, take `b` consecutive indices modulo `T`, truncate to `T`. Default
`b = max(1, round(T^(1/3)))`.

Studentized statistic (LW 2008 section 3.2.2): for resample `m`, `v*` are the resample
moments and `y*_t` the resample moment deviations; with `l = floor(T/b)`,
`f_j = b^{-1/2} sum_{t=1}^{b} y*_{(j-1)b + t}`, `Psi* = (1/l) sum_j f_j f_j'`,
`se* = sqrt(grad f(v*)' Psi* grad f(v*) / T)`. For `b = 1` this is the sample covariance
of the resample (LW footnote 9; a unit test). `se_hat` on the original data is the
prewhitened QS HAC standard error for `b >= 2`, and the iid delta-method SE (sample
covariance of the moment vectors) for `b = 1`, following LW 2008 section 3.2.1.

Block-size calibration (LW Algorithm 3.1, "should", M7): fit VAR(1) to the pair; generate
`K` pseudo series from the VAR with stationary-bootstrap residuals (mean block 5); for each
candidate `b` compute the `1 - alpha` bootstrap interval for the pseudo-truth `Delta_hat`;
`g_hat(b)` = fraction covering; choose `b` minimising `|g_hat(b) - (1 - alpha)|`.

### 4.5 `compare` (M3)

Notation of LW (2008): `v = (mu1, mu2, gamma1, gamma2)`,
`f(a, b, c, d) = a / sqrt(c - a^2) - b / sqrt(d - b^2)`,
`grad f = (c/(c - a^2)^1.5, -d/(d - b^2)^1.5, -a/(2 (c - a^2)^1.5), b/(2 (d - b^2)^1.5))`.

- `jkm_test` (Jobson and Korkie 1981, Memmel 2003):
  `Var(Delta_hat) = (2 - 2 rho + (SR1^2 + SR2^2 - 2 SR1 SR2 rho^2) / 2) / T`, `rho` the
  sample correlation; `z = Delta_hat / sqrt(Var)`; two-sided normal p-value.
- `lw_hac_test`: `y_t = (r1 - mu1, r2 - mu2, r1^2 - gamma1, r2^2 - gamma2)`, QS kernel,
  Andrews bandwidth, prewhitening by default, `dof_adjust = 4`;
  `se = sqrt(grad' Psi grad / T)`; `p = 2 Phi(-|Delta_hat| / se)`.
- `lw_boot_test`: `d = |Delta_hat| / se_hat`, `d*_m = |Delta*_m - Delta_hat| / se*_m`,
  `p = (1 + #{d*_m >= d}) / (M + 1)` (LW eq. 9); interval `Delta_hat -/+ q_{1-alpha}(d*) se_hat`.
  `block = 1` is LW's Boot-IID: as in LW section 3.2.1, `se_hat` is then the iid SE
  from the sample covariance of `(r1, r2, r1^2, r2^2)`, not the HAC SE.
- Many strategies (`bootstrap.studentized_diff_boot`): column 0 is the benchmark; all
  pairs `(0, j)` use the same resampled time indices; `t_j = Delta_hat_j / se_hat_j` and
  `t*_{mj} = (Delta*_{mj} - Delta_hat_j) / se*_{mj}` feed `multiple.romano_wolf`.

### 4.6 `deflated` (M4)

- Gumbel approximation (BLdP 2014, eq. 1):
  `E[max] = sqrt(V) ((1 - gamma) Phi^{-1}(1 - 1/N) + gamma Phi^{-1}(1 - 1/(N e)))`,
  `gamma = 0.5772156649`.
- Exact: `E[M_N] = int z N phi(z) Phi(z)^{N-1} dz`, `E[M_N^2]` likewise, by Gauss-Hermite
  (200 nodes) or adaptive quadrature. Equicorrelated trial SRs with correlation `rho`
  (`SR_i = sqrt(V) (sqrt(rho) X + sqrt(1 - rho) Y_i)`):
  `E[max] = sqrt(V) sqrt(1 - rho) E[M_N]`, `Var[max] = V (rho + (1 - rho) Var[M_N])`.
- `DSR = psr(SR_hat_best, SR0 = E[max], T, g3, g4, variance_at="observed")` (BLdP 2014 eq. 2).
- Implied independent trials (BLdP 2014 eq. 9): `N_hat = rho_bar + (1 - rho_bar) M`,
  `rho_bar` the average off-diagonal sample correlation of the `M` trials.
- pFDR (LZL 2025): `1 / (1 + (1 - beta) p_H1 / (alpha (1 - p_H1)))`.
- oFDR (LZL 2025): `p0 (1 - p_H1) / (p0 (1 - p_H1) + p1 p_H1)` with
  `p_k = 1 - psr(SR_hat, SR_k, T, ..., variance_at="null")`.
- `dsr_test`: `var_sr` defaults to the ddof-1 cross-sectional variance of the trial SRs;
  `n_trials` to the number of columns.

### 4.7 `multiple` (M4; MCS in M7)

- Holm: sorted `p_(1) <= ... <= p_(k)`; `adj_(i) = max_{j <= i} min(1, (k - j + 1) p_(j))`.
- Benjamini-Hochberg: `adj_(i) = min_{j >= i} min(1, k p_(j) / j)`.
- Romano-Wolf stepdown (Romano and Wolf 2005, adjusted p-values by Romano and Wolf 2016,
  two-sided): order hypotheses by `|t_obs|` descending, `r_1, ..., r_k`;
  `p_raw(r_s) = (1 + #{m : max_{l >= s} |t*_{m, r_l}| >= |t_obs(r_s)|}) / (B + 1)`;
  enforce monotonicity `p_adj(r_s) = max(p_raw(r_s), p_adj(r_{s-1}))`.
- `model_confidence_set`: `arch.bootstrap.MCS(losses, size, reps, block_size,
  method="max", bootstrap="stationary")`, returning included models and p-values.

### 4.8 `checkpoint` (M1)

Same design as the sibling project tailcheck: `UnitStore(root, manifest)` writes each unit
as `units/<id>.json` via write-to-temp then `os.replace`; `manifest.json` holds the config
hash, input-data SHA-256s and package version, and a mismatch refuses to resume unless
`--force-new`; `progress.log` gets one timestamped line per finished unit (no absolute
paths). `run_units` uses a `ProcessPoolExecutor` with at most 3 workers, sets
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` to 1 and uses `threadpoolctl`
in workers, and writes each result as soon as it arrives.

## 5. Portfolio application (M7)

### 5.1 Question

Does any of four estimated allocation rules have a Sharpe ratio different from 1/N out of
sample, once the tests are robust to non-normality and serial dependence and adjusted for
the number of comparisons? DeMiguel, Garlappi and Uppal (2009; DGU) compared Sharpe ratios
with the Jobson-Korkie-Memmel test, which LW (2008) show is liberal under fat tails and
dependence.

### 5.2 Data

Kenneth R. French Data Library, monthly, value-weighted, downloaded as CSV zips
(`https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/<file>`):

| Key | File | Use |
|---|---|---|
| `ind10` | `10_Industry_Portfolios_CSV.zip` | 10 industry portfolios |
| `ff25` | `25_Portfolios_5x5_CSV.zip` | 25 size x book-to-market portfolios |
| `ff3` | `F-F_Research_Data_Factors_CSV.zip` | `RF` (one-month T-bill) |

Checked on 2026-10-23: all three return HTTP 200; vintage line "This file was created using
the 202608 CRSP database."; monthly data from 1926-07 to 2026-08; no missing values in the
value-weighted monthly sections of `ind10` and `ff25` after 1963-07. SHA-256 on that day:
`ind10` `ebd937a4...a49ae5`, `ff25` `e02edcd2...6b8429`, `ff3` `593f4fbe...040a66` (full
hashes go in `data/manifest.json` at fetch time; the library revises past data, so the
manifest, not this table, defines a run).

Parsing: files are CRLF text inside a zip; find the CSV member by listing the archive (the
member name's case differs from the zip name, e.g. `25_Portfolios_5x5.csv`). The monthly
value-weighted block starts after the line containing "Average Value Weighted Returns --
Monthly" (factors file: the first header line `,Mkt-RF,SMB,HML,RF`) and ends at the first
blank line; rows are `YYYYMM, v1, ..., vN`. Values are percent; -99.99 and -999 mean
missing. Validate: header width, six-digit consecutive months, finite numbers. Excess
return = portfolio return - RF, both divided by 100.

### 5.3 Rules (`portfolio.weights`)

Window `M = 120` months (DGU), monthly rebalancing, sample from 1963-07 (DGU's start).
Out-of-sample months: 1973-07 to the last month of the vintage (638 months for 202608).

- `equal`: `w = 1/N`.
- `inverse_vol`: `w_i ∝ 1 / sigma_i`.
- `gmv_sample`: `w = S^{-1} 1 / (1' S^{-1} 1)`, `S` the ddof-1 sample covariance;
  short sales allowed (DGU's "min").
- `gmv_lw`: as `gmv_sample` with the Ledoit-Wolf (2004) estimator: demeaned `X` (M x N),
  `S = X'X/M`, `m = tr(S)/N`, `d^2 = ||S - m I||_F^2 / N`,
  `b_bar^2 = (1/M^2) sum_t ||x_t x_t' - S||_F^2 / N`, `b^2 = min(b_bar^2, d^2)`,
  `Sigma = (b^2/d^2) m I + (1 - b^2/d^2) S`. Oracle: `sklearn.covariance.ledoit_wolf`.
- `hrp` (Lopez de Prado 2016): correlation distance `d_ij = sqrt((1 - rho_ij)/2)`;
  following the paper's code, cluster with single linkage on the Euclidean distance between
  columns of `d` (`d~_ij = ||d_.i - d_.j||`; PyPortfolioOpt clusters on `d` directly, so
  results differ from it); quasi-diagonal order from the dendrogram; recursive bisection of
  the ordered list, splitting weight by `alpha = 1 - V1/(V1 + V2)` where `V` is the
  variance of the inverse-variance-weighted sub-portfolio.

Portfolio excess return in month `t`: `w_t' r_t` with weights from months `t-120..t-1`.
Turnover: `sum_i |w_{i,t+1} - w_{i,t}^+|`, `w^+` the drifted weights at the end of `t`.

### 5.4 Inference (pre-registered)

- Families: 4 rules x 2 datasets = 8 comparisons against 1/N, in each of three periods:
  full out-of-sample (1973-07..end; **primary**), DGU era (1973-07..2004-11) and post
  publication (2004-12..end).
- Primary test: LW studentized circular block bootstrap, two-sided, `M = 4999`,
  `b = round(T^(1/3))`, Romano-Wolf adjusted within the family from one joint bootstrap of
  all 10 out-of-sample return series (they share months). Primary claim: the set of rules
  with adjusted p < 0.05 in the full period.
- Secondary: JKM (what DGU used), LW HAC (prewhitened QS), Holm; block-size sensitivity
  `b in {1, 2, 4, 6, 8, 10}`; calibrated block size (Algorithm 3.1, "should"); MCS on
  out-of-sample squared excess returns as a variance comparison ("should").
- Reported per rule: mean, sd, monthly SR with a `boot_ts` 95% interval, annualized SR
  (x sqrt(12), labelled), turnover.
- No numerical comparison with DGU's tables: the library revises history and DGU's sample
  ended in 2004. Qualitative comparison only, and only after reading the published paper
  (doi:10.1093/rfs/hhm075).

## 6. Monte Carlo (M5, M6) — the headline

### 6.1 Experiments

| Id | Question | Unit |
|---|---|---|
| `coverage` | coverage of single-SR intervals; size of one-sided tests | (dgp, sr, T, chunk) |
| `difference` | size of SR-difference tests; LW (2008) Table 1 replication | (dgp, chunk) |
| `dsr` | size (FWER) of selection-adjusted tests under correlated / miscounted trials | (N, rho, T, dist, chunk) |

Seeds: `SeedSequence(config.seed)` spawned per unit with a spawn key derived from a stable
hash (`hashlib.sha256`) of the unit id, so a unit's numbers do not depend on worker count or
order. Within a unit all methods see the same simulated data.

### 6.2 Processes (per-period scale; closed-form truth)

Each process is shifted so that its stationary mean over stationary sd equals the target SR
exactly. Variance does not depend on the shift for any process below, so the calibration is
closed form.

| DGP | Definition | g3 | g4 | rho1 |
|---|---|---|---|---|
| `normal` | iid N(mu, 1) | 0 | 3 | 0 |
| `t5` | iid standardized Student t, nu = 5 | 0 | 9 | 0 |
| `garch-t7` | GARCH(1,1), omega = 0.02, alpha = 0.05, beta = 0.93, standardized t7 innovations, unconditional variance 1, burn-in 1000 | 0 | 6.689 | 0 |
| `ar1-0.2` | Gaussian AR(1), phi = 0.2, innovation variance 1 - phi^2, started in the stationary law | 0 | 3 | 0.2 |
| `regime` | 2-state Markov switching, p11 = 0.98, p22 = 0.90, sd (0.8, 2.0), mean gap mu2 - mu1 = -0.5, rescaled to unit variance, started in the stationary law | closed form | closed form | closed form (small) |
| `crash` | iid mixture: with prob 0.99 N(0, 1), with prob 0.01 N(-8, 2^2), rescaled to unit variance | closed form (strongly negative) | closed form | 0 |

The `garch-t7` fourth moment exists (`beta^2 + 2 alpha beta + kappa alpha^2 = 0.970 < 1`,
`kappa = 5`); `t5` has a finite fourth but infinite eighth moment, so kurtosis estimates are
very noisy (a deliberate stress for `mertens`). Regime moments: stationary
`pi1 = (1 - p22)/(2 - p11 - p22)`; mean and variance of the mixture; lag-1 autocovariance
`(p11 + p22 - 1) pi1 pi2 (mu1 - mu2)^2`; skewness and kurtosis from the mixture moments.
All closed forms are tested against 10^6-observation simulations (M5).

SR levels: 0, 0.1, 0.3 per period. At SR = 0 the iid formulas reduce to `V = 1` whatever
the tails, so SR = 0 cells isolate serial dependence; SR > 0 cells exercise the skewness and
kurtosis terms. T in {60, 120, 250, 1000, 2520} (5 years monthly to 10 years daily).

### 6.3 Methods and metrics

#### 6.3.1 `coverage`

Methods: the seven of section 4.2 at level 0.95. Per cell: two-sided coverage, miss-low and
miss-high rates, one-sided 95% lower-bound violation rate (= size of the one-sided test of
`H0: SR <= SR_true`), the same for `psr_test` with both variance conventions, median width
relative to `iid_normal`, and the count of `nan` intervals (counted as non-covering and
reported).

#### 6.3.2 `difference` (LW 2008 Table 1 replication)

DGPs exactly as LW section 4.2, T = 120, 5000 replications, `M = 499`, levels 1%, 5%, 10%:
Normal-IID (bivariate normal, means 1, variances 1, correlation 0.5), t6-IID (bivariate t6
scaled to unit variance, correlation 0.5, means 1), Normal-GARCH and t6-GARCH (diagonal-vech
bivariate GARCH(1,1) with C = [[0.15, 0.13], [0.13, 0.15]], A = [[0.075, 0.05], [0.05,
0.075]], B = [[0.90, 0.89], [0.89, 0.90]], mean 16.5/52 each, innovations bivariate normal
or bivariate t6 with one shared chi-square, standardized to unit variances and zero
covariance), Normal-VAR and t6-VAR (the IID cases plus AR(1)
coefficient 0.2 in each series). Methods: JKM, HAC (QS, no prewhitening), HACPW, Boot-IID
(`b = 1`), Boot-TS with fixed `b = 4` and `b = 6` (LW calibrate `b`; documented deviation).
Agreement criterion per entry: `|ours - LW| <= 3 sqrt(2 p (1 - p) / 5000)` with `p` the
pooled rate (about 1.3 points at 5%). Ambiguities to resolve and document in M5: whether
"mean 1, variance 1" in the VAR cases refers to the stationary or the innovation law (we
use the stationary law), and the extracted B matrix (read as symmetric). A mismatch is
reported as a mismatch, never as an error in LW, unless a minimal repro shows otherwise.

#### 6.3.3 `dsr`

Trials: `N` zero-SR series with equicorrelation `rho` (one-factor construction), iid normal
or standardized t5, T in {250, 1250}; the "best" trial is the one with the largest SR. Under
the global null every rejection is a false discovery, so the rejection rate is the FWER.
Decision rules at 5%:

1. `dsr_trueN_estV`: DSR with the true N and the cross-sectional variance of trial SRs
   (practice).
2. `dsr_trueN_oracleV`: DSR with the true N and `V = 1/T` (the iid null variance).
3. `dsr_N/10`: DSR with N reported as `max(1, N // 10)` (undercounted trials), estimated V.
4. `dsr_impliedN`: DSR with BLdP eq. 9 from the average sample correlation, estimated V.
5. `psr_best`: PSR(0) of the best trial without deflation (N reported as 1).
6. `holm_psr`: Holm over the N individual PSR(0) p-values (FWER control by construction).
7. `lzl_max`: one-sided normal test of the best SR against the exact mean and sd of the
   maximum (`E[max]` and `Var[max]` of section 4.6 with the true `rho`, `V = 1/T`).

### 6.4 Sizes and budget

Quick profile (`configs/mc_quick.toml`), measured on a pilot before the run (M5):

| Experiment | Cells | Replications | Bootstrap | Estimate |
|---|---|---|---|---|
| coverage | 6 dgp x 3 sr x 5 T = 90 | 2000 (analytic, HAC); boot_ts on the first 1000 | B = 499 | about 40 CPU-min |
| difference | 6 | 5000 | M = 499, 3 bootstrap variants | about 10 CPU-min |
| dsr | 3 N x 3 rho x 2 T x 2 dist = 36 | 2000 | none | about 5 CPU-min |

Scaffold microbenchmark (numpy, one thread): one studentized block bootstrap with B = 499
takes about 2, 7, 23 and 62 ms at T = 60, 250, 1000, 2520. Total about 55 CPU-minutes, so
about 20 minutes on 3 workers if the machine is idle. M5 starts with a timing pilot (50
replications per cell) that writes `results/mc-quick/calibration.json`; if the projection
exceeds 35 wall-minutes, lower the bootstrap replications first, never below 500 for
`boot_ts` coverage cells or 1000 for analytic cells, and record what was used.

Full profile (`configs/mc_full.toml`): 10,000 replications for every method, B = 999, T
adds 500, SR adds 0.5, `difference` adds `b in {1, 2, 4, 6, 8, 10}`, `dsr` uses 5000
replications and adds N = 10,000. Projected 15-25 CPU-hours (a projection, not a
measurement); resumable from `runs/mc-full/units/`.

### 6.5 Report and headline

`sharpecheck mc report` writes `results/mc-quick/`: `coverage.csv`, `onesided.csv`,
`difference.csv` (with the LW Table 1 column and the agreement flag), `dsr.csv`,
`summary.json` (sizes, seeds, versions, wall time), and `coverage_heatmap.png` (rows:
method; columns: dgp x sr; facets: T; colour: coverage, diverging at 0.95). Every rate
carries a 95% Wilson interval. A cell is **flagged as under-covering** when a one-sided
binomial test of coverage < 0.95 is significant after Holm across all coverage cells and
methods (about 630 tests) **and** coverage < 0.93 (material). Over-coverage is flagged
the same way at > 0.97.

### 6.6 Pre-registered expectations (written before any run)

Expectations from theory, to be confirmed or refuted by the quick run and reported either
way:

1. `iid_normal` under-covers for `crash` at SR > 0 and for `ar1-0.2` at every SR and T
   (the AR(1) long-run variance factor is 1.5, so nominal 95% becomes about 89%).
2. `mertens` fixes `t5` and `crash` at large T but not `ar1-0.2`; it is erratic for `t5` at
   small T (noisy kurtosis).
3. `ar1` fixes `ar1-0.2` but not `garch-t7` at SR > 0.
4. `hac` and `hac_pw` are liberal (coverage below 95%) at T <= 120 and close to nominal at
   T >= 1000; prewhitening helps under `ar1-0.2`.
5. `boot_ts` is closest to nominal overall but is not exact at T = 60.
6. `exact_nct` is exact for `normal` and inherits `iid_normal`'s failures elsewhere.
7. In `dsr`, rules 1-2 are conservative (FWER well below 5%) under independent trials,
   rule 5 is badly oversized, rule 3 stays below 5% for moderate undercounting because DSR
   uses the single-trial standard error rather than the spread of the maximum, rule 6
   controls FWER, rule 7 is close to 5%.
8. In `difference`, our rates agree with LW Table 1 within the criterion of 6.3.2 for JKM,
   HAC, HACPW and Boot-IID.

## 7. Validation and test strategy

### 7.1 Reference values

All values below are recomputed independently by `scripts/reference_values.py` (numpy and
scipy only; it does not import sharpecheck). "Paper" values are published; the script's
helpers are themselves pinned by `tests/test_reference_values.py`.

| Quantity | Inputs | Value | Source |
|---|---|---|---|
| sd(SR_hat), non-normal | SR = 0.036/0.079, g3 = -2.448, g4 = 10.164, T = 24 | 0.329 (0.328589) | LZL 2025 numeric example |
| sd(SR_hat), normal | same SR, T = 24 | 0.214 (0.214460) | LZL 2025 |
| PSR, null variance | SR0 = 0 / 0.1 | 0.987 / 0.939 | LZL 2025 |
| MinTRL, null variance | SR0 = 0, alpha = 0.05 | 13.029 | LZL 2025 |
| beta (type II) | SR0 = 0, SR1 = 0.5, T = 24 | 0.315 | LZL 2025 |
| pFDR | p_H1 = 0.05, alpha = 0.05, beta = 0.315 | 0.581 | LZL 2025 |
| oFDR | SR0 = 0, SR1 = 0.5, p_H1 = 0.05, T = 24 | 0.306 | LZL 2025 |
| PSR(0), observed variance, T - 1 | same example | 0.912709 (non-normal), 0.981243 (normal) | derived (BLdP 2012 formula) |
| MinTRL, observed variance | mu = 2, sd = sqrt(12), g3 = -0.72, g4 = 5.78, SR* = 1/sqrt(12) | 59.895 | BLdP 2012 appendix A.3 |
| DSR | SR = 2.5/sqrt(250), T = 1250, V = 0.5/250, g3 = -3, g4 = 10, N = 100 / 46 | SR0 = 0.1132, DSR = 0.9004 / 0.9505 | BLdP 2014 numeric example |
| DSR, normal | g3 = 0, g4 = 3, N = 88 | 0.9505 | BLdP 2014 |
| E[max of N iid N(0,1)] | N = 10 / 100 / 1000 | 1.538753 / 2.507594 / 3.241436 (Gumbel: 1.574598 / 2.530603 / 3.255122) | derived |
| Equicorrelated max | N = 100, rho = 0.5 | E = 1.773136, Var = 0.592202 | derived |
| T Var(SR_hat), Gaussian AR(1) | rho = 0.2, SR = 0.1 | 1.505417 | derived, two ways |
| Exact nct 95% CI | T = 60, SR_hat = 0.2 / T = 250, SR_hat = 0.05 | (-0.056409, 0.454750) / (-0.074087, 0.173987) | derived |
| JKM-Memmel variance | SR = (0.2, 0.1), rho = 0.5, T = 120 | 0.00850000 | derived |
| Lo eta(12) | rho = 0.1 / 0.2 / 0.5 | 3.160111 / 2.878849 / 2.121288 | derived |
| GARCH-t kurtosis | alpha = 0.05, beta = 0.93, nu = 7 | 6.689189 | derived |
| LW 2008 Table 1 | six DGPs, five methods, three levels | see `reference_values.py` | LW 2008 |

Note on the PSR example: the 2012 and 2025 conventions give 0.913 and 0.987 for the same
track record (T = 24). The difference is entirely the variance convention; both are
implemented and the docs must say which one a number uses.

### 7.2 Oracles and property tests

- `hac.long_run_cov` with a fixed bandwidth equals `arch.covariance.QuadraticSpectral(x,
  bandwidth=bw, df_adjust=0).cov.long_run` (and Bartlett/Parzen after checking `arch`'s
  bandwidth convention for finite-support kernels) to 1e-10.
- `portfolio.weights.ledoit_wolf_cov` equals `sklearn.covariance.ledoit_wolf` (shrinkage and
  matrix) to 1e-12.
- Bootstrap: `b = 1` block covariance equals the sample covariance; index matrix values in
  range, blocks contiguous modulo T; fixed seed gives identical output.
- HRP: weights positive and sum to one; two assets give inverse-variance weights; a
  block-diagonal correlation matrix gives the hand-computed allocation.
- No look-ahead: perturbing returns from month `t` onward leaves weights for months `<= t`
  unchanged.
- Romano-Wolf: with one hypothesis it equals the single bootstrap p-value; adjusted p-values
  are monotone in `|t_obs|` and never below the raw bootstrap p-values.
- Statistical tests (`@pytest.mark.slow`): `exact_nct` coverage at T = 60 under `normal`
  within 95 +/- 1 points (2000 replications); DGP closed-form moments within 4 Monte Carlo
  standard errors of 10^6-observation simulations; `difference` Normal-IID rates near 5%.
- Tests use no network; the Ken French parser is tested on a committed synthetic file in
  the same layout (a few rows), never on real data.

### 7.3 Third-party checks

Any statement that another implementation is wrong needs a minimal repro, a check that the
discrepancy is not an adapter mistake (conventions: ddof, excess vs raw kurtosis, T vs
T - 1, per-period vs annual), and a sentence on its effect. At scaffold time, one lead only:
the unlicensed replication code of LZL (2025) (`functions.py`, function `moments_Mk`)
scales the mean of the equicorrelated maximum by `(1 - rho)` where the one-factor
construction gives `sqrt(1 - rho)`; within that file the function is only called with
`rho = 0`, so it may be unused. Do not report it without checking the notebooks and the
paper's equations first. Because that repository has no licence, we cite its numbers and
do not copy its code.

## 8. Architecture

| Path | Role | Milestone |
|---|---|---|
| `src/sharpecheck/results.py` | `SharpeInterval`, `TestResult` | done |
| `src/sharpecheck/_validation.py` | input checks | done |
| `src/sharpecheck/checkpoint.py` | unit store, progress log, capped pool | M1 |
| `src/sharpecheck/moments.py` | sample moments | M1 |
| `src/sharpecheck/single.py` | single-SR intervals, PSR, MinTRL, power | M1, M3 |
| `src/sharpecheck/hac.py` | kernels, Andrews bandwidth, prewhitening | M2 |
| `src/sharpecheck/bootstrap.py` | circular block bootstrap, studentized stats, calibration | M3, M7 |
| `src/sharpecheck/compare.py` | JKM, LW HAC, LW bootstrap | M3 |
| `src/sharpecheck/deflated.py` | expected max, DSR, implied N, pFDR/oFDR | M4 |
| `src/sharpecheck/multiple.py` | Holm, BH, Romano-Wolf, MCS wrapper | M4, M7 |
| `src/sharpecheck/montecarlo/` | DGPs, runner, report | M5, M6 |
| `src/sharpecheck/data.py` | Ken French fetch, cache, parse | M7 |
| `src/sharpecheck/portfolio/` | rules, rolling backtest | M7 |
| `src/sharpecheck/cli.py` | `sr`, `compare`, `mc`, `data`, `portfolio` | M5-M8 |
| `configs/` | quick and full Monte Carlo configs, portfolio config | - |
| `scripts/reference_values.py` | independent reference values | done |
| `scripts/run_with_retry.py` | resume-on-failure wrapper for long runs | M6 |

## 9. Milestones

Each milestone ends with `uv run pytest`, `uv run ruff check .`, `uv run ruff format
--check .` passing, a commit, and an update of the AGENTS.md status section.

| Id | Content | Done when | Budget |
|---|---|---|---|
| M1 | `checkpoint`, `moments`, analytic `single` methods (iid_normal, mertens, ar1, exact_nct), PSR both conventions, MinTRL, critical SR, power, annualize | all section 7.1 single-SR and PSR values reproduced by package tests; nct CIs to 1e-6; checkpoint tests (atomic write, skip done, manifest mismatch refused, workers clipped to 3); slow `exact_nct` coverage test passes | 40 min |
| M2 | `hac` | kernel weights; fixed-bandwidth equality with `arch` to 1e-10; Andrews bandwidth equals hand formula on an AR(1) column; prewhitened estimate within 3% of the true long-run covariance for a VAR(1) at T = 200,000 | 40 min |
| M3 | `bootstrap`, `compare`, `single` hac/hac_pw/boot_ts | b = 1 identity; JKM variance reference value; LW eq. (9) p-value; deterministic with seed; HAC SE for Gaussian AR(1) within 3% of `sqrt(1.505417/T)` at T = 100,000 | 50 min |
| M4 | `deflated`, `multiple` | DSR, E[max], pFDR/oFDR reference values; equicorrelated max moments vs 10^5-draw simulation within 4 SE; Holm/BH hand examples; Romano-Wolf properties | 30 min |
| M5 | `montecarlo.dgps`, runner, timing pilot, `mc run` CLI | closed-form moments vs simulation (slow test); identical unit results with 1 and 3 workers; resume skips finished units; `results/mc-quick/calibration.json` written and config reps set within floors | 45 min |
| M6 | quick Monte Carlo run (retry wrapper, progress log, stall watch), `mc report`, REPORT sections 1-3 | all units finished; `results/mc-quick/` files present; LW Table 1 comparison table; heatmap; pre-registered expectations each marked confirmed / refuted / mixed with numbers | 45 min incl. a run of at most 35 min |
| M7 | `data`, `portfolio`, inference, `portfolio run/report` | parser tests on synthetic fixture; `data/manifest.json` with SHA-256 and vintage; LW shrinkage equals sklearn; HRP and no-look-ahead tests; `results/portfolio/` tables; REPORT section 4 | 45 min |
| M8 | CLI `sr` and `compare`, README, REPORT summary, CHANGELOG, CITATION, final audit | CLI smoke tests; every REPORT number traced to a committed results file; no local paths in committed files; AGENTS.md status current | 30 min |

"Should" items, dropped first under time pressure: LW block calibration, MCS, the
block-size sensitivity table, the `upper` side of intervals.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Monte Carlo over budget on a shared machine | timing pilot, floors on replications, chunked units, resume; full profile only documented |
| "All methods mostly work" is a mild result | SR > 0 stress cells (`crash`, `regime`, `t5` at small T), one-sided size, and the DSR experiment carry the headline; report it plainly if the result is mild |
| HAC prewhitening unstable (near-unit-root VAR, singular I - A) | Andrews-Monahan 0.97 singular-value cap; `nan` with reason instead of a crash |
| `t5` kurtosis estimates make `mertens` erratic | that is a finding, reported with the `nan` count and width |
| Small-T bias of SR_hat (order 1/T) lowers coverage at T = 60 | part of what is measured; truth is the population SR, not a simulated mean |
| LW Table 1 mismatch from DGP ambiguity | ambiguities listed in 6.3.2; investigate before reporting; never claim an LW error without a repro |
| Ken French data revised over time | manifest with SHA-256 and vintage; results name the vintage |
| Unlicensed LZL code | cite values only; independent implementation |
| nct numerics at large noncentrality | bisection on a bracket, fallback to the normal approximation with a `details` flag, tested at T = 2520 |

## 11. References

- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent covariance
  matrix estimation. *Econometrica* 59(3), 817-858.
- Andrews, D. W. K. and Monahan, J. C. (1992). An improved heteroskedasticity and
  autocorrelation consistent covariance matrix estimator. *Econometrica* 60(4), 953-966.
- Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe ratio efficient frontier.
  *Journal of Risk* 15(2), 3-44.
- Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio: correcting for
  selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management*
  40(5), 94-107.
- Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS B*
  57(1), 289-300.
- DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal versus naive diversification:
  how inefficient is the 1/N portfolio strategy? *Review of Financial Studies* 22(5),
  1915-1953. doi:10.1093/rfs/hhm075.
- Goetze, F. and Kuensch, H. R. (1996). Second-order correctness of the blockwise bootstrap
  for stationary observations. *Annals of Statistics* 24(5), 1914-1933.
- Hansen, P. R., Lunde, A. and Nason, J. M. (2011). The model confidence set.
  *Econometrica* 79(2), 453-497.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian
  Journal of Statistics* 6(2), 65-70.
- Jobson, J. D. and Korkie, B. M. (1981). Performance hypothesis testing with the Sharpe and
  Treynor measures. *Journal of Finance* 36(4), 889-908.
- Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional
  covariance matrices. *Journal of Multivariate Analysis* 88(2), 365-411.
- Ledoit, O. and Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe
  ratio. *Journal of Empirical Finance* 15(5), 850-859. doi:10.1016/j.jempfin.2008.03.002.
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal* 58(4),
  36-52.
- Lopez de Prado, M. (2016). Building diversified portfolios that outperform out of sample.
  *Journal of Portfolio Management* 42(4), 59-69.
- Lopez de Prado, M., Lipton, A. and Zoonekynd, V. (2025). How to use the Sharpe ratio.
  SSRN 5520741; published as "Sharpe ratio inference: a new standard for decision making
  and reporting", *Journal of Portfolio Management* 52(6), 2026. Replication code:
  github.com/zoonek/2025-sharpe-ratio (no licence file as of 2026-10-23).
- Memmel, C. (2003). Performance hypothesis testing with the Sharpe ratio. *Finance
  Letters* 1, 21-23.
- Mertens, E. (2002). Comments on variance of the IID estimator in Lo (2002). Working
  paper, University of Basel.
- O'Connor, M. C. (2024). Revisiting hypothesis testing with the Sharpe ratio. *Econ
  Journal Watch* 21(1), 113-155. (Low power of SR-difference tests; motivates reporting
  intervals and power, not only p-values.)
- Opdyke, J. D. (2007). Comparing Sharpe ratios: so where are the p-values? *Journal of
  Asset Management* 8(5), 308-336.
- Politis, D. N. and Romano, J. P. (1992). A circular block-resampling procedure for
  stationary data. In *Exploring the Limits of Bootstrap*, Wiley, 263-270.
- Romano, J. P. and Wolf, M. (2005). Stepwise multiple testing as formalized data snooping.
  *Econometrica* 73(4), 1237-1282.
- Romano, J. P. and Wolf, M. (2016). Efficient computation of adjusted p-values for
  resampling-based stepdown multiple testing. *Statistics & Probability Letters* 113, 38-40.

## 12. Prior art (checked 2026-10-23)

| Project | Version, licence | What it covers | Gap |
|---|---|---|---|
| zoonek/2025-sharpe-ratio (GitHub) | notebooks + `functions.py`, no licence | LZL 2025 formulas: SR variance with AR(1), PSR, MinTRL, power, pFDR/oFDR, FDR control, clustering | paper replication, not a package; no HAC/bootstrap; unusable code licence |
| SharpeR (CRAN) | 1.4.0, LGPL-3 | SR distribution, noncentral-t intervals, equality tests (delta method) | R only |
| PeerPerformance (CRAN) | 2.4.1, GPL (>= 2) | HAC and studentized circular bootstrap tests for (modified) SR differences | R only, GPL |
| PerformanceAnalytics (CRAN) | 2.1.0, GPL-2/3 | SR and many performance measures | R only, no robust inference |
| sharpebench (PyPI) | 0.28.0, MIT or Apache-2.0 | SR, PSR, DSR, MinTRL, stationary-bootstrap p-value and DSR interval (Rust core) | no difference tests, no HAC, no coverage validation |
| deflated-sharpe (PyPI) | 0.1.0, Apache-2.0 | DSR, minimum backtest length, regime-decay heuristics | DSR only |
| arch (PyPI) | 8.0.0, NCSA | HAC kernels, bootstraps, SPA, StepM, MCS on losses | no SR-specific inference (used here as a dependency) |
| quantstats, empyrical-reloaded, pyfolio-reloaded (PyPI) | 0.0.86 / 0.7.0 / 0.10.0, Apache-2.0 | point performance statistics (not reviewed in detail) | no robust SR inference |
| PyPortfolioOpt, riskfolio-lib, skfolio (PyPI) | 1.6.0 MIT / 7.4.0 BSD-3 / 1.10.2 BSD-3 | allocation rules incl. HRP, shrinkage | no SR inference |
| pypbo (GitHub esvhd/pypbo) | not on PyPI | PBO, PSR, DSR | unmaintained, untested |

No Python implementation of the LW (2008) HAC or bootstrap difference test was found
(web and PyPI search, 2026-10-23). The paper points to the authors' own code on M. Wolf's
university page (R, according to a 2008 r-sig-finance post); it was not used here.

## 13. Implementation notes and deviations

Recorded during M1-M8 (2026-10-24). Where an item changes a formula or a reported
quantity, the code docstring says the same.

1. **Checkpoint API.** `UnitStore(root, config_hash, inputs=..., force_new=...)` follows the
   sibling project's tested module instead of the stub's `(root, manifest)` signature. The
   manifest records the config hash, the input-data SHA-256s (a mismatch refuses to
   resume), the source hash (a change only warns) and versions. The package version is
   recorded under `versions` but is not an input, so a version bump neither blocks a
   resume nor forces `--force-new`; reports open the store read-only.
2. **Kernel conventions.** QS equals `arch` at the same bandwidth. For Bartlett and Parzen,
   `arch` truncates at lag `bw` with weights `k(j / (bw + 1))`, so equality holds with our
   `S_T = bw + 1` (tested). The QS weight uses its Taylor series for `z < 0.01`.
3. **Prewhitened residuals** are not re-demeaned before the kernel sum (they have mean
   zero up to rounding); the residual series has `T - 1` rows and its autocovariances use
   divisor `T - 1`; the `T/(T - d)` factor uses the original `T`.
4. **Bootstrap SE via gradient projection.** `grad' Psi* grad = mean_j (grad' f_j)^2`, so
   block sums are projected on the gradient and no matrices are formed. Identical in exact
   arithmetic to section 4.4; `b = 1` reproduces the sample-covariance delta method
   (tested; equals the Mertens variance).
5. **Original-data SE in the bootstrap** is the prewhitened QS HAC SE for `b >= 2`. For
   Boot-IID (`b = 1`) it is the iid sample-covariance SE, as LW 2008 section 3.2.1
   prescribes. Before the 0.1.0 corrections the HAC SE was used for `b = 1` as well, which
   was a deviation from LW, not a convention; the first quick run's Boot-IID VAR rates
   (5.5% and 5.1%) came from that deviation.
6. **Coverage width metric.** `coverage.csv` reports the mean width and the mean ratio of
   width to the `iid_normal` width over replications with finite intervals, not the median
   (medians do not combine across checkpointed chunks).
7. **One-sided size** for every method is the rate at which its one-sided 95% lower bound
   exceeds the true SR (`boot_ts`: percentile-t, `SR_hat - q_0.95(t*) se_hat`). PSR rows
   test `H0: SR <= SR_true` and reject when PSR > 0.95.
8. **`exact_nct` in the Monte Carlo** uses the vectorised bisection with 50 iterations on a
   bracket `t -/+ (12 sqrt(1 + t^2/(2(T-1))) + 2)`; a non-bracketed root falls back to the
   iid-normal bound and is counted (`nct_fallback` in the unit files, `fallback_bounds` in
   `coverage.csv`). SciPy 1.18.1's `stats.nct.cdf` returns NaN far in the tails (for
   example `cdf(-1.37, 249, 11.95)`), inside the bracket; `single.nct_cdf` recomputes such
   entries by the reflection `cdf(t, df, d) = sf(-t, df, -d)` and maps anything still
   non-finite to its 0/1 limit. Before this, the library's brentq path raised on many
   negative t statistics and the Monte Carlo fell back on 2.9% of bounds.
9. **DSR rules.** Rules 1-4 use the Gumbel `E[max]` (what practitioners compute); rule 7
   uses the exact moments. Holm's procedure rejects at least one hypothesis exactly when the
   smallest p-value is at most `alpha / N`, which is how rule 6 is evaluated. The average
   pairwise correlation for rule 4 is computed in O(NT) from the sum of standardized
   columns.
10. **LW 2008 DGPs.** The VAR cases start from a Gaussian draw of the stationary law (for
    t6-VAR too; the start's influence is negligible at T = 120 with phi = 0.2). The
    agreement tolerance is `3 sqrt(p (1 - p) (1/n + 1/5000))` with `p` the pooled rate,
    which equals the section 6.3.2 formula when `n = 5000`. Only the 5% column of LW
    Table 1 is transcribed, so only 5% rows carry the comparison.
11. **Quick profile after the pilot.** The pilot projected 117 CPU-minutes (39 wall
    minutes on 3 workers), above the 35-minute budget, so `coverage.boot_reps` was lowered
    from 1000 to 500 (the floor) and `dsr.reps` from 2000 to 1000
    (`results/mc-quick/calibration.json`).
12. **Turnover** drifts weights with the excess returns (not total returns); for monthly
    data the difference is second order. Reported for description only.
13. **Ken French files.** The vintage regex matches "using the YYYYMM CRSP database", which
    covers both "created using the ..." (current files) and "created by <program> using
    the ..." (older layouts). `ind48` stays in `DATASETS` for
    users but is not used by the study (section 2, item 6).
14. **Corrections before release (0.1.0).** An independent review found and we fixed: the
    noncentral t NaN (item 8), Boot-IID's original-data SE (item 5), the scale dependence of
    the Andrews bandwidth and of the prewhitening cap (section 4.3), the t6-GARCH
    innovations (section 6.3.2: a shared mixing variable, as in LW), missing-value
    alignment in `sharpecheck compare`, and the package version blocking resumes (item 1).
    The coverage and difference experiments of the quick profile and the portfolio report
    were re-run after these fixes; the DSR units do not use any changed code and were kept.
