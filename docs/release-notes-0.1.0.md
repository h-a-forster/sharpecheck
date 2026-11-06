# sharpecheck 0.1.0

First release (alpha; the API may change). Install from GitHub:
`uv add git+https://github.com/h-a-forster/sharpecheck@v0.1.0`.

## What it does

- Sharpe-ratio intervals by iid-normal, exact noncentral t, Mertens/Opdyke,
  AR(1)-adjusted, HAC (QS kernel, Andrews bandwidth, optional prewhitening) and the
  studentized circular block bootstrap; PSR in both published variance conventions,
  MinTRL, critical SR, power.
- Two-strategy tests: Jobson-Korkie-Memmel, Ledoit-Wolf HAC and the Ledoit-Wolf
  studentized bootstrap with block calibration.
- Many strategies: deflated Sharpe ratio, implied number of trials, pFDR/oFDR, Holm,
  Benjamini-Hochberg, Romano-Wolf stepdown, model confidence set (via `arch`).
- A checkpointed, resumable Monte Carlo of interval coverage and test size, and a 1/N
  portfolio study on Ken French data. CLI: `sr`, `compare`, `mc`, `data`, `portfolio`.

## Results in this release (quick profile, `REPORT.md`)

- Under iid returns the analytic and HAC methods cover 92.4-96.0% from T = 60; the block
  bootstrap over-covers at small T (98.0% at T = 60, normal returns, SR = 0).
- Under AR(1) with phi = 0.2 the iid formulas cover 88-90%; `ar1`, `hac_pw` and
  `boot_ts` cover 92.6-97.0%.
- At SR = 0.3 and T = 2520 no method reaches 95% under GARCH(1,1)-t7 (81-83%) or regime
  switching (75-85%).
- Ledoit-Wolf (2008) Table 1 replication: 34 of 36 entries agree.
- Portfolio study: no rule's Sharpe ratio differs from 1/N after Romano-Wolf adjustment
  (closest: Ledoit-Wolf minimum variance on 25 size/book-to-market portfolios, adjusted
  p = 0.072).

## Fixed since the changelog entry

None of these changes a stored Monte Carlo or portfolio result: `mc report` and
`portfolio report` reproduce `results/` byte for byte.

- Constant series: `jkm_test`, `lw_hac_test`, `lw_boot_test`, `dsr_test` and `hac_se` only
  tested for an exactly zero variance. Rounding leaves a constant series a tiny variance,
  so `sharpecheck compare` on a cash-like column reported a huge SR difference with
  p near 0. They now return `nan` with a reason, as `sharpe_ci` already did.
- `ofdr` raised `ZeroDivisionError` for strong track records; it now works from log tail
  probabilities.
- `--json` printed bare `NaN` (the `exact_nct` standard error on every `sr` call), which
  strict JSON parsers reject; it now prints `null`.
- `andrews_bandwidth` returned 0, and `long_run_cov` raised, when no column had any
  lag-1 autocorrelation; the bandwidth is now 1.
- `mc run --experiments` with an unknown name did nothing and exited 0; it now raises.
- The portfolio config's `inference.block` and `inference.calibrate_block` were silently
  ignored. Values the study does not apply now raise.
- `sharpecheck sr --level 0.975` labelled the table "97%".
- The README console example was stale after the pre-release corrections. It now runs on
  the committed `tests/fixtures/example_returns.csv`, and a test keeps it current.

## Known limitations

- Only the quick Monte Carlo profile has been run. The full profile
  (`configs/mc_full.toml`) is resumable; its timing pilot measured about 69 CPU-hours.
- SciPy 1.18.1 `stats.nct.cdf` returns NaN far in the left tail; `single.nct_cdf` works
  around it (`docs/upstream/scipy-nct-cdf.md`).
- No transaction costs in the portfolio study; results depend on the Ken French data
  vintage (202608). The out-of-sample return panel is not committed because the library
  grants no redistribution rights, so the portfolio p-values (including p = 0.072) cannot
  be regenerated after a revision (DESIGN section 13, item 16).

See `CHANGELOG.md` for the full list of changes.
