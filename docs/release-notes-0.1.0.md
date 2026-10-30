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

- Under iid returns every analytic method covers about 95% from T = 60; the block
  bootstrap over-covers at small T (98.0% at T = 60, normal returns, SR = 0).
- Under AR(1) with phi = 0.2 the iid formulas cover 88-90%; `ar1`, `hac_pw` and
  `boot_ts` cover 92.6-97.0%.
- At SR = 0.3 and T = 2520 no method reaches 95% under GARCH(1,1)-t7 (81-83%) or regime
  switching (75-85%).
- Ledoit-Wolf (2008) Table 1 replication: 34 of 36 entries agree.
- Portfolio study: no rule's Sharpe ratio differs from 1/N after Romano-Wolf adjustment
  (closest: Ledoit-Wolf minimum variance on 25 size/book-to-market portfolios, adjusted
  p = 0.072).

## Known limitations

- Only the quick Monte Carlo profile has been run. The full profile
  (`configs/mc_full.toml`) is resumable; its timing pilot measured about 69 CPU-hours.
- SciPy 1.18.1 `stats.nct.cdf` returns NaN far in the left tail; `single.nct_cdf` works
  around it (`docs/upstream/scipy-nct-cdf.md`).
- No transaction costs in the portfolio study; results depend on the Ken French data
  vintage (202608).

See `CHANGELOG.md` for the full list of changes.
