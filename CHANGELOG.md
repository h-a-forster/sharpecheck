# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-24

First release.

### Fixed before release

- `exact_nct` no longer raises on negative t statistics: SciPy's `nct.cdf` returns NaN far
  in the tails, so `single.nct_cdf` uses the reflection identity there; brentq failures
  fall back to the normal approximation as documented.
- The Andrews bandwidth and the prewhitening cap now work on standardised columns, so HAC
  and bootstrap intervals no longer depend on the units of the returns.
- Boot-IID (`block = 1`) studentizes the original statistic with the iid standard error,
  as in Ledoit and Wolf (2008) section 3.2.1.
- The t6-GARCH pair DGP uses bivariate t6 innovations with a shared mixing variable.
- `sharpecheck compare` drops rows with a missing value in either column instead of
  shifting one series against the other.
- The package version is no longer a run-store input, so a version bump does not block
  resuming or reporting; `portfolio report` opens the store read-only.
- `coverage.csv` reports `exact_nct` fallback counts; `summary.json` reports cells below
  0.93 next to the Holm flags per method.

### Added

- `single`: Sharpe-ratio intervals by iid-normal, exact noncentral t, Mertens/Opdyke,
  AR(1)-adjusted (Lopez de Prado, Lipton and Zoonekynd 2025), HAC (QS kernel, Andrews
  bandwidth, optional Andrews-Monahan prewhitening) and the studentized circular block
  bootstrap; PSR in both published variance conventions, MinTRL, critical SR, power,
  annualization.
- `hac`, `bootstrap`: kernel long-run covariance with FFT autocovariances; block
  bootstraps with gradient-projected block sums; LW (2008) Algorithm 3.1 block calibration.
- `compare`: Jobson-Korkie-Memmel, Ledoit-Wolf HAC (with and without prewhitening) and the
  Ledoit-Wolf studentized bootstrap test and interval.
- `deflated`: deflated Sharpe ratio with Gumbel or exact expected maximum and
  equicorrelated trials, implied number of trials, pFDR, oFDR.
- `multiple`: Holm, Benjamini-Hochberg, Romano-Wolf stepdown, model confidence set (via
  `arch`).
- `montecarlo`: six coverage DGPs with closed-form moments, the LW (2008) pair DGPs,
  checkpointed and resumable runner with hashed per-unit seeds, timing pilot, report with
  Wilson intervals, Holm-flagged under-coverage, LW Table 1 agreement and a heatmap.
- `data`, `portfolio`: Ken French downloads with SHA-256 manifest and vintage; 1/N,
  inverse-volatility, sample and Ledoit-Wolf minimum-variance and HRP rules; rolling
  backtest; pre-registered study with a joint Romano-Wolf-adjusted bootstrap.
- CLI: `sr`, `compare`, `mc run|pilot|report`, `data fetch`, `portfolio run|report`.
- `scripts/run_with_retry.py` (resume-on-failure wrapper) and
  `scripts/lw_bootiid_convention.py` (Boot-IID studentization sensitivity check).
- Results of the quick Monte Carlo profile (`results/mc-quick/`) and the portfolio study
  (`results/portfolio/`), summarised in `REPORT.md`.
- Project scaffold, `docs/DESIGN.md` and `scripts/reference_values.py` (independent
  recomputation of published numeric examples).
