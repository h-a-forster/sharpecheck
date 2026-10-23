# AGENTS.md

Guide for coding agents (and humans) working on sharpecheck. Read `docs/DESIGN.md` before
changing any statistics. It holds the formulas, reference values, Monte Carlo design and
milestones this file points to.

## What this is

A Python package for Sharpe-ratio inference (iid-normal, exact noncentral t,
Mertens/Opdyke, AR(1)-adjusted, HAC, studentized circular block bootstrap, PSR, MinTRL,
DSR, Ledoit-Wolf difference tests, Romano-Wolf stepdown), a pre-registered Monte Carlo of
interval coverage and test size (the headline deliverable), and a secondary "1/N versus
optimised portfolios" application on Ken French data.

## Conventions

- uv + hatchling, `src/` layout, Python >= 3.11 (local dev pinned to 3.13 by
  `.python-version`), typed code, `ruff check` + `ruff format` (line length 100), pytest
  (slow tests marked `@pytest.mark.slow`, excluded by default).
- Inputs are per-period excess returns in decimals. Every SR, standard error and interval
  is per period. `g4` is the non-excess kurtosis; moments use ddof 0 (DESIGN 3).
- Say which PSR convention a number uses (`variance_at="observed"`: BLdP 2012;
  `"null"`: LZL 2025). They differ materially at small T.
- Intervals return `SharpeInterval`, tests `TestResult`. Statistically undefined input gives
  `nan` plus `details["reason"]`; invalid input raises `ValueError`.
- Bootstrap and simulation p-values use `(1 + count)/(B + 1)`. Every stochastic function
  takes `seed` or a `numpy.random.Generator`.
- No LLM or model API calls. CPU only. At most 3 worker processes, with BLAS threads set to
  1 in each worker. Keep memory modest (the machine is shared).
- Long jobs: every finished unit goes through `sharpecheck.checkpoint.UnitStore` (atomic
  file per unit, skip on rerun, `progress.log`). Never keep results only in memory. Keep
  each run under about 45 minutes of wall time; wrap it in `scripts/run_with_retry.py`
  (M6) and watch `runs/<name>/progress.log` for stalls.
- Data: only the Kenneth R. French Data Library, with timeouts; record SHA-256 and the CRSP
  vintage in `data/manifest.json`. Never commit raw downloads (`data/`, `cache/` and
  `runs/` are gitignored). Commit only small derived results under `results/`.
- The LZL (2025) replication code has no licence: cite its numbers, never copy its code.
- Scratch files go in `tmp/` (gitignored). Do not create `.bak` copies; use git.
- No personal names, local user paths or machine details in committed files; scrub logs
  before committing anything derived from them.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push
  and do not create remote repositories; publishing is done separately after review.
- Statistical honesty: report sizes, seeds, versions, data vintage and intervals; apply the
  multiple-testing control DESIGN specifies; never overstate. Before calling anything a
  third-party bug, produce a minimal repro and rule out an adapter or convention mistake
  (DESIGN 7.3).

## Architecture map

| Path | Role | Milestone |
|---|---|---|
| `src/sharpecheck/results.py` | `SharpeInterval`, `TestResult` | done |
| `src/sharpecheck/_validation.py` | input checks | done |
| `src/sharpecheck/checkpoint.py` | unit store, progress log, capped pool | done (M1) |
| `src/sharpecheck/moments.py` | sample moments | done (M1) |
| `src/sharpecheck/single.py` | single-SR intervals, PSR, MinTRL, power, annualize | done (M1, M3) |
| `src/sharpecheck/hac.py` | kernels, Andrews bandwidth, prewhitening | done (M2) |
| `src/sharpecheck/bootstrap.py` | circular block bootstrap, studentized stats, calibration | done (M3) |
| `src/sharpecheck/compare.py` | JKM, LW HAC, LW bootstrap | done (M3) |
| `src/sharpecheck/deflated.py` | expected max, DSR, implied N, pFDR/oFDR | M4 |
| `src/sharpecheck/multiple.py` | Holm, BH, Romano-Wolf, MCS wrapper | M4, M7 |
| `src/sharpecheck/montecarlo/` | DGPs, runner, report | M5, M6 |
| `src/sharpecheck/data.py` | Ken French fetch, cache, parse | M7 |
| `src/sharpecheck/portfolio/` | allocation rules, rolling backtest | M7 |
| `src/sharpecheck/cli.py` | `sr`, `compare`, `mc`, `data`, `portfolio` | M5-M8 |
| `configs/` | `mc_quick.toml`, `mc_full.toml`, `portfolio.toml` | - |
| `scripts/reference_values.py` | independent recomputation of DESIGN 7.1 values | done |

## How to test

```sh
uv sync --group dev
uv run pytest                 # fast suite (CI)
uv run pytest --runslow       # plus slow Monte Carlo checks
uv run ruff check .
uv run ruff format --check .
uv run python scripts/reference_values.py
```

## How to run the studies (after M5-M7)

```sh
uv run sharpecheck mc run --config configs/mc_quick.toml --workers 3     # target <= 35 min wall
uv run sharpecheck mc report --config configs/mc_quick.toml              # results/mc-quick/
uv run sharpecheck data fetch --config configs/portfolio.toml
uv run sharpecheck portfolio run --config configs/portfolio.toml
uv run sharpecheck portfolio report --config configs/portfolio.toml      # results/portfolio/
```

Full size: `configs/mc_full.toml` (projected 15-25 CPU-hours, a projection, not a
measurement; resumes from `runs/mc-full/units/`). Before any long run, validate on a small
sample, wrap it in the retry loop and watch the progress log.

## Status and next steps

Updated: 2026-10-24.

Done:
- Scaffold: package layout with typed stubs, CI, hygiene files, configs.
- `docs/DESIGN.md`: scope review, conventions, formulas and references for every module,
  Monte Carlo design with pre-registered expectations (6.6), portfolio design with a
  pre-registered primary test (5.4), validation strategy, milestones M1-M8, risks, prior art.
- Research checks (2026-10-23/24): Ken French CSV zips download (HTTP 200, vintage 202608);
  published numeric examples of BLdP 2012/2014 and LZL 2025 reproduced by
  `scripts/reference_values.py` (all match to the published rounding); LW 2008 Table 1
  transcribed; `arch` 8.0.0's QS kernel uses the same weight function as Andrews (1991)
  but a Newey-West (1994) bandwidth, so it is an oracle for fixed bandwidths only.
- Scaffold microbenchmark: one studentized block bootstrap (B = 499) costs about 2/7/23/62
  ms at T = 60/250/1000/2520 on one thread.

- M1 (2026-10-24): `checkpoint` (ported from tailcheck, 3-worker cap), `moments`
  (scalar and batch), analytic `single` methods (iid_normal, mertens, ar1, exact_nct with
  brentq and a vectorised bisection), PSR in both conventions, MinTRL, critical SR, power,
  annualize. All DESIGN 7.1 single-SR values reproduced by package tests; slow exact_nct
  coverage test passes.
- M2 (2026-10-24): `hac` (Bartlett/Parzen/QS weights, FFT autocovariances, Andrews
  bandwidth, Andrews-Monahan prewhitening). QS equals `arch` at fixed bandwidth to 1e-10;
  Bartlett/Parzen equal `arch` with `S_T = arch bandwidth + 1` (arch's convention);
  prewhitened and plain QS within 3% of the true VAR(1) long-run covariance at T = 200,000.
- M3 (2026-10-24): `bootstrap` (circular block indices, studentized SR and joint
  SR-difference bootstraps via gradient-projected block sums, LW Algorithm 3.1 block
  calibration), `compare` (JKM-Memmel, LW HAC with/without prewhitening, LW bootstrap with
  eq. 9 p-value and symmetric interval), `single` hac/hac_pw/boot_ts. b = 1 block SE equals
  the sample-covariance delta method (= Mertens); HAC SE for Gaussian AR(1) within 3% of
  sqrt(1.505417/T) at T = 100,000. Timing (one thread, B = 499): bootstrap 2/8/30/64 ms,
  HAC + HAC-PW pair 0.8/1.2/1.7/3.4 ms at T = 60/250/1000/2520.

Next:
- M4-M8 in order (DESIGN 9). M5 must start with the timing pilot before the quick
  Monte Carlo is launched.

Open questions recorded for later milestones:
- LW 2008 VAR DGPs: "mean 1, variance 1" read as the stationary law (DESIGN 6.3.2).
- A lead in the LZL replication code (`moments_Mk`, `(1 - rho)` vs `sqrt(1 - rho)`) is
  unverified and must not be reported without a repro (DESIGN 7.3).
