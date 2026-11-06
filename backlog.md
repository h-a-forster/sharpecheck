# Backlog

## In progress

## Waiting on me

- `polish-2026-10-30` is merged into `main` through a PR (merge commit or fast-forward), so CI runs on GitHub for the first time and passes.
- v0.1.0 is tagged after CI passes.
- Decide whether to ask Ken French for permission to publish the derived out-of-sample return panel: the library states no licence, so it is not committed and portfolio p = 0.072 cannot be regenerated after a revision (DESIGN 13.16).
- Optional: SciPy #25470 has a comment linking Boost.Math PR #1529 (`docs/upstream/scipy-nct-cdf.md`).

## Decided

## Proposed

- The full Monte Carlo profile (`configs/mc_full.toml`, about 69 CPU-hours) is run in sessions and folded into REPORT.md.
- The GARCH and regime under-coverage is tested against other HAC bandwidths (Newey-West 1994, fixed-b) and a larger or calibrated bootstrap block.
- The portfolio study applies LW Algorithm 3.1 block calibration and reports the model confidence set (both "should" items, not run).
- The Monte Carlo compares against the 1% and 10% columns of LW Table 1 as well as the 5% column.
- `mypy src` passes and runs in CI (38 errors today, none a confirmed bug, mostly missing stubs and loose dicts), so the `Typing :: Typed` classifier is earned.
- A changed run-store input can be overridden without `--force-new` deleting finished units.
- If redistribution is cleared, `results/portfolio/` holds the out-of-sample return panel and `portfolio report` regenerates every p-value from it without `data/`.
- The Monte Carlo calls one vectorised library entry point, so each interval has one implementation instead of a mirror pinned by a test.
- `single.nct_cdf` is simplified once a SciPy release ships the Boost.Math fix.
