# sharpecheck

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Sharpe-ratio inference for Python, with every interval checked by Monte Carlo.

**Status: 0.1.0, alpha.** All methods below are implemented and tested. The quick
Monte Carlo profile and the portfolio study have been run. Results are in
[REPORT.md](REPORT.md). The API may still change.

## Why

"Is this Sharpe ratio real?" has several published answers:

- the iid-normal standard error (Lo 2002) and the exact noncentral-t interval
- the non-normal correction of Mertens and Opdyke
- the AR(1) adjustment of Lopez de Prado, Lipton and Zoonekynd (2025)
- HAC standard errors
- the studentized block bootstrap of Ledoit and Wolf (2008)
- the probabilistic and deflated Sharpe ratios of Bailey and Lopez de Prado

They are scattered across papers, notebooks and R packages, and each is valid under
different assumptions. sharpecheck puts them behind one typed API. It also measures how
well each one keeps its promise under realistic return processes, and publishes the
results.

| Area | Contents |
|---|---|
| One Sharpe ratio | iid-normal, exact noncentral t, Mertens/Opdyke, AR(1)-adjusted, HAC (QS kernel, Andrews bandwidth, optional prewhitening), studentized circular block bootstrap; PSR (both published variance conventions), MinTRL, critical SR, power, annualization |
| Two Sharpe ratios | Jobson-Korkie-Memmel, Ledoit-Wolf HAC, Ledoit-Wolf studentized bootstrap, block-size calibration (LW Algorithm 3.1) |
| Many strategies | deflated Sharpe ratio (Gumbel and exact expected maximum, correlated trials), implied number of trials, pFDR/oFDR, Holm, Benjamini-Hochberg, Romano-Wolf stepdown, model confidence set (via `arch`) |
| Validation | checkpointed Monte Carlo of coverage and size under six return processes, T = 60 to 2520, and a replication of the Ledoit-Wolf (2008) size table |
| Application | 1/N against minimum-variance (sample and Ledoit-Wolf), inverse-volatility and HRP portfolios on Ken French data, with Romano-Wolf-adjusted tests |

## Install

sharpecheck is installed from GitHub (it is not on PyPI):

```sh
uv add git+https://github.com/h-a-forster/sharpecheck
# or, from a clone:
uv sync
```

It requires Python 3.11 or later, numpy, scipy, pandas and arch. matplotlib is used only for the
Monte Carlo heatmap.

## Quick start

Inputs are per-period **excess** returns. Every SR, standard error and interval is per
period unless a function says otherwise.

```python
import numpy as np
from sharpecheck.single import sharpe_ci
from sharpecheck.compare import lw_boot_test

r = np.loadtxt("fund.csv")  # daily or monthly excess returns

# methods: "iid_normal", "exact_nct", "mertens", "ar1", "hac", "hac_pw", "boot_ts"
ci = sharpe_ci(r, "boot_ts", seed=1)
print(ci.estimate, ci.lower, ci.upper)

res = lw_boot_test(r, benchmark_returns, seed=1)  # H0: equal Sharpe ratios
print(res.statistic, res.pvalue)
```

From the command line, on a CSV with one column per return series:

```console
$ sharpecheck sr returns.csv --column fund
fund: T = 240, per-period SR = 0.0573 (mean 0.002103, sd 0.03671)
skewness 0.047, kurtosis 3.487 (non-excess), lag-1 autocorr -0.093

95% two-sided intervals for the per-period SR:
  method          lower     upper        se
  iid_normal    -0.0693    0.1839    0.0646
  exact_nct     -0.0695    0.1837       nan
  mertens       -0.0692    0.1838    0.0645
  ar1           -0.0579    0.1725    0.0588
  hac           -0.0591    0.1737    0.0594
  hac_pw        -0.0587    0.1732    0.0592
  boot_ts       -0.0541    0.1687    0.0592

PSR(SR0 = 0): 0.8122 (BLdP 2012, observed variance), 0.8126 (LZL 2025, null variance)
MinTRL at 95%: 825.0 (observed), 824.6 (null) periods

$ sharpecheck compare returns.csv fund bench
SR(fund) - SR(bench) = -0.0838 per period, T = 240
  test        statistic   p-value
  jkm            -1.257    0.2088
  lw_hac         -1.403    0.1607
  lw_hac_pw      -1.461    0.1441
  lw_boot        -1.461    0.1544
LW bootstrap 95% interval: [-0.2012, 0.0337] (block 6, 4999 resamples)
```

Add `--json` for machine-readable output. Other subcommands: `sharpecheck mc
run|pilot|report` (Monte Carlo), `sharpecheck data fetch` (Ken French data) and
`sharpecheck portfolio run|report`.

## Which interval should I trust?

These are short answers from the quick Monte Carlo profile ([REPORT.md](REPORT.md),
`results/mc-quick/`):

- **iid returns.** Every method gives about 95% coverage from T = 60. The exact
  noncentral t is exact under normality.
- **Autocorrelated returns** (AR(1), phi = 0.2). The iid formulas cover about 89%. Use
  `ar1`, `hac_pw` or `boot_ts`; these give 93-97%.
- **Skewed, crash-prone returns.** `iid_normal` and `exact_nct` stay near 83% at SR =
  0.3 at any T. `mertens`, `ar1` and the HAC methods recover only at T >= 1000.
- **GARCH volatility clustering or regime switching.** When SR > 0, no method reaches
  95%. At SR = 0.3 and T = 2520 coverage is 81-83% under GARCH and 75-85% under regime
  switching, because the true variance of the estimator is well above the formula value
  (about twice under GARCH). At SR = 0 the GARCH cells are near nominal; regime
  switching stays at 89-93%. Treat every interval as too narrow under these processes
  when the SR is away from zero.
- **Overall**, the studentized block bootstrap (`boot_ts`) is the closest to nominal,
  but it is conservative at T = 60.

Applied to 1/N against optimised portfolios on Ken French data (1973-2026), no rule's
Sharpe ratio differs from 1/N at 5% after Romano-Wolf adjustment. The closest is
Ledoit-Wolf minimum variance on the 25 size/book-to-market portfolios, with adjusted
p = 0.072.

## Validation

- Unit tests reproduce the published numeric examples of Bailey and Lopez de Prado (2012,
  2014) and Lopez de Prado, Lipton and Zoonekynd (2025). `scripts/reference_values.py`
  recomputes them independently.
- The QS HAC estimator equals `arch`'s at a fixed bandwidth to 1e-10. The Ledoit-Wolf
  covariance matches scikit-learn's `ledoit_wolf`.
- A Ledoit-Wolf (2008) Table 1 replication agrees in 34 of 36 entries. The two misses
  are our fixed-block (b = 6) bootstrap, which is conservative; LW calibrate the block.
- The Monte Carlo is checkpointed per unit with hashed seeds, and results are identical
  with 1 and 3 workers. Every rate has a Wilson interval, and under-coverage flags use a
  Holm correction.

To reproduce the quick profile (about 30 minutes on 3 cores) and the portfolio study:

```sh
uv run sharpecheck mc pilot --config configs/mc_quick.toml
uv run python scripts/run_with_retry.py --log runs/mc-quick/retry.log -- \
    uv run sharpecheck mc run --config configs/mc_quick.toml --workers 3
uv run sharpecheck mc report --config configs/mc_quick.toml

uv run sharpecheck data fetch --config configs/portfolio.toml
uv run sharpecheck portfolio run --config configs/portfolio.toml
uv run sharpecheck portfolio report --config configs/portfolio.toml
```

## Development

```sh
uv sync --group dev
uv run pytest                 # add --runslow for the coverage simulations
uv run ruff check .
uv run ruff format --check .
```

See [AGENTS.md](AGENTS.md) for conventions and [docs/DESIGN.md](docs/DESIGN.md) for the
formulas, references, validation strategy and study design.

## Citation

If you use sharpecheck, please cite it with the metadata in [CITATION.cff](CITATION.cff)
and cite the papers behind the methods you use. They are listed in `docs/DESIGN.md`.

## License

MIT. See [LICENSE](LICENSE). Ken French data are not redistributed; `sharpecheck data
fetch` downloads them from the Kenneth R. French Data Library and records SHA-256
checksums.
