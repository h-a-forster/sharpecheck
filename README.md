# sharpecheck

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**Status: pre-alpha. The package skeleton and design are in place; the methods are being
implemented milestone by milestone (see [docs/DESIGN.md](docs/DESIGN.md) section 9).
Nothing below is usable yet.**

Sharpe-ratio inference for Python, with every interval checked by Monte Carlo.

"Is this Sharpe ratio real?" has several published answers: the iid-normal standard error,
the exact noncentral-t interval, the non-normal correction of Mertens and Opdyke, the
AR(1) adjustment of Lopez de Prado, Lipton and Zoonekynd, HAC standard errors (Lo), the
studentized block bootstrap of Ledoit and Wolf, the probabilistic and deflated Sharpe
ratios of Bailey and Lopez de Prado. They are scattered across papers, notebooks and R
packages, and each is valid under different assumptions. sharpecheck puts them behind
one typed API and measures how well each one keeps its promise.

| Area | Planned contents |
|---|---|
| One Sharpe ratio | iid-normal, exact noncentral t, Mertens/Opdyke, AR(1)-adjusted, HAC (QS kernel, optional prewhitening), studentized circular block bootstrap; PSR (both published variance conventions), MinTRL, critical SR, power |
| Two Sharpe ratios | Jobson-Korkie-Memmel, Ledoit-Wolf HAC, Ledoit-Wolf studentized bootstrap (with optional block-size calibration) |
| Many strategies | deflated Sharpe ratio (Gumbel and exact expected maximum, correlated trials), implied number of trials, pFDR/oFDR, Holm, Benjamini-Hochberg, Romano-Wolf stepdown, model confidence set (via `arch`) |
| Validation | a pre-registered Monte Carlo of interval coverage and test size under six return processes, sample sizes 60 to 2520, and a replication of the Ledoit-Wolf (2008) size table |
| Application | 1/N versus minimum-variance (Ledoit-Wolf shrinkage), inverse-volatility and HRP portfolios on Ken French data, with Romano-Wolf-adjusted Sharpe-difference tests |

## Development

```sh
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python scripts/reference_values.py   # independent recomputation of published values
```

See [AGENTS.md](AGENTS.md) for conventions and [docs/DESIGN.md](docs/DESIGN.md) for the
formulas, references, validation strategy and study design.

## License

MIT. See [LICENSE](LICENSE). Ken French data are not redistributed; the data command
downloads them from the Kenneth R. French Data Library.
