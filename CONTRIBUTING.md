# Contributing

Bug reports, new inference methods, reference values and Monte Carlo processes are welcome.

## Setup

You need [uv](https://docs.astral.sh/uv/) and git.

```sh
uv sync --group dev
uv run sharpecheck --version
```

## Checks

```sh
uv run pytest              # fast suite; add --runslow for long Monte Carlo checks
uv run ruff check .
uv run ruff format --check .
```

CI runs these on pull requests (Ubuntu with Python 3.11 and 3.13, Windows with Python 3.13).

Test rules:

- No network. Tests use synthetic data or small committed fixtures.
- Use `tmp_path` for every file.
- Tests must pass on Windows: use `pathlib` and do not assume a POSIX shell.
- Fix every random seed. Statistical tests must be deterministic, not flaky.
- Mark anything over a few seconds with `@pytest.mark.slow`.

## Adding a method

1. Put the formula, its reference and its edge cases in `docs/DESIGN.md` first, including
   which conventions it uses (ddof, excess or raw kurtosis, T or T - 1, per period).
2. Return a `sharpecheck.results.SharpeInterval` or `TestResult`. Return `nan` with
   `details["reason"]` for statistically undefined input; raise `ValueError` for invalid
   input.
3. Add an oracle test (published numeric example, closed form or an independent library)
   and add the method to the Monte Carlo config so its coverage or size is measured.

## Code

- Python 3.11+, typed. Runtime dependencies: numpy, scipy, pandas, arch, threadpoolctl.
  Open an issue before adding one.
- Match the surrounding style; ruff enforces the rest (line length 100).
- Long computations must go through `sharpecheck.checkpoint.UnitStore`, so they can resume
  after a crash.
- Do not copy code from sources without a compatible licence; implement from the paper and
  cite it.
