"""sharpecheck: Sharpe-ratio inference with Monte Carlo-checked intervals.

Conventions (used everywhere, see ``docs/DESIGN.md`` section 3): inputs are per-period
*excess* returns; every Sharpe ratio, standard error and interval is per period (never
annualized unless a function says so); skewness ``g3`` and kurtosis ``g4`` are the
population-moment (biased) estimators and ``g4`` is the non-excess kurtosis (3 for normal).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from sharpecheck.results import SharpeInterval, TestResult

try:
    __version__ = version("sharpecheck")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = ["SharpeInterval", "TestResult", "__version__"]
