"""Monte Carlo report: tables with Wilson intervals and the coverage heatmap (M6). DESIGN 6.5.

Read-only on the run directory; writes ``results/<name>/`` (CSV, JSON summary, PNG
heatmap when matplotlib is available). Fails cleanly when the run is not finished.
"""

from __future__ import annotations

from pathlib import Path


def wilson_interval(successes: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    raise NotImplementedError("M6")


def build_report(config_path: Path, out_dir: Path | None = None) -> Path:
    """Aggregate a finished run into ``results/<name>/``; return that directory."""
    raise NotImplementedError("M6")
