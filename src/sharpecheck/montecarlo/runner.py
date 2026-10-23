"""Monte Carlo runner (milestones M5, M6). DESIGN 6.

Three experiments, each split into checkpointed units:

- ``coverage``: single-SR intervals, unit = (dgp, sr, T, chunk)
- ``difference``: Ledoit-Wolf (2008) Table 1 replication, unit = (dgp, chunk)
- ``dsr``: DSR size under correlated and miscounted trials, unit = (N, rho, T, dist, chunk)

Seeds derive from ``numpy.random.SeedSequence(config seed).spawn`` keyed by the unit id
(hash of the id string mapped to a spawn key), so a unit's result does not depend on which
other units ran or on the worker count.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    """Read and validate a Monte Carlo TOML config."""
    raise NotImplementedError("M5")


def run(config_path: Path, *, workers: int = 3, force_new: bool = False) -> Path:
    """Run (or resume) all units of a config; return the run directory."""
    raise NotImplementedError("M5")
