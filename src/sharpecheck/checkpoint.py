"""Resumable unit store: one file per finished unit, atomic writes, skip-if-done (M1).

Every long computation (Monte Carlo cells and chunks, portfolio backtests) writes each
finished unit through this module and appends a line to ``progress.log``. A rerun with the
same configuration skips completed units; a changed configuration, input-data hash or
package version refuses to mix (``--force-new`` starts a new run). Worker processes are
capped at 3 and run with one BLAS thread each. Design follows tailcheck's checkpoint module.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

MAX_WORKERS = 3


class UnitStore:
    """Directory of completed units keyed by a string id."""

    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root = root
        self.manifest = manifest

    def done(self, unit_id: str) -> bool:
        raise NotImplementedError("M1")

    def write(self, unit_id: str, payload: dict[str, Any]) -> Path:
        raise NotImplementedError("M1")

    def read(self, unit_id: str) -> dict[str, Any]:
        raise NotImplementedError("M1")

    def log(self, message: str) -> None:
        raise NotImplementedError("M1")


def run_units(
    store: UnitStore,
    units: Iterable[tuple[str, dict[str, Any]]],
    work: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    workers: int = MAX_WORKERS,
) -> int:
    """Run ``work`` on every unit not yet in ``store``; return the number of units run.

    ``workers`` is clipped to ``MAX_WORKERS``; workers set BLAS threads to 1.
    """
    raise NotImplementedError("M1")
