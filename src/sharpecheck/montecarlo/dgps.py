"""Data-generating processes with known population Sharpe ratio (milestone M5). DESIGN 6.2.

Every DGP simulates a (reps x T) batch, vectorised over replications, and is calibrated so
that the stationary mean divided by the stationary standard deviation equals the target
per-period SR exactly (closed form; no simulated truth). Burn-in for dependent processes is
documented per DGP and drawn from the same seeded generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sharpecheck._validation import FloatArray

DgpName = Literal["normal", "t5", "garch-t7", "ar1-0.2", "regime", "crash"]


@dataclass(frozen=True)
class PopulationMoments:
    """Stationary moments of a DGP (closed form)."""

    mean: float
    sd: float
    sr: float
    g3: float
    g4: float
    rho1: float


def population_moments(name: DgpName, sr: float) -> PopulationMoments:
    """Closed-form stationary moments of DGP ``name`` calibrated to per-period ``sr``."""
    raise NotImplementedError("M5")


def simulate(
    name: DgpName, sr: float, nobs: int, reps: int, rng: np.random.Generator
) -> FloatArray:
    """(reps x nobs) excess returns from DGP ``name`` with population SR ``sr``."""
    raise NotImplementedError("M5")


def simulate_pair(name: str, nobs: int, reps: int, rng: np.random.Generator) -> FloatArray:
    """(reps x nobs x 2) paired returns with equal SR: the six Ledoit-Wolf (2008) DGPs."""
    raise NotImplementedError("M5")


def simulate_trials(
    n_trials: int,
    nobs: int,
    rho: float,
    rng: np.random.Generator,
    *,
    dist: Literal["normal", "t5"] = "normal",
) -> FloatArray:
    """(nobs x n_trials) zero-SR trial returns with equicorrelation ``rho`` (one factor)."""
    raise NotImplementedError("M5")
