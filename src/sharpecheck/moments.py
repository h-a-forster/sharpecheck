"""Sample moments used by every Sharpe-ratio method (milestone M1).

Conventions (DESIGN 3): ``sr = mean / sd`` with ``sd`` the ddof=0 standard deviation
(the delta-method and PSR literature use moment estimators); ``g3``/``g4`` are the
population-moment skewness and non-excess kurtosis; ``rho1`` is the lag-1 sample
autocorrelation of the returns.
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Moments:
    """Per-period summary statistics of one return series."""

    nobs: int
    mean: float
    sd: float
    sr: float
    g3: float
    g4: float
    rho1: float


def sample_moments(x: ArrayLike) -> Moments:
    """Return the sample moments of a 1-D excess-return series."""
    raise NotImplementedError("M1")
