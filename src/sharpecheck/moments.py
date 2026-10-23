"""Sample moments used by every Sharpe-ratio method (milestone M1).

Conventions (DESIGN 3): ``sr = mean / sd`` with ``sd`` the ddof=0 standard deviation
(the delta-method and PSR literature use moment estimators); ``g3``/``g4`` are the
population-moment skewness and non-excess kurtosis; ``rho1`` is the lag-1 sample
autocorrelation of the returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from sharpecheck._validation import FloatArray, as_float_1d


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
    """Return the sample moments of a 1-D excess-return series.

    Raises ``ValueError`` on fewer than 2 observations or non-finite values. With zero
    variance, ``sr``, ``g3``, ``g4`` and ``rho1`` are ``nan``.
    """
    a = as_float_1d(x, "x")
    n = a.size
    mu = float(a.mean())
    d = a - mu
    m2 = float(np.mean(d * d))
    if not m2 > (1e-14 * abs(mu)) ** 2:  # zero variance up to rounding of the mean
        return Moments(n, mu, 0.0, math.nan, math.nan, math.nan, math.nan)
    m3 = float(np.mean(d**3))
    m4 = float(np.mean(d**4))
    sd = math.sqrt(m2)
    rho1 = float(np.dot(d[1:], d[:-1]) / n / m2)
    return Moments(n, mu, sd, mu / sd, m3 / m2**1.5, m4 / m2**2, rho1)


def batch_moments(x: FloatArray) -> dict[str, FloatArray]:
    """Vectorised moments for a (reps x T) batch (rows are series); same conventions.

    Returns arrays ``mean, sd, sr, g3, g4, rho1``; rows with zero variance give ``nan``.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError(f"x must be (reps x T) with T >= 2, got shape {a.shape}")
    n = a.shape[1]
    mu = a.mean(axis=1)
    d = a - mu[:, None]
    d2 = d * d
    m2 = d2.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        bad = ~(m2 > (1e-14 * np.abs(mu)) ** 2)
        m2s = np.where(bad, np.nan, m2)
        sd = np.sqrt(m2s)
        m3 = (d2 * d).mean(axis=1)
        m4 = (d2 * d2).mean(axis=1)
        rho1 = np.einsum("ij,ij->i", d[:, 1:], d[:, :-1]) / n / m2s
        return {
            "mean": mu,
            "sd": np.where(bad, 0.0, sd),
            "sr": mu / sd,
            "g3": m3 / m2s**1.5,
            "g4": m4 / m2s**2,
            "rho1": rho1,
        }
