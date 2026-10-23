"""Input validation shared by the public functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_float_1d(x: ArrayLike, name: str, min_len: int = 2) -> FloatArray:
    """Return ``x`` as a finite 1-D float64 array of length >= ``min_len`` or raise."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if a.size < min_len:
        raise ValueError(f"{name} needs at least {min_len} observations, got {a.size}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return a


def as_float_2d(x: ArrayLike, name: str, min_rows: int = 2) -> FloatArray:
    """Return ``x`` as a finite 2-D float64 array (T x k) with >= ``min_rows`` rows or raise."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D (T x k), got shape {a.shape}")
    if a.shape[0] < min_rows:
        raise ValueError(f"{name} needs at least {min_rows} rows, got {a.shape[0]}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return a


def check_level(level: float, name: str = "level") -> float:
    """Validate a confidence or significance level in (0, 1)."""
    v = float(level)
    if not 0.0 < v < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {level}")
    return v
