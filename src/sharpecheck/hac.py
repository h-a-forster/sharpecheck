"""Kernel long-run covariance estimation (milestone M2).

Bartlett, Parzen and Quadratic-Spectral kernels, the Andrews (1991) AR(1)-plug-in automatic
bandwidth, and Andrews-Monahan (1992) VAR(1) prewhitening with recolouring. Used by the HAC
Sharpe-ratio methods (Lo 2002; Ledoit and Wolf 2008). Formulas: DESIGN 4.3.
"""

from __future__ import annotations

from typing import Literal

from numpy.typing import ArrayLike

from sharpecheck._validation import FloatArray

Kernel = Literal["bartlett", "parzen", "qs"]


def kernel_weights(lags: ArrayLike, bandwidth: float, kernel: Kernel = "qs") -> FloatArray:
    """Kernel weights k(j / S_T) for integer lags ``j``."""
    raise NotImplementedError("M2")


def andrews_bandwidth(y: ArrayLike, kernel: Kernel = "qs") -> float:
    """Andrews (1991) automatic bandwidth from univariate AR(1) fits to each column of y."""
    raise NotImplementedError("M2")


def long_run_cov(
    y: ArrayLike,
    kernel: Kernel = "qs",
    bandwidth: float | None = None,
    *,
    prewhiten: bool = False,
    dof_adjust: int = 0,
) -> FloatArray:
    """Kernel estimate of the long-run covariance of the (T x k) series ``y``.

    ``y`` is demeaned internally. ``dof_adjust = d`` multiplies by T / (T - d) (Ledoit and
    Wolf 2008 use d = 4 for the difference test). ``bandwidth=None`` uses
    :func:`andrews_bandwidth` (on the prewhitened residuals when ``prewhiten``).
    """
    raise NotImplementedError("M2")
