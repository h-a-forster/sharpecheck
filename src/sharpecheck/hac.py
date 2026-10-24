"""Kernel long-run covariance estimation (milestone M2).

Bartlett, Parzen and Quadratic-Spectral kernels, the Andrews (1991) AR(1)-plug-in automatic
bandwidth, and Andrews-Monahan (1992) VAR(1) prewhitening with recolouring. Used by the HAC
Sharpe-ratio methods (Lo 2002; Ledoit and Wolf 2008). Formulas: DESIGN 4.3.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from sharpecheck._validation import FloatArray, as_float_2d

Kernel = Literal["bartlett", "parzen", "qs"]

KERNELS: tuple[Kernel, ...] = ("bartlett", "parzen", "qs")

#: Andrews (1991) and Andrews-Monahan (1992) cap on AR coefficients / singular values.
AR_CAP = 0.97


def _check_kernel(kernel: str) -> None:
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")


def kernel_weights(lags: ArrayLike, bandwidth: float, kernel: Kernel = "qs") -> FloatArray:
    """Kernel weights k(j / S_T) for integer lags ``j``."""
    _check_kernel(kernel)
    if not bandwidth > 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    x = np.abs(np.asarray(lags, dtype=np.float64)) / float(bandwidth)
    if kernel == "bartlett":
        return np.where(x <= 1.0, 1.0 - x, 0.0)
    if kernel == "parzen":
        return np.where(
            x <= 0.5, 1.0 - 6.0 * x**2 + 6.0 * x**3, np.where(x <= 1.0, 2.0 * (1.0 - x) ** 3, 0.0)
        )
    z = 6.0 * np.pi * x / 5.0
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 3.0 / z**2 * (np.sin(z) / z - np.cos(z))
    small = z < 1e-2  # series 1 - z^2/10 + z^4/280 avoids cancellation near 0
    return np.where(small, 1.0 - z**2 / 10.0 + z**4 / 280.0, w)


def andrews_bandwidth(y: ArrayLike, kernel: Kernel = "qs") -> float:
    """Andrews (1991) automatic bandwidth from univariate AR(1) fits to each column of y.

    Each column is demeaned, divided by its sample standard deviation and fitted by OLS
    without intercept; ``|rho|`` is capped at 0.97. Equal weights across the standardised
    columns, i.e. Andrews' weights ``w_a = 1 / var_a^2`` on the raw columns, so the
    bandwidth does not change when a column is rescaled (DESIGN 4.3). Columns with zero
    variance are ignored; if all are degenerate the bandwidth is 1 (small, harmless because
    the covariance is then ~0).
    """
    _check_kernel(kernel)
    a = as_float_2d(np.asarray(y, dtype=np.float64).reshape(len(y), -1), "y", min_rows=3)
    t = a.shape[0]
    a = a - a.mean(axis=0)
    num1 = num2 = den = 0.0
    for col in a.T:
        x0, x1 = col[:-1], col[1:]
        ss = float(x0 @ x0)
        if not ss > 0:
            continue
        rho = float(np.clip((x0 @ x1) / ss, -AR_CAP, AR_CAP))
        e = x1 - rho * x0
        s2 = float(np.mean(e * e)) / float(np.mean(col * col))
        s4 = s2 * s2
        num1 += 4 * rho**2 * s4 / ((1 - rho) ** 6 * (1 + rho) ** 2)
        num2 += 4 * rho**2 * s4 / (1 - rho) ** 8
        den += s4 / (1 - rho) ** 4
    if not den > 0:
        return 1.0
    if kernel == "bartlett":
        return float(1.1447 * (num1 / den * t) ** (1 / 3))
    if kernel == "parzen":
        return float(2.6614 * (num2 / den * t) ** (1 / 5))
    return float(1.3221 * (num2 / den * t) ** (1 / 5))


def autocovariances(y: FloatArray) -> FloatArray:
    """All sample autocovariance matrices ``Gamma(j) = (1/T) sum_t y_t y_{t-j}'``, j=0..T-1.

    ``y`` (T x k) is used as given (no demeaning). Computed by FFT; returns (T, k, k).
    """
    t = y.shape[0]
    nfft = 1 << int(np.ceil(np.log2(2 * t)))
    f = np.fft.rfft(y, n=nfft, axis=0)
    # c[j, a, b] = sum_t y_{t+j, a} y_{t, b}
    c = np.fft.irfft(f[:, :, None] * np.conj(f[:, None, :]), n=nfft, axis=0)[:t]
    return c / t


def kernel_sum(y: FloatArray, kernel: Kernel, bandwidth: float) -> FloatArray:
    """``sum_{|j| < T} k(j / S_T) Gamma(j)`` with ``Gamma(-j) = Gamma(j)'`` (no demeaning)."""
    t, k = y.shape
    if kernel == "qs":
        g = autocovariances(y)
        w = kernel_weights(np.arange(t), bandwidth, kernel)
    else:
        lmax = min(t - 1, int(np.floor(bandwidth)))
        g = np.empty((lmax + 1, k, k))
        for j in range(lmax + 1):
            g[j] = y[j:].T @ y[: t - j] / t
        w = kernel_weights(np.arange(lmax + 1), bandwidth, kernel)
    s = np.tensordot(w[1:], g[1:], axes=1) if len(w) > 1 else np.zeros((k, k))
    return g[0] + s + s.T


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

    Prewhitening (Andrews and Monahan 1992): OLS VAR(1) ``y_t = A y_{t-1} + e_t``; singular
    values of ``A`` above 0.97 are set to 0.97 and the residuals recomputed; the estimate
    on ``e`` is recoloured by ``(I - A)^{-1} Psi_e (I - A)^{-T}``.

    All steps run on columns divided by their sample standard deviations and the result is
    scaled back, so rescaling a column of ``y`` rescales the estimate exactly (DESIGN 4.3).
    """
    _check_kernel(kernel)
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    a = as_float_2d(arr, "y", min_rows=4)
    t, k = a.shape
    if not 0 <= dof_adjust < t:
        raise ValueError(f"dof_adjust must be in [0, T), got {dof_adjust}")
    if bandwidth is not None and not bandwidth > 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    a = a - a.mean(axis=0)
    # Work on standardised columns and scale back at the end, so the estimate is
    # equivariant under rescaling a column (the singular-value cap is not otherwise).
    sd = a.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    a = a / sd
    scale = t / (t - dof_adjust) * np.outer(sd, sd)
    if not prewhiten:
        bw = andrews_bandwidth(a, kernel) if bandwidth is None else float(bandwidth)
        return scale * kernel_sum(a, kernel, bw)
    y0, y1 = a[:-1], a[1:]
    coef, *_ = np.linalg.lstsq(y0, y1, rcond=None)  # y1 ~ y0 @ coef, so A = coef.T
    amat = coef.T
    u, sv, vt = np.linalg.svd(amat)
    if np.any(sv > AR_CAP):
        amat = u @ np.diag(np.minimum(sv, AR_CAP)) @ vt
    e = y1 - y0 @ amat.T
    bw = andrews_bandwidth(e, kernel) if bandwidth is None else float(bandwidth)
    psi_e = kernel_sum(e, kernel, bw)
    binv = np.linalg.inv(np.eye(k) - amat)
    return scale * (binv @ psi_e @ binv.T)
