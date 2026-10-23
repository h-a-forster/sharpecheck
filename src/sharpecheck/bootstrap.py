"""Circular block bootstrap with studentized statistics (milestones M3, M7).

The bootstrap standard error uses the block-sum covariance of Goetze and Kuensch (1996),
as in Ledoit and Wolf (2008) section 3.2.2. Formulas: DESIGN 4.4.

Implementation note: for a ratio ``f(mu, gamma) = mu / sqrt(gamma - mu^2)`` the quadratic
form ``grad' Psi* grad`` equals ``mean_j (grad' f_j)^2``, so the block sums are projected
on the gradient first and no 2x2 or 4x4 matrices are formed. For a difference
``SR_j - SR_0`` the projection is ``h_j - h_0`` with ``h`` the per-series projection.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from sharpecheck._validation import FloatArray, as_float_1d, as_float_2d, check_level


def default_block(nobs: int) -> int:
    """Default block size ``max(1, round(nobs ** (1/3)))`` (DESIGN 4.4)."""
    if nobs < 1:
        raise ValueError(f"nobs must be positive, got {nobs}")
    return max(1, round(nobs ** (1 / 3)))


def _check_block(nobs: int, block: int) -> int:
    b = int(block)
    if not 1 <= b <= nobs:
        raise ValueError(f"block must be in [1, {nobs}], got {block}")
    return b


def circular_block_indices(
    nobs: int, block: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_boot x nobs) integer index matrix of circular block bootstrap resamples.

    ``ceil(nobs / block)`` block starts are drawn uniformly from ``0..nobs-1``; each block
    takes ``block`` consecutive indices modulo ``nobs``; the result is truncated to ``nobs``.
    """
    b = _check_block(nobs, block)
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    nblocks = -(-nobs // b)
    starts = rng.integers(0, nobs, size=(n_boot, nblocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % nobs
    return idx.reshape(n_boot, nblocks * b)[:, :nobs]


def _projected_block_sums(r: FloatArray, block: int) -> tuple[FloatArray, FloatArray]:
    """For resamples ``r`` (B x T [x k]) return ``(sr*, h)``.

    ``h`` has shape (B x l [x k]): the block sums of the moment deviations projected on the
    SR gradient, scaled by ``block ** -0.5``; ``se*^2 = mean_l h^2 / T``.
    """
    t = r.shape[1]
    mu = r.mean(axis=1)
    gam = np.mean(r * r, axis=1)
    var = gam - mu * mu
    with np.errstate(divide="ignore", invalid="ignore"):
        sd = np.sqrt(var)
        sr = mu / sd
        g_mu = gam / (var * sd)
        g_gam = -mu / (2 * var * sd)
    nb = t // block
    use = r[:, : nb * block]
    dev = g_mu[:, None] * (use - mu[:, None]) + g_gam[:, None] * (use * use - gam[:, None])
    shape = (r.shape[0], nb, block) + r.shape[2:]
    h = dev.reshape(shape).sum(axis=2) / math.sqrt(block)
    return sr, h


def studentized_sr_boot(
    x: ArrayLike, block: int, n_boot: int, seed: int | np.random.Generator | None = None
) -> tuple[float, float, FloatArray]:
    """Return ``(sr_hat, se_hat, t_star)`` for one series.

    ``t_star[m] = (sr*_m - sr_hat) / se*_m`` (signed; take ``abs`` for the symmetric
    interval). ``se_hat`` is the prewhitened-QS HAC standard error of the original data.
    Resamples with zero variance give ``nan`` entries.
    """
    from sharpecheck.single import hac_se

    a = as_float_1d(x, "x", min_len=4)
    t = a.size
    b = _check_block(t, block)
    rng = np.random.default_rng(seed)
    sr_hat, se_hat, _ = hac_se(a, prewhiten=True)
    return sr_hat, se_hat, sr_tstar(a, b, n_boot, rng, sr_hat)


def sr_tstar(
    a: FloatArray, block: int, n_boot: int, rng: np.random.Generator, sr_hat: float
) -> FloatArray:
    """Signed studentized bootstrap statistics ``(SR*_m - sr_hat) / se*_m`` (no checks).

    Low-level helper for the Monte Carlo, which already has ``sr_hat``; non-finite
    entries are ``nan``.
    """
    t = a.size
    idx = circular_block_indices(t, block, n_boot, rng)
    sr_star, h = _projected_block_sums(a[idx], block)
    with np.errstate(divide="ignore", invalid="ignore"):
        se_star = np.sqrt(np.mean(h * h, axis=1) / t)
        tstar = (sr_star - sr_hat) / se_star
    tstar[~np.isfinite(tstar)] = np.nan
    return tstar


def block_se_sr(x: ArrayLike, block: int) -> float:
    """Block-sum (Goetze-Kuensch) standard error of the SR on the data themselves.

    With ``block = 1`` this is the delta-method SE with the sample covariance of
    ``(r, r^2)`` (LW 2008 footnote 9). Exposed for tests.
    """
    a = as_float_1d(x, "x", min_len=2)
    _, h = _projected_block_sums(a[None, :], _check_block(a.size, block))
    return float(np.sqrt(np.mean(h * h) / a.size))


def studentized_diff_boot(
    x: ArrayLike,
    block: int,
    n_boot: int,
    seed: int | np.random.Generator | None = None,
    *,
    prewhiten: bool = True,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Joint studentized bootstrap of SR differences against column 0 of a (T x k) matrix.

    Returns ``(delta_hat, se_hat, t_star)`` with shapes ``(k-1,)``, ``(k-1,)`` and
    ``(n_boot, k-1)``, where ``delta_j = SR_j - SR_0``. All columns share the resampled
    time indices, so ``t_star`` feeds the Romano-Wolf stepdown directly. ``se_hat`` is the
    Ledoit-Wolf HAC standard error (QS, Andrews bandwidth, ``T/(T-4)``; prewhitened by
    default) of each pair on the original data.
    """
    from sharpecheck.compare import lw_hac_se

    a = as_float_2d(x, "x", min_rows=4)
    t, k = a.shape
    if k < 2:
        raise ValueError("x needs at least two columns (benchmark first)")
    b = _check_block(t, block)
    rng = np.random.default_rng(seed)
    delta = np.empty(k - 1)
    se = np.empty(k - 1)
    for j in range(1, k):
        delta[j - 1], se[j - 1] = lw_hac_se(a[:, j], a[:, 0], prewhiten=prewhiten)
    return delta, se, diff_tstar(a, b, n_boot, rng, delta)


def diff_tstar(
    a: FloatArray, block: int, n_boot: int, rng: np.random.Generator, delta_hat: FloatArray
) -> FloatArray:
    """Signed studentized bootstrap statistics for ``SR_j - SR_0`` (no checks).

    ``a`` is (T x k) with the benchmark in column 0; returns (n_boot x k-1).
    """
    t = a.shape[0]
    idx = circular_block_indices(t, block, n_boot, rng)
    sr_star, h = _projected_block_sums(a[idx], block)  # (B, k), (B, l, k)
    dstar = sr_star[:, 1:] - sr_star[:, :1]
    hd = h[:, :, 1:] - h[:, :, :1]
    with np.errstate(divide="ignore", invalid="ignore"):
        se_star = np.sqrt(np.mean(hd * hd, axis=1) / t)
        tstar = (dstar - np.asarray(delta_hat)[None, :]) / se_star
    tstar[~np.isfinite(tstar)] = np.nan
    return tstar


def _stationary_indices(nobs: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One stationary-bootstrap index vector (Politis and Romano 1994), geometric blocks."""
    out = np.empty(nobs, dtype=np.int64)
    p = 1.0 / mean_block
    pos = int(rng.integers(nobs))
    for i in range(nobs):
        if i > 0 and rng.random() < p:
            pos = int(rng.integers(nobs))
        out[i] = pos % nobs
        pos += 1
    return out


def calibrate_block(
    x: ArrayLike,
    candidates: tuple[int, ...] = (1, 2, 4, 6, 8, 10),
    *,
    level: float = 0.95,
    n_pseudo: int = 1000,
    n_boot: int = 499,
    seed: int | None = None,
) -> tuple[int, dict[int, float]]:
    """Ledoit-Wolf (2008) Algorithm 3.1 block-size calibration for a (T x 2) pair.

    Pseudo data: VAR(1) fit with stationary-bootstrap residuals (mean block 5), started at
    an observed row. For each candidate ``b`` the symmetric studentized bootstrap interval
    for ``SR_1 - SR_2`` is computed on every pseudo series, and ``g(b)`` is the fraction
    covering the pseudo-truth ``Delta_hat`` of the original data. Returns the ``b`` that
    minimises ``|g(b) - level|`` and ``g``. Expensive: ``n_pseudo x len(candidates) x
    n_boot`` resamples; for empirical applications only, never inside the Monte Carlo.
    """
    from sharpecheck.compare import lw_hac_se

    check_level(level)
    a = as_float_2d(x, "x", min_rows=10)
    if a.shape[1] != 2:
        raise ValueError("calibrate_block expects a (T x 2) pair")
    t = a.shape[0]
    rng = np.random.default_rng(seed)
    delta_hat, _ = lw_hac_se(a[:, 0], a[:, 1], prewhiten=True)
    mu = a.mean(axis=0)
    y0, y1 = a[:-1] - mu, a[1:] - mu
    coef, *_ = np.linalg.lstsq(y0, y1, rcond=None)
    amat = coef.T
    resid = y1 - y0 @ amat.T
    hits = {b: 0 for b in candidates}
    for _ in range(n_pseudo):
        e = resid[_stationary_indices(t - 1, 5.0, rng)]
        z = np.empty_like(a)
        z[0] = a[int(rng.integers(t))] - mu
        for i in range(1, t):
            z[i] = amat @ z[i - 1] + e[i - 1]
        pseudo = z + mu
        # Benchmark in column 0, so delta = SR(col 1) - SR(col 0); flip to SR_1 - SR_2.
        pair = pseudo[:, ::-1]
        for b in candidates:
            if b > t:
                continue
            d, s, ts = studentized_diff_boot(pair, b, n_boot, rng)
            ts = np.abs(ts[:, 0][np.isfinite(ts[:, 0])])
            if ts.size == 0 or not np.isfinite(s[0]):
                continue
            q = float(np.quantile(ts, level))
            hits[b] += int(abs(d[0] - delta_hat) <= q * s[0])
    g = {b: hits[b] / n_pseudo for b in candidates}
    best = min(candidates, key=lambda b: (abs(g[b] - level), b))
    return best, g
