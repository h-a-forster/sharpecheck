"""Multiple-testing control (milestones M4, M7).

- Holm (1979) and Benjamini-Hochberg (1995) adjusted p-values
- Romano-Wolf stepdown adjusted p-values from a matrix of bootstrap null statistics,
  using the algorithm of Romano and Wolf (2016), on studentized statistics
- thin wrapper around ``arch.bootstrap.MCS`` for the model confidence set on losses

Formulas: DESIGN 4.7.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from sharpecheck._validation import FloatArray


def _pvals(pvalues: ArrayLike) -> FloatArray:
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("pvalues must be 1-D")
    if np.any(np.isnan(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must lie in [0, 1] and not be NaN")
    return p


def holm(pvalues: ArrayLike) -> FloatArray:
    """Holm step-down adjusted p-values (FWER)."""
    p = _pvals(pvalues)
    k = p.size
    if k == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    adj = np.maximum.accumulate(np.minimum(1.0, (k - np.arange(k)) * p[order]))
    out = np.empty(k)
    out[order] = adj
    return out


def benjamini_hochberg(pvalues: ArrayLike) -> FloatArray:
    """Benjamini-Hochberg step-up adjusted p-values (FDR)."""
    p = _pvals(pvalues)
    k = p.size
    if k == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    adj = np.minimum(1.0, k * p[order] / np.arange(1, k + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(k)
    out[order] = adj
    return out


def romano_wolf(t_obs: ArrayLike, t_star: ArrayLike) -> FloatArray:
    """Romano-Wolf stepdown adjusted p-values for two-sided tests.

    ``t_obs`` (k,) are observed studentized statistics, ``t_star`` (B x k) the centred
    studentized bootstrap statistics from one shared resampling scheme. Uses absolute
    values; adjusted p-values are monotone in |t_obs| and use (1 + count)/(B + 1).
    Non-finite bootstrap statistics are treated as 0 (they never exceed a threshold).
    """
    t = np.abs(np.asarray(t_obs, dtype=np.float64))
    ts = np.abs(np.asarray(t_star, dtype=np.float64))
    if t.ndim != 1 or ts.ndim != 2 or ts.shape[1] != t.size:
        raise ValueError(f"shapes must be (k,) and (B, k), got {t.shape} and {ts.shape}")
    if not np.all(np.isfinite(t)):
        raise ValueError("t_obs must be finite")
    ts = np.where(np.isfinite(ts), ts, 0.0)
    b = ts.shape[0]
    order = np.argsort(-t, kind="stable")
    # Max over the remaining hypotheses r_s, ..., r_k: reverse cumulative max.
    tail_max = np.maximum.accumulate(ts[:, order][:, ::-1], axis=1)[:, ::-1]
    raw = (1 + np.sum(tail_max >= t[order][None, :], axis=0)) / (b + 1)
    adj = np.maximum.accumulate(raw)
    out = np.empty(t.size)
    out[order] = adj
    return out


def model_confidence_set(
    losses: ArrayLike, *, size: float = 0.1, block: int = 6, reps: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Hansen-Lunde-Nason MCS (``arch.bootstrap.MCS``, max statistic) on a (T x m) loss matrix.

    Returns ``{"included": [...], "excluded": [...], "pvalues": {model: p}}`` with column
    indices (or names for a DataFrame).
    """
    from arch.bootstrap import MCS

    mcs = MCS(
        losses, size, reps=reps, block_size=block, method="max", bootstrap="stationary", seed=seed
    )
    mcs.compute()
    pv = mcs.pvalues.iloc[:, 0]
    return {
        "included": list(mcs.included),
        "excluded": list(mcs.excluded),
        "pvalues": {k: float(v) for k, v in pv.items()},
    }
