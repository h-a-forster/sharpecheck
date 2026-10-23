"""Multiple-testing control (milestone M4).

- Holm (1979) and Benjamini-Hochberg (1995) adjusted p-values
- Romano-Wolf stepdown adjusted p-values from a matrix of bootstrap null statistics,
  using the algorithm of Romano and Wolf (2016), on studentized statistics
- thin wrapper around ``arch.bootstrap.MCS`` for the model confidence set on losses

Formulas: DESIGN 4.7.
"""

from __future__ import annotations

from typing import Any

from numpy.typing import ArrayLike

from sharpecheck._validation import FloatArray


def holm(pvalues: ArrayLike) -> FloatArray:
    """Holm step-down adjusted p-values (FWER)."""
    raise NotImplementedError("M4")


def benjamini_hochberg(pvalues: ArrayLike) -> FloatArray:
    """Benjamini-Hochberg step-up adjusted p-values (FDR)."""
    raise NotImplementedError("M4")


def romano_wolf(t_obs: ArrayLike, t_star: ArrayLike) -> FloatArray:
    """Romano-Wolf stepdown adjusted p-values for two-sided tests.

    ``t_obs`` (k,) are observed studentized statistics, ``t_star`` (B x k) the centred
    studentized bootstrap statistics from one shared resampling scheme. Uses absolute
    values; adjusted p-values are monotone in |t_obs| and use (1 + count)/(B + 1).
    """
    raise NotImplementedError("M4")


def model_confidence_set(
    losses: ArrayLike, *, size: float = 0.1, block: int = 6, reps: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Hansen-Lunde-Nason MCS (``arch.bootstrap.MCS``, max statistic) on a (T x m) loss matrix."""
    raise NotImplementedError("M7 (should)")
