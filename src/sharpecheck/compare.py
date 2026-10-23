"""Tests for the difference of two Sharpe ratios (milestone M3).

- ``jkm_test``      Jobson and Korkie (1981) with the Memmel (2003) correction (iid normal)
- ``lw_hac_test``   Ledoit and Wolf (2008) section 3.1: delta method, QS kernel, optional
                    prewhitening, T/(T-4) adjustment
- ``lw_boot_test``  Ledoit and Wolf (2008) section 3.2: studentized circular block bootstrap,
                    p-value by eq. (9); ``block=1`` is their Boot-IID variant

Formulas: DESIGN 4.5. All tests are two-sided tests of H0: SR_1 = SR_2 on paired series.
"""

from __future__ import annotations

from numpy.typing import ArrayLike

from sharpecheck.results import TestResult


def jkm_test(x1: ArrayLike, x2: ArrayLike) -> TestResult:
    """Jobson-Korkie-Memmel test of equal Sharpe ratios."""
    raise NotImplementedError("M3")


def lw_hac_test(x1: ArrayLike, x2: ArrayLike, *, prewhiten: bool = True) -> TestResult:
    """Ledoit-Wolf HAC test of equal Sharpe ratios."""
    raise NotImplementedError("M3")


def lw_boot_test(
    x1: ArrayLike,
    x2: ArrayLike,
    *,
    block: int | None = None,
    n_boot: int = 4999,
    seed: int | None = None,
) -> TestResult:
    """Ledoit-Wolf studentized bootstrap test of equal Sharpe ratios.

    ``block=None`` uses :func:`sharpecheck.bootstrap.default_block`.
    ``details`` carries the symmetric bootstrap confidence interval for the difference.
    """
    raise NotImplementedError("M3")
