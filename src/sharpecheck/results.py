"""Result containers shared by the inference functions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Alternative = Literal["two-sided", "less", "greater"]


@dataclass(frozen=True)
class SharpeInterval:
    """A confidence interval for one per-period Sharpe ratio (or a difference of two).

    ``lower``/``upper`` may be ``-inf``/``inf`` for one-sided intervals. ``se`` is the
    standard error used by the method (``nan`` for methods without one, such as the exact
    noncentral-t interval). ``details["reason"]`` explains a ``nan`` interval.
    """

    method: str
    estimate: float
    lower: float
    upper: float
    level: float
    se: float
    nobs: int
    details: Mapping[str, Any] = field(default_factory=dict)

    def covers(self, value: float) -> bool:
        """True if ``value`` lies in the closed interval. A ``nan`` interval never covers."""
        if math.isnan(self.lower) or math.isnan(self.upper):
            return False
        return self.lower <= value <= self.upper

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for JSON/CSV output."""
        return {
            "method": self.method,
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "level": self.level,
            "se": self.se,
            "nobs": self.nobs,
            **{f"details.{k}": v for k, v in self.details.items()},
        }


@dataclass(frozen=True)
class TestResult:
    """Outcome of one hypothesis test (single Sharpe ratio, difference, or DSR).

    A ``pvalue`` of ``nan`` means the test is undefined for this input;
    ``details["reason"]`` then says why.
    """

    __test__ = False  # not a pytest test class

    test: str
    statistic: float
    pvalue: float
    nobs: int
    alternative: Alternative = "two-sided"
    details: Mapping[str, Any] = field(default_factory=dict)

    def reject(self, level: float = 0.05) -> bool:
        """True if the null is rejected at ``level``. Undefined tests never reject."""
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        return (not math.isnan(self.pvalue)) and self.pvalue < level

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for JSON/CSV output."""
        return {
            "test": self.test,
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "nobs": self.nobs,
            "alternative": self.alternative,
            **{f"details.{k}": v for k, v in self.details.items()},
        }
