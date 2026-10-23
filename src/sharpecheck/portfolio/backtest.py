"""Rolling-window backtest and honest inference against 1/N (milestone M7). DESIGN 5.

Weights estimated on months t-M..t-1 are applied to month t (monthly rebalancing,
M = 120 as in DeMiguel, Garlappi and Uppal 2009). Outputs per dataset x rule: monthly
out-of-sample excess returns, turnover, and the SR difference tests against 1/N; across all
comparisons, Romano-Wolf adjusted p-values from one joint block bootstrap.
"""

from __future__ import annotations

import pandas as pd

from sharpecheck.portfolio.weights import Rule


def rolling_backtest(
    excess: pd.DataFrame, rules: tuple[Rule, ...], window: int = 120
) -> dict[Rule, pd.DataFrame]:
    """Out-of-sample monthly returns and weights for each rule."""
    raise NotImplementedError("M7")


def turnover(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.Series:
    """Monthly turnover sum_i |w_{i,t+1} - w_{i,t+}| with drifted pre-rebalance weights."""
    raise NotImplementedError("M7")
