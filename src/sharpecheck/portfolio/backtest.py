"""Rolling-window backtest and honest inference against 1/N (milestone M7). DESIGN 5.

Weights estimated on months t-M..t-1 are applied to month t (monthly rebalancing,
M = 120 as in DeMiguel, Garlappi and Uppal 2009). Outputs per dataset x rule: monthly
out-of-sample excess returns, turnover, and the SR difference tests against 1/N; across all
comparisons, Romano-Wolf adjusted p-values from one joint block bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sharpecheck.portfolio.weights import Rule, weights


def rolling_backtest(
    excess: pd.DataFrame, rules: tuple[Rule, ...], window: int = 120
) -> dict[Rule, pd.DataFrame]:
    """Out-of-sample monthly returns and weights for each rule.

    Returns, per rule, a frame indexed by the out-of-sample months with a ``return``
    column (the portfolio excess return ``w_t' r_t``) followed by one weight column per
    asset (weights chosen from the ``window`` months before ``t``).
    """
    if excess.isna().to_numpy().any():
        raise ValueError("excess returns contain missing values")
    x = excess.to_numpy(dtype=float)
    t_total, n = x.shape
    if t_total <= window:
        raise ValueError(f"need more than {window} months, got {t_total}")
    out: dict[Rule, pd.DataFrame] = {}
    idx = excess.index[window:]
    for rule in rules:
        w = np.empty((t_total - window, n))
        for i, t in enumerate(range(window, t_total)):
            w[i] = weights(rule, x[t - window : t])
        ret = np.einsum("ij,ij->i", w, x[window:])
        df = pd.DataFrame(w, index=idx, columns=list(excess.columns))
        df.insert(0, "return", ret)
        out[rule] = df
    return out


def turnover(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.Series:
    """Monthly turnover sum_i |w_{i,t+1} - w_{i,t+}| with drifted pre-rebalance weights.

    ``w_{t+} = w_t (1 + r_t) / (1 + w_t' r_t)``. The drift uses the returns passed in (the
    study passes excess returns, a close approximation for monthly data). The first month
    has no previous weights and is omitted.
    """
    w = weights.to_numpy(dtype=float)
    r = asset_returns.loc[weights.index, weights.columns].to_numpy(dtype=float)
    grown = w * (1 + r)
    drifted = grown / grown.sum(axis=1, keepdims=True)
    to = np.abs(w[1:] - drifted[:-1]).sum(axis=1)
    return pd.Series(to, index=weights.index[1:], name="turnover")
