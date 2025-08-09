from __future__ import annotations

import numpy as np
import pandas as pd


def perf_metrics(equity: pd.Series) -> dict[str, float]:
    equity = equity.dropna()
    ret = equity.pct_change().fillna(0.0)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    ann = 252
    cagr = (1.0 + total_return) ** (ann / max(1, len(equity))) - 1.0
    vol = ret.std() * np.sqrt(ann)
    sharpe = 0.0 if vol == 0 else (ret.mean() * ann) / vol
    roll_max = equity.cummax()
    dd = (roll_max - equity) / roll_max
    max_dd = dd.max()
    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }
