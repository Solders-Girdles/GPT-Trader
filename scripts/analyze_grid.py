#!/usr/bin/env python3
import sys

import pandas as pd

if len(sys.argv) < 2:
    print("Usage: scripts/analyze_grid.py data/backtests/grid_results_*.csv")
    sys.exit(1)

df = pd.concat([pd.read_csv(p) for p in sys.argv[1:]], ignore_index=True)

# coerce numeric
num_cols = [
    "atr_k",
    "risk_pct",
    "max_positions",
    "entry_confirm",
    "min_rebalance_pct",
    "total_return",
    "cagr",
    "sharpe",
    "max_drawdown",
    "total_costs",
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("\n== Top 10 by Sharpe ==")
print(
    df.sort_values("sharpe", ascending=False)
    .head(10)[
        [
            "atr_k",
            "risk_pct",
            "max_positions",
            "entry_confirm",
            "min_rebalance_pct",
            "sharpe",
            "cagr",
            "total_return",
            "max_drawdown",
            "total_costs",
        ]
    ]
    .to_string(index=False)
)

print("\n== Top 10 by CAGR (Sharpe≥0.75, MaxDD≤12%) ==")
mask = (df["sharpe"] >= 0.75) & (df["max_drawdown"] <= 0.12)
print(
    df[mask]
    .sort_values("cagr", ascending=False)
    .head(10)[
        [
            "atr_k",
            "risk_pct",
            "max_positions",
            "entry_confirm",
            "min_rebalance_pct",
            "cagr",
            "sharpe",
            "max_drawdown",
            "total_costs",
        ]
    ]
    .to_string(index=False)
)

print("\n== Lowest Costs (Top-30 Sharpe subset) ==")
top30 = df.sort_values("sharpe", ascending=False).head(30)
print(
    top30.sort_values("total_costs")
    .head(10)[
        [
            "atr_k",
            "risk_pct",
            "max_positions",
            "entry_confirm",
            "min_rebalance_pct",
            "total_costs",
            "sharpe",
            "cagr",
        ]
    ]
    .to_string(index=False)
)
