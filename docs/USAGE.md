# Usage Guide

This document explains how to **run backtests**, **perform parameter optimizations**, and **review results** using GPT-Trader.

---

## 1. Backtesting a Strategy

Example: Run `trend_breakout` on the S&P 100 universe.

```bash
poetry run python -m bot.cli backtest \
  --strategy trend_breakout \
  --symbol-list data/universe/sp100.csv \
  --start 2019-01-01 --end 2020-12-31 \
  --regime on \
  --cost-bps 5 \
  --exit-mode stop \
  --entry-confirm 2 \
  --min-rebalance-pct 0.002 \
  --atr-k 2.3
