# GPT-Trader

Algorithmic trading research framework in Python, designed for **rapid iteration**, **backtesting**, and **parameter optimization**.

---

## 🚀 Features

### Strategies
- Modular architecture — easily add new strategies (e.g., `trend_breakout`)
- Configurable parameters for entry confirmation, ATR multipliers, position sizing, and rebalancing thresholds

### Backtesting
- ATR-based position sizing
- Risk % control & maximum open positions
- Optional entry confirmation rules

### Regime Filters
- Market regime detection for conditional strategy activation

### Exits
- Signal-based exits
- ATR stop-based exits

### Optimization
- Parameter grid search with **sharding** for parallel execution
- In-sample (IS) vs out-of-sample (OOS) testing workflows
- Quick top-IS preview

### Data
- Historical data via [`yfinance`](https://pypi.org/project/yfinance/)
- Outputs to `.csv`, `.png`, and `.txt` summary formats in `data/backtests/` and `data/opt/`

### Tooling & QA
- Poetry for dependency management
- Ruff, Black, MyPy for linting, formatting, and type checking
- Pytest for testing
- `pre-commit` hooks to enforce standards

---

## 📦 Installation

    poetry install
    pre-commit install

---

## 📊 Usage

1. **Backtest a single run**

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

2. **Grid Search Optimization**

       poetry run python scripts/run_is_oos.py \
         --is-start 2010-01-01 --is-end 2018-12-31 \
         --oos-start 2019-01-01 --oos-end 2024-12-31

3. **Parallel Grid Search (Sharded)**

       mkdir -p logs
       for k in 0 1 2 3; do
         poetry run python scripts/run_is_oos.py \
           --chunks 4 --chunk $k \
           --is-start 2010-01-01 --is-end 2018-12-31 \
           --oos-start 2019-01-01 --oos-end 2024-12-31 \
           > logs/is_part_$k.log 2>&1 &
       done

---

## 📂 Project Layout

    src/bot/
      cli.py                  # Entrypoints for backtest
      config.py               # Global settings
      logging.py              # Logging setup

      backtest/
        engine.py             # Core backtest logic
        engine_portfolio.py   # Portfolio simulation

      dataflow/
        base.py
        sources/
          yfinance_source.py  # Historical data loader

      strategy/
        base.py
        trend_breakout.py     # ATR + Donchian channel breakout

      risk/
        basic.py              # Risk management rules

      exec/
        base.py
        alpaca_paper.py

    scripts/
      run_is_oos.py           # IS/OOS grid search runner
      ...

---

## 📈 Outputs

Example backtest output files:
- `PORT_<strategy>_<timestamp>.csv` — equity curve & positions
- `PORT_<strategy>_<timestamp>.png` — performance chart
- `PORT_<strategy>_<timestamp>_summary.csv` — performance metrics
- `PORT_<strategy>_<timestamp>_trades.csv` — trade-by-trade log

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m 'Add my feature'`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📜 License

MIT License. See `LICENSE` for details.
