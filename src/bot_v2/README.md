# GPT-Trader V2: Vertical Slice Architecture

Important: This document describes the equities baseline slices used earlier in the project (backtests, paper/live trading via generic brokers, and yfinance-driven data). The active focus of this repository is Coinbase Perpetual Futures. For the current perps bot and operations, see:
- docs/reference/coinbase.md
- docs/reference/trading_logic_perps.md
- src/bot_v2/cli.py (entrypoint for the perps bot)

Legacy content below remains as a reference for the slice architecture and equities workflows.

## What This Is

An intelligent trading system built on **vertical slice architecture** optimized for AI development:
- **9 Feature Slices** - Complete isolation, no shared dependencies
- **ML Intelligence** - Strategy selection and market regime detection
- **Agent-First Design** - 92% token efficiency for AI navigation
- **8K Clean Lines** - Down from 159K lines (95% reduction)

## Architecture Overview

```
features/                    # ONLY source of truth
├── backtest/               # Historical testing
├── paper_trade/            # Simulated trading  
├── analyze/                # Market analysis
├── optimize/               # Parameter optimization
├── live_trade/             # Broker integration
├── monitor/                # Health monitoring
├── data/                   # Data management
├── ml_strategy/            # ML strategy selection ✅
└── market_regime/          # Regime detection ✅
```

## Type System

**Important**: The `live_trade` module uses core broker interfaces exclusively. Local types are retained only for non-core structures (AccountInfo, MarketHours, Bar, ExecutionReport). All Order, Position, and Quote types come from `brokerages.core.interfaces`.

## Key Principles

1. **Complete Isolation** - Each slice is self-contained (~500 tokens)
2. **Agent-First Navigation** - AI can load only what it needs
3. **No Shared Dependencies** - Duplication preferred over coupling
4. **ML Intelligence** - Adaptive trading based on market conditions
5. **Token Efficiency** - 92% reduction in context requirements

## Quick Start

```bash
# Run integration tests for vertical slices
pytest -m integration tests/integration/bot_v2 -q

# Run focused slice checks
pytest tests/integration/bot_v2/test_slice_isolation.py -q
pytest tests/integration/bot_v2/test_vertical_slice.py -q
pytest tests/integration/bot_v2/test_workflows.py -q
```

## Example Results

```
Symbol: AAPL
Period: 2024-01-01 to 2024-06-30
Initial Capital: $10,000.00
Final Value: $11,475.79
Total Return: 14.76%
Buy & Hold Return: 13.08%
Sharpe Ratio: 1.90
Max Drawdown: -4.18%
```

## Data Flow

```
YFinance Data
    ↓
Strategy (MA Crossover)
    ↓
Signals (Buy/Sell/Hold)
    ↓
Executor (Position Management)
    ↓
Ledger (Record Keeping)
    ↓
Results (P&L, Metrics)
```

## What Makes This Different

### Clear Trade Recording
- **Transactions**: Every buy/sell is recorded
- **Completed Trades**: Round-trips are tracked separately
- **Open Positions**: Always know what you're holding

### Transparent Calculations
- Portfolio value = Cash + (Shares × Current Price)
- P&L = (Exit Price - Entry Price) × Shares
- No mysterious calculations or hidden state

### Testable Design
- Every component has unit tests
- Integration tests verify the complete flow
- Mock data ensures predictable test results

## Building On This Foundation

This minimal system proves we can have:
- Working backtesting with realistic results
- Clear separation of concerns
- Proper trade accounting
- Trustworthy calculations

To expand:
1. Add more strategies (keep them simple)
2. Add risk management (position sizing, stop losses)
3. Add more symbols (portfolio management)
4. Add optimization (parameter tuning)

But always maintain:
- Clarity
- Tests
- Transparency

## The Lesson

Sometimes starting over with a minimal, correct implementation is better than trying to fix a complex, broken system. This 500-line system does more reliably than the 159,000-line main codebase.

**Trust is earned through understanding, not complexity.**
