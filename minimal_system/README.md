# Minimal Trading System

## Why This Exists

After auditing the main GPT-Trader codebase, we discovered:
- 7 different orchestrators doing the same job
- 21 execution engines
- 228 dead modules (70% of code never used)
- Only 28% test coverage
- Suspicious results we couldn't trust

This minimal system is a clean slate - built from scratch with clarity as the #1 priority.

## What This Is

A complete trading system in ~500 lines of clean, tested Python:
- **data.py** - Simple YFinance data loader with caching
- **strategy.py** - ONE moving average crossover strategy
- **executor.py** - Clear position management
- **ledger.py** - Sensible trade recording (transactions AND completed trades)
- **backtest.py** - Basic backtesting engine
- **test_all.py** - 100% test coverage

## Key Principles

1. **Clarity over features** - Every line should be obvious
2. **Test everything** - 18 tests covering all components
3. **No hidden complexity** - No magic, no surprises
4. **Correct accounting** - Track both transactions and completed trades
5. **Trust through transparency** - You can understand everything

## Quick Start

```bash
# Run tests (should be 100% pass)
python test_all.py

# Run demo with real data
python demo.py
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