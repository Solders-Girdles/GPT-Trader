# Paper Trading Feature Slice

Simulated live trading with real-time data - completely self-contained.

## Quick Start

```python
from features.paper_trade import start_paper_trading, stop_paper_trading, get_status

# Start paper trading
start_paper_trading(
    strategy="SimpleMAStrategy",
    symbols=["AAPL", "MSFT", "GOOGL"],
    initial_capital=100000,
    fast_period=10,
    slow_period=30
)

# Check status
status = get_status()
print(status.summary())

# Stop and get results
results = stop_paper_trading()
print(results.summary())
```

## Complete Isolation

This slice has **zero external dependencies**. Everything needed is local:

- ✅ Local strategies (5 implementations)
- ✅ Local data feed
- ✅ Local risk management
- ✅ Local execution engine
- ✅ Local types and validation

## Module Structure

```
paper_trade/
├── paper_trade.py   # Main orchestration (300 lines)
├── strategies.py    # Local strategies (200 lines)
├── data.py         # Real-time data feed (150 lines)
├── execution.py    # Order execution (200 lines)
├── risk.py         # Risk management (120 lines)
└── types.py        # Local types (100 lines)
```

## Available Strategies

All strategies are implemented locally in `strategies.py`:

1. **SimpleMAStrategy** - Moving average crossover
2. **MomentumStrategy** - Momentum-based signals
3. **MeanReversionStrategy** - Bollinger band mean reversion
4. **VolatilityStrategy** - Low volatility trading
5. **BreakoutStrategy** - Price breakout detection

## Features

### Real-Time Data
- Fetches live market data during market hours
- Falls back to historical data when markets closed
- Maintains configurable lookback window
- Updates at configurable intervals

### Order Execution
- Simulates realistic order execution
- Applies commission and slippage
- Tracks positions and P&L
- Enforces position limits

### Risk Management
- Maximum position size limits
- Daily loss limits
- Maximum drawdown protection
- Minimum cash reserve requirements

### Performance Tracking
- Real-time equity curve
- Trade logging
- Performance metrics calculation
- Position tracking

## Configuration

```python
start_paper_trading(
    strategy="MomentumStrategy",
    symbols=["SPY", "QQQ"],
    initial_capital=50000,
    
    # Strategy parameters
    lookback=20,
    threshold=0.02,
    
    # Execution parameters
    commission=0.001,      # 0.1% commission
    slippage=0.0005,      # 0.05% slippage
    position_size=0.95,   # Use 95% of available capital
    max_positions=10,     # Max 10 concurrent positions
    
    # Update frequency
    update_interval=60    # Update every 60 seconds
)
```

## Token Efficiency

To use paper trading, agents only need to load this directory:
- **Token cost**: ~400 tokens for entire slice
- **No external dependencies**: Everything is local
- **Complete functionality**: Full paper trading system

## Example Output

```
Paper Trading Summary
====================
Duration: 2:15:30
Total Equity: $102,345.67
Total Return: 2.35%
Sharpe Ratio: 1.82
Max Drawdown: 3.24%
Win Rate: 65.00%
Total Trades: 20
Open Positions: 3
```

## Notes

- Paper trading runs in a background thread
- Only one session can be active at a time
- Positions are automatically closed when stopping
- All data is fetched from Yahoo Finance
- Market hours detection is simplified (doesn't handle holidays)