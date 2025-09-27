# Live Paper Trading Monitoring Guide

## Overview

We've implemented a comprehensive live monitoring system for paper trading that provides real-time updates and visualization of your trading sessions.

## Components

### 1. Live Paper Trading Script (`paper_trade_live.py`)
Runs paper trading sessions with continuous file updates for monitoring.

**Features:**
- Real-time session data updates every 2 seconds
- Automatic position and P&L tracking
- Strategy performance metrics
- Trade logging with timestamps

**Usage:**
```bash
python scripts/paper_trade_live.py \
    --strategy momentum \
    --duration 30 \
    --capital 10000 \
    --symbols BTC-USD,ETH-USD,SOL-USD
```

**Options:**
- `--strategy`: Choose from momentum, mean_reversion, breakout, ma_crossover, volatility
- `--duration`: Session duration in minutes
- `--capital`: Initial capital (default: $10,000)
- `--symbols`: Comma-separated list of symbols to trade

### 2. Terminal Monitor (`live_monitor.py`)
Console-based live monitoring with auto-refresh.

**Features:**
- Real-time metrics display
- Open positions tracking
- Recent trades log
- Equity curve visualization (text-based)
- Auto-updates every 2 seconds

**Usage:**
```bash
python scripts/live_monitor.py
```

**Controls:**
- `q`: Quit monitor
- `r`: Force refresh
- Updates automatically every 2 seconds

### 3. Web Dashboard (`dashboard.html` + `dashboard_server.py`)
Beautiful web-based dashboard with charts and real-time updates.

**Features:**
- Live equity chart (Chart.js)
- Performance metrics cards
- Open positions table with P&L
- Recent trades history
- Auto-refresh every 3 seconds
- Responsive design

**Usage:**
```bash
# Start the dashboard server
python scripts/dashboard_server.py --port 8888

# Open in browser
http://localhost:8888
```

## How to Use

### Option 1: Terminal Monitoring

**Step 1:** Start paper trading in one terminal
```bash
python scripts/paper_trade_live.py --strategy momentum --duration 60
```

**Step 2:** Start monitor in another terminal
```bash
python scripts/live_monitor.py
```

### Option 2: Web Dashboard

**Step 1:** Start the dashboard server
```bash
python scripts/dashboard_server.py
```

**Step 2:** Open browser to http://localhost:8888

**Step 3:** Start paper trading in another terminal
```bash
python scripts/paper_trade_live.py --strategy mean_reversion --duration 30
```

## Monitoring Features

### Real-Time Metrics
- **Equity**: Current account value
- **Total Return**: Percentage gain/loss
- **Cash**: Available cash balance
- **Positions Value**: Value of open positions
- **Win Rate**: Percentage of profitable trades
- **Drawdown**: Maximum peak-to-trough decline
- **Number of Trades**: Total executed trades
- **Number of Positions**: Currently open positions

### Position Tracking
- Symbol and quantity held
- Entry price vs current price
- Real-time P&L calculation
- Percentage gain/loss per position

### Trade History
- Timestamp of execution
- Buy/Sell side indication
- Execution price and quantity
- P&L for closed trades
- Strategy that triggered the trade

### Visual Indicators
- 🟢 Green for profits/buys
- 🔴 Red for losses/sells
- 📊 Charts for equity progression
- ⚡ Live status indicators

## Session Data Storage

All session data is saved to `/results/` directory:
- `live_YYYYMMDD_HHMMSS.json`: Live session data with continuous updates
- Contains metrics, trades, positions, and equity history
- Updates every 2 seconds during active sessions

## Example Session Output

### Terminal Monitor
```
📊 LIVE PAPER TRADING MONITOR
Session Time: 0:15:30        Last Update: 14:32:45

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Equity: $10,245.67           Return: +2.46%
Cash: $5,123.45              Positions: $5,122.22
Trades: 8                    Win Rate: 62.5%     Drawdown: 1.23%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPEN POSITIONS (3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Symbol      Qty        Entry      Current    P&L
BTC-USD     0.021656   $114,750   $115,230   $10.39 (+0.4%)
ETH-USD     0.527831   $4,740     $4,755     $7.92 (+0.3%)
```

### Web Dashboard
- Interactive equity chart showing account value over time
- Color-coded metrics cards (green for positive, red for negative)
- Sortable tables for positions and trades
- Auto-refreshing every 3 seconds
- Mobile-responsive design

## Tips for Effective Monitoring

1. **Multiple Monitors**: Run both terminal and web monitoring for different perspectives
2. **Strategy Comparison**: Run different strategies in sequence and compare results
3. **Market Hours**: Best results during active market hours
4. **Position Limits**: Monitor position count to ensure diversification
5. **Risk Metrics**: Pay attention to drawdown and win rate

## Troubleshooting

### No Data Showing
- Ensure paper trading session is running
- Check that results directory exists
- Verify file permissions

### Dashboard Not Loading
- Check server is running on correct port
- Try different port if 8888 is in use
- Ensure dashboard.html exists in scripts directory

### Monitor Not Updating
- Check for active paper trading session
- Verify results files are being created
- Restart monitor if needed

## Performance Considerations

- **File I/O**: Session data writes every 2 seconds (minimal impact)
- **Memory**: Keeps last 100 equity points and 50 trades in memory
- **CPU**: Very low usage (~1-2%)
- **Network**: Dashboard uses local server only

## Next Steps

1. **Extended Sessions**: Run 24-hour paper trading sessions
2. **Strategy Optimization**: Use monitoring to identify best parameters
3. **Alert System**: Add notifications for significant events
4. **Historical Analysis**: Build database from session files
5. **Multi-Strategy**: Run multiple strategies simultaneously

## Summary

The live monitoring system provides comprehensive real-time visibility into paper trading performance with:
- ✅ Terminal-based monitoring
- ✅ Web dashboard with charts
- ✅ Real-time position tracking
- ✅ Performance metrics
- ✅ Trade history
- ✅ Auto-refresh capabilities

This allows you to track paper trading sessions in real-time and make informed decisions about strategy performance before risking real capital.

---

*Created: 2025-08-24*
*Version: 1.0*
*Status: Fully Operational*