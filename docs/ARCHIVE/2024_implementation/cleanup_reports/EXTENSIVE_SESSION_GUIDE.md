# Extensive Paper Trading Session Guide

## Quick Start

### Option 1: Automated Launch (Recommended)
```bash
# Make script executable (first time only)
chmod +x scripts/start_paper_trading.sh

# Run the launcher
./scripts/start_paper_trading.sh
```

Choose from:
1. **Quick Test** - 5 minutes per strategy
2. **Standard Session** - 30 minutes per strategy (2.5 hours total)
3. **Extended Session** - 60 minutes per strategy (5 hours total)
4. **Custom Single Strategy** - Your choice
5. **Dashboard Only** - Just monitoring

### Option 2: Manual Setup

**Terminal 1 - Dashboard:**
```bash
python scripts/dashboard_server.py
# Open browser to http://localhost:8888
```

**Terminal 2 - Paper Trading:**
```bash
# Single strategy
python scripts/paper_trade_live.py \
    --strategy momentum \
    --duration 60 \
    --symbols BTC-USD,ETH-USD,SOL-USD,LINK-USD

# Or run all strategies in rotation
python scripts/run_extensive_session.py
```

**Terminal 3 - Monitor (Optional):**
```bash
python scripts/live_monitor.py
```

## Session Configurations

### Quick Test (25 minutes total)
- **Purpose**: Verify everything works
- **Duration**: 5 minutes per strategy
- **Strategies**: All 5 (momentum, mean_reversion, breakout, ma_crossover, volatility)
- **Best for**: Initial testing, debugging

### Standard Session (2.5 hours)
- **Purpose**: Gather meaningful data
- **Duration**: 30 minutes per strategy
- **Strategies**: All 5 strategies
- **Best for**: Daily testing, parameter tuning

### Extended Session (5 hours)
- **Purpose**: Comprehensive testing
- **Duration**: 60 minutes per strategy
- **Strategies**: All 5 strategies
- **Best for**: Weekend runs, final validation

### 24-Hour Marathon
```bash
# Run single strategy for 24 hours
python scripts/paper_trade_live.py \
    --strategy momentum \
    --duration 1440 \
    --symbols BTC-USD,ETH-USD,SOL-USD,LINK-USD,MATIC-USD,AVAX-USD
```

## Monitoring Your Session

### 1. Web Dashboard (Best Visual)
- **URL**: http://localhost:8888
- **Updates**: Every 3 seconds
- **Features**: Charts, metrics cards, position tracking
- **Best for**: Visual learners, presentations

### 2. Terminal Monitor (Low Resource)
- **Command**: `python scripts/live_monitor.py`
- **Updates**: Every 2 seconds
- **Features**: Colored text UI, compact display
- **Best for**: Server environments, SSH sessions

### 3. Status Check (Quick Snapshot)
- **Command**: `python scripts/check_status.py`
- **Purpose**: One-time status check
- **Best for**: Quick verification

### 4. Results Analysis (Post-Session)
- **Command**: `python scripts/monitor_paper_trading.py`
- **Purpose**: Analyze completed sessions
- **Best for**: Performance review

## Recommended Symbols by Market

### Crypto Majors (Most Liquid)
```
BTC-USD,ETH-USD,SOL-USD
```

### Extended Crypto
```
BTC-USD,ETH-USD,SOL-USD,LINK-USD,MATIC-USD,AVAX-USD
```

### DeFi Focus
```
UNI-USD,AAVE-USD,SUSHI-USD,COMP-USD
```

### Layer 2 Focus
```
MATIC-USD,ARB-USD,OP-USD
```

## Strategy Performance Expectations

### Momentum Strategy
- **Best Market**: Trending markets
- **Expected Win Rate**: 40-60%
- **Trade Frequency**: Moderate
- **Risk Level**: Medium

### Mean Reversion
- **Best Market**: Range-bound markets
- **Expected Win Rate**: 50-70%
- **Trade Frequency**: High
- **Risk Level**: Low-Medium

### Breakout
- **Best Market**: Volatile markets
- **Expected Win Rate**: 35-50%
- **Trade Frequency**: Low
- **Risk Level**: High

### MA Crossover
- **Best Market**: Trending markets
- **Expected Win Rate**: 45-55%
- **Trade Frequency**: Low-Moderate
- **Risk Level**: Medium

### Volatility
- **Best Market**: Changing volatility
- **Expected Win Rate**: 45-60%
- **Trade Frequency**: Moderate
- **Risk Level**: Medium

## Session Management

### Starting a Session
1. **Check market status** - Best during active trading hours
2. **Start dashboard** - For monitoring
3. **Launch paper trading** - Choose strategy and duration
4. **Verify connection** - Check status script
5. **Monitor progress** - Watch dashboard

### During the Session
- Check dashboard every 15-30 minutes
- Note any unusual behavior
- Don't interrupt unless necessary
- Let strategies run their course

### Ending a Session
- Sessions auto-close positions at end
- Results saved automatically
- Review performance metrics
- Compare strategy results

## Data Collection Goals

### Minimum Viable Data
- **Trades**: 20+ per strategy
- **Duration**: 30+ minutes per strategy
- **Market Conditions**: Various (trending, range-bound)

### Statistical Significance
- **Trades**: 50+ per strategy
- **Duration**: 2+ hours per strategy
- **Sessions**: 5+ different sessions

### Production Ready
- **Trades**: 100+ per strategy
- **Duration**: 10+ hours per strategy
- **Market Conditions**: Bull, bear, and sideways
- **Time Period**: 30+ days

## Troubleshooting

### Session Not Starting
```bash
# Check if ports are in use
lsof -i :8888

# Kill existing processes
pkill -f dashboard_server
pkill -f paper_trade_live
```

### No Data Showing
```bash
# Check results directory
ls -la results/

# Check latest file
python scripts/check_status.py
```

### Dashboard Not Updating
```bash
# Restart dashboard
pkill -f dashboard_server
python scripts/dashboard_server.py
```

## Performance Benchmarks

### Good Performance Indicators
- ✅ Positive returns over 2+ hours
- ✅ Win rate > 50%
- ✅ Drawdown < 5%
- ✅ Consistent trade execution
- ✅ Profit factor > 1.2

### Warning Signs
- ⚠️ Win rate < 40%
- ⚠️ Drawdown > 10%
- ⚠️ No trades for 30+ minutes
- ⚠️ Profit factor < 1.0
- ⚠️ Excessive position concentration

## Next Steps After Sessions

1. **Analyze Results**
   ```bash
   python scripts/monitor_paper_trading.py
   ```

2. **Compare Strategies**
   - Identify best performers
   - Note market conditions
   - Review trade patterns

3. **Optimize Parameters**
   - Adjust based on performance
   - Test new configurations
   - Document changes

4. **Build Confidence**
   - Run multiple sessions
   - Different market conditions
   - Consistent profitability

5. **Prepare for Live Trading**
   - Only after 30+ days paper trading
   - Positive returns demonstrated
   - Risk management proven

## Session Checklist

### Before Starting
- [ ] Check Coinbase connection
- [ ] Verify market hours
- [ ] Choose appropriate duration
- [ ] Start monitoring tools
- [ ] Clear old result files (optional)

### During Session
- [ ] Monitor dashboard periodically
- [ ] Check for errors
- [ ] Note market conditions
- [ ] Document observations

### After Session
- [ ] Review final metrics
- [ ] Save important results
- [ ] Compare strategy performance
- [ ] Plan next session

## Summary

You now have everything needed to run extensive paper trading sessions:

- **Automated launcher** for easy starts
- **Multiple monitoring options** for real-time tracking
- **Various session configurations** for different needs
- **Performance benchmarks** to measure success
- **Clear next steps** toward live trading

Remember: **More data = Better confidence**. Run sessions regularly to build a comprehensive performance history before considering live trading.

---

*Created: 2025-08-24*
*Version: 1.0*
*Status: Ready for Extensive Testing*