# Alpaca Paper Trading Setup Guide

This guide explains how to set up and use the safe Alpaca paper trading integration with GPT-Trader.

## Overview

The Alpaca paper trading integration provides:
- **Complete safety**: Multiple verification layers prevent accidental live trading
- **Real paper trading**: Uses actual Alpaca paper trading API
- **Order execution**: Market, limit, and stop orders with realistic slippage
- **Position tracking**: Real-time P&L calculation and monitoring
- **Trade logging**: Comprehensive audit trail for compliance
- **Risk management**: Built-in position and order size limits
- **System integration**: Works with existing strategies and orchestrator

## Safety Features

### 🛡️ Multi-Layer Safety Protection

1. **Environment Validation**
   - Requires `ALPACA_PAPER=true` environment variable
   - Validates API key formats for paper trading
   - Checks for missing credentials

2. **Paper Mode Verification**
   - Connects to paper trading endpoints only
   - Verifies account characteristics (high balances, no PDT restrictions)
   - Double-checks API response patterns

3. **Order Risk Controls**
   - Maximum order value limits ($5,000 default)
   - Daily trade frequency limits (100 trades/day default)
   - Position size limits (10% of portfolio default)
   - Execution delay simulation

4. **Audit & Compliance**
   - All orders logged to files
   - Complete audit trail with timestamps
   - Session data saved for analysis
   - Error tracking and reporting

## Setup Instructions

### Step 1: Get Alpaca Paper Trading Account

1. Visit [Alpaca Markets](https://alpaca.markets/)
2. Sign up for a free account
3. Navigate to Paper Trading section
4. Generate API keys for paper trading

### Step 2: Set Environment Variables

Create a `.env.local` file in your project root:

```bash
# Alpaca Paper Trading Configuration
ALPACA_API_KEY="your-paper-api-key-here"
ALPACA_SECRET_KEY="your-paper-secret-key-here"
ALPACA_PAPER="true"  # CRITICAL: Must be "true" for safety

# Optional: Explicit paper trading URL
ALPACA_BASE_URL="https://paper-api.alpaca.markets"
```

**⚠️ IMPORTANT**: Always set `ALPACA_PAPER="true"` to prevent live trading!

### Step 3: Verify Installation

Run the safety test suite to verify everything is working:

```bash
poetry run python scripts/test_paper_trading_safe.py
```

This test runs without API keys and verifies all safety mechanisms.

### Step 4: Test Paper Trading Connection

Run the connection test with your API keys:

```bash
poetry run python scripts/alpaca_paper_trading_demo.py
```

This will:
- Verify your API credentials
- Test paper trading connection
- Execute sample orders
- Generate audit logs

## Usage Examples

### Basic Paper Trading

```python
import asyncio
from bot.live.alpaca_paper_trader import AlpacaPaperTrader

async def basic_paper_trading():
    # Create paper trader with safety features
    async with AlpacaPaperTrader(
        symbols=["AAPL", "MSFT", "GOOGL"],
        strategy_name="your_strategy",
        initial_capital=100000.0
    ) as trader:
        
        # Run a trading session
        await trader.execute_trading_session(duration_minutes=30)
        
        # Get performance report
        report = await trader.generate_performance_report()
        print(f"Total Return: {report['performance']['total_return']:.2%}")

# Run it
asyncio.run(basic_paper_trading())
```

### Integration with Existing Strategies

```python
from bot.live.alpaca_paper_trader import create_alpaca_paper_trader

async def strategy_integration():
    # Create trader with custom configuration
    trader = await create_alpaca_paper_trader(
        symbols=["SPY", "QQQ", "IWM"],
        strategy_name="trend_breakout",
        initial_capital=50000.0,
        config_overrides={
            "max_order_value": 2500.0,  # $2.5k max per order
            "max_daily_trades": 50,     # 50 trades per day max
        }
    )
    
    try:
        # Execute trading session
        await trader.execute_trading_session(duration_minutes=60)
        
        # Monitor positions
        positions = trader.bridge.get_positions()
        for pos in positions:
            print(f"{pos.symbol}: {int(pos.qty)} shares, P&L: ${float(pos.unrealized_pl):+,.2f}")
            
    finally:
        await trader.stop_trading()
```

### Manual Order Execution

```python
async def manual_trading():
    async with AlpacaPaperTrader() as trader:
        bridge = trader.bridge
        
        # Get current quote
        quote = bridge.get_latest_quote("AAPL")
        current_price = (quote["bid_price"] + quote["ask_price"]) / 2
        print(f"AAPL current price: ${current_price:.2f}")
        
        # Submit market order
        result = bridge.submit_order(
            symbol="AAPL",
            side="buy",
            qty=10,
            order_type="market"
        )
        
        if result.success:
            print(f"Order submitted: {result.order_id}")
            
            # Check order status
            await asyncio.sleep(2)
            status = bridge.get_order_status(result.order_id)
            print(f"Order status: {status['status']}")
        else:
            print(f"Order failed: {result.error}")
```

## Configuration Options

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ALPACA_API_KEY` | Yes | - | Paper trading API key |
| `ALPACA_SECRET_KEY` | Yes | - | Paper trading secret key |
| `ALPACA_PAPER` | Yes | `"true"` | Must be "true" for safety |
| `ALPACA_BASE_URL` | No | Auto-detected | Paper trading API URL |

### PaperTradingConfig Options

```python
config_overrides = {
    "max_order_value": 5000.0,        # Max $ value per order
    "max_daily_trades": 100,          # Max trades per day
    "simulate_execution_delay": True, # Add realistic delays
    "min_execution_delay_ms": 100,    # Min delay in milliseconds
    "max_execution_delay_ms": 500,    # Max delay in milliseconds
    "log_all_orders": True,           # Log every order
    "save_execution_log": True,       # Save logs to files
}
```

### Risk Management Settings

```python
from bot.live.alpaca_paper_trader import AlpacaPaperTrader

trader = AlpacaPaperTrader(
    symbols=["AAPL", "MSFT"],
    initial_capital=100000.0,
    config_overrides={
        # Order limits
        "max_order_value": 10000.0,     # $10k max per order
        "max_daily_trades": 50,         # 50 trades/day max
        
        # Execution settings
        "simulate_execution_delay": True,
        "min_execution_delay_ms": 50,
        "max_execution_delay_ms": 200,
        
        # Monitoring
        "log_all_orders": True,
        "save_execution_log": True,
    }
)
```

## Monitoring and Analysis

### Real-time Position Monitoring

```python
async def monitor_positions(trader):
    while trader.is_running:
        # Get account info
        account = trader.bridge.get_account()
        positions = trader.bridge.get_positions()
        
        print(f"Portfolio Value: ${account.portfolio_value:,.2f}")
        print(f"Cash: ${account.cash:,.2f}")
        print(f"Positions: {len(positions)}")
        
        for pos in positions:
            print(f"  {pos.symbol}: {int(pos.qty)} @ ${float(pos.current_price):.2f}")
        
        await asyncio.sleep(30)  # Update every 30 seconds
```

### Performance Analysis

```python
async def analyze_performance(trader):
    report = await trader.generate_performance_report()
    
    print("📊 Performance Summary:")
    print(f"Total Return: {report['performance']['total_return']:.2%}")
    print(f"Sharpe Ratio: {report['performance']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {report['performance']['max_drawdown']:.2%}")
    print(f"Total Trades: {report['trading_activity']['total_trades']}")
    
    # Analyze positions
    for position in report['positions']['details']:
        symbol = position['symbol']
        pnl = position['unrealized_pnl']
        print(f"{symbol}: ${pnl:+,.2f} P&L")
```

## Audit and Compliance

### Audit Log Structure

All trading activity is logged with:

```json
{
  "timestamp": "2025-08-15T19:45:00.000Z",
  "event": "order_execution",
  "order_id": "order_20250815_194500_001",
  "symbol": "AAPL",
  "side": "buy",
  "quantity": 10,
  "price": 150.25,
  "execution_time_ms": 125.3,
  "success": true
}
```

### Session Data Export

Complete session data is saved including:
- Configuration settings
- All orders and executions
- Position history
- Equity curve
- Performance metrics
- Risk events

### Compliance Features

- **Order Validation**: All orders checked against risk limits
- **Trade Recording**: Every trade logged with full details
- **Error Tracking**: All failures and errors recorded
- **Performance Metrics**: Real-time calculation and storage
- **Audit Trail**: Complete history preserved for analysis

## Troubleshooting

### Common Issues

1. **"Environment validation failed"**
   - Check that `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` are set
   - Verify `ALPACA_PAPER="true"` is set
   - Ensure API keys are for paper trading (not live)

2. **"Connection test failed"**
   - Verify API keys are correct
   - Check internet connection
   - Confirm Alpaca API is accessible

3. **"Order rejected"**
   - Check order size against limits
   - Verify sufficient buying power
   - Review daily trade count limits

4. **"Paper mode verification failed"**
   - Ensure using paper trading API keys
   - Check that account is paper trading account
   - Verify API endpoint configuration

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger("alpaca_paper_trader").setLevel(logging.DEBUG)
```

### Support

- Review logs in `logs/paper_trading/` directory
- Check audit files for detailed execution history
- Use the test suite to verify system integrity
- Monitor execution metrics for performance issues

## Next Steps

1. **Start Small**: Begin with small position sizes and few symbols
2. **Monitor Closely**: Watch execution logs and performance metrics
3. **Gradual Scale**: Increase position sizes as confidence grows
4. **Strategy Integration**: Connect with your existing trading strategies
5. **Risk Management**: Customize limits based on your risk tolerance

## Safety Reminder

🚨 **NEVER set `ALPACA_PAPER="false"` unless you intend live trading with real money!**

Always verify paper mode before running:
```bash
echo $ALPACA_PAPER  # Should show "true"
```

The system includes multiple safety checks, but you are ultimately responsible for ensuring paper mode is active.