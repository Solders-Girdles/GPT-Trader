# GPT-Trader Week 1 Validation Runbook

## Quick Reference

### Mock Validation (Safe - Default)
```bash
# Set environment for mock testing
export RUN_SANDBOX_VALIDATIONS=1
export PERPS_FORCE_MOCK=1

# Run validations
python scripts/validate_perps_client_week1.py
python scripts/validate_ws_week1.py
```

### Real Sandbox Validation (Requires Credentials) 
```bash
# Set Coinbase sandbox credentials
export COINBASE_SANDBOX=1
export COINBASE_API_KEY="your_sandbox_key"
export COINBASE_API_SECRET="your_sandbox_secret" 
export COINBASE_PASSPHRASE="your_sandbox_passphrase"
export COINBASE_ENABLE_DERIVATIVES=1

# Enable real adapter and sandbox validations
export USE_REAL_ADAPTER=1
export RUN_SANDBOX_VALIDATIONS=1

# Run validations
python scripts/validate_perps_client_week1.py
python scripts/validate_ws_week1.py
```

## Quick Command Reference

### Mock vs Real Adapter Commands

**Mock Mode (Safe Local Testing):**
```bash
RUN_SANDBOX_VALIDATIONS=1 python scripts/validate_perps_client_week1.py
```

**Real Sandbox Mode (Requires Credentials):**
```bash
USE_REAL_ADAPTER=1 COINBASE_SANDBOX=1 \
COINBASE_API_KEY="your_key" COINBASE_API_SECRET="your_secret" \
COINBASE_PASSPHRASE="your_passphrase" COINBASE_ENABLE_DERIVATIVES=1 \
RUN_SANDBOX_VALIDATIONS=1 python scripts/validate_perps_client_week1.py
```

## Validation Overview

### Week 1 Scope
- ✅ **Perpetuals Discovery**: Dynamic product listing for BTC/ETH/SOL/XRP
- ✅ **Order Management**: Market/limit orders with quantization and TIF support
- ✅ **WebSocket Data**: Real-time ticker, trades, and level2 orderbook
- ✅ **Risk Metrics**: Spread calculation, depth analysis, rolling volume
- ✅ **Staleness Guards**: Data freshness validation with configurable thresholds

### Architecture Validation
- **Mock Mode**: Uses MinimalCoinbaseBrokerage for safe local testing
- **Sandbox Mode**: Uses real CoinbaseBrokerage against Coinbase sandbox
- **Production Isolation**: broker_factory still returns CoinbaseBrokerage for live usage

## Environment Variables

### Safety Controls
- `RUN_SANDBOX_VALIDATIONS=1` - Required to enable any validation testing
- `USE_REAL_ADAPTER=1` - Switch from mock to real Coinbase adapter
- `PERPS_FORCE_MOCK=1` - Force mock mode for runner integration tests

### Coinbase Configuration
- `COINBASE_SANDBOX=1` - Use Coinbase sandbox environment
- `COINBASE_API_KEY` - Your Coinbase Advanced API key
- `COINBASE_API_SECRET` - Your Coinbase API secret  
- `COINBASE_PASSPHRASE` - Your Coinbase API passphrase
- `COINBASE_ENABLE_DERIVATIVES=1` - Enable perpetuals/futures trading

### Alternative CDP API
- `CDP_API_KEY` - Use CDP API instead of Advanced Trade (JWT auth)

## Validation Scripts

### 1. Perpetuals Client (`validate_perps_client_week1.py`)

**Tests:**
- Product discovery and metadata validation
- Order placement with quantization
- TIF support (GTC, IOC, FOK)
- Client-ID and reduce-only order features

**Expected Results:**
```
✅ Product Discovery: PASS
✅ Product Metadata: PASS  
✅ Order Placement: PASS
✅ TIF Support: PASS
✅ Quantization: PASS

🎉 All perpetuals client tests PASSED!
```

**Common Issues:**
- Missing perps: Market may not have all expected symbols (SOL/XRP)
- TIF FOK errors: FOK may not be supported, use feature flag
- Quantization errors: Check step_size and price_increment alignment

### 2. WebSocket Data (`validate_ws_week1.py`)

**Tests:**
- WebSocket connection and subscription 
- Market data updates (ticker, trades, level2)
- Staleness detection with configurable thresholds
- Rolling volume calculation over time windows

**Expected Results:**
```
✅ WebSocket Connection: PASS
✅ Market Data Updates: PASS
✅ Staleness Detection: PASS  
✅ Rolling Volume: PASS

🎉 All WebSocket tests PASSED!
```

**Common Issues:**
- Connection timeouts: Check network and credentials
- No market data: Market may be quiet, tests handle gracefully
- Staleness false positives: Check system clock and thresholds

## Integration Testing

### Runner Integration
```bash
# Test with mock perps adapter
export PERPS_FORCE_MOCK=1
export RUN_SANDBOX_VALIDATIONS=1

python -m src.bot_v2.features.live_trade.runner
```

### Strategy Integration
```bash
# Test strategy with mock data
export PERPS_FORCE_MOCK=1

python demos/test_momentum_integration.py
python demos/test_volatility_integration.py
```

## Week 2 Preparation

### Strategy Filters Implementation
- **Spread/Depth Guards**: Reject entries when market conditions poor
- **RSI Confirmation**: Add RSI filter to MA crossover signals
- **Volume Thresholds**: Ensure minimum liquidity before trading

### Risk Management Enhancements
- **Liquidation Distance**: Calculate and maintain safe margin buffer
- **Slippage Guards**: Estimate and cap market impact of orders
- **Position Sizing**: Dynamic sizing based on volatility and depth

### Architecture Enhancements
- **MarketSnapshot Integration**: Thread WS data into strategy/execution
- **Strategy Factory**: Support configurable filter/guard parameters
- **Monitoring Integration**: Add strategy performance dashboards

## Troubleshooting

### Mock Mode Issues
```bash
# Verify mock dependencies
python -c "from src.bot_v2.features.brokerages.coinbase.test_adapter import MinimalCoinbaseBrokerage; print('Mock adapter OK')"

# Check test data generation
python -c "from src.bot_v2.features.brokerages.coinbase.endpoints import get_perps_symbols; print(get_perps_symbols())"
```

### Sandbox Mode Issues  
```bash
# Test credentials
curl -H "CB-ACCESS-KEY: $COINBASE_API_KEY" \
     -H "CB-ACCESS-PASSPHRASE: $COINBASE_PASSPHRASE" \
     "https://api.sandbox.coinbase.com/api/v3/brokerage/accounts"

# Check derivative permissions
curl -H "CB-ACCESS-KEY: $COINBASE_API_KEY" \
     -H "CB-ACCESS-PASSPHRASE: $COINBASE_PASSPHRASE" \
     "https://api.sandbox.coinbase.com/api/v3/brokerage/products?product_type=FUTURE"
```

### Common Error Patterns

**"No perpetual products found"**
- Check `COINBASE_ENABLE_DERIVATIVES=1`
- Verify sandbox account has futures permissions
- Market may not have expected perps available

**"WebSocket connection failed"**
- Check network connectivity to sandbox.exchange.coinbase.com
- Verify WebSocket credentials and subscription format
- Try with reduced symbol count (1-2 symbols max)

**"Order placement failed"**
- Verify product quantization (step_size, price_increment)  
- Check order size meets minimum requirements
- Ensure TIF is supported (FOK may not be available)

**"Staleness detection not working"**
- Check system clock synchronization
- Verify WebSocket data is being received and parsed
- Adjust staleness thresholds for testing environment

## Monitoring & Metrics

### Validation Metrics
- **Test Pass Rate**: Track % of tests passing over time
- **API Latency**: Monitor response times for order placement
- **WebSocket Uptime**: Track connection stability and reconnects
- **Data Quality**: Monitor spread, depth, and volume metrics

### Performance Baselines
- **Order Placement**: <500ms for limit orders
- **Market Data Latency**: <100ms ticker updates
- **Memory Usage**: <100MB for 4-symbol WebSocket feeds
- **CPU Usage**: <10% during normal market data processing

## Next Steps

### Week 2 Roadmap
1. **Strategy Enhancements**: Implement RSI + spread/volume filters
2. **Risk Guards**: Add liquidation distance and slippage protection  
3. **Integration Testing**: End-to-end with real market conditions
4. **Performance Optimization**: Reduce latency and memory usage
5. **Monitoring**: Add dashboards for strategy health and performance

### Week 3+ Planning
- **Multi-Strategy Support**: Portfolio allocation across strategies
- **Advanced Risk**: VaR calculations and stress testing
- **ML Integration**: Predictive features and regime detection
- **Production Deployment**: Live trading infrastructure and monitoring

---

For support and issues: See project README and GitHub issues.