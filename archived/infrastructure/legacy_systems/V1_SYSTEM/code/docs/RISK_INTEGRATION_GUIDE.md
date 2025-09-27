# Risk Management Integration Guide

## Overview

The Risk Management Integration Layer provides comprehensive risk validation and control for all trading operations in GPT-Trader. This system ensures that all portfolio allocations comply with risk limits and protects the portfolio from excessive exposure.

## Architecture

### Core Components

1. **RiskIntegration** (`src/bot/risk/integration.py`)
   - Main integration layer between portfolio allocation and risk management
   - Validates allocations against comprehensive risk limits
   - Provides real-time risk monitoring and alerts

2. **RiskConfig** (`src/bot/risk/config.py`)
   - Comprehensive risk configuration management
   - Multiple risk profiles (Conservative, Moderate, Aggressive)
   - Validation and factory patterns for risk limits

3. **RiskDashboard** (`src/bot/risk/dashboard.py`)
   - Real-time risk monitoring and alerting
   - Risk metrics visualization and reporting
   - Historical risk tracking

4. **Risk Utilities** (`src/bot/risk/utils.py`)
   - Risk calculation functions (VaR, CVaR, drawdown, etc.)
   - Portfolio risk metrics
   - Risk reporting utilities

## Key Features

### Position-Level Risk Controls

- **Position Size Limits**: Maximum percentage of portfolio per position (default 10%)
- **Sector Exposure Limits**: Maximum exposure per sector/industry
- **Correlation Limits**: Maximum correlation between positions
- **Stop-Loss Management**: Automatic stop-loss and take-profit calculations

### Portfolio-Level Risk Controls

- **Total Exposure Limits**: Maximum portfolio exposure (default 95%)
- **Daily Loss Limits**: Maximum daily loss percentage (default 3%)
- **Drawdown Limits**: Maximum portfolio drawdown (default 15%)
- **Concentration Limits**: Herfindahl index and top-N concentration monitoring

### Risk Validation Process

```python
from bot.risk.integration import RiskIntegration, RiskConfig
from bot.portfolio.allocator import PortfolioRules

# Initialize risk integration
risk_config = RiskConfig(
    max_position_size=0.10,        # 10% max per position
    max_portfolio_exposure=0.95,   # 95% max total exposure
    default_stop_loss_pct=0.05,    # 5% stop loss
    max_daily_loss=0.03            # 3% max daily loss
)

risk_integration = RiskIntegration(risk_config=risk_config)

# Validate allocations
allocations = {
    'AAPL': 100,
    'GOOGL': 50,
    'MSFT': 75
}

current_prices = {
    'AAPL': 150.0,
    'GOOGL': 2500.0,
    'MSFT': 300.0
}

result = risk_integration.validate_allocations(
    allocations=allocations,
    current_prices=current_prices,
    portfolio_value=500000.0
)

# Check results
if result.passed_validation:
    print("Allocations approved")
    final_allocations = result.adjusted_allocations
else:
    print(f"Risk violations: {result.warnings}")
```

## Risk Configuration Profiles

### Conservative Profile
```python
from bot.risk.config import RiskConfigFactory

conservative_config = RiskConfigFactory.create_conservative_config()
# - 5% max position size
# - 80% max portfolio exposure
# - 10% max drawdown
# - 20% cash reserve
# - 3% stop loss
```

### Moderate Profile (Default)
```python
moderate_config = RiskConfigFactory.create_moderate_config()
# - 10% max position size
# - 95% max portfolio exposure
# - 15% max drawdown
# - 5% cash reserve
# - 5% stop loss
```

### Aggressive Profile
```python
aggressive_config = RiskConfigFactory.create_aggressive_config()
# - 20% max position size
# - 98% max portfolio exposure
# - 25% max drawdown
# - 2% cash reserve
# - 8% stop loss
# - 2x leverage allowed
```

## Risk Metrics and Calculations

### Value at Risk (VaR)
```python
from bot.risk.utils import calculate_var

# Calculate 95% VaR using historical method
var = calculate_var(returns, confidence_level=0.95, method='historical')

# Calculate using parametric method
var_parametric = calculate_var(returns, confidence_level=0.95, method='parametric')
```

### Conditional VaR (Expected Shortfall)
```python
from bot.risk.utils import calculate_cvar

cvar = calculate_cvar(returns, confidence_level=0.95)
```

### Maximum Drawdown
```python
from bot.risk.utils import calculate_max_drawdown

dd_metrics = calculate_max_drawdown(equity_curve)
print(f"Max Drawdown: {dd_metrics['max_drawdown_pct']:.1%}")
print(f"Duration: {dd_metrics['drawdown_duration']} periods")
```

### Risk-Adjusted Returns
```python
from bot.risk.utils import calculate_risk_adjusted_returns

risk_metrics = calculate_risk_adjusted_returns(returns)
print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
print(f"Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}")
print(f"Calmar Ratio: {risk_metrics['calmar_ratio']:.2f}")
```

## Real-Time Risk Monitoring

### Risk Dashboard
```python
from bot.risk.dashboard import RiskDashboard
from bot.risk.config import RiskManagementConfig

# Initialize dashboard
risk_config = RiskManagementConfig()
dashboard = RiskDashboard(risk_config, risk_integration)

# Update with current portfolio state
dashboard_data = dashboard.update_dashboard(
    portfolio_value=500000.0,
    positions=position_data,
    market_data=historical_data,
    current_pnl=-1500.0
)

# Check for alerts
if dashboard_data['alerts']['critical'] > 0:
    print("CRITICAL RISK ALERTS!")
    for alert in dashboard_data['alerts']['recent']:
        print(f"- {alert['message']}")
```

### Alert Types

- **PORTFOLIO_EXPOSURE**: Total exposure exceeds limits
- **POSITION_SIZE**: Individual position too large
- **DAILY_LOSS**: Daily loss limit exceeded
- **CONCENTRATION**: Portfolio too concentrated
- **CORRELATION**: High correlation detected
- **RISK_BUDGET**: Total risk exceeds VaR limits

## Stop-Loss and Take-Profit Management

### Automatic Stop-Loss Calculation
```python
# Calculate stop-loss levels for positions
stop_levels = result.stop_levels['AAPL']

print(f"Current Price: ${stop_levels['current_price']}")
print(f"Stop Loss: ${stop_levels['stop_loss']}")
print(f"Trailing Stop: ${stop_levels['trailing_stop']}")
print(f"Take Profit: ${stop_levels['take_profit']}")
print(f"Risk per Share: ${stop_levels['risk_per_share']}")
```

### Dynamic Stop-Loss Updates
```python
# Update stop-losses based on current market conditions
positions = {
    'AAPL': {
        'current_price': 155.0,
        'entry_price': 150.0,
        'highest_price': 160.0
    }
}

stop_updates = risk_integration.update_stop_losses(positions)
print(f"New stop level: ${stop_updates['AAPL']['effective_stop']}")
```

### Triggered Stops Detection
```python
# Check for triggered stop-losses
current_prices = {
    'AAPL': 142.0,  # Below stop level
    'GOOGL': 2600.0
}

triggered_stops = risk_integration.check_triggered_stops(current_prices)
for stop in triggered_stops:
    print(f"STOP TRIGGERED: {stop['symbol']} at ${stop['current_price']} (stop: ${stop['stop_price']})")
```

## Integration with Trading Pipeline

### Pre-Trade Validation
```python
# Validate new position before execution
is_valid, reason, adjusted_shares = risk_integration.validate_new_position(
    symbol='AAPL',
    proposed_shares=200,
    current_price=150.0,
    portfolio_value=500000.0
)

if not is_valid:
    print(f"Position rejected: {reason}")
    print(f"Suggested size: {adjusted_shares} shares")
```

### Post-Trade Risk Update
```python
# Check daily loss limits after trades
if risk_integration.check_daily_loss_limit(current_pnl=-2500.0):
    print("DAILY LOSS LIMIT BREACHED - HALT TRADING")
```

## Risk Reporting

### Comprehensive Risk Report
```python
# Generate detailed risk report
report = dashboard.generate_risk_report(positions, portfolio_value)
print(report)
```

Sample output:
```
============================================================
RISK MANAGEMENT REPORT
============================================================

PORTFOLIO METRICS:
------------------------------
Total Exposure: 85.2%
Total Risk: 2.1%
Largest Position: 12.5%
Number of Positions: 8
Concentration Ratio: 0.156

POSITION RISKS:
------------------------------
  AAPL: Size=12.5%, Risk=0.63%, P&L=+2.3%
 GOOGL: Size=10.0%, Risk=0.50%, P&L=-1.1%
  MSFT: Size=8.7%, Risk=0.44%, P&L=+0.8%

RISK LIMITS:
------------------------------
Max Position Size: 10.0%
Max Daily Loss: 3.0%
Stop Loss: 5.0%
Max Drawdown: 15.0%

LIMIT CHECKS:
------------------------------
⚠️ Position size limit exceeded: 12.5% > 10.0%
✅ All other risk limits within acceptable ranges
============================================================
```

### Export Dashboard Data
```python
# Export risk data for analysis
filename = dashboard.export_dashboard_data()
print(f"Risk data exported to {filename}")
```

## Configuration Management

### Environment-Specific Configs
```python
# Development environment - more permissive
dev_config = RiskConfig(
    max_position_size=0.15,
    max_daily_loss=0.05,
    enable_realtime_monitoring=False
)

# Production environment - strict limits
prod_config = RiskConfig(
    max_position_size=0.08,
    max_daily_loss=0.02,
    enable_realtime_monitoring=True,
    alert_on_limit_breach=True
)
```

### Risk Limit Validation
```python
from bot.risk.utils import validate_risk_parameters

risk_params = {
    'max_position_size': 0.10,
    'max_daily_loss': 0.03,
    'stop_loss_pct': 0.05,
    'take_profit_pct': 0.10
}

errors = validate_risk_parameters(risk_params)
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"- {error}")
```

## Best Practices

### 1. Risk-First Design
- Always validate allocations through risk integration before execution
- Set conservative limits initially and adjust based on experience
- Monitor risk metrics continuously, not just at trade time

### 2. Layered Risk Controls
- Position-level limits (individual stock exposure)
- Portfolio-level limits (total exposure, concentration)
- Time-based limits (daily/weekly/monthly loss)
- Market condition adjustments (volatility-based sizing)

### 3. Dynamic Risk Management
- Adjust position sizes based on volatility
- Use correlation analysis to avoid concentrated bets
- Implement circuit breakers for extreme market conditions
- Regular risk limit reviews and adjustments

### 4. Risk Monitoring
- Set up real-time alerts for limit breaches
- Daily risk reports for portfolio oversight
- Historical risk analysis for strategy improvement
- Stress testing under various market scenarios

### 5. Documentation and Audit Trail
- Log all risk decisions and adjustments
- Maintain detailed records of limit breaches
- Regular risk configuration reviews
- Clear escalation procedures for risk events

## Error Handling

### Common Issues and Solutions

1. **Missing Price Data**
   ```python
   if symbol not in current_prices:
       result.warnings[symbol] = "No price data available"
       result.adjusted_allocations[symbol] = 0
   ```

2. **Zero Portfolio Value**
   ```python
   position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
   ```

3. **Invalid Risk Parameters**
   ```python
   errors = risk_config.validate()
   if errors:
       raise ValueError(f"Invalid risk configuration: {errors}")
   ```

## Performance Considerations

- Risk calculations are optimized for real-time use
- Historical data caching for correlation and volatility calculations
- Efficient matrix operations for portfolio-level metrics
- Configurable calculation frequencies to balance accuracy and performance

## Testing

Run the validation script to verify risk integration:
```bash
python validate_risk_integration.py
```

Run unit tests (when dependencies are available):
```bash
python -m pytest tests/unit/risk/test_integration.py -v
```

## Summary

The Risk Management Integration Layer provides:

✅ **Comprehensive Risk Validation** - All allocations checked against limits

✅ **Real-Time Monitoring** - Continuous risk tracking and alerting

✅ **Flexible Configuration** - Multiple risk profiles and custom limits

✅ **Stop-Loss Management** - Automatic calculation and monitoring

✅ **Portfolio Protection** - Multi-layered risk controls

✅ **Detailed Reporting** - Rich risk metrics and dashboards

✅ **Production Ready** - Robust error handling and performance optimization

This system ensures that GPT-Trader operates within defined risk parameters while maximizing trading opportunities within those constraints.
