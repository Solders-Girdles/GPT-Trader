# Risk Management

Pre-trade and runtime risk controls for live trading.

## Purpose

This package enforces risk limits to protect trading capital:

- **Leverage limits**: Global and per-symbol caps
- **Exposure limits**: Maximum portfolio exposure percentage
- **Daily loss limits**: Circuit breakers for excessive losses
- **Liquidation buffers**: Maintain safe distance from liquidation

## Key Components

| Module | Purpose |
|--------|---------|
| `manager/` | LiveRiskManager implementation |
| `protocols.py` | RiskManagerProtocol interface |

## Configuration

Risk parameters are defined in `RiskConfig`:

```python
RiskConfig(
    max_leverage=5,              # Global leverage cap
    daily_loss_limit_pct=0.05,   # 5% daily loss circuit breaker
    max_exposure_pct=0.8,        # 80% max portfolio exposure
    min_liquidation_buffer_pct=0.1,  # 10% buffer from liquidation
)
```

## Validation Flow

```
Order Request
    ↓
pre_trade_validate()
    ├── Check leverage limits
    ├── Check exposure caps
    ├── Check liquidation buffer
    └── Check day/night leverage caps
    ↓
Order Execution (if passed)
```

## Circuit Breakers

When limits are breached, the system enters **reduce-only mode**:
- New position-increasing orders blocked
- Only position-reducing trades allowed
- Automatic reset at start of next trading day

## Related Packages

- `risk_runtime/` - Runtime monitoring and guards
- `guard_errors.py` - Error hierarchy
