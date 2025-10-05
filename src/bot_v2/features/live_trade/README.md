# Live Trade Feature

**Purpose**: Real-money trading execution with comprehensive risk management, liquidity analysis, and advanced order handling.

---

## Overview

The `live_trade` feature is the production trading engine responsible for:
- Order validation and execution
- Real-time risk management
- Position and portfolio valuation
- Liquidity analysis and market impact estimation
- Fee calculation and PnL tracking
- Advanced execution models (TWAP, VWAP, iceberg)

**Complexity**: HIGH (33+ modules) - Candidate for future modularization

---

## Interface Contract

### Inputs

#### Required Dependencies
```python
from bot_v2.features.brokerages.core.interfaces import IBrokerage
from bot_v2.shared.types import OrderRequest, TradingSignal
from bot_v2.config.live_trade_config import LiveTradeConfig
```

#### Configuration
- **Config File**: `config/live_trade_config.yaml`
- **Risk Config**: `config/risk/*.yaml` (per environment)
- **Environment Variables**:
  - `RISK_MAX_LEVERAGE`
  - `RISK_MAX_EXPOSURE_PCT`
  - `RISK_DAILY_LOSS_LIMIT`
  - See `.env.template` for full list

#### Runtime Inputs
- `broker: IBrokerage` - Connected brokerage adapter
- `symbols: list[str]` - Trading symbols
- `risk_config: RiskConfig` - Risk parameters
- `signals: list[TradingSignal]` - Strategy signals

### Outputs

#### Data Structures
```python
from bot_v2.shared.types import (
    OrderResult,
    TradingPosition,
    RiskMetrics,
    PositionUpdate
)
```

#### Return Values
- **Order Execution**: `OrderResult` with fill details or error
- **Position Updates**: `PositionUpdate` events on position changes
- **Risk Metrics**: `RiskMetrics` snapshot on each update

### Side Effects

#### State Modifications
- ✅ Updates position state in broker
- ✅ Modifies cash balance via trades
- ✅ Records fills in event store
- ✅ Updates PnL tracking

#### External Interactions
- 🌐 Places orders via broker API
- 🌐 Fetches real-time prices and orderbook depth
- 📊 Emits metrics to Prometheus
- 💾 Persists trades to OrdersStore

#### Logging & Monitoring
- **Log Level**: INFO (trades, fills), DEBUG (validation steps)
- **Metrics**: Order latency, fill rate, slippage, risk utilization
- **Alerts**: Margin call warnings, kill switch triggers

---

## Core Modules

### Execution Pipeline
1. **Order Validation** (`order_validation_pipeline.py`)
   - Input: `OrderRequest`
   - Validates: Size, price, risk limits, margin
   - Output: Validated order or rejection

2. **Advanced Execution** (`advanced_execution.py`)
   - TWAP, VWAP, Iceberg order execution
   - Input: `OrderRequest` + execution parameters
   - Output: Multiple child orders or error

3. **Broker Adapter** (`broker_adapter.py`)
   - Normalizes broker-specific order formats
   - Handles retry logic and error mapping

### Risk Management (`risk/`)
- **Risk Calculations** (`risk_calculations.py`)
  - Calculates leverage, exposure, liquidation distance
- **Risk Metrics** (`risk_metrics.py`)
  - Tracks VAR, drawdown, Sharpe ratio
- **Risk Runtime** (`risk_runtime.py`)
  - Real-time risk gate enforcement

### Liquidity & Market Impact
- **Liquidity Service** (`liquidity_service.py`)
  - Analyzes orderbook depth and spread
- **Impact Estimator** (`impact_estimator.py`)
  - Estimates market impact before order
- **Depth Analyzer** (`depth_analyzer.py`)
  - Parses L2/L3 orderbook data

### Position & Valuation
- **Position Valuer** (`position_valuer.py`)
  - Calculates position value and unrealized PnL
- **Portfolio Valuation** (`portfolio_valuation.py`)
  - Aggregates portfolio-level metrics
- **Equity Calculator** (`equity_calculator.py`)
  - Computes total account equity

### Fees & Margin
- **Fees Engine** (`fees_engine.py`)
  - Calculates trading fees by symbol/tier
- **Margin Calculator** (`margin_calculator.py`)
  - Computes required margin for positions

### Strategies (`strategies/`)
- Multiple strategy implementations
- Uses `bot_v2.shared.types.TradingSignal` for outputs

---

## Usage Examples

### Basic Order Execution
```python
from bot_v2.features.live_trade import execute_order
from bot_v2.shared.types import OrderRequest
from decimal import Decimal

order = OrderRequest(
    symbol="BTC-USD",
    side="buy",
    quantity=Decimal("0.01"),
    order_type="limit",
    price=Decimal("50000.00")
)

result = execute_order(broker, order, risk_manager)
if result.success:
    print(f"Order filled: {result.order_id}")
else:
    print(f"Order rejected: {result.error_message}")
```

### Risk-Gated Execution
```python
from bot_v2.features.live_trade.risk import LiveRiskManager

risk_manager = LiveRiskManager(config=risk_config)

# Check if order passes risk gates
if risk_manager.validate_order(order, current_positions):
    result = execute_order(broker, order, risk_manager)
else:
    print("Order rejected by risk gates")
```

### Advanced Execution (TWAP)
```python
from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine

engine = AdvancedExecutionEngine(broker)

result = engine.execute_twap(
    symbol="ETH-USD",
    quantity=Decimal("10.0"),
    duration_minutes=60,
    num_slices=12
)
```

---

## Testing Strategy

### Unit Tests (`tests/unit/bot_v2/features/live_trade/`)
- Mock broker adapter for deterministic tests
- Test risk calculations with edge cases
- Validate fee calculations across tiers

### Integration Tests (`tests/integration/live_trade/`)
- Use recorded Coinbase API responses
- Test full order pipeline: validation → execution → fill
- Verify risk gates block invalid orders

### Scenario Tests (Recommended)
- **Margin Call Scenario**: Simulate position loss triggering margin call
- **Liquidity Crunch**: Test order execution with thin orderbook
- **Kill Switch**: Verify kill switch halts all trading

---

## Configuration

### Risk Parameters
```yaml
# config/risk/coinbase_perps.prod.yaml
max_leverage: 5
daily_loss_limit: 50
max_exposure_pct: 0.5
max_position_pct_per_symbol: 0.2
slippage_guard_bps: 30
```

### Execution Settings
```yaml
# config/live_trade_config.yaml
default_order_type: limit
max_order_retries: 3
retry_delay_ms: 500
post_only_default: false
reduce_only_default: false
```

---

## Error Handling

### Common Errors
- `InsufficientMarginError`: Raised when order exceeds available margin
- `RiskLimitExceededError`: Risk gate rejection
- `LiquidityError`: Orderbook too thin for order size
- `BrokerageError`: API errors from broker

### Error Recovery
- Orders are retried up to `max_order_retries` times
- Risk violations are logged and alerted
- Margin calls trigger `reduce_only` mode

---

## Metrics & Monitoring

### Key Metrics (Prometheus)
- `live_trade_order_latency_seconds` - Order placement latency
- `live_trade_fill_rate` - Percentage of orders filled
- `live_trade_slippage_bps` - Actual slippage vs expected
- `live_trade_risk_utilization` - Current risk vs limits

### Alerts
- **Critical**: `RiskLimitExceeded`, `MarginCallWarning`
- **Warning**: `HighSlippage`, `LowLiquidity`
- **Info**: `OrderFilled`, `PositionClosed`

---

## Dependencies

### Internal
- `bot_v2.features.brokerages.core.interfaces` - Broker abstraction
- `bot_v2.shared.types` - Shared DTOs
- `bot_v2.persistence` - Event and order storage
- `bot_v2.monitoring` - Metrics and alerts

### External
- `pydantic` - Config validation
- `decimal` - Precision arithmetic
- `prometheus_client` - Metrics export

---

## Future Refactoring (Phase 1 Recommendations)

### Modularization Opportunities
1. **Extract Liquidity Module**
   - Move `liquidity_*.py` to `features/liquidity/`
   - Create dedicated `LiquidityService` interface

2. **Extract Fees Module**
   - Move `fees_engine.py`, `margin_calculator.py` to `features/fees/`
   - Consolidate fee calculation logic

3. **Simplify Risk Module**
   - Current: 4 files in `risk/` subdir
   - Recommended: Single `RiskGate` interface with pluggable validators

4. **Consolidate Execution Models**
   - Move `advanced_execution_models/` to top-level
   - Create `ExecutionStrategy` protocol

---

## Ownership & Contacts

**Primary Maintainer**: Trading Team
**Code Review**: Requires 2 approvals for changes to risk/* and order_validation*
**On-Call**: trading-oncall@company.com

---

**Last Updated**: 2025-10-05
**Status**: ✅ Production (Active Development)
