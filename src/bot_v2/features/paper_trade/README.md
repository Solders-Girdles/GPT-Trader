# Paper Trade Feature

**Purpose**: Simulated trading with live market data and performance dashboard.

---

## Overview

The `paper_trade` feature provides:
- Paper trading simulation with real-time data
- Portfolio tracking and PnL calculation
- Live dashboard (console and HTML)
- Performance analytics and reporting

**Coverage**: 🟢 96.8% (Excellent)

---

## Interface Contract

### Inputs

#### Required Dependencies
```python
from bot_v2.features.paper_trade import start_paper_trading
from bot_v2.shared.types import StrategyConfig
```

#### Configuration
- **Initial Capital**: Starting portfolio value
- **Strategy**: Strategy configuration or callable
- **Symbols**: List of trading symbols
- **Dashboard Mode**: console, html, or both

### Outputs

#### Data Structures
```python
from bot_v2.features.paper_trade.types import (
    PaperTradeResult,
    SessionMetrics,
    PortfolioSnapshot
)
```

#### Return Values
- **Paper Trade Result**: Final portfolio state, trades, performance
- **Session Metrics**: Real-time performance metrics
- **Dashboard Output**: Console rendering or HTML report

### Side Effects

#### State Modifications
- ✅ Tracks simulated positions in memory
- ✅ Records fills in performance tracker
- ✅ Updates dashboard display

#### External Interactions
- 🌐 Fetches live prices from broker API
- 📊 Emits performance metrics
- 💾 Saves session results to file (optional)
- 📺 Renders live console dashboard

---

## Core Modules

### Trading Loop (`trading_loop.py`)
```python
def run_trading_loop(
    strategy: StrategyConfig,
    symbols: list[str],
    initial_capital: Decimal,
    dashboard_mode: str = "console"
) -> PaperTradeResult:
    """Main paper trading loop."""
```

### Execution (`execution.py`)
```python
class PaperExecutionEngine:
    """Simulates order execution with slippage."""

    def execute_order(self, order: OrderRequest) -> OrderResult:
        """Simulate order fill at current market price."""

    def update_positions(self, prices: dict[str, Decimal]) -> None:
        """Update position values with latest prices."""
```

### Dashboard (`dashboard/`)
- **Console Renderer** (`console_renderer.py`): Live terminal dashboard
- **HTML Report** (`html_report_generator.py`): Post-session HTML report
- **Metrics Calculator** (`metrics.py`): Performance calculations

---

## Usage Examples

### Basic Paper Trading
```python
from bot_v2.features.paper_trade import start_paper_trading
from bot_v2.shared.types import StrategyConfig
from decimal import Decimal

config = StrategyConfig(
    strategy_name="momentum",
    risk_per_trade_pct=0.02,
    stop_loss_pct=0.05
)

result = start_paper_trading(
    strategy=config,
    symbols=["BTC-USD", "ETH-USD"],
    initial_capital=Decimal("10000"),
    duration_minutes=60,
    dashboard_mode="console"
)

print(f"Final Equity: ${result.final_equity}")
print(f"Total Return: {result.performance.total_return:.2%}")
print(f"Trades: {result.performance.trades_count}")
```

### With Live Dashboard
```python
# Console dashboard updates every 5 seconds
result = start_paper_trading(
    strategy=momentum_strategy,
    symbols=["BTC-USD", "ETH-USD", "SOL-USD"],
    initial_capital=Decimal("50000"),
    dashboard_mode="console",
    update_interval_secs=5
)

# Generate HTML report after session
from bot_v2.features.paper_trade.dashboard import generate_html_report

generate_html_report(
    result=result,
    output_path="reports/paper_trade_session.html"
)
```

---

## Dashboard Features

### Console Dashboard
- **Real-time Updates**: Portfolio value, positions, PnL
- **Performance Metrics**: Return %, Sharpe, win rate
- **Recent Trades**: Last 5 trades with P&L
- **Position Summary**: Current holdings and values

### HTML Report
- **Session Summary**: Initial/final capital, return, drawdown
- **Equity Curve**: Interactive chart showing portfolio value over time
- **Trade Log**: Complete trade history with entry/exit prices
- **Performance Analytics**: Sharpe ratio, max drawdown, win rate

---

## Testing Strategy

### Unit Tests (`tests/unit/bot_v2/features/paper_trade/`)
- Execution engine with mock prices
- Dashboard rendering with sample data
- Performance calculations accuracy

### Integration Tests
- Full paper trading session with live data
- Dashboard output verification
- Multi-symbol trading scenarios

---

## Configuration

```python
# Paper trading settings
PAPER_INITIAL_CAPITAL = 10000
PAPER_COMMISSION_PCT = 0.001
PAPER_SLIPPAGE_BPS = 10
PAPER_UPDATE_INTERVAL = 5  # Dashboard update frequency (seconds)

# Dashboard settings
DASHBOARD_MODE = "console"  # console, html, both
DASHBOARD_SAVE_PATH = "./reports"
```

---

## Performance Characteristics

- **Latency**: ~100ms per trade execution (simulated)
- **Memory**: O(n) where n = number of trades
- **Dashboard Update**: 5-10 FPS for console rendering

---

## Dependencies

### Internal
- `bot_v2.features.strategies` - Strategy implementations
- `bot_v2.shared.types` - Type definitions
- `bot_v2.features.data` - Market data fetching

### External
- `rich` (optional) - Enhanced console rendering
- `plotly` (optional) - HTML report charts

---

**Last Updated**: 2025-10-05
**Status**: ✅ Production (Stable)
