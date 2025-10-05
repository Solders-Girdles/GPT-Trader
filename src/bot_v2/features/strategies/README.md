# Strategies Feature

**Purpose**: Collection of trading strategy implementations.

---

## Overview

The `strategies` feature provides ready-to-use trading strategies:
- **Momentum**: Trend-following based on price momentum
- **Mean Reversion**: Counter-trend trading at extremes
- **Breakout**: Trade breakouts from consolidation
- **MA Crossover**: Moving average crossover signals
- **Volatility**: Volatility-based entries
- **Scalping**: High-frequency scalping strategies

**Coverage**: 🟢 98.0% (Excellent)

---

## Interface Contract

### Inputs

#### Required Dependencies
```python
from bot_v2.features.strategies import (
    MomentumStrategy,
    MeanReversionStrategy,
    BreakoutStrategy
)
from bot_v2.shared.types import StrategyConfig, TradingSignal
import pandas as pd
```

#### Data Requirements
- **OHLCV Data**: pandas DataFrame with price and volume
- **Strategy Config**: Parameters specific to each strategy
- **Market Context**: (Optional) Regime, volatility, etc.

### Outputs

#### Data Structures
```python
from bot_v2.shared.types import TradingSignal, SignalAction
```

#### Return Values
- **Trading Signals**: List of `TradingSignal` objects with action, confidence, reasoning
- **Strategy State**: Internal state for stateful strategies (optional)

### Side Effects
- ✅ **Mostly Stateless**: Strategies are pure functions of data + config
- ✅ **No I/O**: No external API calls or database writes
- 📊 **Metrics**: Some strategies emit diagnostic metrics

---

## Available Strategies

### 1. Momentum Strategy (`momentum.py`)
```python
class MomentumStrategy:
    """Trend-following strategy using RSI and price momentum."""

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        momentum_lookback: int = 20
    ):
        ...

    def generate_signals(self, ohlc: pd.DataFrame) -> list[TradingSignal]:
        """Generate buy signals when oversold, sell when overbought."""
```

### 2. Mean Reversion (`mean_reversion.py`)
```python
class MeanReversionStrategy:
    """Fade extremes using Bollinger Bands and Z-score."""

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        zscore_threshold: float = 2.0
    ):
        ...

    def generate_signals(self, ohlc: pd.DataFrame) -> list[TradingSignal]:
        """Buy at lower band, sell at upper band."""
```

### 3. Breakout Strategy (`breakout.py`)
```python
class BreakoutStrategy:
    """Trade breakouts from consolidation ranges."""

    def __init__(
        self,
        lookback_period: int = 50,
        breakout_threshold: float = 0.02,  # 2% above resistance
        volume_confirm: bool = True
    ):
        ...

    def generate_signals(self, ohlc: pd.DataFrame) -> list[TradingSignal]:
        """Buy on upside breakout, sell on downside breakout."""
```

### 4. MA Crossover (`ma_crossover.py`)
```python
class MACrossoverStrategy:
    """Classic moving average crossover."""

    def __init__(
        self,
        fast_period: int = 50,
        slow_period: int = 200,
        ma_type: str = "EMA"  # SMA, EMA, WMA
    ):
        ...

    def generate_signals(self, ohlc: pd.DataFrame) -> list[TradingSignal]:
        """Buy on golden cross, sell on death cross."""
```

### 5. Volatility Strategy (`volatility.py`)
```python
class VolatilityStrategy:
    """Enter during volatility expansion, exit during contraction."""

    def __init__(
        self,
        atr_period: int = 14,
        vol_threshold: float = 1.5  # Enter when ATR > 1.5x average
    ):
        ...

    def generate_signals(self, ohlc: pd.DataFrame) -> list[TradingSignal]:
        """Trade volatility breakouts."""
```

### 6. Scalping Strategy (`scalp.py`)
```python
class ScalpingStrategy:
    """High-frequency scalping using order flow."""

    def __init__(
        self,
        tick_size: Decimal,
        spread_threshold: Decimal,
        min_volume: int
    ):
        ...

    def generate_signals(self, ohlc: pd.DataFrame) -> list[TradingSignal]:
        """Scalp bid-ask spread."""
```

---

## Usage Examples

### Basic Strategy Usage
```python
from bot_v2.features.strategies import MomentumStrategy
import pandas as pd

# Load data
ohlc = pd.read_csv("prices.csv")

# Initialize strategy
strategy = MomentumStrategy(
    rsi_period=14,
    rsi_oversold=30,
    rsi_overbought=70
)

# Generate signals
signals = strategy.generate_signals(ohlc)

for signal in signals:
    print(f"{signal.action} {signal.symbol} - Confidence: {signal.confidence:.2%}")
    print(f"  Reasoning: {signal.reasoning}")
```

### Multi-Strategy Portfolio
```python
from bot_v2.features.strategies import (
    MomentumStrategy,
    MeanReversionStrategy,
    BreakoutStrategy
)

strategies = [
    MomentumStrategy(rsi_period=14),
    MeanReversionStrategy(bb_std=2.0),
    BreakoutStrategy(lookback_period=50)
]

# Aggregate signals from all strategies
all_signals = []
for strategy in strategies:
    signals = strategy.generate_signals(ohlc)
    all_signals.extend(signals)

# Filter for high-confidence signals (>70%)
high_conf = [s for s in all_signals if s.confidence > 0.7]
```

### Strategy with Config
```python
from bot_v2.shared.types import StrategyConfig

config = StrategyConfig(
    strategy_name="momentum",
    enabled=True,
    risk_per_trade_pct=0.02,
    stop_loss_pct=0.05,
    parameters={
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70
    }
)

strategy = MomentumStrategy(**config.parameters)
signals = strategy.generate_signals(ohlc)
```

---

## Strategy Interface

All strategies implement a common interface:

```python
from typing import Protocol

class TradingStrategy(Protocol):
    """Common strategy interface."""

    def generate_signals(self, ohlc: pd.DataFrame) -> list[TradingSignal]:
        """Generate trading signals from OHLCV data."""
        ...

    def get_state(self) -> dict:
        """Get current strategy state (for stateful strategies)."""
        ...

    def reset(self) -> None:
        """Reset strategy state."""
        ...
```

---

## Testing Strategy

### Unit Tests (`tests/unit/bot_v2/features/strategies/`)
- Each strategy with synthetic price data
- Boundary conditions (flat prices, extreme volatility)
- Signal generation consistency

### Backtesting
- All strategies should be backtested before production use
- Recommended: Walk-forward analysis with out-of-sample validation
- See `features/optimize` for backtesting tools

---

## Configuration

Strategy parameters can be configured via:

1. **Direct Instantiation**
   ```python
   strategy = MomentumStrategy(rsi_period=14)
   ```

2. **Config File**
   ```yaml
   # config/strategies/momentum.yaml
   rsi_period: 14
   rsi_oversold: 30
   rsi_overbought: 70
   ```

3. **Environment Variables**
   ```bash
   STRATEGY_RSI_PERIOD=14
   STRATEGY_RSI_OVERSOLD=30
   ```

---

## Performance Considerations

- **Latency**: < 100ms for signal generation (per symbol)
- **Memory**: O(n) where n = lookback period
- **CPU**: Vectorized operations via pandas/numpy (fast)

---

## Dependencies

### Internal
- `bot_v2.features.analyze` - Indicators
- `bot_v2.shared.types` - Type definitions
- `bot_v2.features.strategy_tools` - Filters and enhancements

### External
- `pandas` - Data manipulation
- `numpy` - Numerical operations

---

**Last Updated**: 2025-10-05
**Status**: ✅ Production (Stable)
