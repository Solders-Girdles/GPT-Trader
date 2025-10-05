# Analyze Feature

**Purpose**: Technical analysis, pattern detection, and indicator calculation.

---

## Overview

The `analyze` feature provides:
- Technical indicator calculation (RSI, MACD, Bollinger Bands, etc.)
- Chart pattern detection (head & shoulders, triangles, channels)
- Strategy signal generation from technical patterns
- Market condition analysis

**Coverage**: 🟢 94.4% (Well-tested)

---

## Interface Contract

### Inputs

#### Required Dependencies
```python
from bot_v2.features.analyze import (
    calculate_indicators,
    detect_patterns,
    analyze_strategy
)
import pandas as pd
```

#### Data Requirements
- **OHLCV Data**: pandas DataFrame with columns `['open', 'high', 'low', 'close', 'volume']`
- **Timeframe**: String (e.g., "1h", "1d", "1w")
- **Indicator Parameters**: Dict with indicator-specific settings

### Outputs

#### Data Structures
```python
from bot_v2.features.analyze.types import (
    IndicatorResult,
    PatternMatch,
    StrategySignal
)
```

#### Return Values
- **Indicators**: Dict mapping indicator names to calculated values
- **Patterns**: List of detected patterns with confidence scores
- **Strategy Signals**: Typed `StrategySignal` objects (BUY/SELL/HOLD)

### Side Effects
- ✅ **Pure Functions**: No state modifications
- ✅ **Stateless**: Can be called concurrently
- ❌ **No I/O**: No external API calls or database writes

---

## Core Modules

### Indicators (`indicators.py`)
```python
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""

def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> dict[str, pd.Series]:
    """Calculate MACD, signal line, and histogram."""
```

### Pattern Detection (`patterns.py`)
```python
def detect_head_and_shoulders(ohlc: pd.DataFrame) -> list[PatternMatch]:
    """Detect head and shoulders pattern."""

def detect_support_resistance(ohlc: pd.DataFrame) -> dict[str, list[float]]:
    """Identify support and resistance levels."""
```

### Strategy Analysis (`strategies.py`)
```python
def analyze_momentum_strategy(
    ohlc: pd.DataFrame,
    config: StrategyConfig
) -> list[StrategySignal]:
    """Generate signals from momentum indicators."""
```

---

## Usage Examples

### Calculate Indicators
```python
import pandas as pd
from bot_v2.features.analyze import calculate_indicators

# Price data
prices = pd.Series([100, 102, 101, 105, 103, ...])

# Calculate RSI
rsi = calculate_indicators(prices, indicators=["rsi"], period=14)
print(f"Current RSI: {rsi['rsi'].iloc[-1]}")

# Calculate multiple indicators
indicators = calculate_indicators(
    prices,
    indicators=["rsi", "macd", "bollinger"],
    rsi_period=14,
    macd_fast=12,
    macd_slow=26
)
```

### Detect Patterns
```python
from bot_v2.features.analyze import detect_patterns

# OHLC data
ohlc = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Detect patterns
patterns = detect_patterns(ohlc, patterns=["head_and_shoulders", "triangle"])

for pattern in patterns:
    print(f"Found {pattern.name} with {pattern.confidence:.2f} confidence")
```

### Generate Strategy Signals
```python
from bot_v2.features.analyze import analyze_strategy
from bot_v2.shared.types import StrategyConfig

config = StrategyConfig(
    strategy_name="momentum",
    rsi_oversold=30,
    rsi_overbought=70
)

signals = analyze_strategy(ohlc, config)
for signal in signals:
    print(f"{signal.action} {signal.symbol} at {signal.confidence}")
```

---

## Available Indicators

### Momentum Indicators
- **RSI** (Relative Strength Index)
- **Stochastic Oscillator**
- **Williams %R**
- **ROC** (Rate of Change)

### Trend Indicators
- **MACD** (Moving Average Convergence Divergence)
- **ADX** (Average Directional Index)
- **Moving Averages** (SMA, EMA, WMA)

### Volatility Indicators
- **Bollinger Bands**
- **ATR** (Average True Range)
- **Keltner Channels**

### Volume Indicators
- **OBV** (On-Balance Volume)
- **Volume Profile**
- **VWAP** (Volume Weighted Average Price)

---

## Testing Strategy

### Unit Tests (`tests/unit/bot_v2/features/analyze/`)
- Test each indicator with known inputs/outputs
- Verify pattern detection accuracy
- Edge cases: zero volume, flat prices, extreme volatility

### Property-Based Tests (Recommended)
- Use `hypothesis` to test indicator properties:
  - RSI should be 0-100
  - MACD crossovers should align with price trends
  - Bollinger Bands should contain price X% of time

---

## Configuration

No external configuration required. All parameters passed via function arguments.

---

## Dependencies

### Internal
- `pandas` - Time series calculations
- `numpy` - Numerical operations
- `bot_v2.shared.types` - Type definitions

### External
- None (pure Python calculations)

---

## Performance Considerations

- **Vectorized Operations**: Uses pandas/numpy for speed
- **Memory**: Scales with DataFrame size (O(n) memory)
- **Caching**: Consider caching indicator results for repeated analysis

---

**Last Updated**: 2025-10-05
**Status**: ✅ Production (Stable)
