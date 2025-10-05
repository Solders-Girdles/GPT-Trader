# Strategy Tools Feature

**Purpose**: Cross-cutting utilities for strategy enhancement, filtering, and guards.

---

## Overview

The `strategy_tools` feature provides:
- **Signal Filters**: Remove low-quality or conflicting signals
- **Signal Enhancements**: Combine and refine signals
- **Strategy Guards**: Safety checks and risk gates

**Coverage**: 🟢 100.0% (Perfect!)

---

## Interface Contract

### Inputs

#### Required Dependencies
```python
from bot_v2.features.strategy_tools import (
    filter_signals,
    enhance_signals,
    apply_guards
)
from bot_v2.shared.types import TradingSignal
```

#### Data Requirements
- **Signals**: List of `TradingSignal` objects from strategies
- **Filter Config**: Minimum confidence, max signals, etc.
- **Market Context**: Current positions, risk limits

### Outputs

#### Data Structures
```python
from bot_v2.shared.types import TradingSignal
```

#### Return Values
- **Filtered Signals**: Subset of input signals passing filters
- **Enhanced Signals**: Signals with adjusted confidence/sizing
- **Guard Results**: Signals that pass safety checks

### Side Effects
- ✅ **Pure Functions**: No state modifications
- ✅ **Stateless**: Deterministic given same inputs

---

## Core Modules

### Signal Filters (`filters.py`)
```python
def filter_by_confidence(
    signals: list[TradingSignal],
    min_confidence: float = 0.5
) -> list[TradingSignal]:
    """Remove low-confidence signals."""

def filter_conflicting_signals(
    signals: list[TradingSignal]
) -> list[TradingSignal]:
    """Remove conflicting signals for same symbol."""

def filter_by_time(
    signals: list[TradingSignal],
    min_age_seconds: int = 0,
    max_age_seconds: int = 300
) -> list[TradingSignal]:
    """Filter by signal age."""
```

### Signal Enhancements (`enhancements.py`)
```python
def combine_signals(
    signals: list[TradingSignal],
    method: str = "vote"  # vote, average, max
) -> list[TradingSignal]:
    """Combine multiple signals for same symbol."""

def adjust_confidence(
    signals: list[TradingSignal],
    market_regime: str,
    regime_adjustments: dict[str, float]
) -> list[TradingSignal]:
    """Adjust confidence based on market regime."""

def add_stop_loss(
    signals: list[TradingSignal],
    stop_loss_pct: float
) -> list[TradingSignal]:
    """Add stop loss metadata to signals."""
```

### Strategy Guards (`guards.py`)
```python
def check_position_limits(
    signals: list[TradingSignal],
    current_positions: dict[str, Position],
    max_positions: int
) -> list[TradingSignal]:
    """Block signals that would exceed position limits."""

def check_exposure_limits(
    signals: list[TradingSignal],
    current_exposure: Decimal,
    max_exposure_pct: float
) -> list[TradingSignal]:
    """Block signals that would exceed exposure limits."""

def check_correlated_positions(
    signals: list[TradingSignal],
    correlation_matrix: pd.DataFrame,
    max_correlation: float = 0.8
) -> list[TradingSignal]:
    """Block highly correlated positions."""
```

---

## Usage Examples

### Filter Low-Quality Signals
```python
from bot_v2.features.strategy_tools import filter_by_confidence

# Strategy generated multiple signals
raw_signals = strategy.generate_signals(ohlc)  # 10 signals

# Keep only high-confidence signals
filtered = filter_by_confidence(raw_signals, min_confidence=0.7)

print(f"Filtered {len(raw_signals)} → {len(filtered)} signals")
```

### Combine Multi-Strategy Signals
```python
from bot_v2.features.strategy_tools import combine_signals

# Multiple strategies generate signals
momentum_signals = momentum_strategy.generate_signals(ohlc)
reversion_signals = reversion_strategy.generate_signals(ohlc)

# Combine using voting
all_signals = momentum_signals + reversion_signals
combined = combine_signals(all_signals, method="vote")

# A signal needs ≥2 strategies to agree
```

### Apply Risk Guards
```python
from bot_v2.features.strategy_tools import check_position_limits

signals = strategy.generate_signals(ohlc)

# Check if we can take new positions
safe_signals = check_position_limits(
    signals=signals,
    current_positions=portfolio.positions,
    max_positions=10
)

if len(safe_signals) < len(signals):
    print(f"Blocked {len(signals) - len(safe_signals)} signals (position limit)")
```

### Full Pipeline
```python
from bot_v2.features.strategy_tools import (
    filter_by_confidence,
    filter_conflicting_signals,
    combine_signals,
    check_position_limits,
    check_exposure_limits
)

# 1. Generate signals from multiple strategies
signals = []
for strategy in strategies:
    signals.extend(strategy.generate_signals(ohlc))

# 2. Filter low confidence
signals = filter_by_confidence(signals, min_confidence=0.6)

# 3. Combine conflicting signals
signals = combine_signals(signals, method="average")

# 4. Apply risk guards
signals = check_position_limits(signals, current_positions, max_positions=10)
signals = check_exposure_limits(signals, current_exposure, max_exposure_pct=0.8)

# 5. Execute remaining signals
for signal in signals:
    execute_signal(signal)
```

---

## Signal Combination Methods

### Vote Method
- Each strategy gets one vote
- Signal passes if ≥50% of strategies agree
- Confidence = (votes / total_strategies)

### Average Method
- Average confidence across all signals
- Filters out if average < threshold
- Useful when strategies have calibrated confidences

### Max Method
- Take highest confidence signal
- Use when strategies are independent
- Risk: May amplify overconfident signals

---

## Guard Implementations

### Position Limit Guard
- Checks `len(current_positions) < max_positions`
- Blocks new signals if at limit
- Allows closing signals always

### Exposure Limit Guard
- Calculates total portfolio exposure
- Blocks signals that would exceed `max_exposure_pct`
- Considers signal size and current positions

### Correlation Guard
- Checks correlation between signal symbol and current positions
- Blocks highly correlated (>0.8) new positions
- Promotes diversification

---

## Testing Strategy

### Unit Tests (`tests/unit/bot_v2/features/strategy_tools/`)
- Filter functions with edge cases (empty list, all filtered)
- Combination methods with conflicting signals
- Guards with boundary conditions (exactly at limit)

### Property Tests
- Filtering should never increase signal count
- Combination should preserve symbols
- Guards should be monotonic (adding positions → fewer allowed signals)

---

## Configuration

```python
# Signal filtering
MIN_SIGNAL_CONFIDENCE = 0.5
MAX_SIGNAL_AGE_SECONDS = 300
ENABLE_CONFLICT_RESOLUTION = True

# Signal combination
COMBINATION_METHOD = "vote"  # vote, average, max
MIN_AGREEMENT_PCT = 0.5      # For vote method

# Guards
MAX_POSITIONS = 10
MAX_EXPOSURE_PCT = 0.8
MAX_CORRELATION = 0.7
ENABLE_CORRELATION_GUARD = True
```

---

## Best Practices

1. **Always Filter First**: Remove low-quality signals before combining
2. **Combine Before Guards**: Consolidate signals, then check limits
3. **Guard Order Matters**: Check position limits before exposure limits
4. **Log Blocked Signals**: Track what guards are blocking for tuning

---

## Dependencies

### Internal
- `bot_v2.shared.types` - Type definitions
- `bot_v2.features.analyze` (optional) - Correlation calculations

### External
- `pandas` (optional) - Correlation matrix operations

---

**Last Updated**: 2025-10-05
**Status**: ✅ Production (Stable)
