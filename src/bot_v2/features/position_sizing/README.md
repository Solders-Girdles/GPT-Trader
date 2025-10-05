# Position Sizing Feature

**Purpose**: Calculate optimal position sizes using Kelly Criterion, regime analysis, and confidence weighting.

---

## Overview

The `position_sizing` feature provides:
- Kelly Criterion-based sizing
- Regime-aware position adjustments
- Confidence-weighted sizing
- Risk-adjusted position calculation

**Coverage**: 🟡 89.2% (Good)

---

## Interface Contract

### Inputs

#### Required Dependencies
```python
from bot_v2.features.position_sizing import (
    calculate_kelly_size,
    calculate_regime_adjusted_size,
    calculate_confidence_weighted_size
)
from decimal import Decimal
```

#### Required Data
- **Win Rate**: Historical strategy win rate (0.0 to 1.0)
- **Win/Loss Ratio**: Average win / average loss
- **Account Equity**: Current portfolio value
- **Risk Parameters**: Max position size, max risk per trade

### Outputs

#### Data Structures
```python
from bot_v2.features.position_sizing.types import (
    PositionSize,
    KellyCriterion,
    RegimeAdjustment
)
```

#### Return Values
- **Position Size**: Recommended size in base currency or percentage
- **Kelly Fraction**: Optimal Kelly percentage (often fractional)
- **Adjusted Size**: Size after regime and confidence adjustments

### Side Effects
- ✅ **Pure Functions**: No state modifications
- ✅ **Stateless**: Deterministic calculations

---

## Core Modules

### Kelly Criterion (`kelly.py`)
```python
def calculate_kelly_size(
    win_rate: float,
    win_loss_ratio: float,
    account_equity: Decimal,
    kelly_fraction: float = 0.25  # Safety factor
) -> PositionSize:
    """Calculate position size using Kelly Criterion.

    Formula: f* = (p * b - q) / b
    where:
        p = win rate
        q = 1 - p (loss rate)
        b = win/loss ratio
    """
```

### Regime-Based Sizing (`regime.py`)
```python
def calculate_regime_adjusted_size(
    base_size: Decimal,
    market_regime: str,  # "bull", "bear", "sideways", "volatile"
    regime_multipliers: dict[str, float]
) -> Decimal:
    """Adjust position size based on market regime."""

def detect_market_regime(prices: pd.Series) -> str:
    """Detect current market regime from price data."""
```

### Confidence-Weighted Sizing (`confidence.py`)
```python
def calculate_confidence_weighted_size(
    base_size: Decimal,
    signal_confidence: float,  # 0.0 to 1.0
    min_confidence: float = 0.5,
    scaling: str = "linear"  # linear, quadratic, exponential
) -> Decimal:
    """Scale position size by signal confidence."""
```

---

## Usage Examples

### Kelly Criterion Sizing
```python
from bot_v2.features.position_sizing import calculate_kelly_size
from decimal import Decimal

# Strategy historical stats
win_rate = 0.55  # 55% win rate
win_loss_ratio = 2.0  # Average win is 2x average loss
account_equity = Decimal("10000")

size = calculate_kelly_size(
    win_rate=win_rate,
    win_loss_ratio=win_loss_ratio,
    account_equity=account_equity,
    kelly_fraction=0.25  # Use 25% of Kelly for safety
)

print(f"Recommended size: ${size.amount}")
print(f"Kelly percentage: {size.kelly_pct:.2%}")
```

### Regime-Adjusted Sizing
```python
from bot_v2.features.position_sizing import (
    calculate_regime_adjusted_size,
    detect_market_regime
)

# Detect current regime
regime = detect_market_regime(price_history)

# Define regime multipliers
multipliers = {
    "bull": 1.2,      # Increase size in bull markets
    "bear": 0.6,      # Reduce size in bear markets
    "sideways": 0.8,  # Slightly reduce in choppy markets
    "volatile": 0.5   # Significantly reduce in volatile markets
}

adjusted_size = calculate_regime_adjusted_size(
    base_size=base_size,
    market_regime=regime,
    regime_multipliers=multipliers
)

print(f"Regime: {regime}")
print(f"Adjusted size: ${adjusted_size}")
```

### Confidence-Weighted Sizing
```python
from bot_v2.features.position_sizing import calculate_confidence_weighted_size

# Strategy signal with confidence
signal_confidence = 0.75  # 75% confidence

final_size = calculate_confidence_weighted_size(
    base_size=Decimal("1000"),
    signal_confidence=signal_confidence,
    min_confidence=0.5,
    scaling="quadratic"  # More aggressive scaling
)

print(f"Confidence: {signal_confidence:.0%}")
print(f"Final size: ${final_size}")
```

### Combined Approach
```python
from bot_v2.features.position_sizing import position_sizing

# Calculate position size using all methods
result = position_sizing.calculate(
    account_equity=Decimal("10000"),
    win_rate=0.55,
    win_loss_ratio=2.0,
    signal_confidence=0.7,
    market_regime="bull",
    kelly_fraction=0.25
)

print(f"Kelly size: ${result.kelly_size}")
print(f"Regime-adjusted: ${result.regime_adjusted_size}")
print(f"Final size: ${result.final_size}")
```

---

## Position Sizing Methods

### 1. Kelly Criterion
- **Pros**: Optimal long-term growth, mathematically sound
- **Cons**: Can be aggressive, requires accurate win rate estimates
- **Recommendation**: Use fractional Kelly (0.25-0.5) for safety

### 2. Regime-Based
- **Pros**: Adapts to market conditions
- **Cons**: Requires accurate regime detection
- **Recommendation**: Combine with other methods for best results

### 3. Confidence-Weighted
- **Pros**: Scales with signal quality
- **Cons**: Depends on reliable confidence estimates
- **Recommendation**: Use with well-calibrated ML models or technical indicators

### 4. Fixed Percentage
- **Pros**: Simple, predictable
- **Cons**: Doesn't adapt to market conditions or signal quality
- **Recommendation**: Use as baseline or fallback

---

## Testing Strategy

### Unit Tests (`tests/unit/bot_v2/features/position_sizing/`)
- Kelly formula verification with known inputs
- Regime detection with synthetic price data
- Confidence scaling edge cases (0%, 100%, boundary conditions)

### Property-Based Tests (Recommended)
- Kelly size should be 0 when win_rate ≤ 1/(1+win_loss_ratio)
- Regime adjustments should preserve sign
- Confidence weighting should be monotonic

---

## Configuration

```python
# Position sizing defaults
DEFAULT_KELLY_FRACTION = 0.25       # Conservative Kelly
MIN_POSITION_SIZE_USD = 10          # Minimum trade size
MAX_POSITION_SIZE_PCT = 0.10        # Max 10% of portfolio per position
MIN_SIGNAL_CONFIDENCE = 0.5         # Only trade signals ≥ 50% confidence

# Regime multipliers
REGIME_MULTIPLIERS = {
    "bull": 1.2,
    "bear": 0.6,
    "sideways": 0.8,
    "volatile": 0.5
}
```

---

## Risk Considerations

### Kelly Criterion Risks
- **Overestimation**: Overestimating win rate leads to oversizing
- **Fat Tails**: Kelly assumes normal distribution (use fractional Kelly for safety)
- **Ruin Risk**: Full Kelly can lead to significant drawdowns

### Recommended Safeguards
1. Use fractional Kelly (25-50% of full Kelly)
2. Cap maximum position size (e.g., 10% of portfolio)
3. Implement minimum position size to avoid dust trades
4. Regularly recalibrate win rate and win/loss ratio

---

## Dependencies

### Internal
- `bot_v2.shared.types` - Type definitions
- `bot_v2.features.analyze` - Regime detection (optional)

### External
- `pandas` - Price data analysis for regime detection

---

**Last Updated**: 2025-10-05
**Status**: ✅ Production (Stable)
