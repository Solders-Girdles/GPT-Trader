# Vertical Slice Architecture

## Core Principles

### 1. Complete Isolation
- No cross-slice imports
- Each slice is self-contained
- Duplication over dependencies
- Independent deployment possible

### 2. Token Efficiency
- ~500 tokens per slice
- Load only what you need
- No cascading dependencies
- Optimal for AI agents

### 3. Independent Operation
- Each slice can run alone
- No shared state
- Local configuration
- Self-contained tests

## Slice Organization

```
src/bot_v2/
└── features/
    ├── backtest/        # Historical testing
    ├── paper_trade/     # Simulated trading
    ├── analyze/         # Technical analysis
    ├── optimize/        # Parameter optimization
    ├── live_trade/      # Broker integration
    ├── monitor/         # Health monitoring
    ├── data/           # Data management
    ├── ml_strategy/    # ML strategy selection
    ├── market_regime/  # Regime detection
    └── position_sizing/# Position management
```

## Slice Anatomy

```python
features/[slice]/
├── __init__.py         # Public API
├── core.py            # Business logic
├── models.py          # Data structures
├── utils.py           # Helpers
├── config.py          # Configuration
└── validators.py      # Validation
```

## Working with Slices

### Creating New Features

1. Create slice directory:
```bash
mkdir -p src/bot_v2/features/new_feature
```

2. Implement functionality:
```python
# new_feature/__init__.py
from .core import main_function
__all__ = ['main_function']
```

3. Test independently:
```python
# src/bot_v2/test_new_feature.py
from src.bot_v2.features.new_feature import main_function
```

### Modifying Features

1. Locate the slice
2. Make changes locally
3. Test in isolation
4. No side effects on other slices

## Best Practices

### DO
- Keep slices small and focused
- Implement locally rather than import
- Test each slice independently
- Document public API
- Use type hints

### DON'T
- Import between slices
- Create shared utilities
- Use global configuration
- Share state between slices

## System Metrics

- **Code**: 8,000 lines total
- **Slices**: 10 operational
- **Token load**: ~500 per slice
- **Isolation**: 100% complete