# Navigation Guide

## Quick Slice Reference

Each slice is COMPLETELY ISOLATED. Never import between slices.

### Available Slices

| Slice | Purpose | Test Command |
|-------|---------|--------------|
| `backtest` | Historical testing | `poetry run python src/bot_v2/test_backtest.py` |
| `paper_trade` | Simulated trading | `poetry run python src/bot_v2/test_paper_trade.py` |
| `analyze` | Technical analysis | `poetry run python src/bot_v2/test_analyze.py` |
| `optimize` | Parameter tuning | `poetry run python src/bot_v2/test_optimize.py` |
| `live_trade` | Broker integration | `poetry run python src/bot_v2/test_live_trade.py` |
| `monitor` | Health monitoring | `poetry run python src/bot_v2/test_monitor.py` |
| `data` | Data management | `poetry run python src/bot_v2/test_data.py` |
| `ml_strategy` | ML strategy selection | `poetry run python src/bot_v2/test_ml_strategy.py` |
| `market_regime` | Regime detection | `poetry run python src/bot_v2/test_market_regime.py` |
| `position_sizing` | Position management | `poetry run python src/bot_v2/test_position_sizing.py` |

## Slice Structure

```
src/bot_v2/features/[slice]/
├── __init__.py       # Exports
├── core.py          # Main logic
├── models.py        # Data models
├── utils.py         # Utilities
└── config.py        # Configuration
```

## Import Pattern

```python
# Always use this pattern
from src.bot_v2.features.[slice] import function_name
```

## Testing

```bash
# Test single slice
poetry run python src/bot_v2/test_[slice].py

# Test all slices
poetry run python src/bot_v2/test_all_slices.py

# Verify isolation (should return nothing)
grep -r "from bot_v2.features" src/bot_v2/features/[slice]/
```

## Navigation by Task

- **Test a strategy** → `backtest` slice
- **Analyze market data** → `analyze` slice
- **Get ML recommendations** → `ml_strategy` slice
- **Detect market conditions** → `market_regime` slice
- **Optimize parameters** → `optimize` slice
- **Simulate trading** → `paper_trade` slice
- **Monitor system health** → `monitor` slice
- **Manage data** → `data` slice
- **Execute real trades** → `live_trade` slice
- **Size positions** → `position_sizing` slice