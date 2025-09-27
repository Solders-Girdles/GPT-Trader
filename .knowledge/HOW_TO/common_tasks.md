# Common Tasks

## Run a Backtest
```python
from src.bot_v2.features.backtest import run_backtest
results = run_backtest(symbol="AAPL", start="2024-01-01", end="2024-03-01")
```

## Get ML Strategy Recommendation
```python
from src.bot_v2.features.ml_strategy import predict_best_strategy
strategy = predict_best_strategy("AAPL")
```

## Detect Market Regime
```python
from src.bot_v2.features.market_regime import detect_regime
regime = detect_regime("AAPL")
```

## Analyze Technical Indicators
```python
from src.bot_v2.features.analyze import calculate_indicators
indicators = calculate_indicators(data)
```

## Load and Save Data
```python
from src.bot_v2.features.data import load_data, save_data
data = load_data("AAPL", "2024-01-01", "2024-03-01")
save_data(data, "output.csv")
```

## Check System Health
```python
from src.bot_v2.features.monitor import check_health
status = check_health()
print(f"System health: {status}")
```

## Optimize Strategy Parameters
```python
from src.bot_v2.features.optimize import optimize_parameters
params = optimize_parameters(strategy="momentum", symbol="AAPL")
```

## Size a Position
```python
from src.bot_v2.features.position_sizing import calculate_position_size
size = calculate_position_size(capital=10000, confidence=0.8)
```

## Quick Commands

### Install Dependencies
```bash
# Dependencies are in config/ directory
cd config && poetry install && cd ..
```

### Format Code
```bash
poetry run black src/bot_v2/
```

### Check Types
```bash
poetry run mypy src/bot_v2/
```

### Find Functionality
```bash
# Search for a class or function
grep -r "class.*Strategy" src/bot_v2/features/
```

### List All Slices
```bash
ls src/bot_v2/features/
```

## Navigation Tips

- Each slice in `src/bot_v2/features/[slice]/`
- Tests in `src/bot_v2/test_[slice].py`
- Check `__init__.py` for public API
- Core logic usually in `core.py`