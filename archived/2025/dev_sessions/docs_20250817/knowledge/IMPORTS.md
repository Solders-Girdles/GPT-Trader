# V2 Slice Import Pattern Guide

## The V2 Golden Rule
**ALWAYS import directly from individual slices. NEVER cross-import between slices.**

```python
# ✅ CORRECT - V2 slice import pattern
from src.bot_v2.features.backtest import BacktestEngine
from src.bot_v2.features.ml_strategy import predict_best_strategy

# ❌ WRONG - V1 patterns (archived)
from bot_v2.features.module.submodule import ClassName  # V1 archived
from bot_v2.features.strategy import AVAILABLE_STRATEGIES  # V1 archived

# ❌ FORBIDDEN - Cross-slice imports (breaks isolation)
from src.bot_v2.features.backtest import BacktestEngine
from src.bot_v2.features.paper_trade import PositionTracker  # VIOLATION!
```

---

## 📦 V2 Slice Import Patterns (Complete Isolation)

### Backtest Slice
```python
# ✅ CORRECT - Import everything from backtest slice
from src.bot_v2.features.backtest import BacktestEngine
from src.bot_v2.features.backtest import PerformanceMetrics
from src.bot_v2.features.backtest import DataLoader  # Local implementation

# ❌ WRONG - No cross-slice imports
from src.bot_v2.features.data import DataPipeline  # ISOLATION VIOLATION
from src.bot_v2.features.paper_trade import PositionTracker  # ISOLATION VIOLATION
```

### Paper Trade Slice
```python
# ✅ CORRECT - Complete isolation
from src.bot_v2.features.paper_trade import PaperTrader
from src.bot_v2.features.paper_trade import PositionTracker  # Local implementation
from src.bot_v2.features.paper_trade import PnLCalculator  # Local implementation

# ❌ WRONG - Cannot import from other slices
from src.bot_v2.features.backtest import BacktestEngine  # ISOLATION VIOLATION
```

### Analyze Slice
```python
# ✅ CORRECT - Self-contained analysis
from src.bot_v2.features.analyze import TechnicalIndicators  # Local implementation
from src.bot_v2.features.analyze import SignalGenerator
from src.bot_v2.features.analyze import PatternDetector

# ❌ WRONG - Must implement indicators locally
from src.bot_v2.features.optimize import ParameterOptimizer  # ISOLATION VIOLATION
```

### Optimize Slice
```python
# ✅ CORRECT - Independent optimization
from src.bot_v2.features.optimize import ParameterOptimizer  # Local implementation
from src.bot_v2.features.optimize import SearchAlgorithms
from src.bot_v2.features.optimize import ResultAnalyzer

# ❌ WRONG - Cannot use external optimization
from src.bot_v2.features.backtest import BacktestEngine  # ISOLATION VIOLATION
```

### ML Strategy Slice (Intelligence Week 1-2)
```python
# ✅ CORRECT - ML strategy selection
from src.bot_v2.features.ml_strategy import predict_best_strategy
from src.bot_v2.features.ml_strategy import MLModelTrainer  # Local implementation
from src.bot_v2.features.ml_strategy import ConfidenceScorer

# ❌ WRONG - Must implement ML locally
from src.bot_v2.features.analyze import TechnicalIndicators  # ISOLATION VIOLATION
```

### Market Regime Slice (Intelligence Week 3)
```python
# ✅ CORRECT - Regime detection
from src.bot_v2.features.market_regime import detect_regime
from src.bot_v2.features.market_regime import MarketClassifier  # Local implementation
from src.bot_v2.features.market_regime import TransitionPredictor

# ❌ WRONG - Cannot use external ML
from src.bot_v2.features.ml_strategy import MLModelTrainer  # ISOLATION VIOLATION
```

### Live Trade Slice
```python
# ✅ CORRECT - Broker integration
from src.bot_v2.features.live_trade import BrokerAPI  # Local implementation
from src.bot_v2.features.live_trade import OrderManager
from src.bot_v2.features.live_trade import PositionTracker  # Local implementation

# ❌ WRONG - Must implement position tracking locally
from src.bot_v2.features.paper_trade import PositionTracker  # ISOLATION VIOLATION
```

### Tests (V2 Slice Testing)
```python
# ✅ CORRECT - Test individual slices
from src.bot_v2.features.backtest import BacktestEngine
from src.bot_v2.test_backtest import test_backtest_functionality

# ❌ WRONG - V1 test patterns
from tests.fixtures.factories import create_sample_data  # V1 archived
from bot_v2.features.strategy.demo_ma import DemoMAStrategy  # V1 archived
```

---

## 🔧 V2 Slice Structure (No `__init__.py` Required!)

V2 slices use direct imports - no complex `__init__.py` hierarchy needed:

```
src/bot_v2/
├── features/
│   ├── backtest/          # Self-contained slice
│   ├── paper_trade/       # Self-contained slice  
│   ├── analyze/           # Self-contained slice
│   ├── optimize/          # Self-contained slice
│   ├── live_trade/        # Self-contained slice
│   ├── monitor/           # Self-contained slice
│   ├── data/              # Self-contained slice
│   ├── ml_strategy/       # Intelligence slice (Week 1-2)
│   └── market_regime/     # Intelligence slice (Week 3)
└── test_*.py              # Slice integration tests
```

### V2 Slice Validation
```bash
# Check V2 slice structure exists
ls -la src/bot_v2/features/

# Verify all 9 slices present
required_slices="backtest paper_trade analyze optimize live_trade monitor data ml_strategy market_regime"
for slice in $required_slices; do
    if [ -d "src/bot_v2/features/$slice" ]; then
        echo "✅ $slice slice exists"
    else
        echo "❌ $slice slice missing"
    fi
done

# Test slice imports work
poetry run python -c "from src.bot_v2.features.backtest import *; print('✅ V2 imports work')"
```

---

## 🎯 V2 Common Import Fixes

### Fix 1: V2 Slice Not Found
```bash
# Diagnosis
python -c "from src.bot_v2.features.missing_slice import Class"

# Solution 1: Check slice exists
ls src/bot_v2/features/  # List actual slices

# Solution 2: Check for typo in slice name
# Valid slices: backtest, paper_trade, analyze, optimize, live_trade, monitor, data, ml_strategy, market_regime

# Solution 3: Slice not implemented yet
grep -r "class ClassName" src/bot_v2/features/  # Search in V2 slices
```

### Fix 2: Cross-Slice Import Violation
```python
# Error: Isolation violation detected

# Check for cross-slice imports (should be empty)
grep -r "from bot_v2.features" src/bot_v2/features/

# Fix: Remove cross-imports, duplicate code instead
# V2 principle: Duplication > Dependencies

# Example fix:
# Instead of: from src.bot_v2.features.data import DataLoader
# Create: Local DataLoader implementation in your slice
```

### Fix 3: V1 Import in V2 Context
```python
# Symptom: ImportError for V1 paths

# V1 imports (archived, don't use):
from bot_v2.features.strategy import AVAILABLE_STRATEGIES  # ❌ V1 archived
from bot_v2.features.dataflow.pipeline import DataPipeline  # ❌ V1 archived

# V2 replacements:
from src.bot_v2.features.backtest import BacktestEngine  # ✅ V2 slice
from src.bot_v2.features.analyze import TechnicalIndicators  # ✅ V2 slice
```

### Fix 4: V2 Slice Isolation Testing
```python
# Test slice isolation compliance

# Check single slice imports cleanly
python -c "from src.bot_v2.features.backtest import *; print('✅ Backtest isolated')"

# Check no cross-slice dependencies
for slice in backtest paper_trade analyze optimize; do
    echo "Testing $slice isolation..."
    grep -r "from bot_v2.features" "src/bot_v2/features/$slice/" && echo "❌ $slice violation" || echo "✅ $slice isolated"
done
```

---

## 📋 V2 Slice Import Order Convention

```python
# Standard library imports
import os
import sys
from datetime import datetime
from pathlib import Path

# Third-party imports
import pandas as pd
import numpy as np
import yfinance as yf

# V2 Slice imports (single slice only!)
from src.bot_v2.features.backtest import BacktestEngine
from src.bot_v2.features.backtest import PerformanceMetrics
from src.bot_v2.features.backtest import DataLoader  # Local implementation

# NO cross-slice imports allowed!
# Each slice contains everything it needs
```

---

## 🚫 V2 Import Anti-Patterns to Avoid

### 1. Never Cross-Import Between Slices
```python
# ❌ FORBIDDEN - Breaks V2 isolation
from src.bot_v2.features.backtest import BacktestEngine
from src.bot_v2.features.paper_trade import PositionTracker  # VIOLATION!

# ✅ CORRECT - Import from single slice only
from src.bot_v2.features.backtest import BacktestEngine
from src.bot_v2.features.backtest import LocalPositionTracker  # Local implementation
```

### 2. Never Use V1 Imports in V2 Context
```python
# ❌ V1 ARCHIVED - Don't use these
from bot_v2.features.strategy import AVAILABLE_STRATEGIES  # V1 archived
from bot_v2.features.dataflow.pipeline import DataPipeline  # V1 archived

# ✅ V2 SLICES - Use these instead
from src.bot_v2.features.analyze import TechnicalIndicators
from src.bot_v2.features.backtest import BacktestEngine
```

### 3. Never Star Import from Slices (Except for Testing)
```python
# ❌ AVOID in production
from src.bot_v2.features.backtest import *

# ✅ BETTER - Explicit imports
from src.bot_v2.features.backtest import BacktestEngine, PerformanceMetrics

# ✅ EXCEPTION - OK for testing/diagnostics
from src.bot_v2.features.backtest import *  # Testing purposes only
```

### 4. Never Modify sys.path for V2 Slices
```python
# ❌ NEVER DO THIS (V2 slices work without path hacking)
import sys
sys.path.append('src/bot_v2')
from features.backtest import BacktestEngine

# ✅ RIGHT - Direct V2 slice imports
from src.bot_v2.features.backtest import BacktestEngine
```

---

## 🔍 Debugging V2 Slice Import Issues

### V2 Step-by-Step Process
```bash
# 1. Try direct V2 slice import
python -c "from src.bot_v2.features.backtest import BacktestEngine"

# 2. Check slice directory exists
ls -la src/bot_v2/features/backtest/

# 3. Check class exists in slice
grep "class BacktestEngine" src/bot_v2/features/backtest/*.py

# 4. Check slice structure
ls -la src/bot_v2/features/

# 5. Test slice isolation (should be empty)
grep -r "from bot_v2.features" src/bot_v2/features/backtest/

# 6. Check for syntax errors in slice
python -m py_compile src/bot_v2/features/backtest/*.py

# 7. Test slice loads independently
python -c "from src.bot_v2.features.backtest import *; print('✅ Slice loads')"
```

---

## ✅ V2 Slice Import Validation

```python
# Run this to validate all V2 slices work
python -c "
slices = [
    'src.bot_v2.features.backtest',
    'src.bot_v2.features.paper_trade', 
    'src.bot_v2.features.analyze',
    'src.bot_v2.features.optimize',
    'src.bot_v2.features.live_trade',
    'src.bot_v2.features.monitor',
    'src.bot_v2.features.data',
    'src.bot_v2.features.ml_strategy',
    'src.bot_v2.features.market_regime',
]

for slice_module in slices:
    try:
        exec(f'from {slice_module} import *')
        slice_name = slice_module.split('.')[-1]
        print(f'✅ {slice_name}: Working')
    except ImportError as e:
        slice_name = slice_module.split('.')[-1]
        print(f'❌ {slice_name}: {e}')
"
```

### V2 Isolation Compliance Check
```bash
# Verify complete slice isolation (should return no results)
grep -r "from bot_v2.features" src/bot_v2/features/ && echo "❌ Isolation violations found" || echo "✅ Complete V2 isolation verified"

# Check individual slice isolation
for slice in backtest paper_trade analyze optimize live_trade monitor data ml_strategy market_regime; do
    violations=$(grep -r "from bot_v2.features" "src/bot_v2/features/$slice/" 2>/dev/null | wc -l)
    if [ "$violations" -eq 0 ]; then
        echo "✅ $slice: Isolated"
    else
        echo "❌ $slice: $violations violations"
    fi
done
```

---

## 📝 V2 Summary

1. **Always use**: `from src.bot_v2.features.slice_name import Class`
2. **Never use**: Cross-slice imports (breaks isolation)
3. **Check**: All 9 slices exist and load independently  
4. **Verify**: No `from bot_v2.features` imports within slices
5. **Debug**: Use V2 step-by-step process above

**V2 Principle**: Each slice is completely self-contained (~500 tokens to load vs 10K+ for V1).