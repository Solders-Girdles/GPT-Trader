# V2 Slice Diagnostic Commands

## Purpose
Fast one-liners to diagnose V2 slice issues WITHOUT running full test suites. Each slice loads independently (~500 tokens vs 10K+ for V1).

---

## 🚀 V2 System Health Checks

### 1. V2 Slice Import Health
```bash
# Check if individual slices import (complete isolation)
poetry run python -c "from src.bot_v2.features.backtest import *; print('✅ Backtest slice imports')"
poetry run python -c "from src.bot_v2.features.paper_trade import *; print('✅ Paper Trade slice imports')"
poetry run python -c "from src.bot_v2.features.analyze import *; print('✅ Analyze slice imports')"
poetry run python -c "from src.bot_v2.features.optimize import *; print('✅ Optimize slice imports')"

# Check ML intelligence slices
poetry run python -c "from src.bot_v2.features.ml_strategy import *; print('✅ ML Strategy slice imports')"
poetry run python -c "from src.bot_v2.features.market_regime import *; print('✅ Market Regime slice imports')"
```

### 2. V2 Slice Isolation Verification
```bash
# CRITICAL: Check no cross-slice dependencies exist (should return empty)
grep -r "from bot_v2.features" src/bot_v2/features/ && echo "❌ Isolation violation found" || echo "✅ Complete slice isolation verified"

# Check specific slice isolation
grep -r "from bot_v2.features" src/bot_v2/features/backtest/ && echo "❌ Backtest isolation violated" || echo "✅ Backtest isolated"
grep -r "from bot_v2.features" src/bot_v2/features/ml_strategy/ && echo "❌ ML Strategy isolation violated" || echo "✅ ML Strategy isolated"
```

### 3. V2 Architecture Status  
```bash
# Check V2 slice status from PROJECT_STATE.json
poetry run python -c "import json; s = json.load(open('.knowledge/PROJECT_STATE.json')); print('V2 Slices:', list(s.get('v2_slices', {}).keys()))"

# Verify slice completion status
poetry run python -c "
import json
s = json.load(open('.knowledge/PROJECT_STATE.json'))
for name, slice_info in s.get('v2_slices', {}).items():
    status = slice_info.get('status', 'unknown')
    isolation = slice_info.get('isolation', 'unknown')
    print(f'{name}: {status} (isolation: {isolation})')
"
```

---

## 🔍 V2 Slice-Specific Diagnostics

### Backtest Slice
```bash
# Check slice imports and test
poetry run python -c "from src.bot_v2.features.backtest import *; print('✅ Backtest slice loads')"
poetry run python src/bot_v2/test_backtest.py

# Check local implementations
poetry run python -c "
from src.bot_v2.features.backtest import BacktestEngine
engine = BacktestEngine()
print('✅ Backtest engine instantiates')
"
```

### Paper Trade Slice  
```bash
# Check slice imports and test
poetry run python -c "from src.bot_v2.features.paper_trade import *; print('✅ Paper Trade slice loads')"
poetry run python src/bot_v2/test_paper_trade.py

# Check position tracking (local implementation)
poetry run python -c "
from src.bot_v2.features.paper_trade import PositionTracker
tracker = PositionTracker()
print('✅ Position tracker instantiates')
"
```

### Analyze Slice
```bash
# Check slice imports and test
poetry run python -c "from src.bot_v2.features.analyze import *; print('✅ Analyze slice loads')"
poetry run python src/bot_v2/test_analyze.py

# Check technical indicators (local implementation)
poetry run python -c "
from src.bot_v2.features.analyze import TechnicalIndicators
indicators = TechnicalIndicators()
print('✅ Technical indicators instantiate')
"
```

### Optimize Slice
```bash
# Check slice imports and test
poetry run python -c "from src.bot_v2.features.optimize import *; print('✅ Optimize slice loads')"
poetry run python src/bot_v2/test_optimize.py

# Check parameter optimization (local implementation)  
poetry run python -c "
from src.bot_v2.features.optimize import ParameterOptimizer
optimizer = ParameterOptimizer()
print('✅ Parameter optimizer instantiates')
"
```

### ML Strategy Slice (Intelligence Week 1-2)
```bash
# Check ML strategy slice imports and test
poetry run python -c "from src.bot_v2.features.ml_strategy import *; print('✅ ML Strategy slice loads')"
poetry run python src/bot_v2/test_ml_strategy.py

# Test strategy selection functionality
poetry run python -c "
from src.bot_v2.features.ml_strategy import predict_best_strategy
result = predict_best_strategy('AAPL')
print(f'✅ ML strategy selection working: {result}')
"
```

### Market Regime Slice (Intelligence Week 3)
```bash
# Check market regime slice imports and test
poetry run python -c "from src.bot_v2.features.market_regime import *; print('✅ Market Regime slice loads')"
poetry run python src/bot_v2/test_market_regime.py

# Test regime detection functionality
poetry run python -c "
from src.bot_v2.features.market_regime import detect_regime
result = detect_regime('AAPL')
print(f'✅ Market regime detection working: {result}')
"
```

### Live Trade Slice
```bash
# Check slice imports and test
poetry run python -c "from src.bot_v2.features.live_trade import *; print('✅ Live Trade slice loads')"
poetry run python src/bot_v2/test_live_trade.py

# Check broker API integration (local implementation)
poetry run python -c "
from src.bot_v2.features.live_trade import BrokerAPI
api = BrokerAPI()
print('✅ Broker API instantiates')
"
```

---

## 🐛 V2 Test Diagnostics

### V2 Slice Test Validation
```bash
# Test all V2 slices independently (should all pass)
poetry run python src/bot_v2/test_backtest.py
poetry run python src/bot_v2/test_paper_trade.py
poetry run python src/bot_v2/test_analyze.py
poetry run python src/bot_v2/test_optimize.py
poetry run python src/bot_v2/test_ml_strategy.py
poetry run python src/bot_v2/test_market_regime.py

# Run all V2 slice tests in parallel
poetry run python src/bot_v2/test_all_slices.py
```

### V2 Isolation Testing
```bash
# Test that each slice works independently (no cross-imports)
for slice in backtest paper_trade analyze optimize live_trade monitor data ml_strategy market_regime; do
    echo "Testing $slice isolation..."
    poetry run python -c "from src.bot_v2.features.$slice import *; print('✅ $slice isolated')" || echo "❌ $slice has dependencies"
done
```

### V2 Integration Testing  
```bash
# Test slice combinations work together (when needed)
poetry run python -c "
from src.bot_v2.features.backtest import BacktestEngine
from src.bot_v2.features.ml_strategy import predict_best_strategy
print('✅ Slices can be used together')
"
```

---

## 📦 V2 Dependency Diagnostics

### V2 Architecture Validation
```bash
# Check V2 isolation compliance
poetry run python -c "
import json
deps = json.load(open('docs/knowledge/DEPENDENCIES.json'))
print('V2 Isolation Principle:', deps['v2_slice_isolation']['principle'])
print('Verification Command:', deps['v2_slice_isolation']['isolation_verification']['command'])
"

# Run isolation verification
grep -r "from bot_v2.features" src/bot_v2/features/ && echo "❌ V2 isolation violated" || echo "✅ V2 complete isolation verified"
```

### V2 vs V1 Status
```bash
# Show V2 active vs V1 archived status
poetry run python -c "
import os
v1_archived = os.path.exists('archived/bot_v1_20250817/')
v2_active = os.path.exists('src/bot_v2/')
print(f'V1 Archived: {v1_archived}')
print(f'V2 Active: {v2_active}')
if v2_active and v1_archived:
    print('✅ Clean V2 architecture with V1 properly archived')
else:
    print('❌ Architecture migration incomplete')
"
```

### External Dependencies (V2 Compatible)
```bash
# Key packages that work with V2 slices
poetry run python -c "
import pandas as pd
import numpy as np
print(f'Pandas: {pd.__version__}')
print(f'Numpy: {np.__version__}')
"

# Check if packages support slice isolation
poetry show | grep -E "yfinance|pandas|numpy|scikit-learn"
```

---

## 🔥 V2 Emergency Diagnostics

### When V2 Slices Fail
```bash
# Reset V2 slice caches only
find src/bot_v2 -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find src/bot_v2 -name "*.pyc" -delete 2>/dev/null || true

# Test fresh V2 slice import
poetry run python -c "from src.bot_v2.features.backtest import *; print('✅ V2 slices work after reset')"
```

### Find V2 Isolation Violations
```bash
# Check for cross-slice imports (should be empty)
poetry run python -c "
import os
violations = []
for root, dirs, files in os.walk('src/bot_v2/features'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                content = f.read()
                if 'from bot_v2.features' in content:
                    violations.append(filepath)
if violations:
    print('❌ Isolation violations found:')
    for v in violations:
        print(f'  {v}')
else:
    print('✅ No isolation violations detected')
"
```

### V2 File Structure Validation
```bash
# Ensure V2 slice structure is correct
ls -la src/bot_v2/features/
ls -la src/bot_v2/test_*.py

# Check if all required slices exist
required_slices="backtest paper_trade analyze optimize live_trade monitor data ml_strategy market_regime"
for slice in $required_slices; do
    if [ -d "src/bot_v2/features/$slice" ]; then
        echo "✅ $slice slice exists"
    else
        echo "❌ $slice slice missing"
    fi
done
```

---

## 💡 V2 Performance Diagnostics

### V2 Slice Load Time (Token Efficiency)
```bash
# Time how long each slice takes to import (~500 token efficiency)
for slice in backtest paper_trade analyze optimize ml_strategy market_regime; do
    echo "Testing $slice load time..."
    poetry run python -c "
import time
start = time.time()
exec('from src.bot_v2.features.$slice import *')
load_time = time.time() - start
print(f'$slice: {load_time:.3f}s')
"
done
```

### V2 Memory Efficiency
```bash
# Check V2 slice memory usage (should be minimal)
poetry run python -c "
import tracemalloc
import sys

tracemalloc.start()

# Load one slice
from src.bot_v2.features.backtest import *
current, peak = tracemalloc.get_traced_memory()
print(f'Single slice memory: {current / 10**6:.1f} MB')

tracemalloc.stop()
"
```

---

## 📊 V2 Quick Status Report

### Generate V2 Health Report
```bash
# Run all V2 slice checks
poetry run python -c "
import json
import os

print('=== V2 SLICE HEALTH CHECK ===')

# Check slice existence
slices = ['backtest', 'paper_trade', 'analyze', 'optimize', 'live_trade', 'monitor', 'data', 'ml_strategy', 'market_regime']
for slice in slices:
    slice_dir = f'src/bot_v2/features/{slice}'
    if os.path.exists(slice_dir):
        try:
            exec(f'from src.bot_v2.features.{slice} import *')
            print(f'✅ {slice}: Working')
        except Exception as e:
            print(f'❌ {slice}: {e.__class__.__name__}')
    else:
        print(f'❌ {slice}: Directory missing')

# Check V2 project state
try:
    with open('.knowledge/PROJECT_STATE.json') as f:
        state = json.load(f)
        v2_slices = state.get('v2_slices', {})
        print(f'\nV2 Slices Configured: {len(v2_slices)}')
        
        # Check intelligence status
        ml_status = v2_slices.get('ml_strategy', {}).get('status', 'unknown')
        regime_status = v2_slices.get('market_regime', {}).get('status', 'unknown')
        print(f'ML Strategy: {ml_status}')
        print(f'Market Regime: {regime_status}')
except:
    print('❌ Could not read .knowledge/PROJECT_STATE.json')

# Check isolation
print('\n=== ISOLATION VERIFICATION ===')
import subprocess
result = subprocess.run(['grep', '-r', 'from bot_v2.features', 'src/bot_v2/features/'], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print('❌ Isolation violations found')
else:
    print('✅ Complete slice isolation verified')
"
```

---

## 🔄 V2 Usage Pattern

1. **Load single slice** for specific task (~500 tokens)
2. **Run slice diagnostic** (seconds vs minutes for V1)
3. **Check slice isolation** if imports fail
4. **Verify in PROJECT_STATE.json** for status
5. **Document in .knowledge/KNOWN_FAILURES.md** if new issue

V2 slices give instant feedback with minimal context loading.