# Quick Diagnostic Commands

## Purpose
Fast one-liners to diagnose issues WITHOUT running full test suites. Run these BEFORE diving into debugging.

---

## 🚀 System Health Checks

### 1. Overall Import Health
```bash
# Check if bot package imports at all
poetry run poetry run python -c "import bot; print('✅ Bot package imports')"

# Check all major components import
poetry run poetry run python -c "
from bot import config, dataflow, strategy, risk, portfolio, backtest
print('✅ All major components import')
"

# QUIET VERSION (no logging noise)
poetry run poetry run python -c "import os; os.environ['SUPPRESS_LOGS']='1'; import bot; print('✅ Bot imports (quiet)')" 2>/dev/null
```

### 2. Configuration Health
```bash
# Check config initializes
poetry run poetry run python -c "from bot.config import get_config; c = get_config(); print('✅ Config works')"

# Check environment
poetry run poetry run python -c "from bot.config import get_config; c = get_config(); print(f'Environment: {c.environment}')"

# QUIET VERSION
poetry run poetry run python -c "import os; os.environ['SUPPRESS_LOGS']='1'; from bot.config import get_config; c = get_config(); print('✅ Config works (quiet)')" 2>/dev/null
```

### 3. Quick Component Status
```bash
# One-line status check
poetry run python -c "import json; s = json.load(open('PROJECT_STATE.json')); print(f\"System: {s['system_status']['functional_percentage']}% functional\")"

# List broken components
poetry run python -c "import json; s = json.load(open('PROJECT_STATE.json')); print('Broken:', [k for k,v in s['components'].items() if v['status'] == 'failed'])"
```

---

## 🔍 Component-Specific Diagnostics

### Data Pipeline
```bash
# Check import
poetry run python -c "from bot.dataflow.pipeline import DataPipeline; print('✅ Pipeline imports')"

# Test data fetch (quick)
poetry run python -c "
from bot.dataflow.pipeline import DataPipeline
p = DataPipeline()
print('✅ Pipeline instantiates')
"

# Check if YFinance works
poetry run python -c "import yfinance as yf; print('YFinance version:', yf.__version__)"
```

### Strategies
```bash
# List available strategies
poetry run python -c "
from bot.strategy import AVAILABLE_STRATEGIES
print('Available strategies:', list(AVAILABLE_STRATEGIES.keys()))
"

# Check specific strategy imports
poetry run python -c "from bot.strategy.demo_ma import DemoMAStrategy; print('✅ DemoMA imports')"
poetry run python -c "from bot.strategy.trend_breakout import TrendBreakoutStrategy; print('✅ TrendBreakout imports')"
```

### Risk Management
```bash
# Check risk config
poetry run python -c "from bot.risk.config import RiskConfig; RiskConfig(); print('✅ RiskConfig works')"

# Check risk integration
poetry run python -c "from bot.risk.integration import RiskIntegration; print('✅ RiskIntegration imports')"
```

### Portfolio
```bash
# Check allocator
poetry run python -c "from bot.portfolio.allocator import PortfolioAllocator; print('✅ Allocator imports')"

# Check rules
poetry run python -c "from bot.portfolio.allocator import PortfolioRules; print('✅ Rules import')"
```

### Backtesting
```bash
# Check engine imports
poetry run python -c "from bot.backtest.engine import BacktestEngine; print('✅ Engine imports')"

# Check orchestrator
poetry run python -c "from bot.integration.orchestrator import IntegratedOrchestrator; print('✅ Orchestrator imports')"
```

### ML Pipeline
```bash
# Check ML imports (often broken)
poetry run python -c "from bot.ml import models; print('✅ ML models import')"

# Check if sklearn installed
poetry run python -c "import sklearn; print('Sklearn version:', sklearn.__version__)"
```

---

## 🐛 Test Diagnostics

### Find Test Failures Pattern
```bash
# Count test collection errors (import issues)
poetry run pytest --collect-only 2>&1 | grep ERROR | wc -l

# Show first import error
poetry run pytest --collect-only 2>&1 | grep -A5 "ERROR collecting" | head -10

# List working test files
poetry run pytest --collect-only -q 2>/dev/null | grep "\.py::" | cut -d: -f1 | sort -u | head -10
```

### Check Test Fixtures
```bash
# List available fixtures
poetry run pytest --fixtures -q | grep "@pytest.fixture" | head -10

# Check if test fixtures import
poetry run python -c "from tests.fixtures.factories import *; print('✅ Test fixtures import')"
```

---

## 📦 Dependency Diagnostics

### Check Package Versions
```bash
# Key packages
poetry run python -c "
import pandas as pd
import numpy as np
print(f'Pandas: {pd.__version__}')
print(f'Numpy: {np.__version__}')
"

# Trading packages
poetry show | grep -E "yfinance|alpaca|ta-lib"

# ML packages  
poetry show | grep -E "scikit-learn|joblib|tensorflow"
```

### Find Missing Imports
```bash
# Scan for import errors in last run
poetry run python -c "
import sys
import importlib
modules = ['bot.dataflow', 'bot.strategy', 'bot.risk', 'bot.ml']
for m in modules:
    try:
        importlib.import_module(m)
        print(f'✅ {m}')
    except ImportError as e:
        print(f'❌ {m}: {e}')
"
```

---

## 🔥 Emergency Diagnostics

### When Nothing Works
```bash
# Nuclear reset
rm -rf .venv __pycache__ .pytest_cache
poetry install
poetry run poetry run python -c "import bot; print('Fresh install works')"
```

### Find Circular Imports
```bash
# Will hang or error if circular imports exist
poetry run python -c "
import sys
sys.setrecursionlimit(50)  # Make circular imports fail fast
import bot
print('No circular imports detected')
"
```

### Check File Permissions
```bash
# Ensure all Python files are readable
find src -name "*.py" ! -perm -444 -ls

# Fix permissions if needed
find src -name "*.py" -exec chmod 644 {} \;
```

---

## 💡 Performance Diagnostics

### Memory Usage
```bash
# Check if memory leak in imports
poetry run python -c "
import tracemalloc
tracemalloc.start()
import bot
current, peak = tracemalloc.get_traced_memory()
print(f'Memory used: {current / 10**6:.1f} MB')
tracemalloc.stop()
"
```

### Import Time
```bash
# Time how long imports take
poetry run python -c "
import time
start = time.time()
import bot
print(f'Import time: {time.time() - start:.2f} seconds')
"
```

---

## 📊 Quick Status Report

### Generate Instant Health Report
```bash
# Run all quick checks
poetry run python -c "
import json
import subprocess
import sys

checks = {
    'Config': 'from bot.config import get_config; get_config()',
    'Pipeline': 'from bot.dataflow.pipeline import DataPipeline',
    'Strategy': 'from bot.strategy import AVAILABLE_STRATEGIES',
    'Risk': 'from bot.risk.config import RiskConfig',
    'Tests': 'from tests.fixtures.factories import *'
}

print('=== QUICK HEALTH CHECK ===')
for name, check in checks.items():
    try:
        exec(check)
        print(f'✅ {name}: Working')
    except Exception as e:
        print(f'❌ {name}: {e.__class__.__name__}')

# Show system percentage
try:
    with open('PROJECT_STATE.json') as f:
        state = json.load(f)
        print(f\"\nSystem: {state['system_status']['functional_percentage']}% functional\")
except:
    print('Could not read .knowledge/PROJECT_STATE.json')
"
```

---

## 🔄 Usage Pattern

1. **Start here** when component fails
2. **Run relevant diagnostic** (takes seconds)
3. **If diagnostic fails**, check `.knowledge/KNOWN_FAILURES.md`
4. **If not listed**, investigate with full tests
5. **Document new failure** in .knowledge/KNOWN_FAILURES.md

These commands give instant feedback without waiting for full test suites.