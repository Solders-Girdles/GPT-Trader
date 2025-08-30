# Import Pattern Guide

## The Golden Rule
**ALWAYS use absolute imports from `bot` root. NEVER use relative imports.**

```python
# ✅ CORRECT - Always use this pattern
from bot.module.submodule import ClassName

# ❌ WRONG - Never use these patterns
from ..module import ClassName  # Relative
from src.bot.module import ClassName  # Including src
from module import ClassName  # Missing bot prefix
import sys; sys.path.append()  # Path hacking
```

---

## 📦 Correct Import Patterns by Module

### Configuration (Import First!)
```python
# ✅ CORRECT
from bot.config import get_config
config = get_config()

# ❌ WRONG
from bot.config import Config  # Don't import class directly
from bot.config.unified_config import Config  # Don't dig into submodules
```

### Data Pipeline
```python
# ✅ CORRECT
from bot.dataflow.pipeline import DataPipeline, PipelineConfig
from bot.dataflow.pipeline import DataQualityMetrics

# ❌ WRONG
from dataflow.pipeline import DataPipeline  # Missing bot
from ..dataflow.pipeline import DataPipeline  # Relative
```

### Strategies
```python
# ✅ CORRECT
from bot.strategy import AVAILABLE_STRATEGIES
from bot.strategy.demo_ma import DemoMAStrategy
from bot.strategy.base import BaseStrategy

# ❌ WRONG
from strategy.demo_ma import DemoMAStrategy  # Missing bot
from bot.strategy import DemoMAStrategy  # Wrong level
```

### Risk Management
```python
# ✅ CORRECT
from bot.risk.config import RiskConfig
from bot.risk.integration import RiskIntegration
from bot.risk.simple_risk_manager import SimpleRiskManager

# ❌ WRONG
from risk.config import RiskConfig  # Missing bot
from bot.risk import RiskConfig  # Not exported from __init__
```

### Portfolio
```python
# ✅ CORRECT
from bot.portfolio.allocator import PortfolioAllocator, PortfolioRules
from bot.portfolio import PortfolioAllocator  # If exported in __init__.py

# ❌ WRONG
from portfolio.allocator import PortfolioAllocator  # Missing bot
```

### ML Pipeline
```python
# ✅ CORRECT
from bot.ml.models.strategy_selector import StrategySelector
from bot.ml.integrated_pipeline import IntegratedMLPipeline

# ❌ WRONG
from bot.ml import StrategySelector  # Not at this level
from ml.models import StrategySelector  # Missing bot
```

### Tests
```python
# ✅ CORRECT (in test files)
from bot.strategy.demo_ma import DemoMAStrategy
from tests.fixtures.factories import create_sample_data

# ❌ WRONG
from ..fixtures.factories import create_sample_data  # Relative
from src.bot.strategy import DemoMAStrategy  # Including src
```

---

## 🔧 Required `__init__.py` Files

These MUST exist for imports to work:

```
src/bot/__init__.py
src/bot/config/__init__.py
src/bot/dataflow/__init__.py
src/bot/strategy/__init__.py
src/bot/risk/__init__.py
src/bot/portfolio/__init__.py
src/bot/backtest/__init__.py
src/bot/ml/__init__.py
src/bot/ml/models/__init__.py
src/bot/indicators/__init__.py
src/bot/live/__init__.py
src/bot/dashboard/__init__.py
tests/__init__.py
tests/fixtures/__init__.py
tests/unit/__init__.py
tests/integration/__init__.py
```

### Check & Create Missing `__init__.py`
```bash
# Find directories missing __init__.py
find src/bot -type d '!' -exec test -e "{}/__init__.py" ';' -print

# Create all missing __init__.py files
find src/bot -type d '!' -exec test -e "{}/__init__.py" ';' -exec touch "{}/__init__.py" ';'
```

---

## 🎯 Common Import Fixes

### Fix 1: Module Not Found
```bash
# Diagnosis
python -c "from bot.missing.module import Class"

# Solution 1: Create missing __init__.py
touch src/bot/missing/__init__.py

# Solution 2: Check typo in module name
ls src/bot/  # List actual module names

# Solution 3: Module not implemented yet
grep -r "class ClassName" src/  # Search for the class
```

### Fix 2: Cannot Import Name
```python
# Error: cannot import name 'Thing' from 'bot.module'

# Check if Thing exists
grep "class Thing\|def Thing" src/bot/module/*.py

# Check if exported in __init__.py
cat src/bot/module/__init__.py

# Fix: Add to __init__.py
echo "from .submodule import Thing" >> src/bot/module/__init__.py
```

### Fix 3: Circular Import
```python
# Symptom: ImportError or RecursionError

# Diagnosis: Find circular dependency
python -c "
import sys
sys.setrecursionlimit(50)
from bot.module import Class  # Will fail fast if circular
"

# Solution: Move shared code to separate module
# Instead of A imports B, B imports A
# Create C that both A and B import
```

### Fix 4: Import Works in Script but Not in Test
```python
# Problem: Different working directories

# Solution: Always use absolute imports
# Never rely on working directory

# In tests/conftest.py add:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

---

## 📋 Import Order Convention

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

# Local application imports
from bot.config import get_config
from bot.dataflow.pipeline import DataPipeline
from bot.strategy.base import BaseStrategy

# Initialize config first if needed
config = get_config()
```

---

## 🚫 Import Anti-Patterns to Avoid

### 1. Never Modify sys.path
```python
# ❌ NEVER DO THIS
import sys
sys.path.append('../src')
sys.path.insert(0, '/path/to/module')
```

### 2. Never Use Star Imports in Production
```python
# ❌ AVOID
from bot.strategy import *

# ✅ BETTER
from bot.strategy import DemoMAStrategy, TrendBreakoutStrategy
```

### 3. Never Import from Private Modules
```python
# ❌ WRONG
from bot.module._private import InternalClass

# ✅ RIGHT
from bot.module import PublicClass
```

### 4. Never Use Relative Imports
```python
# ❌ WRONG
from . import module
from .. import parent_module
from ...package import module

# ✅ RIGHT
from bot.package.module import Class
```

---

## 🔍 Debugging Import Issues

### Step-by-Step Process
```bash
# 1. Try direct import
python -c "from bot.module.submodule import Class"

# 2. Check module exists
ls -la src/bot/module/submodule.py

# 3. Check __init__.py exists
ls -la src/bot/module/__init__.py

# 4. Check class exists in file
grep "class Class" src/bot/module/submodule.py

# 5. Check if exported
grep "Class" src/bot/module/__init__.py

# 6. Try importing module only
python -c "import bot.module; print(dir(bot.module))"

# 7. Check for syntax errors
python -m py_compile src/bot/module/submodule.py
```

---

## ✅ Quick Import Validation

```python
# Run this to validate all imports work
python -c "
modules = [
    'bot.config',
    'bot.dataflow.pipeline',
    'bot.strategy.base',
    'bot.strategy.demo_ma',
    'bot.risk.config',
    'bot.portfolio.allocator',
    'bot.backtest.engine',
    'bot.ml.models',
]

for module in modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError as e:
        print(f'❌ {module}: {e}')
"
```

---

## 📝 Summary

1. **Always use**: `from bot.module import Class`
2. **Never use**: Relative imports or path hacking
3. **Check**: `__init__.py` files exist
4. **Export**: Classes in `__init__.py` for clean imports
5. **Debug**: Use step-by-step process above

When in doubt, check how existing working code does imports and copy that pattern.