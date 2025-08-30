# Known Failure Patterns & Solutions

## Purpose
This documents recurring failures and their proven solutions. Check here FIRST when encountering errors.

---

## 🔴 Import Errors (80% of all failures)

### Error: `ModuleNotFoundError: No module named 'bot.xxx'`
**Causes & Solutions:**
1. Missing `__init__.py` file
   ```bash
   # Fix: Create the missing __init__.py
   touch src/bot/xxx/__init__.py
   ```

2. Wrong import pattern
   ```python
   # ❌ BAD: Relative imports often fail
   from ..module import Class
   
   # ✅ GOOD: Absolute imports from bot
   from bot.module import Class
   ```

3. Module doesn't exist
   ```bash
   # Check if module exists
   ls src/bot/xxx/
   # If not, the feature might not be implemented
   ```

### Error: `ImportError: cannot import name 'XXX' from 'bot.module'`
**Solutions:**
1. Check if the class/function exists:
   ```bash
   grep -n "class XXX\|def XXX" src/bot/module/*.py
   ```

2. Check the `__init__.py` exports:
   ```python
   # src/bot/module/__init__.py should have:
   from .submodule import XXX
   __all__ = ['XXX']
   ```

### Error: `ERROR collecting tests/...` (pytest can't import)
**Solution:**
```bash
# Missing test fixtures
cp tests/fixtures/factories.py tests/fixtures/factories_backup.py
# Check fixture imports in conftest.py
cat tests/unit/conftest.py
```

---

## 🔴 Test Failures

### Error: `AttributeError: 'NoneType' object has no attribute 'xxx'`
**Common Locations & Fixes:**

1. Strategy returns None instead of DataFrame:
   ```python
   # ❌ BAD
   def calculate_signals(self, data):
       if data.empty:
           return None  # Causes AttributeError
   
   # ✅ GOOD
   def calculate_signals(self, data):
       if data.empty:
           return pd.DataFrame()  # Return empty DataFrame
   ```

2. Config not initialized:
   ```python
   # ❌ BAD
   config = None
   config.get_value()  # AttributeError
   
   # ✅ GOOD
   from bot.config import get_config
   config = get_config()
   ```

### Error: `KeyError: 'column_name'`
**Solution:**
```python
# Check DataFrame columns exist
if 'column_name' not in df.columns:
    df['column_name'] = default_value

# Or use get() with default
value = df.get('column_name', default_value)
```

### Error: Test parameters don't match
**Solution:**
```python
# Check fixture parameters
# Old fixtures use: short_window, long_window
# New fixtures use: fast, slow
# Update test to match fixture
```

---

## 🔴 Configuration Errors

### Error: `Config not initialized`
**Solution:**
```python
# Always initialize config first
from bot.config import get_config
config = get_config()

# Don't import Config directly
# ❌ from bot.config import Config
# ✅ from bot.config import get_config
```

### Error: Environment variables not found
**Solution:**
```bash
# Check .env file exists
ls -la .env
# Copy from template if missing
cp .env.template .env
```

---

## 🔴 CLI Errors

### Error: `usage: gpt-trader backtest: error: the following arguments are required`
**Solution:**
```bash
# Backtest requires specific arguments
poetry run gpt-trader backtest --symbol AAPL --start 2024-01-01 --end 2024-12-31

# Not just:
poetry run gpt-trader backtest --symbol AAPL --start 2024-01-01
```

### Error: Command times out
**Solution:**
```bash
# Add timeout flag for long operations
timeout 30 poetry run gpt-trader dashboard

# Or run in background
poetry run gpt-trader dashboard &
sleep 5
kill %1
```

---

## 🔴 Dependency Errors

### Error: `No module named 'talib'`
**Solution:**
```bash
# TA-Lib requires system library first
brew install ta-lib  # macOS
# or
sudo apt-get install ta-lib  # Ubuntu

# Then install Python wrapper
poetry add TA-Lib
```

### Error: `No module named 'alpaca'`
**Solution:**
```bash
poetry add alpaca-py
# NOT alpaca-trade-api (old version)
```

---

## 🔴 Data Pipeline Errors

### Error: YFinance download fails
**Solution:**
```python
# Add retry logic
import time
for attempt in range(3):
    try:
        data = yf.download(symbol)
        break
    except Exception as e:
        if attempt == 2:
            raise
        time.sleep(2)
```

### Error: Empty DataFrame operations
**Solution:**
```python
# Always check if DataFrame is empty
if df.empty:
    return pd.DataFrame(columns=['expected', 'columns'])
    
# Don't perform operations on empty DataFrame
```

---

## 🔴 Async/Threading Errors

### Error: `RuntimeError: There is no current event loop`
**Solution:**
```python
import asyncio

# For scripts/tests
if __name__ == "__main__":
    asyncio.run(main())

# For Jupyter/interactive
import nest_asyncio
nest_asyncio.apply()
```

### Error: Thread pool shutdown errors
**Solution:**
```python
# Always cleanup thread pools
from bot.core.concurrency import get_concurrency_manager
manager = get_concurrency_manager()
# Use manager...
manager.shutdown()  # Important!
```

---

## 🔴 File Path Errors

### Error: Relative path not found
**Solution:**
```python
# ❌ BAD: Relative paths break when run from different directories
with open("../data/file.csv")

# ✅ GOOD: Use absolute paths
from pathlib import Path
file_path = Path(__file__).parent.parent / "data" / "file.csv"
with open(file_path)
```

---

## 🟡 Common Warning Patterns

### Warning: `DeprecationWarning: use options instead of chrome_options`
**Ignore** - From selenium, doesn't affect functionality

### Warning: `RuntimeWarning: divide by zero`
**Fix:**
```python
# Add zero check
if denominator != 0:
    result = numerator / denominator
else:
    result = 0  # or np.nan
```

---

## 🔧 Quick Fixes Checklist

When component fails, try these in order:

1. **Check imports work**
   ```bash
   python -c "from bot.component import *"
   ```

2. **Check test fixtures exist**
   ```bash
   ls tests/fixtures/*.py
   ```

3. **Initialize config**
   ```bash
   python -c "from bot.config import get_config; get_config()"
   ```

4. **Install missing packages**
   ```bash
   poetry install
   poetry update
   ```

5. **Clear caches**
   ```bash
   rm -rf .pytest_cache __pycache__
   ```

6. **Check file exists**
   ```bash
   ls -la src/bot/module/file.py
   ```

---

## 📝 When Adding New Failures

Format:
```markdown
### Error: `Exact error message`
**Common Cause:** Brief description
**Solution:**
\```language
Exact code or command that fixes it
\```
**Files Often Affected:** List of files
```

Keep solutions SPECIFIC and TESTED.
### Error: `@patch("src.bot.module")` in tests
**Added**: 2025-08-16
**Solution:**
Change to `@patch("bot.module")` - no "src" prefix
```python
# ❌ WRONG
@patch("src.bot.dataflow.validate.function")

# ✅ CORRECT  
@patch("bot.dataflow.validate.function")
```
**Files Often Affected:** All test files using @patch

### Error: Mock not being called (AssertionError: Expected 'X' to have been called)
**Common Causes:**
1. Wrong patch path - check actual import in code
2. Mock not attached to right object
3. Function not actually called in test

**Solution:**
```python
# Check what the code actually imports
grep "from .* import X" src/bot/module/file.py

# Patch at the usage point, not definition
@patch("bot.module.file.X")  # Where it's used
# Not: @patch("bot.other.X")  # Where it's defined
```

### Error: Mock 'adjust_to_adjclose' not called / Mock 'validate_daily_bars' not called
**Added**: 2025-08-16
**Common Cause:** Patching imported functions at wrong location
**Solution:**
```python
# If pipeline.py has: from bot.dataflow.validate import adjust_to_adjclose
# Then patch where it's USED, not where it's DEFINED:

# ❌ WRONG - patches at definition location
@patch("bot.dataflow.validate.adjust_to_adjclose")

# ✅ CORRECT - patches at usage location  
@patch("bot.dataflow.pipeline.adjust_to_adjclose")
```
**Files Affected:** 
- tests/unit/dataflow/test_pipeline.py (6 occurrences)
- tests/unit/dataflow/test_pipeline_multisource.py (2 occurrences)

### Error: Fixture not found
**Solution:**
```python
# Check available fixtures
poetry run pytest --fixtures tests/unit/

# Common fixtures location
tests/fixtures/factories.py
tests/unit/conftest.py
tests/integration/conftest.py

# If missing, check fixture name matches
@pytest.fixture
def sample_dataframe():  # Must match parameter name
    return pd.DataFrame()
```

### Error: Test collection error with cryptic message
**Solution:**
```bash
# Get better error message
poetry run python -c "import tests.unit.module.test_file"

# Common causes:
# 1. Missing __init__.py in tests/unit/module/
# 2. Syntax error in test file
# 3. Import error at module level
```

### Error: Tests pass individually but fail together
**Common Cause:** Shared state between tests
**Solution:**
```python
# Add cleanup in tests
@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Reset any global state
    if hasattr(SomeClass, '_instance'):
        SomeClass._instance = None
```


### Error: `AttributeError: object has no attribute method_name`
**Added**: 2025-08-16  
**Common Cause:** Test expects functionality that is not implemented
**Solutions:**
1. Check if feature is actually implemented:
   ```bash
   grep -n "def method_name" src/bot/module/file.py
   ```
2. If not implemented, either:
   - Skip test: `@pytest.mark.skip("Not implemented")`
   - Implement the method
   - Remove test if feature deprecated

### Error: Strategies generate signals but no trades executed
**Added**: 2025-08-16
**Root Cause:** Allocator only checks LAST bar's signal value
**Location:** `src/bot/portfolio/allocator.py` lines 82-96
**Impact:** Strategies that exit before backtest end get 0 allocation
**STATUS:** ✅ FIXED 2025-08-16

**Solution Applied:**
```python
# Fixed code in allocate_signals():
sig = df["signal"].iloc[-1]
# If no signal on last bar, check recent window (last 120 bars)
if int(_to_float(sig)) <= 0:
    recent_window = df["signal"].iloc[-120:]  # Last 120 bars (~6 months)
    active_signals = recent_window[recent_window > 0]
    if not active_signals.empty:
        sig = active_signals.iloc[-1]  # Use most recent active signal
```

### Error: Strategy parameters too conservative for autonomous trading
**Added**: 2025-08-16
**Root Cause:** Strategy defaults designed for extreme market conditions
**Impact:** 4/7 strategies generated 0 signals in normal market conditions
**Affected:** mean_reversion, momentum, optimized_ma, enhanced_trend_breakout

**Solution Applied:**
- **Mean Reversion**: RSI 30/70 → 40/60, period 14 → 10 ✅ FIXED
- **Momentum**: 3% → 1.5% threshold, 1.5x → 1.2x volume ✅ Parameters fixed  
- **OptimizedMA**: MA 10/20 → 5/15, disabled restrictive filters ✅ Parameters fixed
- **Enhanced Trend**: 55 → 20 day lookback, disabled volume filter ✅ Parameters fixed

**Current Status:** 1 completely working, 3 generate signals but need allocator compatibility fix

### Error: Allocator compatibility issue with 3 strategies
**Added**: 2025-08-16
**Root Cause:** Bridge passed ALL signal columns including indicators with NaN values
**Evidence:** "Generated 12 momentum entries, 159 total signal periods" but "Allocated 0 positions"
**Affected:** momentum, optimized_ma, enhanced_trend_breakout
**STATUS:** ✅ FIXED 2025-08-16

**Solution Applied:**
```python
# strategy_allocator_bridge.py:89-102
# Extract only essential columns to avoid NaN issues
essential_signal_cols = []
if "signal" in signals.columns:
    essential_signal_cols.append("signal")
if "atr" in signals.columns:
    essential_signal_cols.append("atr")

if essential_signal_cols:
    essential_signals = signals[essential_signal_cols]
    combined = data.join(essential_signals, how="left")
```

**Result:** All 3 strategies now execute trades successfully

