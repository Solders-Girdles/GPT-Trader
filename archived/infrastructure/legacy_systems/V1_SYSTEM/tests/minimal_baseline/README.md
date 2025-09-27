# Minimal Test Baseline for GPT-Trader

## Overview

This directory contains the **minimal set of 25+ tests** that must pass for GPT-Trader to be considered minimally functional. These tests focus on **actual working functionality** rather than comprehensive coverage.

## Current Status (TEST-004)

**Target**: 80% pass rate for minimal functionality
**Focus**: Real functionality, not mocks of mocks

## Test Categories

### 1. Core Imports (`test_core_imports.py`)
- ✅ Configuration system loads
- ✅ Strategy classes import
- ✅ Data sources import
- ✅ Indicators import
- ✅ Backtest engine imports (correct one)
- ✅ Logging system works

### 2. Data Pipeline (`test_data_pipeline.py`)
- ✅ YFinance source initialization
- ✅ Data structure validation
- ✅ ATR indicator calculations
- ✅ Data caching system
- 🐌 Actual network data download (slow test)

### 3. Strategy Tests (`test_strategies.py`)
- ✅ Demo MA strategy creation and signals
- ✅ Trend breakout strategy creation
- ✅ Signal value validation
- ✅ Edge case handling (insufficient data, flat prices)
- ✅ Parameter validation

### 4. Integration Tests (`test_integration.py`)
- ✅ Config + Strategy integration
- ✅ Data + Strategy integration
- ✅ Portfolio allocator basic functionality
- ✅ Backtest engine creation
- 🐌 Simple end-to-end backtest flow (slow test)
- ✅ Logging across components

### 5. Risk Management (`test_risk.py`)
- ✅ Stop loss generation
- ✅ Take profit generation
- ✅ ATR-based risk calculations
- ✅ Position sizing basics
- ✅ Risk parameter validation
- ✅ Extreme market condition handling

## Running the Tests

### Quick Import Test
```bash
python scripts/test_baseline_quick.py
```

### Quick Strategy Test
```bash
python scripts/test_strategy_quick.py
```

### Full Baseline Suite
```bash
python scripts/run_baseline_tests.py
```

### With Slow Tests
```bash
python scripts/run_baseline_tests.py --slow
```

### With Coverage
```bash
python scripts/run_baseline_tests.py --coverage
```

### Direct pytest
```bash
pytest tests/minimal_baseline -c pytest_baseline.ini -v
```

## Test Philosophy

### What We Test
- **Real imports** - Can we actually import the modules?
- **Real data** - Do indicators work with actual DataFrames?
- **Real signals** - Do strategies generate valid outputs?
- **Real integration** - Do components talk to each other?

### What We Don't Test
- **Perfect accuracy** - We don't test if strategies are profitable
- **Complete features** - We only test what currently works
- **Performance** - We test functionality, not speed
- **Complex scenarios** - We test basic happy paths

### Test Design Principles
1. **No import errors** - Every test must actually run
2. **Test reality** - Use real data structures and calls
3. **Fail fast** - Use `-x` flag to stop on first failure
4. **Clear feedback** - Tests explain what they verify
5. **Minimal dependencies** - Don't test complex integrations

## Expected Results

**Week 1 Target**: 80% pass rate
- All import tests should pass
- Most strategy tests should pass
- Basic integration should work
- Some slow tests may fail (network issues)

**Known Issues**:
- Network-dependent tests may be flaky
- TrendBreakout strategy may have calculation issues
- Some integration tests depend on modules that don't exist yet

## Adding Tests

When adding new tests to this baseline:

1. **Test must actually run** - No import errors allowed
2. **Test real functionality** - Not mocks or stubs
3. **Keep it simple** - Test one thing clearly
4. **Use existing fixtures** - From `conftest.py`
5. **Mark slow tests** - Use `@pytest.mark.slow` for >1 second tests

## Test Markers

- `@pytest.mark.slow` - Tests that take >1 second (network, large data)
- `@pytest.mark.critical` - Tests that absolutely must pass
- `@pytest.mark.integration` - Tests that verify component interaction

## File Structure

```
tests/minimal_baseline/
├── README.md                 # This file
├── __init__.py              # Package marker
├── test_core_imports.py     # Core import verification (9 tests)
├── test_data_pipeline.py    # Data loading and processing (7 tests)
├── test_strategies.py       # Strategy signal generation (8 tests)
├── test_integration.py      # Component integration (9 tests)
└── test_risk.py            # Risk management basics (9 tests)
```

**Total: ~42 tests covering critical functionality**

## Success Criteria

✅ **Minimal Functionality Achieved When**:
- All import tests pass (no ImportError)
- Strategies generate signals without crashing
- Data pipeline loads and validates data
- Basic integration between components works
- Risk calculations produce reasonable results

This baseline represents the **minimum viable system** - if these tests pass, users can run basic backtests and strategies work.
