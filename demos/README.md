# GPT-Trader Demos - Emergency Recovery

This directory contains working demonstrations that prove the GPT-Trader system has functioning components.

## Available Demos

### 1. Download Data Demo (`download_data.py`)
**Status: ✅ WORKING**

A simple demonstration that:
- Downloads market data for AAPL, MSFT, GOOGL
- Uses the working YFinanceSource
- Saves data to `data/historical/` directory
- Shows clear output of what was downloaded

**Usage:**
```bash
poetry run python demos/download_data.py
```

**Expected Output:**
- Downloads ~90 days of data for 3 symbols
- Creates CSV files in `data/historical/`
- Shows success summary with file sizes and latest prices

### 2. Standalone Online Learning Demo (`../standalone_demo.py`)
**Status: ✅ WORKING (but verbose)**

A comprehensive demonstration of the ML online learning pipeline that:
- Tests learning rate schedulers
- Tests concept drift detectors
- Runs online learning simulation with synthetic data
- Shows all Phase 3 ADAPT tasks are implemented

**Usage:**
```bash
poetry run python standalone_demo.py
```

**Note:** This demo is quite verbose with lots of drift detection output, but proves the ML components work.

## Recovery Status

✅ **Data Download**: Working - can fetch real market data
✅ **ML Pipeline**: Working - online learning components functional
✅ **YFinance Integration**: Working - caching and retry logic functional
✅ **Poetry Environment**: Working - dependencies resolved

## Next Steps for Full Recovery

1. Test strategy implementations (demo_ma, trend_breakout)
2. Test backtesting engine with real data
3. Create end-to-end trading simulation demo
4. Verify CLI functionality beyond data download

## Files Created

- `download_data.py` - Main data download demo
- `README.md` - This file
- `../data/historical/*.csv` - Downloaded market data files

This proves the system isn't completely broken and has working components that can be built upon.
