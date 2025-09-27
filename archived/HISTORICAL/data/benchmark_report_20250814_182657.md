
# Performance Report – GPT-Trader System (2025-08-15)

## Executive Summary
| Metric | Value | Status |
|--------|-------|--------|
| Total Bottlenecks | 5 | ⚠️ |
| Import Failures | 3 | ❌ |
| System Memory | 12.3GB available | ℹ️ |

## Import Performance Analysis
### Successful Imports
| Module | Time (ms) | Memory (MB) |
|--------|-----------|-------------|
| Configuration System | 4512.12 | 117.73 |
| CLI Commands | 40.02 | 1.87 |
| Integration Orchestrator | 4.91 | 0.12 |
| Risk Management | 3.77 | 0.09 |
| Data Pipeline | 2.99 | 0.07 |
| Portfolio Backtest Engine | 2.15 | 0.08 |
| YFinance Data Source | 1.4 | 0.03 |
| Trend Breakout Strategy | 1.27 | 0.02 |
| Demo MA Strategy | 0.35 | 0.01 |
| Strategy Base | 0.3 | 0.01 |

### Failed Imports
- **Core Base Module**: bot.core.exceptions.GPTTraderException.__init__() got multiple values for keyword argument 'context'
- **ML Baseline Models**: cannot import name 'FeatureSelectionConfig' from 'bot.ml.feature_selector' (/Users/rj/PycharmProjects/GPT-Trader/src/bot/ml/feature_selector.py)
- **ML Performance Benchmark**: cannot import name 'FeatureSelectionConfig' from 'bot.ml.feature_selector' (/Users/rj/PycharmProjects/GPT-Trader/src/bot/ml/feature_selector.py)

## Component Performance

### Data Pipeline
- **Symbols Tested**: 3
- **Total Time Seconds**: 0.1
- **Time Per Symbol Seconds**: 0.03
- **Throughput Symbols Per Minute**: 1820.55

### Strategy Demoma
- **Name**: DemoMA
- **Data Points**: 366
- **Calculation Time Seconds**: 0.0072
- **Signals Generated**: 366
- **Throughput Points Per Second**: 51006.61

### Strategy Trendbreakout
- **Name**: TrendBreakout
- **Data Points**: 366
- **Calculation Time Seconds**: 0.0069
- **Signals Generated**: 366
- **Throughput Points Per Second**: 53012.75

## Bottlenecks Identified
1. **Slow Import**: Slow import: Configuration System
2. **Import Failure**: Import failure: Core Base Module
3. **Import Failure**: Import failure: ML Baseline Models
4. **Import Failure**: Import failure: ML Performance Benchmark
5. **Backtest Failure**: Backtest execution failure

## Recommendations

### High Priority
- **Dependencies**: Fix missing dependencies or import errors
- **Dependencies**: Fix missing dependencies or import errors
- **Dependencies**: Fix missing dependencies or import errors

### Medium Priority
- **Imports**: Consider lazy imports or module restructuring
- **Parallelization**: Implement parallel processing for backtests and data fetching

---
**Report generated**: 2025-08-14T18:26:52.463679
