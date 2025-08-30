
# Performance Report – GPT-Trader System (2025-08-15)

## Executive Summary
| Metric | Value | Status |
|--------|-------|--------|
| Total Bottlenecks | 5 | ⚠️ |
| Import Failures | 3 | ❌ |
| System Memory | 12.1GB available | ℹ️ |

## Import Performance Analysis

### Import Times (sorted by slowest)
| Module | Time (ms) | Memory (MB) |
|--------|-----------|-------------|
| Risk Management | 523.42 | 6.38 |
| CLI Commands | 44.74 | 1.9 |
| Integration Orchestrator | 5.76 | 0.11 |
| Data Pipeline | 4.12 | 0.07 |
| Portfolio Backtest Engine | 2.18 | 0.08 |
| YFinance Data Source | 1.42 | 0.03 |
| Trend Breakout Strategy | 1.27 | 0.02 |
| Demo MA Strategy | 1.13 | 0.01 |
| Strategy Base | 0.3 | 0.01 |

### Failed Imports
- **Configuration System**: bot.core.exceptions.GPTTraderException.__init__() got multiple values for keyword argument 'context'
- **Core Base Module**: bot.core.exceptions.GPTTraderException.__init__() got multiple values for keyword argument 'context'
- **ML Baseline Models**: cannot import name 'AttentionType' from 'bot.ml.deep_learning' (/Users/rj/PycharmProjects/GPT-Trader/src/bot/ml/deep_learning/__init__.py)

## Bottlenecks Identified
1. **Import Failure**: Import failure: Configuration System
2. **Import Failure**: Import failure: Core Base Module
3. **Import Failure**: Import failure: ML Baseline Models
4. **Data Pipeline Failure**: Data pipeline test failure
5. **Backtest Failure**: Backtest execution failure

---
**Report generated**: 2025-08-14T17:37:19.769431
