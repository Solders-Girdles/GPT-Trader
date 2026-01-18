# Backtesting Flow Map

Generated: 2026-01-18T01:50:49.741733+00:00

## Event Store
| ID | Label | Path |
|----|-------|------|
| event_store | EventStore.events | `src/gpt_trader/persistence/event_store.py` |

## Historical Loader
| ID | Label | Path |
|----|-------|------|
| data_loader | HistoricalDataLoader.load_symbol | `src/gpt_trader/features/research/backtesting/data_loader.py` |
| data_result | DataLoadResult | `src/gpt_trader/features/research/backtesting/data_loader.py` |
| data_point | HistoricalDataPoint | `src/gpt_trader/features/research/backtesting/data_loader.py` |

## Backtest Simulation
| ID | Label | Path |
|----|-------|------|
| backtest_simulator | BacktestSimulator.run | `src/gpt_trader/features/research/backtesting/simulator.py` |

## Strategy
| ID | Label | Path |
|----|-------|------|
| market_data_context | MarketDataContext | `src/gpt_trader/features/live_trade/strategies/base.py` |
| strategy_decide | Strategy.decide | `src/gpt_trader/features/live_trade/strategies/base.py` |

## Metrics + Output
| ID | Label | Path |
|----|-------|------|
| backtest_result | BacktestResult | `src/gpt_trader/features/research/backtesting/simulator.py` |
| performance_metrics | PerformanceMetrics.from_result | `src/gpt_trader/features/research/backtesting/metrics.py` |

## Edges
| From | To | Description |
|------|----|-------------|
| event_store | data_loader | load events |
| data_loader | data_result | build result |
| data_result | data_point | points list |
| data_point | backtest_simulator | data_points |
| backtest_simulator | market_data_context | build market data |
| market_data_context | strategy_decide | strategy input |
| strategy_decide | backtest_simulator | decision |
| backtest_simulator | backtest_result | result |
| backtest_result | performance_metrics | compute metrics |

## Notes
- HistoricalDataLoader reconstructs market state from EventStore events.
- BacktestSimulator replays HistoricalDataPoint sequences into strategy decisions.
- PerformanceMetrics summarizes outcomes from BacktestResult.
