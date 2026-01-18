# Backtest Reporting Flow Map

Generated: 2026-01-18T01:50:49.746236+00:00

## Simulation Broker
| ID | Label | Path |
|----|-------|------|
| simulated_broker | SimulatedBroker | `src/gpt_trader/backtesting/simulation/broker.py` |
| broker_stats | SimulatedBroker.get_statistics | `src/gpt_trader/backtesting/simulation/broker.py` |

## Metrics
| ID | Label | Path |
|----|-------|------|
| trade_stats_compute | calculate_trade_statistics | `src/gpt_trader/backtesting/metrics/statistics.py` |
| trade_stats | TradeStatistics | `src/gpt_trader/backtesting/metrics/statistics.py` |
| risk_metrics_compute | calculate_risk_metrics | `src/gpt_trader/backtesting/metrics/risk.py` |
| risk_metrics | RiskMetrics | `src/gpt_trader/backtesting/metrics/risk.py` |

## Reporter
| ID | Label | Path |
|----|-------|------|
| backtest_reporter | BacktestReporter.generate_result | `src/gpt_trader/backtesting/metrics/report.py` |
| generate_backtest_report | generate_backtest_report | `src/gpt_trader/backtesting/metrics/report.py` |

## Outputs
| ID | Label | Path |
|----|-------|------|
| backtest_result_report | BacktestResult | `src/gpt_trader/backtesting/types.py` |
| report_summary | BacktestReporter.generate_summary | `src/gpt_trader/backtesting/metrics/report.py` |
| report_csv | BacktestReporter.generate_csv_row | `src/gpt_trader/backtesting/metrics/report.py` |

## Edges
| From | To | Description |
|------|----|-------------|
| simulated_broker | trade_stats_compute | trade history |
| trade_stats_compute | trade_stats | build stats |
| simulated_broker | risk_metrics_compute | equity curve |
| risk_metrics_compute | risk_metrics | build metrics |
| simulated_broker | broker_stats | broker stats |
| trade_stats | backtest_reporter | trade statistics |
| risk_metrics | backtest_reporter | risk metrics |
| broker_stats | backtest_reporter | summary stats |
| backtest_reporter | backtest_result_report | BacktestResult |
| backtest_reporter | report_summary | summary text |
| backtest_reporter | report_csv | csv row |
| simulated_broker | generate_backtest_report | input broker |
| generate_backtest_report | backtest_reporter | construct reporter |
| generate_backtest_report | backtest_result_report | return result |

## Notes
- BacktestReporter lazily computes TradeStatistics and RiskMetrics from the broker.
- generate_backtest_report is a convenience wrapper returning BacktestResult.
