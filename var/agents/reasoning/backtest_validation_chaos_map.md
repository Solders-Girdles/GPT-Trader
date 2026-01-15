# Backtest Validation + Chaos Flow Map

Generated: 2026-01-15T13:32:26.240490+00:00

## Simulation Loop
| ID | Label | Path |
|----|-------|------|
| clocked_bar_runner | ClockedBarRunner.run | `src/gpt_trader/backtesting/engine/bar_runner.py` |
| simulated_broker_place_order | SimulatedBroker.place_order | `src/gpt_trader/backtesting/simulation/broker.py` |

## Guarded Execution
| ID | Label | Path |
|----|-------|------|
| backtest_execution_context | BacktestExecutionContext | `src/gpt_trader/backtesting/engine/guarded_execution.py` |
| backtest_decision_context | BacktestDecisionContext | `src/gpt_trader/backtesting/engine/guarded_execution.py` |
| backtest_guarded_submit | BacktestGuardedExecutor.submit_order | `src/gpt_trader/backtesting/engine/guarded_execution.py` |
| state_collector | StateCollector.collect_account_state | `src/gpt_trader/features/live_trade/execution/state_collection.py` |
| order_validator | OrderValidator.run_pre_trade_validation | `src/gpt_trader/features/live_trade/execution/validation.py` |
| order_submitter | OrderSubmitter.submit_order | `src/gpt_trader/features/live_trade/execution/order_submission.py` |

## Decision Logging
| ID | Label | Path |
|----|-------|------|
| strategy_decision | StrategyDecision | `src/gpt_trader/backtesting/validation/decision_logger.py` |
| decision_logger_log | DecisionLogger.log_decision | `src/gpt_trader/backtesting/validation/decision_logger.py` |
| decision_logger_export | DecisionLogger.export_to_json | `src/gpt_trader/backtesting/validation/decision_logger.py` |

## Golden-Path Validation
| ID | Label | Path |
|----|-------|------|
| replay_decisions | replay_decisions_through_simulator | `src/gpt_trader/backtesting/validation/validator.py` |
| golden_validate | GoldenPathValidator.validate_decision | `src/gpt_trader/backtesting/validation/validator.py` |
| golden_report | GoldenPathValidator.generate_report | `src/gpt_trader/backtesting/validation/validator.py` |

## Chaos Injection
| ID | Label | Path |
|----|-------|------|
| chaos_scenario | ChaosScenario | `src/gpt_trader/backtesting/types.py` |
| chaos_scenario_factories | create_*_scenario | `src/gpt_trader/backtesting/chaos/scenarios.py` |
| chaos_engine_add | ChaosEngine.add_scenario | `src/gpt_trader/backtesting/chaos/engine.py` |
| chaos_engine_process_candle | ChaosEngine.process_candle | `src/gpt_trader/backtesting/chaos/engine.py` |
| chaos_engine_process_order | ChaosEngine.process_order | `src/gpt_trader/backtesting/chaos/engine.py` |
| chaos_engine_apply_latency | ChaosEngine.apply_latency | `src/gpt_trader/backtesting/chaos/engine.py` |

## Outputs
| ID | Label | Path |
|----|-------|------|
| decision_log_json | Decision log JSON | `src/gpt_trader/backtesting/validation/decision_logger.py` |
| validation_report | ValidationReport | `src/gpt_trader/backtesting/types.py` |
| validation_divergence | ValidationDivergence | `src/gpt_trader/backtesting/types.py` |
| chaos_event | ChaosEvent | `src/gpt_trader/backtesting/chaos/engine.py` |
| chaos_stats | ChaosEngine.get_statistics | `src/gpt_trader/backtesting/chaos/engine.py` |

## Edges
| From | To | Description |
|------|----|-------------|
| backtest_execution_context | backtest_guarded_submit | configure executor |
| backtest_decision_context | backtest_guarded_submit | decision metadata |
| backtest_guarded_submit | state_collector | collect state |
| state_collector | order_validator | equity/positions |
| order_validator | order_submitter | validated order |
| order_submitter | simulated_broker_place_order | broker execution |
| backtest_guarded_submit | strategy_decision | build decision |
| strategy_decision | decision_logger_log | log decision |
| decision_logger_log | decision_logger_export | export JSON |
| decision_logger_export | decision_log_json | write file |
| decision_logger_log | replay_decisions | recorded decisions |
| replay_decisions | golden_validate | validate decisions |
| golden_validate | validation_divergence | divergence |
| golden_validate | golden_report | collect divergences |
| golden_report | validation_report | report |
| chaos_scenario | chaos_engine_add | scenario config |
| chaos_scenario_factories | chaos_engine_add | factory scenario |
| chaos_engine_add | chaos_engine_process_candle | inject candles |
| chaos_engine_add | chaos_engine_process_order | inject orders |
| chaos_engine_add | chaos_engine_apply_latency | inject latency |
| clocked_bar_runner | chaos_engine_process_candle | optional hook |
| simulated_broker_place_order | chaos_engine_process_order | optional hook |
| clocked_bar_runner | chaos_engine_apply_latency | optional hook |
| chaos_engine_process_candle | chaos_event | record event |
| chaos_engine_process_order | chaos_event | record event |
| chaos_event | chaos_stats | aggregate |

## Notes
- BacktestGuardedExecutor reuses live guard stack components for parity.
- DecisionLogger captures StrategyDecision entries for replay and reporting.
- GoldenPathValidator compares live vs simulated decisions and emits ValidationReport.
- ChaosEngine hooks are optional; inject via simulation loops when enabled.
