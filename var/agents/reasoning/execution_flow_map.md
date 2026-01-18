# Execution Flow Map

Generated: 2026-01-18T01:50:49.723695+00:00

## Entry + Decision
| ID | Label | Path |
|----|-------|------|
| trading_engine_cycle | TradingEngine._cycle | `src/gpt_trader/features/live_trade/engines/strategy.py` |
| strategy_decide | Strategy.decide | `src/gpt_trader/features/live_trade/strategies/base.py` |
| order_router | OrderRouter.execute_async (external) | `src/gpt_trader/features/live_trade/execution/router.py` |
| engine_submit | TradingEngine.submit_order | `src/gpt_trader/features/live_trade/engines/strategy.py` |

## Guard + Validation
| ID | Label | Path |
|----|-------|------|
| engine_validate | TradingEngine._validate_and_place_order | `src/gpt_trader/features/live_trade/engines/strategy.py` |
| degradation_gate | Degradation gate | `src/gpt_trader/features/live_trade/degradation.py` |
| security_validator | Security validator | `src/gpt_trader/security/security_validator.py` |
| risk_manager | LiveRiskManager.pre_trade_validate | `src/gpt_trader/features/live_trade/risk/manager/__init__.py` |
| order_validator | OrderValidator (exchange/slippage/preview) | `src/gpt_trader/features/live_trade/execution/validation.py` |
| engine_mark_staleness | TradingEngine._check_mark_staleness | `src/gpt_trader/features/live_trade/engines/strategy.py` |
| risk_check_mark_staleness | LiveRiskManager.check_mark_staleness | `src/gpt_trader/features/live_trade/risk/manager/__init__.py` |

## Submission + Telemetry
| ID | Label | Path |
|----|-------|------|
| order_submitter | OrderSubmitter.submit_order | `src/gpt_trader/features/live_trade/execution/order_submission.py` |
| order_rejection | OrderSubmitter.record_rejection | `src/gpt_trader/features/live_trade/execution/order_submission.py` |
| broker_executor | BrokerExecutor.execute_order | `src/gpt_trader/features/live_trade/execution/broker_executor.py` |
| broker_adapter | BrokerProtocol.place_order | `src/gpt_trader/features/brokerages/core/protocols.py` |
| order_event_recorder | OrderEventRecorder | `src/gpt_trader/features/live_trade/execution/order_event_recorder.py` |
| orders_store | OrdersStore | `src/gpt_trader/persistence/orders_store.py` |

## Event Store
| ID | Label | Path |
|----|-------|------|
| engine_append_event | TradingEngine._append_event | `src/gpt_trader/features/live_trade/engines/strategy.py` |
| emit_metric | emit_metric | `src/gpt_trader/utilities/telemetry.py` |
| event_store_append_metric | EventStoreProtocol.append_metric | `src/gpt_trader/app/protocols.py` |
| event_store_append | EventStoreProtocol.append | `src/gpt_trader/app/protocols.py` |

## Outcomes
| ID | Label | Path |
|----|-------|------|
| decision_trace | OrderDecisionTrace | `src/gpt_trader/features/live_trade/execution/decision_trace.py` |
| submission_result | OrderSubmissionResult | `src/gpt_trader/features/live_trade/execution/submission_result.py` |

## Edges
| From | To | Description |
|------|----|-------------|
| trading_engine_cycle | strategy_decide | produce decision |
| strategy_decide | engine_validate | submit decision |
| order_router | engine_submit | external entry |
| engine_submit | engine_validate | delegate to guard stack |
| engine_validate | decision_trace | record outcomes |
| engine_validate | degradation_gate | pause/allow |
| degradation_gate | security_validator | validate request |
| security_validator | engine_mark_staleness | staleness gate |
| engine_mark_staleness | risk_check_mark_staleness | risk check |
| engine_mark_staleness | engine_append_event | stale_mark_detected |
| engine_append_event | event_store_append | append event |
| engine_mark_staleness | order_validator | continue guards |
| order_validator | risk_manager | pre-trade validate |
| engine_validate | order_rejection | guard rejection |
| order_rejection | order_event_recorder | record rejection |
| order_validator | order_submitter | submit order |
| order_submitter | broker_executor | execute broker call |
| broker_executor | broker_adapter | place order |
| order_submitter | order_event_recorder | record events |
| order_event_recorder | emit_metric | emit metrics |
| emit_metric | event_store_append_metric | append_metric |
| order_event_recorder | event_store_append | decision trace |
| order_submitter | orders_store | persist order |
| engine_validate | submission_result | return status |

## Notes
- TradingEngine._cycle submits strategy decisions directly to the guard stack.
- OrderRouter routes external decisions to TradingEngine.submit_order.
- OrderSubmitter handles broker IO, telemetry, and persistence.
- Guard rejections emit metrics; stale marks append EventStore events.
