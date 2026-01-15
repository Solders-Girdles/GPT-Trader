# Guard Stack Map

Generated: 2026-01-15T05:17:14.751629+00:00

## Preflight
| ID | Label | Path |
|----|-------|------|
| preflight_entry | Preflight entrypoint | `scripts/production_preflight.py` |
| preflight_cli | Preflight CLI | `src/gpt_trader/preflight/cli.py` |
| preflight_core | PreflightCheck | `src/gpt_trader/preflight/core.py` |
| preflight_checks | Preflight checks | `src/gpt_trader/preflight/checks/` |
| preflight_report | Preflight report | `src/gpt_trader/preflight/report.py` |

## Runtime Guards + Monitoring
| ID | Label | Path |
|----|-------|------|
| trading_engine | TradingEngine | `src/gpt_trader/features/live_trade/engines/strategy.py` |
| execution_guard_manager | GuardManager (execution) | `src/gpt_trader/features/live_trade/execution/guard_manager.py` |
| execution_guards | Execution guards | `src/gpt_trader/features/live_trade/execution/guards/` |
| monitoring_guard_manager | RuntimeGuardManager | `src/gpt_trader/monitoring/guards/manager.py` |
| monitoring_guards | Monitoring guards | `src/gpt_trader/monitoring/guards/builtins.py` |
| health_signals | Health signals | `src/gpt_trader/monitoring/health_signals.py` |
| health_checks | Health checks | `src/gpt_trader/monitoring/health_checks.py` |
| status_reporter | Status reporter | `src/gpt_trader/monitoring/status_reporter.py` |

## Edges
| From | To | Description |
|------|----|-------------|
| preflight_entry | preflight_cli | delegate CLI |
| preflight_cli | preflight_core | create PreflightCheck |
| preflight_core | preflight_checks | run checks |
| preflight_core | preflight_report | generate report |
| trading_engine | execution_guard_manager | runtime guard sweep |
| execution_guard_manager | execution_guards | execute runtime guards |
| trading_engine | execution_guards | pre-trade guard stack |
| trading_engine | monitoring_guard_manager | emit guard events |
| monitoring_guard_manager | monitoring_guards | evaluate guards |
| monitoring_guards | health_signals | emit health signals |
| health_signals | health_checks | evaluate thresholds |
| health_checks | status_reporter | report status |

## Notes
- Preflight checks run via the preflight CLI and PreflightCheck facade.
- Runtime guard sweep is owned by TradingEngine and GuardManager.
