# Claude Agent Notes

All core orientation details now live in `docs/agents/Agents.md`. Keep that doc open while you work.

## Key Architecture Patterns

### Modular Subpackages (Prefer Over Monoliths)
When navigating the codebase, expect these patterns:
- **Risk Management:** `features/live_trade/risk/` (manager, position_sizing, pre_trade_checks, runtime_monitoring, state_management)
- **Execution Layer:** `orchestration/execution/` (guards, validation, order_submission, state_collection)
- **Coinbase Client:** `client/` directory with mixins, not a single `client.py` file
- **REST Services:** `rest/` layer for business logic (orders, portfolio, products, pnl)

### Archived Features
Experimental slices are periodically archived to keep the active codebase lean:
- Look in `archived/` or git history if referencing backtest, ml_strategy, market_regime, or monitoring_dashboard
- Core production features: `live_trade/`, `brokerages/`, `position_sizing/`, `paper_trade/`

### Test Expectations
- **Command:** `poetry run pytest --collect-only` shows current count and selection
- **Target:** 100% pass rate on active code (spot trading + INTX-gated perps)
- **Reality Check:** If tests fail, investigate before assuming code is broken—dependencies may need refresh

## Claude-Specific Tips
- Start with the planning tool when tasks span multiple slices; call out your plan explicitly in replies.
- Surface risk-impact summaries and testing commands in every response so maintainers can assess changes quickly.
- Prefer `rg`/`fd` for navigation—large `grep`/`find` calls slow the Mac mini CI boxes.
- When referencing refactored modules, use new subpackage paths (e.g., `features/live_trade/risk/manager.py` not `features/live_trade/risk.py`)

Need more context? Jump to the relevant sections in `docs/agents/Agents.md` (Directory Compass, Core Commands, and Agent-Specific Notes).
