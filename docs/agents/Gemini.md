# Gemini Agent Notes

`docs/agents/Agents.md` is the canonical reference—use it for architecture, commands, and workflows.

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

## Gemini-Specific Tips
- Keep responses concise and enumerate follow-up actions so the human maintainer can respond with a single number when possible.
- Include the exact commands you ran (or recommend) with `rg`/`fd` snippets for context gathering; this keeps the assistant workflow reproducible.
- Call out environment prerequisites (`poetry install`, credentials) whenever you suggest running tests or scripts.
- When referencing refactored modules, use new subpackage paths (e.g., `features/live_trade/risk/manager.py` not `features/live_trade/risk.py`)

Check the Agent guide's *Agent-Specific Notes* section for any new expectations that apply to all assistants.
