# Legacy Equities Workflows

This repository has pivoted to Coinbase Perpetual Futures as the primary focus. The
equities-oriented modules remain in-place to preserve tests, examples, and historical
context. They will be incrementally deprecated.

Active perps entry points:
- docs/COINBASE_README.md
- docs/PERPS_TRADING_LOGIC_REPORT.md
- src/bot_v2/cli.py (`poetry run perps-bot`)

Legacy modules (kept for tests/examples):
- src/bot_v2/orchestration/orchestrator.py (Core orchestration)
- src/bot_v2/orchestration/enhanced_orchestrator.py (ML-integrated orchestrator)
- src/bot_v2/features/live_trade/live_trade.py (template live trading)
- src/bot_v2/features/live_trade/brokers.py (Alpaca/IBKR/Simulated templates)
- src/bot_v2/data_providers/__init__.py (YFinance/Alpaca providers)

Notes:
- These modules contain “shares” semantics and equities-specific examples. They are not
  used by the perps bot.
- Tests referencing these modules are maintained to validate type usage and APIs.

Migration guidance:
- Use the perps orchestration (`src/bot_v2/orchestration/perps_bot.py`) and the Coinbase
  brokerage slice for production workflows.
- If adding new features, prefer perps-first abstractions and avoid equities-specific
  semantics unless explicitly working on legacy paths.

