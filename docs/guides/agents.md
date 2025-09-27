# AI Agent Development Guide

This guide points AI agents to the canonical resources for GPT-Trader V2. The bot operates **spot-first** with dormant perps logic that activates only when Coinbase grants INTX access.

## Where to Start
- `Agents.md` (repo root) – shared playbook for all assistants
- `CLAUDE.md`, `Gemini.md` – per-agent quick-start instructions
- `docs/ARCHITECTURE.md` – architecture overview (spot-first refresh)
- `docs/guides/complete_setup_guide.md` – environment + credential walkthrough
- `docs/guides/testing.md` – current test metrics and commands

## Core Commands
```bash
poetry install                                        # install deps
poetry run perps-bot --profile dev --dev-fast         # spot dev cycle
poetry run perps-bot --profile canary --dry-run       # canary dry run
poetry run perps-bot --profile canary                 # live spot (tiny)
poetry run python scripts/stage3_runner.py ...        # legacy wrapper → perps-bot
poetry run python scripts/monitoring/export_metrics.py --metrics-file data/perps_bot/prod/metrics.json

# Tests
poetry run pytest --collect-only                      # expect 480 collected / 422 selected
poetry run pytest tests/unit/bot_v2 tests/unit/test_foundation.py -q
poetry run pytest -m integration tests/integration/bot_v2 -q
```

Perps execution remains dormant until INTX access is approved and `COINBASE_ENABLE_DERIVATIVES=1` is set.

## Environment Basics
```bash
# Spot trading defaults
COINBASE_API_KEY=your_hmac_key
COINBASE_API_SECRET=your_hmac_secret
COINBASE_ENABLE_DERIVATIVES=0

# Enable derivatives only when Coinbase grants INTX access
# COINBASE_ENABLE_DERIVATIVES=1
# COINBASE_PROD_CDP_API_KEY=organizations/{org}/apiKeys/{key_id}
# COINBASE_PROD_CDP_PRIVATE_KEY="""-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----"""

# Risk controls
RISK_DAILY_LOSS_LIMIT=100
DRY_RUN=1
```

## Running Tests
- Always run `poetry install` after pulling to pick up dependency changes (`pyotp` is now required for security tests).
- Use `poetry run pytest --collect-only` to confirm suite counts (480/422/58).
- The primary enforcement suite is `poetry run pytest tests/unit/bot_v2 tests/unit/test_foundation.py -q`.
- Integration and real-API suites require explicit markers and credentials.

## Operational Notes
- Stage 3 runner is a wrapper—prefer `poetry run perps-bot` directly.
- Metrics exporter lives at `scripts/monitoring/export_metrics.py`.
- Experimental slices (`backtest`, `ml_strategy`, `market_regime`, `workflows`, `monitoring_dashboard`) are tagged `__experimental__`; avoid production changes there unless requested.

## Keeping Docs in Sync
Whenever behavior changes:
1. Update README, docs/ARCHITECTURE.md, and the relevant guide.
2. Sync `Agents.md`, `CLAUDE.md`, `Gemini.md` (and this file).
3. Note INTX gating in any perps-related instructions.

Refer back to the agent-specific guides for deeper workflows and delegation patterns.
