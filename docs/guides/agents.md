# AI Agent Development Guide

This guide points AI agents to the canonical resources for GPT-Trader V2. The bot operates **spot-first** with dormant perps logic that activates only when Coinbase grants INTX access.

## Where to Start
- `docs/agents/Agents.md` – shared playbook for all assistants
- `docs/agents/CLAUDE.md`, `docs/agents/Gemini.md` – per-agent quick-start instructions
- `docs/ARCHITECTURE.md` – architecture overview (spot-first refresh)
- `docs/guides/complete_setup_guide.md` – environment + credential walkthrough
- `docs/guides/testing.md` – current test metrics and commands

## Core Commands
```bash
poetry install                                        # install deps
poetry run coinbase-trader run --profile dev --dev-fast         # spot dev cycle
poetry run coinbase-trader run --profile canary --dry-run       # canary dry run
poetry run coinbase-trader run --profile canary                 # live spot (tiny)
poetry run python scripts/monitoring/export_metrics.py --metrics-file var/data/coinbase_trader/prod/metrics.json

# Tests
poetry run pytest --collect-only                      # expect 1484 collected / 1483 selected / 1 deselected
poetry run pytest -q                                   # full bot_v2 regression suite
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
- After pulling, run `poetry install --with security` when working on authentication flows so optional dependencies like `pyotp` are available for the security tests.
- Use `poetry run pytest --collect-only` to confirm suite counts (1484 collected / 1483 selected / 1 deselected).
- The enforcement suite is `poetry run pytest -q`.
- Real-API or integration flows are archived; build new coverage inside `tests/unit/bot_v2/` when adding features.

## Operational Notes
- Metrics exporter lives at `scripts/monitoring/export_metrics.py`.
- Legacy experimental slices are no longer present in the workspace. Retrieve
  them via `docs/archive/legacy_recovery.md` if a task explicitly requires the
  historical code.

## Keeping Docs in Sync
Whenever behavior changes:
1. Update README, docs/ARCHITECTURE.md, and the relevant guide.
2. Sync `docs/agents/Agents.md`, `docs/agents/CLAUDE.md`, `docs/agents/Gemini.md` (and this file).
3. Note INTX gating in any perps-related instructions.

Refer back to the agent-specific guides for deeper workflows and delegation patterns.
