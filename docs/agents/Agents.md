# GPT-Trader Agent Guide

This is the shared orientation document for all AI agents working in this repository. Keep it open while you plan or execute tasks.

## 0. Critical Orientation: Trust & Confusion Awareness

**⚠️ BEFORE ANYTHING ELSE:** The repository contains significant sources of agent confusion due to architectural changes. **YOU CANNOT FULLY TRUST THIS DOCUMENTATION.**

**Essential Reading:** `docs/AGENT_CONFUSION_POINTS.md` - Comprehensive guide to known confusion points and verification protocols.

**Trust Verification:** `docs/agents/Document_Verification_Matrix.md` - Matrix showing which documents are trustworthy vs historical traps.

**Verification Required:** Always cross-check against current command output (`poetry run pytest --collect-only` should report 1484 collected / 1483 selected / 1 deselected).

---

## 1. Current Mission Snapshot
- **Live focus:** Coinbase **spot** trading. Perpetual futures logic remains in the tree but real endpoints stay locked behind the Coinbase INTX gate (`COINBASE_ENABLE_DERIVATIVES` must be `1` *and* INTX access is required).
- **Primary entry point:** `poetry run coinbase-trader run --profile dev --dev-fast`.
- **Architecture style:** Vertical slices under `src/gpt_trader/features/`, but the codebase has grown to 181 Python files—expect multi-file workflows instead of single 500-token modules.

## 2. Directory Compass
| Area | Purpose |
|------|---------|
| `src/gpt_trader/orchestration/trading_bot/bot.py` | Core orchestrator used for spot profiles; enforces risk guards, telemetry, and optional perps hooks. |
| `src/gpt_trader/cli/` | CLI wiring (run/account/orders/treasury subcommands). |
| `src/gpt_trader/features/brokerages/coinbase/` | Coinbase adapter, account manager, telemetry helpers. |
| `src/gpt_trader/features/brokerages/coinbase/account_manager.py` | Fee/limit snapshots plus treasury helpers (`convert`, `move-funds`). |
| `src/gpt_trader/orchestration/live_execution.py` | Runtime safety rails (PnL caps, liquidation buffer, volatility CB, correlation checks). |
| `scripts/monitoring/export_metrics.py` | Prometheus/JSON metrics service for runtime telemetry. |
| `src/gpt_trader/monitoring/` | Additional observability helpers (metrics serialisation, dashboards). |
| `docs/guides/paper_trading.md` | Deep dive on mock/paper trading workflows. |
| `docs/ARCHITECTURE.md` | High-level design doc—update alongside code changes. |
| `README.md` | Fast-install + day-to-day runbook. |

## 3. Core Commands

```bash
poetry install                                   # Set up or refresh dependencies
poetry run coinbase-trader run --profile dev --dev-fast    # Spot trading (mock fills)
poetry run coinbase-trader run --profile canary --dry-run  # Canary validation without live orders
poetry run coinbase-trader account snapshot                # Coinbase fee/limit snapshot
poetry run coinbase-trader treasury convert --from USD --to USDC --amount 1000   # Treasury helpers
poetry run coinbase-trader treasury move --from-portfolio a --to-portfolio b --amount 50   # Treasury helpers
poetry run python scripts/monitoring/export_metrics.py --metrics-file var/data/coinbase_trader/prod/metrics.json
poetry run pytest --collect-only                 # Test discovery snapshot
poetry run pytest -q                             # Full unit suite
```

## 4. Trading Modes & Perps Status
1. **Spot (default)**
   - Profiles `dev`, `demo`, `prod`, `canary` auto-normalize symbols to spot markets.
   - `trading_bot` turns on the deterministic broker stub unless derivatives are explicitly enabled *and* credentials pass validation.
2. **Perps (future-ready)**
   - Keep the code paths compiling and tested, but call out the INTX dependency in any user-facing change.
   - Guard new work behind checks for `COINBASE_ENABLE_DERIVATIVES` to avoid surprise production enablement.

## 5. Experimental vs Production Slices
All legacy experimental slices were removed from the active tree. Retrieve them
through the legacy bundle when explicitly required (`docs/archive/legacy_recovery.md`).
Everything under `src/gpt_trader/features/` is considered current unless marked
otherwise in this guide.

## 6. Operational Tooling
- **Account telemetry:** `poetry run coinbase-trader account snapshot` (dumps permissions, fee schedule, and limits).
- **Treasury helpers:**
  - `poetry run coinbase-trader treasury convert --from USD --to USDC --amount 1000`
  - `poetry run coinbase-trader treasury move --from-portfolio from_portfolio_uuid --to-portfolio to_portfolio_uuid --amount 50`
- **Metrics:** `poetry run python scripts/monitoring/export_metrics.py --metrics-file var/data/coinbase_trader/prod/metrics.json` exposes `/metrics` (Prometheus) and `/metrics.json`.
- **Risk guards (runtime):** Daily PnL stops, liquidation-buffer checks, mark staleness, volatility circuit breakers, and correlation checks all live inside `LiveExecutionEngine`.
- **Documentation templates:** Copy/paste matrices, interview outlines, and backlog seeds from `docs/archive/agents/templates.md` during Sprint 0 and ongoing maintenance.

## 7. Testing Expectations
- **Command:** `poetry run pytest --collect-only` currently discovers 1484 tests (1483 selected / 1 deselected).
- **Dependencies:** Install the security extras (`poetry install --with security`) when working on auth flows so libraries like `pyotp` are available for tests.
- Keep unit tests under `tests/unit/gpt_trader/` up to date, and add coverage for new risk or telemetry paths.

## 8. Common Workflows for Agents
1. **Feature work:**
   - Read the relevant slice README + tests.
   - Implement in the slice + orchestration glue.
   - Update docs (`README.md`, `docs/ARCHITECTURE.md`) if behavior shifts.
2. **Bugfix:**
   - Reproduce with the dev profile (`--dev-fast` is useful).
   - Add or adjust regression tests before patching.
3. **Documentation pass:**
   - Sync this guide, `docs/agents/CLAUDE.md`, and `docs/agents/Gemini.md` whenever the architecture or operations change.

## 9. Source of Truth Checklist
Whenever you ship a change, confirm:
- [ ] README reflects the new instructions.
- [ ] Architecture doc matches the live system.
- [ ] Tests either pass or document any optional dependency gaps (e.g., missing `pyotp` if security extras are skipped).
- [ ] Agent guides (this file + per-agent files) stay consistent.

Stay explicit about spot vs perps mode, and note the INTX gate in every perps-related change description.

## 10. Agent-Specific Notes

- **Claude Code:** Lean on the planning tool for multi-slice edits, keep responses explicit about testing status, and highlight any risk-surface changes inline for reviewers.
- **Gemini Code Assistants:** Prefer concise diffs with command references (keep `rg`/`fd` usage visible) and call out environment prerequisites such as `poetry install` when suggesting test runs.
