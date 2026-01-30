# Development Guidelines (gpt_trader)

---
status: current
last-updated: 2026-01-30
---

These guidelines cover contributions to the spot-first `gpt_trader` stack. Older
guides from the pre-DI era were removed from the tree; use git history if you
need to review historical practices.

## Architectural Principles

- **Vertical slices**: Add features within `src/gpt_trader/features/<slice>/` and
  keep cross-slice coupling minimal.
- **Explicit wiring**: Register new dependencies in `ApplicationContainer`
  (`src/gpt_trader/app/container.py`) instead of hidden imports. See
  `docs/DI_POLICY.md` for detailed guidance on when to use container vs
  singletons.
- **Strategy contracts**: Standard strategies return `gpt_trader.core.Action` +
  `gpt_trader.core.Decision`. Hybrid strategies emit `HybridDecision` and are
  adapted to the standard contract via `HybridStrategyBase`.
- **Research handoff**: Use Strategy Artifacts for research -> live promotion
  (`docs/STRATEGY_ARTIFACTS.md`).
- **Research backtests**: The research backtesting adapter accepts order intent via
  `Decision.indicators` (`order_type`, `price`/`limit_price`, `stop_price`,
  `tif`/`time_in_force`, `reduce_only`) when using the canonical broker adapter.
- **Configuration-first**: Extend `BotConfig` when new runtime options are
  required; expose overrides through the CLI when appropriate.
- **Modular refactoring**: Extract large modules (>500 lines) into focused
  subpackages with clear separation of concerns. See `features/live_trade/execution/`,
  `src/gpt_trader/monitoring/guards/`, and `features/live_trade/risk/` as examples of successful refactorings.

## Slice Scaffolding

- Use `scripts/maintenance/feature_slice_scaffold.py --name <slice>` to bootstrap new
  vertical slices under `src/gpt_trader/features/<slice>/`.
- Add `--with-readme` and `--with-tests` so documentation and unit tests live beside
  the slice (`tests/unit/gpt_trader/features/<slice>/`).
- Use `--dry-run` for previews; the scaffold tool refuses overwrites by design.
- Keep slice names snake_case, prefer explicit imports, and avoid cross-slice
  dependencies (see `src/gpt_trader/scripts/README.md`).

## Where to Change Things

| Intent | Start Here |
|--------|------------|
| Add a new trading strategy | `src/gpt_trader/features/live_trade/strategies/` + register in `src/gpt_trader/features/live_trade/factory.py` |
| Add a new runtime guard | `src/gpt_trader/features/live_trade/execution/guards/` + register in `src/gpt_trader/features/live_trade/execution/guard_manager.py` |
| Add a new pre-trade validation | `src/gpt_trader/features/live_trade/execution/validation.py` + `src/gpt_trader/features/live_trade/engines/strategy.py` |
| Change order submission behavior | `src/gpt_trader/features/live_trade/execution/order_submission.py` + `src/gpt_trader/features/live_trade/execution/broker_executor.py` |
| Modify risk rules | `src/gpt_trader/features/live_trade/risk/manager/__init__.py` + `src/gpt_trader/features/live_trade/risk/config.py` |
| Add a new env config field | `src/gpt_trader/app/config/bot_config.py` (bot-level) or `src/gpt_trader/features/live_trade/risk/config.py` (risk manager) + `config/environments/.env.template` |
| Modify degradation behavior | `src/gpt_trader/features/live_trade/degradation.py` |
| Add/modify a health check | `src/gpt_trader/monitoring/health_checks.py` |
| Add a Coinbase REST/WS endpoint | `src/gpt_trader/features/brokerages/coinbase/client/` + `src/gpt_trader/features/brokerages/coinbase/endpoints.py` |
| Update TUI screens/widgets | `src/gpt_trader/tui/screens/` or `src/gpt_trader/tui/widgets/` |

## Intentional Guard-Stack Bypasses

The canonical order path routes through `TradingEngine._validate_and_place_order()` (live loop),
with `TradingEngine.submit_order()` as the external wrapper. The following locations intentionally
bypass guards:

| Location | Purpose |
|----------|---------|
| `src/gpt_trader/features/live_trade/bot.py` (`TradingBot.flatten_and_stop()`) | Emergency position closure (must succeed even during risk trips) |
| `src/gpt_trader/features/optimize/` | Optimization/backtesting flows using simulated brokers |

When a direct broker call is required (emergency shutdown only), use the
`bypass_order_guard()` context manager to document the reason in logs.

## Code Style

- Python 3.12 with Ruff + Black defaults (line length 100).
- Type annotations for public interfaces; prefer `typing.Protocol` for guard or
  strategy contracts.
- Prefer `pathlib.Path` for filesystem access.
- Use structured logging via `gpt_trader/logging` helpers (call `configure_logging`)
  whenever a new module emits logs.

## Error Handling

- Raise domain-specific exceptions from `src/gpt_trader/features/live_trade/guard_errors.py`
  or define new ones in the relevant slice.
- Avoid swallowing exceptions; propagate up to the trading engine/guard manager so guard rails
  can respond.
- Provide actionable log messages (symbol, profile, guard name, etc.).

## Testing

- Place unit tests under `tests/unit/gpt_trader/` mirroring the module path.
- Use fixtures for Coinbase mocks (`tests/fixtures/brokerages/` when available).
- Run `uv run pytest -q` locally before submitting a pull request.
- Add regression coverage for new guard conditions, telemetry counters, or CLI
  flags.
- **Subpackage testing**: When refactoring into subpackages, ensure each
  submodule has independent test coverage. Maintain backward compatibility by
  keeping facade modules (e.g., `risk/__init__.py`) that re-export the public
  API.
- **Offline backtests**: Set `BACKTEST_DATA_SOURCE=offline` (and optionally
  `BACKTEST_DATA_DIR`) to force cache-only datasets and avoid API rate limits.

## Continuous Integration

- `Python CI`: Default PR gate; runs selective tests on pull requests and full coverage on direct pushes.
- `Targeted Suites`: Matrix guarding Coinbase, perps, and live-trade regressions when relevant paths change.
- `Perps Validation`: Focused derivatives/perps health check; formerly the “phase6” workflow.
- `Nightly Validation`: Coinbase websocket and harness smoke tests on a nightly cadence.
- `Nightly Full Suite`: Scheduled full pytest run (slow markers included); manually triggerable for debugging.
- `Security Audit`: Weekly pip-audit export to catch dependency vulnerabilities.
- `GPT-Trader CI/CD Pipeline`: End-to-end build and deployment flow for staging/production releases; Docker publish is skipped on pull requests.

## Documentation

- Update `docs/ARCHITECTURE.md`, `docs/RISK_INTEGRATION_GUIDE.md`, or other
  relevant guides whenever behaviour changes.
- Note INTX gating whenever derivatives-resident code paths are touched.
- Keep agent-facing references (`AGENTS.md`, `docs/agents/CODEBASE_MAP.md`, and generated `var/agents/**`) aligned with new workflows.

## Operational Hygiene

- Validate new behaviour with `uv run gpt-trader run --profile dev --dev-fast`.
- Confirm metrics output updates when telemetry changes (`metrics.json`).
- Coordinate with operations before altering risk guard thresholds or order
  routing.

## Submitting Changes

1. Create a descriptive branch.
2. Implement code + tests + docs.
3. Run `uv run pytest -q` and any targeted integration scripts.
4. Open a pull request summarising risk impact, telemetry changes, and rollout
   steps.

Legacy contribution guides were removed from the tree; if you need to review
them, pull from repository history. Do not base new development on those
documents.
