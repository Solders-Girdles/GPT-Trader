# Sprint 1 Gap Report (Baseline & Gaps)
Date: 2026-01-13

## Tool Baseline
- `agent-map`: 538 modules / 158 dependencies; components include `cli`, `app`, `brokerages`, `live_trade`.
- `agent-impact`: No changed files detected (impact level `low`).
- `agent-tests`: 11,087 tests across 409 files; trading markers defined for `spot`, `perps`, `portfolio`, `strategy`, `liquidity`, but only `perps` shows counts.
- `agent-check`: FAILED (32 format issues, 16 type errors, 1 snapshot failure).
  - Snapshot failures: `tests/unit/gpt_trader/cli/test_cli_integration.py::test_cli_dev_fast_single_cycle`, `tests/unit/gpt_trader/tui/test_snapshots.py::TestModeSelectionSnapshots::test_mode_selection_screen`.

## Gaps & Recommended Actions
- Spot/perps entrypoints are not explicit in `docs/agents/CODEBASE_MAP.md`. Action: add pointers for `src/gpt_trader/features/live_trade/symbols.py`, `src/gpt_trader/features/live_trade/strategies/perps_baseline/`, and `src/gpt_trader/features/live_trade/strategies/hybrid/base.py` with notes on CFM/INTX gating.
- Config → code linkages are not mapped from CLI config build (`src/gpt_trader/cli/services.py`) to `BotConfig` and `ProfileLoader`. Action: add a config/profile row pointing to `src/gpt_trader/app/config/bot_config.py` and `src/gpt_trader/app/config/profile_loader.py` and mention `config/profiles/`.
- CLI → container path is only partially captured (CLI entrypoint without the `cli/services.py` wiring). Action: reference `src/gpt_trader/cli/commands/run.py` and `cli/services.instantiate_bot` in the CLI row.
- Trust matrix excludes `docs/agents/CODEBASE_MAP.md` even though it’s used by agents. Action: add it to Medium trust with a verification note.
- Test inventory defines `spot` marker but no counts appear. Action: decide whether to add `@pytest.mark.spot` to spot-focused suites or remove references to spot markers in docs.

## Doc Drift Checks
- `docs/ARCHITECTURE.md` flow (CLI → BotConfig → ApplicationContainer → TradingBot) matches code in `src/gpt_trader/cli/commands/run.py`, `src/gpt_trader/cli/services.py`, and `src/gpt_trader/app/container.py`.
