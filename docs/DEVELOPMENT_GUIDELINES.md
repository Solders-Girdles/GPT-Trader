# Development Guidelines (bot_v2)

These guidelines cover contributions to the spot-first `bot_v2` stack. Older
rules that referenced the monolithic `src/bot` package have been archived.

## Architectural Principles

- **Vertical slices**: Add features within `src/bot_v2/features/<slice>/` and
  keep cross-slice coupling minimal.
- **Explicit wiring**: Register new dependencies in
  `bot_v2/orchestration/service_registry.py` instead of hidden imports.
- **Configuration-first**: Extend `BotConfig` when new runtime options are
  required; expose overrides through the CLI when appropriate.
- **Modular refactoring**: Extract large modules (>500 lines) into focused
  subpackages with clear separation of concerns. See `orchestration/execution/`
  and `features/live_trade/risk/` as examples of successful refactorings.

## Code Style

- Python 3.12 with Ruff + Black defaults (line length 100).
- Type annotations for public interfaces; prefer `typing.Protocol` for guard or
  strategy contracts.
- Prefer `pathlib.Path` for filesystem access.
- Use structured logging via `bot_v2/logging` helpers (call `configure_logging`)
  whenever a new module emits logs.

## Error Handling

- Raise domain-specific exceptions from `bot_v2/features/live_trade/guard_errors`
  or define new ones in the relevant slice.
- Avoid swallowing exceptions; propagate up to the orchestrator so guard rails
  can respond.
- Provide actionable log messages (symbol, profile, guard name, etc.).

## Testing

- Place unit tests under `tests/unit/bot_v2/` mirroring the module path.
- Use fixtures for Coinbase mocks (`tests/fixtures/coinbase_*` when available).
- Run `poetry run pytest -q` locally before submitting a pull request.
- Add regression coverage for new guard conditions, telemetry counters, or CLI
  flags.
- **Subpackage testing**: When refactoring into subpackages, ensure each
  submodule has independent test coverage. Maintain backward compatibility by
  keeping facade modules (e.g., `risk/__init__.py`) that re-export the public
  API.

## Documentation

- Update `docs/ARCHITECTURE.md`, `docs/RISK_INTEGRATION_GUIDE.md`, or other
  relevant guides whenever behaviour changes.
- Note INTX gating whenever derivatives-resident code paths are touched.
- Keep agent-facing docs (`docs/agents/Agents.md`, `docs/agents/CLAUDE.md`, `docs/agents/Gemini.md`) aligned with
  new workflows.

## Operational Hygiene

- Validate new behaviour with `poetry run perps-bot run --profile dev --dev-fast`.
- Confirm metrics output updates when telemetry changes (`metrics.json`).
- Coordinate with operations before altering risk guard thresholds or order
  routing.

## Submitting Changes

1. Create a descriptive branch.
2. Implement code + tests + docs.
3. Run `poetry run pytest -q` and any targeted integration scripts.
4. Open a pull request summarising risk impact, telemetry changes, and rollout
   steps.

Legacy contribution guides were removed from the tree; if you need to review
them, pull from repository history. Do not base new development on those
documents.
