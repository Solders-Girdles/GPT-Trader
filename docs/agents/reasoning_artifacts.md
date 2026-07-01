# Reasoning Artifacts

---
status: current
---

Curated summaries for generated reasoning maps and health schema outputs.

Flow definitions live in `config/agents/flows/*.yaml`. Run the generator with
`--validate` to check that node paths still exist; add `--strict` to fail when
paths are missing.

Only the curated `.md` summaries are committed. The `.json`/`.dot` machine
forms are gitignored (high-churn, regenerated on nearly any `src/**` change);
produce them locally with `uv run agent-regenerate --only reasoning`. The
scheduled agent-artifacts refresh workflow regenerates and packages all three
forms.

## CLI → Config → Container → Engine Flow

- Committed summary: `var/agents/reasoning/cli_flow_map.md` (machine forms `cli_flow_map.json` / `cli_flow_map.dot` are gitignored)
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Trace CLI argument parsing through config construction, container wiring, and engine execution.

## Config → Code Linkage Map

- Committed summary: `var/agents/reasoning/config_code_map.md` (machine forms `config_code_map.json` / `config_code_map.dot` are gitignored)
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Identify which modules consume specific `BotConfig` fields (static scan).

## Guard Stack Map

- Committed summary: `var/agents/reasoning/guard_stack_map.md` (machine forms `guard_stack_map.json` / `guard_stack_map.dot` are gitignored)
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Trace preflight checks vs runtime guard + monitoring flow.

## Execution Flow Map

- Committed summary: `var/agents/reasoning/execution_flow_map.md` (machine forms `execution_flow_map.json` / `execution_flow_map.dot` are gitignored)
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Trace decision → guard stack → submission → telemetry for order execution.

## Market Data Flow Map

- Committed summary: `var/agents/reasoning/market_data_flow_map.md` (machine forms `market_data_flow_map.json` / `market_data_flow_map.dot` are gitignored)
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Trace REST polling + WebSocket streaming inputs into strategy decisions, risk staleness, and EventStore persistence.

## Backtest Reporting Flow Map

- Committed summary: `var/agents/reasoning/backtest_reporting_flow_map.md` (machine forms `backtest_reporting_flow_map.json` / `backtest_reporting_flow_map.dot` are gitignored)
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Trace SimulatedBroker metrics into BacktestReporter outputs and BacktestResult summaries.

## Backtest Entrypoints Map

- Committed summary: `var/agents/reasoning/backtest_entrypoints_map.md` (machine forms `backtest_entrypoints_map.json` / `backtest_entrypoints_map.dot` are gitignored)
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Locate CLI, script, and library entrypoints into backtesting engines.

## Backtest Validation + Chaos Flow Map

- Committed summary: `var/agents/reasoning/backtest_validation_chaos_map.md` (machine forms `backtest_validation_chaos_map.json` / `backtest_validation_chaos_map.dot` are gitignored)
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Trace guarded execution parity, decision logging, golden-path validation, and chaos injection hooks.

## Agent Health Schema

- Schema: `var/agents/health/agent_health_schema.json`
- Example: `var/agents/health/agent_health_example.json`
- Generator: `scripts/agents/generate_agent_health_schema.py`
