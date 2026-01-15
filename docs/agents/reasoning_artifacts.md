# Reasoning Artifacts

Curated summaries for generated reasoning maps and health schema outputs.

Flow definitions live in `config/agents/flows/*.yaml`. Run the generator with
`--validate` to check that node paths still exist; add `--strict` to fail when
paths are missing.

## CLI → Config → Container → Engine Flow

- Generated outputs: `var/agents/reasoning/cli_flow_map.json`, `var/agents/reasoning/cli_flow_map.md`, `var/agents/reasoning/cli_flow_map.dot`
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Trace CLI argument parsing through config construction, container wiring, and engine execution.

## Config → Code Linkage Map

- Generated outputs: `var/agents/reasoning/config_code_map.json`, `var/agents/reasoning/config_code_map.md`, `var/agents/reasoning/config_code_map.dot`
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Identify which modules consume specific `BotConfig` fields (static scan).

## Guard Stack Map

- Generated outputs: `var/agents/reasoning/guard_stack_map.json`, `var/agents/reasoning/guard_stack_map.md`, `var/agents/reasoning/guard_stack_map.dot`
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Trace preflight checks vs runtime guard + monitoring flow.

## Execution Flow Map

- Generated outputs: `var/agents/reasoning/execution_flow_map.json`, `var/agents/reasoning/execution_flow_map.md`, `var/agents/reasoning/execution_flow_map.dot`
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Trace decision → guard stack → submission → telemetry for order execution.

## Market Data Flow Map

- Generated outputs: `var/agents/reasoning/market_data_flow_map.json`, `var/agents/reasoning/market_data_flow_map.md`, `var/agents/reasoning/market_data_flow_map.dot`
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Trace REST polling + WebSocket streaming inputs into strategy decisions, risk staleness, and EventStore persistence.

## Backtesting Flow Map

- Generated outputs: `var/agents/reasoning/backtesting_flow_map.json`, `var/agents/reasoning/backtesting_flow_map.md`, `var/agents/reasoning/backtesting_flow_map.dot`
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Trace EventStore historical data into backtest simulations and performance metrics.

## Backtest Reporting Flow Map

- Generated outputs: `var/agents/reasoning/backtest_reporting_flow_map.json`, `var/agents/reasoning/backtest_reporting_flow_map.md`, `var/agents/reasoning/backtest_reporting_flow_map.dot`
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Trace SimulatedBroker metrics into BacktestReporter outputs and BacktestResult summaries.

## Backtest Entrypoints Map

- Generated outputs: `var/agents/reasoning/backtest_entrypoints_map.json`, `var/agents/reasoning/backtest_entrypoints_map.md`, `var/agents/reasoning/backtest_entrypoints_map.dot`
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Locate CLI, script, and library entrypoints into backtesting engines.

## Backtest Validation + Chaos Flow Map

- Generated outputs: `var/agents/reasoning/backtest_validation_chaos_map.json`, `var/agents/reasoning/backtest_validation_chaos_map.md`, `var/agents/reasoning/backtest_validation_chaos_map.dot`
- Generator: `scripts/agents/generate_reasoning_artifacts.py`
- Use case: Trace guarded execution parity, decision logging, golden-path validation, and chaos injection hooks.

## Agent Health Schema

- Schema: `var/agents/health/agent_health_schema.json`
- Example: `var/agents/health/agent_health_example.json`
- Generator: `scripts/agents/generate_agent_health_schema.py`
