# Reasoning Artifacts

Curated summaries for generated reasoning maps and health schema outputs.

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

## Agent Health Schema

- Schema: `var/agents/health/agent_health_schema.json`
- Example: `var/agents/health/agent_health_example.json`
- Generator: `scripts/agents/generate_agent_health_schema.py`
