# Entrypoints

---
status: current
last-updated: 2026-01-31
---

This document lists the canonical runtime entrypoints and where they are wired.

## gpt-trader CLI (primary)

- Script entrypoint: `gpt-trader`
- Registered in: `pyproject.toml` `[project.scripts]` -> `gpt_trader.cli:main`
- Implementation: `src/gpt_trader/cli/__init__.py` (`main`)
- Module runner: `python -m gpt_trader.cli` -> `src/gpt_trader/cli/__main__.py`
- Command wiring: `src/gpt_trader/cli/commands/*`

## Live bot (trading loop)

- Invoked via: `gpt-trader run ...`
- Handler: `src/gpt_trader/cli/commands/run.py` (`execute`)
- Bot wiring chain:
  - `src/gpt_trader/cli/services.py` (`instantiate_bot`)
  - `src/gpt_trader/app/container.py` (`ApplicationContainer.create_bot`)
  - `src/gpt_trader/features/live_trade/bot.py` (`TradingBot.run`)

## TUI demo

- Standalone demo module: `python -m gpt_trader.tui.demo`
  - Entry: `src/gpt_trader/tui/demo/__main__.py` (`main`)
- CLI demo modes (preferred for env/logging setup):
  - `gpt-trader run --tui --demo` -> `src/gpt_trader/cli/commands/run.py` (`_run_demo_tui`)
  - `gpt-trader tui --mode demo` -> `src/gpt_trader/cli/commands/tui.py` (`execute`)

## Preflight CLI

- Script entrypoint: `python scripts/production_preflight.py`
  - Implementation: `scripts/production_preflight.py` (`main`)
- CLI subcommand: `gpt-trader preflight`
  - Implementation: `src/gpt_trader/cli/commands/preflight.py` (`execute`)
- Preflight CLI function:
  - `src/gpt_trader/preflight/__init__.py` (`run_preflight_cli`)
  - `src/gpt_trader/preflight/cli.py` (`main`)
- Checks live in: `src/gpt_trader/preflight/checks/`
