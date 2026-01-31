# Canonical Entrypoints

---
status: current
last-updated: 2026-01-31
---

This document lists the supported entrypoints and the wiring they activate.
Use these paths when documenting or invoking the system.

## gpt-trader (Primary CLI)

**Definition**
- `pyproject.toml` `[project.scripts]`: `gpt-trader = "gpt_trader.cli:main"`

**Wiring**
- `gpt_trader.cli:main`
- `gpt_trader.cli.commands.*` (command routing)
- `gpt_trader.cli.services.instantiate_bot`
- `gpt_trader.app.container.create_application_container`
- `ApplicationContainer.create_bot()` -> `TradingBot`

**Module entrypoint**
- `python -m gpt_trader.cli` -> `gpt_trader.cli.__main__` -> `gpt_trader.cli:main`

## Preflight CLI

**Definition**
- Script entrypoint: `scripts/production_preflight.py`

**Wiring**
- `scripts/production_preflight.py:main`
- `gpt_trader.preflight.run_preflight_cli`
- `gpt_trader.preflight.cli:main`
- `gpt_trader.preflight.core.PreflightCheck` -> `gpt_trader.preflight.checks.*`

**Notes**
- This is a standalone CLI; it is not a `gpt-trader` subcommand.

## TUI Demo (Module Entrypoint)

**Definition**
- Module entrypoint: `python -m gpt_trader.tui.demo`

**Wiring**
- `gpt_trader.tui.demo.__main__:main`
- `gpt_trader.logging.setup.configure_logging(tui_mode=True)`
- `gpt_trader.tui.demo.demo_bot.DemoBot`
- `gpt_trader.tui.app.TraderApp` -> `gpt_trader.tui.helpers.run_tui_app_with_cleanup`
