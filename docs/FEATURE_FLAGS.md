# Feature Flags Reference

---
status: current
last-updated: 2026-01-24
---

This page is intentionally thin. Feature flags and configuration drift quickly, so canonical references live in code and generated inventories.

## Canonical References

- Operator defaults (minimal): [Environment template](../config/environments/.env.template)
- Generated env var inventory (code + template): [var/agents/configuration/environment_variables.md](../var/agents/configuration/environment_variables.md) ([.json](../var/agents/configuration/environment_variables.json))
- Generated config schemas: [`var/agents/schemas/bot_config_schema.json`](../var/agents/schemas/bot_config_schema.json), [`var/agents/schemas/risk_config_schema.json`](../var/agents/schemas/risk_config_schema.json)

## Precedence

Highest → lowest:

1. CLI arguments
2. Profile settings (`--profile ...`)
3. Environment variables (including `RISK_*` / `HEALTH_*` prefixes)
4. Dataclass defaults

Implementation entrypoints:

- `src/gpt_trader/app/config/bot_config.py`
- `src/gpt_trader/features/live_trade/risk/config.py`

## Regeneration

```bash
uv run agent-regenerate --only configuration,schemas
```
