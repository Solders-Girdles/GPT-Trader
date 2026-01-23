Legacy Risk Templates (Removed)

The legacy risk templates were removed from the tree during docs cleanup.
Use git history if you need to review historical examples.

Active risk settings come from:
- Profile YAML (`config/profiles/*.yaml`) + env overrides (`RISK_*`, `CFM_*`)
- `BotRiskConfig` (bot-level sizing/leverage) adapted into the risk manager config

`RISK_CONFIG_PATH` exists on `BotConfig` but is currently unused by the runtime loader.
