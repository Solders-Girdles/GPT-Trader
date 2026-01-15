# CLI → Config → Container → Engine Flow

Generated: 2026-01-15T13:32:26.175858+00:00

## Entrypoints
- `uv run gpt-trader run --profile dev --dev-fast`
- `uv run coinbase-trader run --profile dev --dev-fast`

## Nodes
| ID | Label | Path |
|----|-------|------|
| cli_entrypoint | CLI entrypoint (gpt_trader.cli:main) | `src/gpt_trader/cli/__init__.py` |
| cli_run_command | CLI run command | `src/gpt_trader/cli/commands/run.py` |
| cli_services | CLI config/services | `src/gpt_trader/cli/services.py` |
| profile_loader | ProfileLoader | `src/gpt_trader/app/config/profile_loader.py` |
| bot_config | BotConfig | `src/gpt_trader/app/config/bot_config.py` |
| bootstrap | build_bot / bot_from_profile | `src/gpt_trader/app/bootstrap.py` |
| container | ApplicationContainer | `src/gpt_trader/app/container.py` |
| trading_bot | TradingBot | `src/gpt_trader/features/live_trade/bot.py` |
| trading_engine | TradingEngine | `src/gpt_trader/features/live_trade/engines/strategy.py` |
| strategy_factory | create_strategy | `src/gpt_trader/features/live_trade/factory.py` |

## Edges
| From | To | Description |
|------|----|-------------|
| cli_entrypoint | cli_run_command | dispatch to run command |
| cli_run_command | cli_services | build config + instantiate bot |
| cli_services | profile_loader | load profile schema |
| profile_loader | bot_config | construct BotConfig |
| cli_services | container | create ApplicationContainer |
| bootstrap | container | optional bootstrap path |
| container | trading_bot | create bot |
| trading_bot | trading_engine | instantiate engine |
| trading_engine | strategy_factory | select strategy |
