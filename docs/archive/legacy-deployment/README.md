# Legacy Deployment (Deprecated)

This directory contains the archived Docker assets that previously lived under `deploy/docker/`.
They targeted the original `src.bot.paper_trading.ml_paper_trader` entry point and relied on a
Poetry-based build with TA-Lib compilation. That stack has been superseded by the `bot_v2`
deployment found at `deploy/bot_v2/`.

## Why it was retired
- The legacy entry point (`src.bot.paper_trading.ml_paper_trader`) has been removed from the
  codebase.
- Dependency management diverged from the current project conventions and required manual
  maintenance.
- The modern deployment introduces a different service topology (RabbitMQ, Vault, Elasticsearch,
  etc.) that the legacy stack does not support.

## Migration guidance
If you previously relied on this stack:
1. Adopt the `deploy/bot_v2/docker/` pipeline for all new builds.
2. Migrate environment variables and configuration files to the bot_v2 format.
3. Rebuild Docker images using the bot_v2 workflow and validate them with the updated test suites.

These files are preserved here strictly for historical reference and should not be used for
production or development deployments.
