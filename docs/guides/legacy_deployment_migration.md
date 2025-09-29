# Migrating from Legacy Docker Deployment

The legacy deployment that lived under `deploy/docker/` targeted the now-removed
`src.bot.paper_trading.ml_paper_trader` entry point. The modern stack is driven from
`deploy/bot_v2/docker/` and introduces new supporting services (RabbitMQ, Vault,
Elasticsearch, Kibana, Jaeger) along with an updated dependency build based on Poetry.
This guide walks through the migration process.

## Key Differences to Keep in Mind
- **Entry point:** The bot now starts via `python -m bot_v2` instead of the legacy
  paper-trading module.
- **Dependency management:** Poetry + `pyproject.toml` replace the old `requirements.txt`
  files; TA-Lib is compiled during the Docker build.
- **Service topology:** The compose file provisions PostgreSQL, Redis, RabbitMQ, Vault,
  Prometheus, Grafana, Elasticsearch, Kibana, Jaeger, and optional Nginx.
- **Configuration:** Environment variables and secrets now follow the bot_v2 naming
  conventions (see `config/environments/.env.template`).

## Migration Checklist
1. **Archive the old stack**
   - Remove any local copies of `deploy/docker/` or copy them to `docs/archive/legacy-deployment/`
     for reference only.
   - Stop and remove containers started from the legacy compose file.

2. **Migrate persistent data**
   - Export PostgreSQL data (`pg_dump`) from the legacy database and import it into the
     new `trading_db` schema provided by `deploy/bot_v2/docker/docker-compose.yaml`.
   - Migrate Redis state if needed using `redis-cli --rdb` exports.
   - Review any files stored under `var/` and copy only the artifacts still required by bot_v2.

3. **Update configuration**
   - Start from `config/environments/.env.template` and recreate environment variable files.
   - Update secrets and service credentials to match the new compose defaults (RabbitMQ,
     Vault, etc.) or override them via `.env`.
   - Review `deploy/bot_v2/docker/docker-compose.yaml` for additional configuration knobs
     such as `BUILD_TARGET` and service profiles.

4. **Build and run the new images**
   - Build the development image: `BUILD_TARGET=development docker compose -f deploy/bot_v2/docker/docker-compose.yaml build trading-bot`
   - For production-ready images set `BUILD_TARGET=production` and disable the source-code bind
     mount (see the README in `deploy/bot_v2/`).
   - Start the stack: `docker compose -f deploy/bot_v2/docker/docker-compose.yaml up -d`

5. **Validate the environment**
   - Check the trading bot health endpoint at `http://localhost:8080/health`.
   - Ensure metrics and logs flow into Prometheus, Grafana, Elasticsearch, and Kibana.
   - Exercise message flows through RabbitMQ and verify Vault secrets resolution.

## Need the Old Files?
The legacy Docker assets now live under `docs/archive/legacy-deployment/`. They are
kept for historical reference only and should not be used for active deployments.
