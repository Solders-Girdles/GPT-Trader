# bot_v2 Docker Deployment

This directory hosts the active Docker assets for the bot_v2 trading system. The
multi-stage Dockerfile builds Poetry-managed dependencies and exposes
separate targets for development, testing, production, and optional security scanning. The
compose file (`docker/docker-compose.yaml`) wires the trading bot together with its supporting
infrastructure: PostgreSQL, Redis, RabbitMQ, Vault, Prometheus, Grafana, Elasticsearch, Kibana,
Jaeger, and an optional Nginx reverse proxy.

## Image Targets
The Dockerfile defines several stages selectable via the `BUILD_TARGET` build argument:

| Target        | Use Case                                      |
|---------------|-----------------------------------------------|
| `development` | Iterative development with source bind mounts |
| `testing`     | CI-oriented build that runs unit tests        |
| `production`  | Minimal runtime image for deployment         |
| `security`    | Runs `safety` and `bandit` scanners           |

Example build command:
```bash
docker build   --file deploy/bot_v2/docker/Dockerfile   --target production   --tag gpt-trader/bot_v2:production   .
```

## Compose Stack
The compose file expects Docker Compose v2 and can load environment overrides from a `.env`
file in the same directory. Key variables:

- `BUILD_TARGET`: Dockerfile target to use (`development` by default).
- `ENV` and `LOG_LEVEL`: forwarded to the bot runtime.
- `JWT_SECRET_KEY`, `VAULT_TOKEN`, database credentials, and other secrets – override them in
  your local `.env` or secrets manager.

### Development Quick Start
```bash
# Build the dev image and start the full stack
cd deploy/bot_v2/docker
cp ../../../config/environments/.env.template .env  # customize as needed
docker compose build
docker compose up -d
```
The trading bot container binds the repository root into `/app` for rapid iteration. Restart
just that container after making code changes: `docker compose restart trading-bot`.

### Production Notes
- Use `docker compose --profile production` to include the Nginx reverse proxy.
- Switch `BUILD_TARGET=production` and remove the source bind mount (edit the compose file or
  supply an override file) before shipping images to a registry.
- Vault, RabbitMQ, Redis, and PostgreSQL credentials should be replaced with hardened values.

### Deployment Script
- `deploy/scripts/deploy.sh` wraps the build, backup, and stack startup steps.
- Pass the target environment as the first argument (defaults to `production`).
- Override the Dockerfile stage with `BUILD_TARGET=development|testing|production`.

## Monitoring & Observability
Prometheus, Grafana, Elasticsearch, Kibana, and Jaeger ship as part of the compose stack. Their
volumes are declared at the bottom of the file so metrics and logs persist across restarts.

## Migrating from the Legacy Stack
If you previously relied on `deploy/docker/`, consult
[`docs/guides/legacy_deployment_migration.md`](../../docs/guides/legacy_deployment_migration.md)
for migration notes. The legacy assets now live under `docs/archive/legacy-deployment/`.
