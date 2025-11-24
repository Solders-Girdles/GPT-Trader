# gpt_trader Docker Deployment

This directory hosts the active Docker assets for the gpt_trader trading system. The
multi-stage Dockerfile builds Poetry-managed dependencies and exposes
separate targets for development, testing, production, and optional security scanning. The base
compose file (`docker/docker-compose.yaml`) now focuses on the essentials: the trading bot and
opt-in Prometheus/Grafana metrics via the `observability` profile. Heavier infrastructure blocks
are available through the optional `docker-compose.infrastructure.yaml` override.

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
docker build   --file deploy/gpt_trader/docker/Dockerfile   --target production   --tag gpt-trader/gpt_trader:production   .
```

## Compose Stack
The compose files expect Docker Compose v2 and can load environment overrides from a `.env`
file in the same directory. Base stack variables:

- `BUILD_TARGET`: Dockerfile target to use (`development` by default).
- `ENV` and `LOG_LEVEL`: forwarded to the bot runtime.
- `GF_SECURITY_ADMIN_*`: default Grafana credentials when you enable the observability profile.

Override-specific environment variables (database, Redis, RabbitMQ, Vault secrets) only apply
when you include `docker-compose.infrastructure.yaml`.

### Development Quick Start
```bash
# Build the dev image and start the lightweight stack
cd deploy/gpt_trader/docker
cp ../../../config/environments/.env.template .env  # customize as needed
docker compose build
docker compose up -d
```
The trading bot container binds the repository root into `/app` for rapid iteration. Restart
just that container after making code changes: `docker compose restart trading-bot`.
No profile flag is required in the trimmed stack—the bot starts by default.

Need the historical databases, queues, or Vault mock? Layer the override on top:

```bash
docker compose \
  -f docker-compose.yaml \
  -f docker-compose.infrastructure.yaml \
  --profile infra \
  up -d
```

Enable the `observability` profile alongside the override if you also want Elasticsearch/Kibana
and Jaeger:

```bash
docker compose \
  -f docker-compose.yaml \
  -f docker-compose.infrastructure.yaml \
  --profile observability \
  --profile infra \
  up -d
```

### Production Notes
- Use `docker compose --profile production` (with the infrastructure override) to include the
  Nginx reverse proxy.
- Switch `BUILD_TARGET=production` and remove the source bind mount (edit the compose file or
  supply an additional override file) before shipping images to a registry.
- Vault, RabbitMQ, Redis, and PostgreSQL credentials should be replaced with hardened values
  when the override stack is in play.

### Deployment Script
- `deploy/scripts/deploy.sh` wraps the build, backup, and stack startup steps.
- Pass the target environment as the first argument (defaults to `production`).
- Override the Dockerfile stage with `BUILD_TARGET=development|testing|production`.

## Monitoring & Observability
Prometheus and Grafana are part of the base stack (use `--profile observability`). Elasticsearch,
Kibana, and Jaeger remain available through the infrastructure override for teams that still need
their historical traces and log searches.

## Migrating from the Legacy Stack
If you previously relied on `deploy/docker/`, consult
for migration notes. The legacy assets now live under `docs/archive/legacy-deployment/`.
