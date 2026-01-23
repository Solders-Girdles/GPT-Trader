# Monitoring Stack

This directory contains the Prometheus/Grafana configuration used by GPT-Trader.

It is referenced by:
- `deploy/gpt_trader/docker/docker-compose.yaml` (the `observability` profile mounts these files).
- `monitoring/docker-compose.monitoring.yml` (standalone monitoring stack for local/dev use).

## Quick Start (Standalone)

From the repo root:

```bash
docker compose -f monitoring/docker-compose.monitoring.yml up -d
```

Grafana runs on `http://localhost:3000` and Prometheus on `http://localhost:9090`.
