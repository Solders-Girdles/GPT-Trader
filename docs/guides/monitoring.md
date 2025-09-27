# Monitoring & Metrics Exporter

This guide shows how to surface GPT‑Trader telemetry in Prometheus/Grafana (or any monitoring stack that can scrape HTTP JSON/text). The bot already writes snapshots to disk; the exporter bundles them up for external use.

## 1. Install the optional monitoring extra

```bash
poetry install -E monitoring
```

This pulls in `flask` so the exporter can run.

## 2. Start the exporter

```bash
poetry run python scripts/monitoring/export_metrics.py \
  --metrics-file data/perps_bot/prod/metrics.json \
  --events-file results/managed/events.jsonl \
  --host 0.0.0.0 --port 9000
```

Endpoints:

- `/metrics` – Prometheus text format (equity, open orders, CPU%, memory%, fee tier, remaining limit, preview failure counter, metrics timestamp).
- `/metrics.json` – Raw JSON including the latest account snapshot and order preview events.

## 3. Prometheus scrape configuration

Add a job similar to the sample below (`scripts/monitoring/prometheus.yml.example`):

```yaml
scrape_configs:
  - job_name: gpt_trader
    scrape_interval: 15s
    static_configs:
      - targets: ['gpt-trader-host:9000']
```

## 4. Grafana panels & alerts

- Read `gpt_trader_equity`, `gpt_trader_open_orders`, and the `account_snapshot` fields to build dashboards.
- Alert ideas: equity drawdown, open orders stuck > X minutes, account limits approaching zero, missing `account_snapshot` samples.

## 5. Shipping events to Loki (optional)

Use the Promtail config template (`scripts/monitoring/promtail-config.yml.example`) to tail `results/managed/events.jsonl` into Loki. Key event types to monitor:

- `order_preview`, `order_submit`, `order_reject`
- `account_snapshot`
- `convert_commit`, `portfolio_move`

## 6. Updating metrics frequency

`BotConfig.account_telemetry_interval` (CLI `--account-interval`) controls how often snapshots are captured. Default: 300 seconds.

## 7. Troubleshooting

- Exporter returns empty metrics → ensure the bot has written `metrics.json` (run at least one cycle).
- Preview metrics missing → enable previews (`ORDER_PREVIEW_ENABLED=1` or `--enable-preview`) and confirm the broker account allows it.
- Multiple bots → run one exporter per profile or point the exporter at the desired metrics file.

With the exporter running and Prometheus scraping it, you can build Grafana dashboards and alerting that closely track bot health and Coinbase account state.

## 8. Docker Compose bundle (optional)

The repo ships a sample stack under `scripts/monitoring/docker-compose.yml.example`. It runs the exporter, Prometheus, Grafana, Loki, and Promtail. Adjust volume paths, then start with:

```bash
cd scripts/monitoring
cp docker-compose.yml.example docker-compose.yml
cp prometheus.yml.example prometheus.yml
cp prometheus-alerts.yml.example prometheus-alerts.yml
cp promtail-config.yml.example promtail-config.yml
cp loki-config.yml.example loki-config.yml
# edit paths/targets as needed
podman-compose up -d   # or docker-compose up -d
```

Load the provided Grafana dashboard (`scripts/monitoring/grafana-dashboard.json.example`) to get starter panels, and point Prometheus to `prometheus-alerts.yml` for sample alert rules (stale metrics, low limit, preview failures).
