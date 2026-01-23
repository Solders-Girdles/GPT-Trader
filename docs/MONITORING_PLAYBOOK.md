# Monitoring & Operations Guide

This playbook covers the spot-first GPT-Trader monitoring stack including metrics export, dashboards, alerting, and incident response.

## Quick Start

### 1. Install Monitoring Dependencies

```bash
uv sync --extra monitoring
```

### 2. Expose Metrics

**Option A (recommended): Built-in health server**

- If you run via `gpt-trader run`, set `GPT_TRADER_HEALTH_SERVER_ENABLED=1`
  (optional: `GPT_TRADER_HEALTH_PORT=8080`).
- If you embed the bot, start `gpt_trader.app.health_server.start_health_server`
  (port 8080 by default) with the shared `HealthState`.
- Enable metrics: `GPT_TRADER_METRICS_ENDPOINT_ENABLED=1`

```bash
curl http://localhost:8080/health
curl http://localhost:8080/metrics
```

**Option B (exporter script, events.db-first):**

```bash
uv run python scripts/monitoring/export_metrics.py \
  --profile <profile> \
  --runtime-root . \
  --host 0.0.0.0 --port 9000
```

The exporter reads `runtime_data/<profile>/events.db` first and falls back to
`runtime_data/<profile>/events.jsonl` if the DB is unavailable.

**Exporter Endpoints:**
- `/metrics` - Prometheus text format
- `/metrics.json` - Raw JSON with account snapshots

### 3. Start Observability Stack (Optional)

```bash
docker compose --project-directory deploy/gpt_trader/docker \
  -f deploy/gpt_trader/docker/docker-compose.yaml \
  --profile observability up -d
```

This is the canonical local monitoring stack. It provisions Grafana from
`monitoring/grafana/provisioning/` + `monitoring/grafana/dashboards/` and uses
`deploy/gpt_trader/docker/prometheus.minimal.yml` for Prometheus.

For full stack (Elasticsearch, Kibana, Jaeger):
```bash
docker compose --project-directory deploy/gpt_trader/docker \
  -f deploy/gpt_trader/docker/docker-compose.yaml \
  -f deploy/gpt_trader/docker/docker-compose.infrastructure.yaml \
  --profile observability up -d
```

## Available Metrics

Metrics are exposed with the `gpt_trader_` prefix. Common signals (non-exhaustive):

| Metric | Description |
|--------|-------------|
| `gpt_trader_equity_dollars` | Account equity in USD |
| `gpt_trader_cycle_duration_seconds` | Control loop latency histogram |
| `gpt_trader_order_submission_total{result,side}` | Order submissions by outcome |
| `gpt_trader_order_submission_latency_seconds` | Order submission latency histogram |
| `gpt_trader_broker_call_latency_seconds{operation,outcome}` | Broker call latency histogram |
| `gpt_trader_guard_trips_total{guard,category,recoverable}` | Guard trip counts |
| `gpt_trader_guard_checks_total{guard,result}` | Guard evaluation counts |
| `gpt_trader_api_requests_total{endpoint,method,outcome}` | Coinbase REST request counts |
| `gpt_trader_api_latency_seconds{endpoint,method}` | Coinbase REST latency histogram |
| `gpt_trader_rate_limit_usage_ratio` | Coinbase rate-limit usage ratio |
| `gpt_trader_api_error_rate` | Coinbase API error rate |
| `gpt_trader_ws_gap_count` | WebSocket gap count |

## Key Thresholds

| Signal | Warning | Critical |
|--------|---------|----------|
| `order_error_rate` | > 5% | > 15% |
| `order_retry_rate` | > 10% | > 25% |
| `broker_latency_p95_ms` | > 1000 ms | > 3000 ms |
| `ws_staleness_seconds` | > 30s | > 60s |
| `guard_trip_count` | > 3 (5m) | > 10 (5m) |

## Prometheus Configuration

If you use the deploy compose stack, Prometheus is preconfigured via
`deploy/gpt_trader/docker/prometheus.minimal.yml` to scrape `trading-bot:8080`.

For a standalone Prometheus, add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'gpt-trader'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8080']  # health server /metrics
```

If you use the exporter script, target port `9000` instead. For Docker compose,
swap the target to `trading-bot:8080`.

## Alert Rules

```yaml
groups:
  - name: gpt_trader_alerts
    rules:
      - alert: MetricsEndpointDown
        expr: up{job="gpt-trader"} == 0
        for: 2m
        labels:
          severity: critical

      - alert: OrderErrorRateHigh
        expr: sum(rate(gpt_trader_order_submission_total{result=~"failed|rejected"}[5m])) / sum(rate(gpt_trader_order_submission_total[5m])) > 0.15
        for: 2m
        labels:
          severity: high

      - alert: BrokerLatencyP95High
        expr: histogram_quantile(0.95, sum(rate(gpt_trader_broker_call_latency_seconds_bucket{operation="submit"}[5m])) by (le)) > 3
        for: 5m
        labels:
          severity: high

      - alert: GuardTripSpike
        expr: increase(gpt_trader_guard_trips_total[5m]) > 10
        for: 5m
        labels:
          severity: high
```

## Daily Reports

Generate performance reports:

```bash
# Generate today's report
uv run gpt-trader report daily

# Specific date
uv run gpt-trader report daily --date 2025-10-22

# JSON format
uv run gpt-trader report daily --report-format json

# Custom lookback
uv run gpt-trader report daily --lookback-hours 48
```

Reports saved to: `runtime_data/{profile}/reports/daily_report_{date}.txt`

Note: Daily reports read `runtime_data/<profile>/events.db` (canonical) and fall back to
`runtime_data/<profile>/events.jsonl` for legacy runs. Metrics come from
`runtime_data/<profile>/metrics.json` when present (otherwise the latest `cycle_metrics`
event), and unfilled-order health uses `runtime_data/<profile>/orders.db`. The exporter
prefers `events.db` and uses `events.jsonl` only as a fallback.

### Cron Job Setup

```bash
# Daily report at 6 AM
0 6 * * * cd /path/to/GPT-Trader && uv run gpt-trader report daily --profile prod
```

## Grafana Dashboard

Dashboards are auto-provisioned from `monitoring/grafana/dashboards/` when using
the deploy compose stack. For manual import, use the JSON files in that
directory (for example, `monitoring/grafana/dashboards/grafana_dashboard.json`).

Legacy manual import (older example) uses `scripts/monitoring/grafana-dashboard.json.example`:
1. In Grafana: **+ → Import**
2. Paste JSON content
3. Select Prometheus data source

**Dashboard Panels:**
- Account Equity (time series)
- PnL Breakdown (realized, unrealized, funding)
- Per-Symbol Exposure (stacked area)
- Circuit Breaker State
- Cycle Latency with thresholds
- WebSocket Reconnects
- Stale Marks Detected
- System Resources (CPU, memory)

## Incident Workflows

### Exporter Not Reporting

1. Confirm exporter health (container, process, systemd)
2. Tail bot logs: `grep "CRITICAL"`
3. Restart using deployment tooling
4. Validate recovery in staging before live

### Daily Loss Guard Triggered

1. Review guard output in logs (`risk.guards.daily_loss.*`)
2. Flatten positions if needed:
   ```bash
   uv run gpt-trader orders preview \
     --symbol BTC-USD --side sell --type market \
     --quantity 0.01 --reduce-only
   ```
3. Keep trading halted until root cause identified
4. Document for post-mortem

### Market Data Staleness

1. Check upstream connectivity (Coinbase REST/WS)
2. Verify network access from host
3. Inspect logs for reconnection attempts
4. Restart data pump if exchange healthy but bot stale

### WebSocket Reconnect Loop

1. Check `/health` for websocket `reconnect_count` and staleness details
2. Verify network stability
3. Check Coinbase status page
4. Consider backoff or graceful degradation

## Post-Incident Checklist

- [ ] Exporter metrics recovered (no alerting)
- [ ] Order submission failures stopped incrementing
- [ ] Run `uv run pytest -q` for code-related incidents
- [ ] Update runbooks if threshold changes needed

## Health Checks

```python
from gpt_trader.monitoring.health_checks import (
    check_broker_ping,
    check_degradation_state,
    check_ws_freshness,
)

broker_ok, broker_details = check_broker_ping(broker)
ws_ok, ws_details = check_ws_freshness(broker, message_stale_seconds=30.0)
degradation_ok, degradation_details = check_degradation_state(degradation_state, risk_manager)
```

**Status Conditions:**
- `HEALTHY`: All systems normal
- `DEGRADED`: Minor issues detected
- `UNHEALTHY`: Critical issues requiring action

## Troubleshooting

**No metrics in Prometheus:**
- Check endpoint: `curl http://localhost:8080/metrics` (health server) or `http://localhost:9000/metrics` (exporter)
- Verify `runtime_data/<profile>/events.db` exists and is updating (JSONL is legacy fallback)
- Verify Prometheus targets: `http://localhost:9090/targets`

**Daily report has no data:**
- Verify `runtime_data/<profile>/events.db` exists (canonical) or `events.jsonl` for legacy runs
- Check `runtime_data/<profile>/metrics.json` (or ensure `cycle_metrics` events exist)
- Confirm `runtime_data/<profile>/orders.db` has recent orders for unfilled-order health
- Increase `--lookback-hours`

**Exporter returns empty metrics:**
- Ensure bot has written `runtime_data/<profile>/metrics.json` (run at least one cycle)
- Check file permissions

The older `scripts/monitoring/*.example` files are legacy references.
