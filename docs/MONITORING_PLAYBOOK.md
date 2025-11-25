# Monitoring & Operations Guide

This playbook covers the spot-first `coinbase-trader` monitoring stack including metrics export, dashboards, alerting, and incident response.

## Quick Start

### 1. Install Monitoring Dependencies

```bash
poetry install -E monitoring
```

### 2. Start the Metrics Exporter

```bash
poetry run python scripts/monitoring/export_metrics.py \
  --metrics-file var/data/coinbase_trader/prod/metrics.json \
  --events-file var/data/coinbase_trader/prod/events.jsonl \
  --host 0.0.0.0 --port 9000
```

**Endpoints:**
- `/metrics` - Prometheus text format
- `/metrics.json` - Raw JSON with account snapshots

### 3. Start Observability Stack (Optional)

```bash
docker compose --project-directory deploy/gpt_trader/docker \
  -f deploy/gpt_trader/docker/docker-compose.yaml \
  --profile observability up -d
```

For full stack (Elasticsearch, Kibana, Jaeger):
```bash
docker compose --project-directory deploy/gpt_trader/docker \
  -f deploy/gpt_trader/docker/docker-compose.yaml \
  -f deploy/gpt_trader/docker/docker-compose.infrastructure.yaml \
  --profile observability up -d
```

## Available Metrics

### Account Metrics
| Metric | Description |
|--------|-------------|
| `coinbase_trader_equity` | Account equity in USD |
| `coinbase_trader_total_pnl_usd` | Total PnL |
| `coinbase_trader_realized_pnl_usd` | Realized PnL |
| `coinbase_trader_unrealized_pnl_usd` | Unrealized PnL |
| `coinbase_trader_funding_pnl_usd` | Funding PnL |

### Performance Metrics
| Metric | Description |
|--------|-------------|
| `coinbase_trader_cycle_duration_seconds` | Control loop wall time |
| `coinbase_trader_cycle_errors_total` | Exceptions per cycle |
| `coinbase_trader_cycle_latency_ms` | Trading cycle latency |

### Risk Metrics
| Metric | Description |
|--------|-------------|
| `coinbase_trader_risk_daily_loss` | Running PnL vs guard |
| `coinbase_trader_circuit_breaker_triggered` | 0=normal, 1=triggered |

### Health Metrics
| Metric | Description |
|--------|-------------|
| `coinbase_trader_market_data_staleness_seconds` | Age of last mark price |
| `coinbase_trader_telemetry_account_age_seconds` | Age of account snapshot |
| `coinbase_trader_websocket_reconnects_total` | WebSocket reconnections |
| `coinbase_trader_stale_marks_total` | Stale mark detections |

## Key Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| `cycle_duration_seconds` | > 3s | > 6s |
| `cycle_errors_total` | +1 in 5m | +3 in 5m |
| `risk_daily_loss` | > 80% of limit | ≥ limit |
| `market_data_staleness_seconds` | > 10s | > 30s |
| `telemetry_account_age_seconds` | > 15m | > 30m |

## Prometheus Configuration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'coinbase-trader'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:9000']
```

## Alert Rules

```yaml
groups:
  - name: gpt_trader_alerts
    rules:
      - alert: CoinbaseTraderExporterDown
        expr: up{job="coinbase-trader-exporter"} == 0
        for: 2m
        labels:
          severity: critical

      - alert: DailyLossGuardWarn
        expr: coinbase_trader_risk_daily_loss / coinbase_trader_risk_daily_loss_limit > 0.8
        for: 1m
        labels:
          severity: high

      - alert: MarketDataStale
        expr: coinbase_trader_market_data_staleness_seconds > 30
        for: 30s
        labels:
          severity: high

      - alert: CircuitBreakerTriggered
        expr: coinbase_trader_circuit_breaker_triggered > 0
        for: 1m
        labels:
          severity: critical

      - alert: WebSocketReconnectLoop
        expr: increase(coinbase_trader_websocket_reconnects_total[1m]) >= 5
        labels:
          severity: critical
```

## Daily Reports

Generate performance reports:

```bash
# Generate today's report
python -m gpt_trader.cli report daily

# Specific date
python -m gpt_trader.cli report daily --date 2025-10-22

# JSON format
python -m gpt_trader.cli report daily --format json

# Custom lookback
python -m gpt_trader.cli report daily --lookback-hours 48
```

Reports saved to: `var/data/coinbase_trader/{profile}/reports/daily_report_{date}.txt`

### Cron Job Setup

```bash
# Daily report at 6 AM
0 6 * * * cd /path/to/GPT-Trader && python -m gpt_trader.cli report daily --profile prod
```

## Grafana Dashboard

Import dashboard from `monitoring/grafana_dashboard.json`:
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
   poetry run coinbase-trader orders preview \
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

1. Check `websocket_reconnects_total` metric
2. Verify network stability
3. Check Coinbase status page
4. Consider backoff or graceful degradation

## Post-Incident Checklist

- [ ] Exporter metrics recovered (no alerting)
- [ ] `cycle_errors_total` stopped incrementing
- [ ] Run `poetry run pytest -q` for code-related incidents
- [ ] Update runbooks if threshold changes needed

## Health Checks

```python
from gpt_trader.monitoring.health.checks import (
    StaleFillsHealthCheck,
    StaleMarksHealthCheck,
    WebSocketReconnectHealthCheck,
)
from gpt_trader.monitoring.health.registry import HealthCheckRegistry

registry = HealthCheckRegistry()
registry.register(StaleFillsHealthCheck(orders_store, max_age_minutes=10.0))
registry.register(StaleMarksHealthCheck(market_data, max_age_seconds=30.0))
registry.register(WebSocketReconnectHealthCheck(ws_handler))

results = await registry.check_all()
```

**Status Conditions:**
- `HEALTHY`: All systems normal
- `DEGRADED`: Minor issues detected
- `UNHEALTHY`: Critical issues requiring action

## Troubleshooting

**No metrics in Prometheus:**
- Check exporter: `curl http://localhost:9000/metrics`
- Verify Prometheus targets: `http://localhost:9090/targets`

**Daily report has no data:**
- Verify events file exists
- Check `metrics.json`
- Increase `--lookback-hours`

**Exporter returns empty metrics:**
- Ensure bot has written `metrics.json` (run at least one cycle)
- Check file permissions

## Docker Compose Bundle

Sample stack in `scripts/monitoring/docker-compose.yml.example`:

```bash
cd scripts/monitoring
cp docker-compose.yml.example docker-compose.yml
cp prometheus.yml.example prometheus.yml
cp prometheus-alerts.yml.example prometheus-alerts.yml
docker-compose up -d
```
