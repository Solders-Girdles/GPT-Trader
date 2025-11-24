# GPT-Trader Monitoring & Incident Response

---
status: current
last-updated: 2025-10-07
---

This playbook reflects the spot-first `coinbase-trader` architecture. It supersedes
the retired monitoring guides that targeted the legacy `gpt-trader` service and
its ML pipeline.

## Observability Stack

- **Metrics**: `scripts/monitoring/export_metrics.py` exposes `/metrics` and
  `/metrics.json` from the bot's `metrics.json` output.
- **Logs**: Structured JSON (or text) emitted by `coinbase-trader`; forward to your
  logging stack for search and alerting.
- **Account telemetry**: `poetry run coinbase-trader account snapshot` when a manual
  check is required.

Start Prometheus/Grafana when needed (base stack):

```bash
docker compose --project-directory deploy/gpt_trader/docker \
  -f deploy/gpt_trader/docker/docker-compose.yaml \
  --profile observability up -d
```

Need Elasticsearch, Kibana, or Jaeger? Include the infrastructure override:

```bash
docker compose --project-directory deploy/gpt_trader/docker \
  -f deploy/gpt_trader/docker/docker-compose.yaml \
  -f deploy/gpt_trader/docker/docker-compose.infrastructure.yaml \
  --profile observability up -d
```

Run the exporter alongside the bot:

```bash
poetry run python scripts/monitoring/export_metrics.py \
  --metrics-file var/data/coinbase_trader/prod/metrics.json \
  --port 9102
```

## Key Metrics & Suggested Thresholds

| Metric | Description | Warning | Critical |
|--------|-------------|---------|----------|
| `coinbase_trader_cycle_duration_seconds` | Control loop wall time | > 3s | > 6s |
| `coinbase_trader_cycle_errors_total` | Exceptions raised during cycle | +1 in 5m | +3 in 5m |
| `coinbase_trader_orders_submitted_total` | Orders sent in the last cycle | Deviation from baseline | Confirm unexpected surge |
| `coinbase_trader_risk_daily_loss` | Running PnL vs guard | > 80% of limit | ≥ limit (trading halts) |
| `coinbase_trader_market_data_staleness_seconds` | Age of last mark price | > 10s | > 30s |
| `coinbase_trader_telemetry_account_age_seconds` | Age of account snapshot | > 15m | > 30m |

Adjust thresholds per deployment, but keep the relationship (warning < critical)
to preserve alert semantics.

## Alert Reference

```yaml
# Bot not scraping metrics / exporter down
alert: CoinbaseTraderExporterDown
expr: up{job="coinbase-trader-exporter"} == 0
for: 2m
labels:
  severity: critical
annotations:
  runbook: docs/MONITORING_PLAYBOOK.md#coinbase-trader-not-reporting

# Daily loss approaching the configured stop
alert: DailyLossGuardWarn
expr: coinbase_trader_risk_daily_loss / coinbase_trader_risk_daily_loss_limit > 0.8
for: 1m
labels:
  severity: high
annotations:
  runbook: docs/MONITORING_PLAYBOOK.md#daily-loss-guard

# Market data stale beyond guardrails
alert: MarketDataStale
expr: coinbase_trader_market_data_staleness_seconds > 30
for: 30s
labels:
  severity: high
annotations:
  runbook: docs/MONITORING_PLAYBOOK.md#market-data-staleness
```

## Incident Workflows

### Coinbase Trader Not Reporting

1. Confirm exporter health (container, process, or systemd service).
2. Tail the bot logs for errors (`grep "CRITICAL"`).
3. If the bot crashed, restart using your deployment tooling
   (e.g., `supervisorctl restart coinbase-trader` or redeploy via CI).
4. Validate recovery with `poetry run coinbase-trader run --profile dev --dev-fast` in a
   staging environment before enabling live trading if the crash was caused by a
   code change.

### Daily Loss Guard

1. Review the guard output in logs (`risk.guards.daily_loss.*`).
2. Ensure positions are flattened; if not, use manual order tooling:

   ```bash
   poetry run coinbase-trader orders preview \
     --symbol BTC-USD --side sell --type market \
     --quantity 0.01 --reduce-only
   ```

3. Keep trading halted until root cause (market regime, bug, configuration) is
   identified.
4. Document incident notes for post-mortem.

### Market Data Staleness

1. Check upstream connectivity (Coinbase REST/WS). If the exchange is degraded,
   remain in reduce-only mode.
2. Verify network access from the host (ping, traceroute, etc.).
3. Inspect logs for reconnection attempts or authentication failures.
4. Restart the data pump if the exchange is healthy but the bot remains stale.

### Order Tooling Errors

1. Review `coinbase_trader_order_tooling_errors_total` (if enabled) or log lines.
2. Confirm the profile has appropriate permissions (account snapshot).
3. Retry tooling commands after verifying parameters (symbol, quantity, TIF).

## Post-Incident Checklist

1. Confirm exporter metrics recovered (no alerting).
2. Ensure `coinbase_trader_cycle_errors_total` stopped incrementing.
3. Run `poetry run pytest -q` for code-related incidents before redeploying.
4. Update relevant runbooks or thresholds if changes were required.

## Legacy Material

Retired guides covering the old `gpt-trader` dashboard were removed from the
tree. If you need them for reference, retrieve them from repository history and
do not use them for production operations.
