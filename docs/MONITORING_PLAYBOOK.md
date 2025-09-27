# GPT-Trader Monitoring & Incident Response

This playbook reflects the spot-first `perps-bot` architecture. It supersedes
the retired monitoring guides that targeted the legacy `gpt-trader` service and
its ML pipeline.

## Observability Stack

- **Metrics**: `scripts/monitoring/export_metrics.py` exposes `/metrics` and
  `/metrics.json` from the bot's `metrics.json` output.
- **Logs**: Structured JSON (or text) emitted by `perps-bot`; forward to your
  logging stack for search and alerting.
- **Account telemetry**: `poetry run perps-bot --account-snapshot` when a manual
  check is required.

Run the exporter alongside the bot:

```bash
poetry run python scripts/monitoring/export_metrics.py \
  --metrics-file var/data/perps_bot/prod/metrics.json \
  --port 9102
```

## Key Metrics & Suggested Thresholds

| Metric | Description | Warning | Critical |
|--------|-------------|---------|----------|
| `perps_bot_cycle_duration_seconds` | Control loop wall time | > 3s | > 6s |
| `perps_bot_cycle_errors_total` | Exceptions raised during cycle | +1 in 5m | +3 in 5m |
| `perps_bot_orders_submitted_total` | Orders sent in the last cycle | Deviation from baseline | Confirm unexpected surge |
| `perps_bot_risk_daily_loss` | Running PnL vs guard | > 80% of limit | ≥ limit (trading halts) |
| `perps_bot_market_data_staleness_seconds` | Age of last mark price | > 10s | > 30s |
| `perps_bot_telemetry_account_age_seconds` | Age of account snapshot | > 15m | > 30m |

Adjust thresholds per deployment, but keep the relationship (warning < critical)
to preserve alert semantics.

## Alert Reference

```yaml
# Bot not scraping metrics / exporter down
alert: PerpsBotExporterDown
expr: up{job="perps-bot-exporter"} == 0
for: 2m
labels:
  severity: critical
annotations:
  runbook: docs/MONITORING_PLAYBOOK.md#perps-bot-not-reporting

# Daily loss approaching the configured stop
alert: DailyLossGuardWarn
expr: perps_bot_risk_daily_loss / perps_bot_risk_daily_loss_limit > 0.8
for: 1m
labels:
  severity: high
annotations:
  runbook: docs/MONITORING_PLAYBOOK.md#daily-loss-guard

# Market data stale beyond guardrails
alert: MarketDataStale
expr: perps_bot_market_data_staleness_seconds > 30
for: 30s
labels:
  severity: high
annotations:
  runbook: docs/MONITORING_PLAYBOOK.md#market-data-staleness
```

## Incident Workflows

### Perps-Bot Not Reporting

1. Confirm exporter health (container, process, or systemd service).
2. Tail the bot logs for errors (`grep "CRITICAL"`).
3. If the bot crashed, restart using your deployment tooling
   (e.g., `supervisorctl restart perps-bot` or redeploy via CI).
4. Validate recovery with `poetry run perps-bot --profile dev --dev-fast` in a
   staging environment before enabling live trading if the crash was caused by a
   code change.

### Daily Loss Guard

1. Review the guard output in logs (`risk.guards.daily_loss.*`).
2. Ensure positions are flattened; if not, use manual order tooling:

   ```bash
   poetry run perps-bot --profile spot --preview-order \
     --order-symbol BTC-USD --order-side sell --order-type market \
     --order-qty 0.01 --order-reduce-only
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

1. Review `perps_bot_order_tooling_errors_total` (if enabled) or log lines.
2. Confirm the profile has appropriate permissions (account snapshot).
3. Retry tooling commands after verifying parameters (symbol, quantity, TIF).

## Post-Incident Checklist

1. Confirm exporter metrics recovered (no alerting).
2. Ensure `perps_bot_cycle_errors_total` stopped incrementing.
3. Run `poetry run pytest -q` for code-related incidents before redeploying.
4. Update relevant runbooks or thresholds if changes were required.

## Legacy Material

Retired guides covering the old `gpt-trader` dashboard were removed from the
tree. If you need them for reference, retrieve them from repository history and
do not use them for production operations.
