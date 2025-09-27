# GPT-Trader Monitoring Surfaces

The Streamlit dashboard and launcher scripts shipped with the v1 stack have
been retired. Runtime visibility now comes from telemetry emitted by the
`perps-bot` orchestrator and the Prometheus-compatible exporter. If you need
the old UI assets, retrieve them from repository history.

## Runtime Telemetry

- `var/data/perps_bot/<profile>/metrics.json` captures the latest cycle metrics for
  each profile. Use standard tooling (`jq`, spreadsheets) for quick reviews.
- Application logs stream to stdout; redirect to your logging stack of choice.
- `poetry run perps-bot --account-snapshot` prints balances, fee tiers, and
  permissions without executing the trading loop.

## Prometheus Exporter

```bash
poetry run python scripts/monitoring/export_metrics.py \
  --metrics-file var/data/perps_bot/prod/metrics.json \
  --port 9102
```

- Exposes `/metrics` for Prometheus and `/metrics.json` for JSON polling.
- Use separate processes per profile if you publish multiple metric files.
- The exporter reads files written by the bot; ensure `metrics.json` updates are
  enabled in your deployment configuration.

## Alerting Hooks

- Integrate exporter endpoints with Grafana, PagerDuty, or Slack via your
  existing observability stack.
- Risk guard breaches raise structured log events
  (`risk.guards.<name>.*`). Tail logs or ship them to your log aggregator for
  alerting rules.

## Legacy UI

The deprecated Streamlit interface depended on modules removed from the current
code base. Avoid resurrecting those scripts; pull them from git history only if
you need to study the legacy implementation.
