# GPT-Trader Monitoring & Dashboards

The current monitoring stack centres on runtime telemetry emitted by
`coinbase-trader`, the live risk engine, and the Prometheus-compatible exporter. This
guide covers the primary surfaces now that the legacy Streamlit UI has been
retired.

## Runtime Telemetry

- `var/data/coinbase_trader/<profile>/metrics.json` captures the latest cycle metrics
  for each profile. Tools like `jq` or spreadsheets work well for ad-hoc
  reviews.
- `poetry run coinbase-trader account snapshot` prints balances, fee tiers, and
  permissions without executing the trading loop.
- The risk manager records periodic snapshots via
  `gpt_trader/features/live_trade/risk_metrics.py`; these feed dashboards and the
  monitoring APIs.
- Application logs stream to stdout. Ship them to your logging stack (e.g.,
  Loki, Splunk, CloudWatch) for long-term retention and alerting.

## Prometheus Exporter

```bash
poetry run python scripts/monitoring/export_metrics.py \
  --metrics-file var/data/coinbase_trader/prod/metrics.json \
  --port 9102
```

- Exposes `/metrics` for Prometheus scraping and `/metrics.json` for raw JSON
  polling.
- Run separate exporter processes per profile if you publish multiple metric
  files.
- Ensure the bot is configured to persist `metrics.json`; the exporter consumes
  that file as its data source.

## Real-Time Dashboard

Use `scripts/perps_dashboard.py` for a lightweight real-time view of guard
status, risk metrics, and system health. The script consumes the same telemetry
as the exporter without depending on the deprecated dashboard stack.

```bash
poetry run python scripts/perps_dashboard.py \
  --metrics-file var/data/coinbase_trader/canary/metrics.json \
  --refresh-seconds 5
```

- Supports local terminal output and optional curses-based rendering.
- Displays runtime guard statuses (healthy/warning/breached), recent alerts, and
  equity/notional snapshots.
- Pulls resource telemetry from `gpt_trader/monitoring/system/` collectors when
  available.

## Runtime Monitoring Components

- **Runtime guards** (`gpt_trader/monitoring/runtime_guards.py`): support comparison
  modes (`gt`, `lt`, `abs_gt`, etc.), warning bands, and contextual messaging.
- **Guard alert dispatcher** (`gpt_trader/features/live_trade/guard_errors.py`):
  wraps `AlertManager` to emit guard failures without the archived alert stack.
- **Validation framework** (`gpt_trader/validation`): declarative validators with
  inline predicate support keep inputs clean before they reach monitoring
  surfaces.


## Alerting Hooks

- Integrate exporter endpoints with Grafana, Prometheus Alertmanager, PagerDuty,
  or Slack according to your observability stack.
- Guard breaches emit structured log events (`risk.guards.<name>.*`) and can be
  surfaced via the dashboard or your log pipeline.
- The alert dispatcher logs a warning when no transport is configured, allowing
  safe degradation in lower environments.
