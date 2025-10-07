# GPT-Trader Monitoring & Dashboards

The current monitoring stack centres on runtime telemetry emitted by
`perps-bot`, the live risk engine, and the Prometheus-compatible exporter. This
guide covers the primary surfaces now that the legacy Streamlit UI has been
retired.

## Runtime Telemetry

- `var/data/perps_bot/<profile>/metrics.json` captures the latest cycle metrics
  for each profile. Tools like `jq` or spreadsheets work well for ad-hoc
  reviews.
- `poetry run perps-bot account snapshot` prints balances, fee tiers, and
  permissions without executing the trading loop.
- The risk manager records periodic snapshots via
  `bot_v2/features/live_trade/risk_metrics.py`; these feed dashboards and the
  monitoring APIs.
- Application logs stream to stdout. Ship them to your logging stack (e.g.,
  Loki, Splunk, CloudWatch) for long-term retention and alerting.

## Prometheus Exporter

```bash
poetry run python scripts/monitoring/export_metrics.py \
  --metrics-file var/data/perps_bot/prod/metrics.json \
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
as the exporter and uses the monitoring framework directly.

```bash
poetry run python scripts/perps_dashboard.py \
  --metrics-file var/data/perps_bot/canary/metrics.json \
  --refresh-seconds 5
```

- Supports local terminal output and optional curses-based rendering.
- Displays runtime guard statuses (healthy/warning/breached), recent alerts, and
  equity/notional snapshots.
- Pulls resource telemetry from `bot_v2/monitoring/system/` collectors when
  available.

## Advanced Monitoring Components

- **Runtime guards** (`bot_v2/monitoring/runtime_guards.py`): support comparison
  modes (`gt`, `lt`, `abs_gt`, etc.), warning bands, and contextual messaging.
- **Alert channels** (`bot_v2/monitoring/alerts.py`): Slack, PagerDuty, email,
  webhook, and a safe no-op fallback. Configure channels via your orchestration
  or deployment scripts.
- **Validation framework** (`bot_v2/validation`): declarative validators with
  inline predicate support keep inputs clean before they reach monitoring
  surfaces.
- **Monitoring dashboard module** (`bot_v2/monitoring/monitoring_dashboard.py`):
  provides snapshot storage and helper utilities for CLI dashboards or future
  UI integrations.

## Alerting Hooks

- Integrate exporter endpoints with Grafana, Prometheus Alertmanager, PagerDuty,
  or Slack according to your observability stack.
- Guard breaches emit structured log events (`risk.guards.<name>.*`) and can be
  surfaced via the dashboard or your log pipeline.
- The alert dispatcher logs a warning when no transport is configured, allowing
  safe degradation in lower environments.

## Legacy UI

The Streamlit dashboard bundled with earlier versions relied on modules that no
longer ship with GPT-Trader V2. For historical reference, retrieve the assets
from repository history; avoid reintroducing them into active deployments.
