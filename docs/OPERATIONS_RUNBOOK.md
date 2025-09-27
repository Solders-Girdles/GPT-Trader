# GPT-Trader Operations Runbook (Spot Profiles)

This runbook documents the operational expectations for the `perps-bot`
orchestrator. It replaces the legacy Phase 3 documentation that referenced
retired ML services, dashboards, and the `gpt-trader` CLI.

## Daily Checklist

1. **Creds & Env** – Ensure the deployment environment exports the correct
   Coinbase credentials and leaves `COINBASE_ENABLE_DERIVATIVES` unset unless
   INTX access is confirmed.
2. **Dry-run smoke** – `poetry run perps-bot --profile dev --dev-fast` should
   complete without errors on staging infrastructure.
3. **Metric scrape** – Verify `/metrics` is reachable from the Prometheus
   exporter and confirms recent timestamps.
4. **Risk guard lookback** – Review guard counters (drawdown, staleness,
   volatility) for unexpected spikes.

## Incident Response

| Severity | Criteria | First Actions |
|----------|----------|---------------|
| `SEV-1` | Bot offline, or guard-triggered halt with open exposure | Halt trading if not already stopped, contact on-call, collect logs, confirm balances |
| `SEV-2` | Partial functionality loss (e.g., account telemetry stalled) | Investigate recent deploys, restart affected service, monitor recovery |
| `SEV-3` | Degraded performance without financial impact | Create ticket, plan fix, monitor |

### Generic Recovery Flow

1. Identify failing component (bot loop, exporter, network).
2. Capture logs (`journalctl -u perps-bot` or container logs).
3. Restart the component using the supervisor used in your environment.
4. Confirm recovery via metrics and a controlled `--dev-fast` run.
5. File or update the incident record.

## Common Playbooks

### Exchange/API Degradation

1. Guard will raise `market_data_staleness`. Switch to reduce-only mode with
   `poetry run perps-bot --profile spot --reduce-only`.
2. Confirm Coinbase status and rate limits.
3. Once restored, re-enable full trading and watch the next five cycles.

### Risk Guard Triggered

1. Check logs for the specific guard (daily loss, volatility, correlation).
2. Ensure the guard forced reduce-only or flat positions; if not, issue manual
   market orders using the preview tooling.
3. Document root cause and required config tweaks.

### Metrics Exporter Gap

1. Validate the path passed to `--metrics-file` still updates.
2. Restart the exporter process.
3. Use `curl http://localhost:9102/metrics` to confirm Prometheus scrape
   readiness.

## Operational Commands

- `poetry run perps-bot --profile canary --dry-run` – Protective canary run.
- `poetry run perps-bot --account-snapshot` – On-demand permissions audit.
- `poetry run python scripts/monitoring/export_metrics.py --metrics-file ...` –
  Start exporter locally.
- `poetry run pytest -q` – Full regression suite (run pre-deploy when code
  changes land).

## Change Management

1. Stage changes in feature branches; ensure documentation and tests update
   together.
2. CI must include `poetry run pytest -q` and any per-profile smoke tests.
3. Deployment should follow canary → prod progression with live monitoring at
   each stage.

## Knowledge Base

- `docs/ARCHITECTURE.md` – High-level system overview.
- `docs/MONITORING_PLAYBOOK.md` – Metrics and alert details.
- Repository history – Contains the deprecated runbooks if you need to review
  them.
