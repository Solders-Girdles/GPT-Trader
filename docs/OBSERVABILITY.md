# Observability Reference

---
status: current
last-updated: 2026-02-06
---

This page is intentionally thin. Metric names/labels and event schemas are generated from code to minimize drift.

## Generated Catalogs

- Metrics: [var/agents/observability/metrics_catalog.md](../var/agents/observability/metrics_catalog.md) ([.json](../var/agents/observability/metrics_catalog.json))
- Structured logging:
  - Event catalog: [var/agents/logging/event_catalog.json](../var/agents/logging/event_catalog.json)
  - Log schema: [var/agents/logging/log_schema.json](../var/agents/logging/log_schema.json)

## Key Entrypoints

- Metrics collection: `src/gpt_trader/monitoring/metrics_collector.py`
- Metrics export: `scripts/monitoring/export_metrics.py`
- Tracing helpers: `src/gpt_trader/observability/tracing.py`

## Health-Check Extension Contract

Health checks are registered in the health check runner and surfaced by the
health server's `/health` response. Registration entrypoints live in
`src/gpt_trader/monitoring/health_checks.py` (`HealthCheckRunner._health_check_registry`)
with the runner instantiated in
`src/gpt_trader/features/live_trade/engines/strategy.py`.

**Registration flow**
1. Add a check function in `src/gpt_trader/monitoring/health_checks.py` that returns
   `(healthy: bool, details: dict)`.
2. Register it in `HealthCheckRunner._health_check_registry` with a
   `HealthCheckDescriptor(name=..., mode="blocking"|"fast", run=...)`.
3. Ensure required dependencies are plumbed into `HealthCheckRunner` (broker,
   degradation state, risk manager, market data service).

**Expected result shape**
- `HealthState.add_check` stores results as:
  ```json
  {
    "checks": {
      "<name>": {
        "status": "pass" | "fail",
        "details": { "...": "..." }
      }
    }
  }
  ```
- Include `details["severity"]` as `warning` or `critical` when a failure should
  degrade or fail `/health`. The health server also appends `performance` and
  `signals` summaries to the `checks` payload.

**Extension checklist**
- Pick a unique check `name` and add the check function in
  `src/gpt_trader/monitoring/health_checks.py`.
- Return `(healthy, details)` with `details["severity"]` when failing.
- Register in `HealthCheckRunner._health_check_registry` with the correct `mode`.
- Confirm the `/health` payload renders as expected (see [Monitoring Playbook](MONITORING_PLAYBOOK.md)).

## Regeneration

```bash
uv run agent-regenerate --only observability,logging
```

## Dashboards & Runbooks

See [Monitoring Playbook](MONITORING_PLAYBOOK.md) and [Alert Runbooks](RUNBOOKS.md).
