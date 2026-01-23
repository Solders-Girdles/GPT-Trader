# Observability Reference

---
status: current
last-updated: 2026-01-23
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

## Regeneration

```bash
uv run agent-regenerate --only observability,logging
```

## Dashboards & Runbooks

See [Monitoring Playbook](MONITORING_PLAYBOOK.md) and [Alert Runbooks](operations/RUNBOOKS.md).
