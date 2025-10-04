# Recovery & Backup Telemetry Enhancement Plan

**Date**: October 2025
**Status**: Pending

Telemetric instrumentation will surface key outcomes from recovery and backup workflows, improve
observability in dashboards (Prometheus/DataDog), and provide actionable KPIs (success rates,
latencies, bottlenecks).

## Recovery Workflow Metrics

### Metrics Collector Integration
- Extend `RecoveryWorkflow` to accept optional `MetricsCollector` dependency.
- Provide accessor for orchestrator/monitor to inject the global collector.
- Use namespace `recovery.*` for all gauges.

### Metrics to Capture
- `recovery.operations.started` (counter)
- `recovery.operations.completed_total` (counter)
  - `recovery.operations.completed_success`
  - `recovery.operations.completed_partial`
  - `recovery.operations.completed_failed`
- `recovery.operations.escalated` (counter)
- `recovery.operation.duration_seconds` (histogram / summary)
  - Prometheus safe approach: export sum/count as gauges.
- `recovery.operation.rto_exceeded` (counter)
- `recovery.operation.retry_attempts` (histogram or gauge per operation)

### Instrumentation Points
1. Start: increment `started` as soon as `execute()` begins.
2. Handler retry loop: export retry count after execution.
3. Outcome: increment success/partial/failure and `escalated` when relevant.
4. Duration: measure elapsed time from pre-handler timestamps to outcome.

### Testing Strategy
- Mock `MetricsCollector` to assert increments in unit tests (list of calls).
- Add new tests in `tests/unit/bot_v2/state/recovery/test_recovery_workflow.py` using fixture that provides `collector`.
- Ensure metrics not emitted on exceptions (e.g., handler missing).

### Documentation
- Update `RECOVERY_ORCHESTRATOR_REFACTOR.md` with metrics section.
- Provide Prometheus queries (e.g., recovery success rate, escalations).

## Backup Workflow Metrics

### Metrics Collector Integration
- Inject `MetricsCollector` into `BackupWorkflow` and `RetentionManager` via `BackupManager`.
- Namespace: `backup.*`.

### Metrics to Capture
- `backup.operations.created_total`
  - `backup.operations.created_success`
  - `backup.operations.created_failed`
- `backup.operations.duration_seconds` (histogram via sum/count gauges)
- `backup.operations.size_bytes_total`
- `backup.retention.cleaned_total`
  - `backup.retention.cleaned_success`
  - `backup.retention.cleaned_failed`
- `backup.retention.removed_count`

### Instrumentation Points
1. Start/finish of `BackupWorkflow.create_backup`.
2. Catch exceptions to increment failure gauge.
3. Record size (sum of payload lengths) and normalized state entries.
4. `RetentionManager.cleanup`: record number of backups removed, batch vs. sequential path, errors.

### Testing Strategy
- Mock collector in workflow/retention unit tests.
- Verify metrics increments for success/failure, size accumulation, cleanup counts.

### Documentation
- Update `BACKUP_OPERATIONS_REFACTOR.md` with telemetry section.
- Provide sample dashboard filters: backup success rate, average durations, retention backlog.

## Repository Operation Metrics

### Metrics Collector Integration
- Optional extension: add instrumentation to repository base classes.
- Namespace: `state.repository.<tier>.*`

### Metrics to Capture
- `<tier>.operations.fetch_total`, `.store_total`, `.delete_total`
- `<tier>.operations.errors_total`
- `<tier>.operations.batch_sizes` (summary of batch operations)
- `<tier>.operations.latency_seconds` (if adapters expose timing)

### Implementation Notes
- Wrap public methods in each repository with metric increments.
- For batch operations capture counts and failure counts separately.
- Keep instrumentation lightweight (no logging changes required).

### Testing Strategy
- Extend existing repository unit tests to assert metric updates via mocked collector.

## Deliverables
- Instrumented modules (`RecoveryWorkflow`, `BackupWorkflow`, `RetentionManager`, optionally repositories).
- New unit tests per component verifying metrics emission.
- Prometheus/DataDog integration examples referencing new gauges.
- Documentation updates in respective architecture files and Phase summary.

## Sequencing
1. **Phase 1** — Recovery workflow metrics (highest impact for incident response).
2. **Phase 2** — Backup workflow + retention metrics (RPO/RTO visibility).
3. **Phase 3** — Repository metrics (operational observability).
4. Documentation updates after each phase.

---
Next action: inject `MetricsCollector` into `RecoveryWorkflow` and add success/failure gauges.
