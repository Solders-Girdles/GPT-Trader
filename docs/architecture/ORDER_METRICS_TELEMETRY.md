# Order Metrics Telemetry Integration

**Date**: 2025-10-02
**Status**: ✅ Complete

## Overview

OrderMetricsReporter metrics are now automatically exported to MetricsCollector for telemetry dashboards (Prometheus, DataDog, etc.).

## Architecture

```
AdvancedExecutionEngine
    ├── OrderMetricsReporter (tracks order events)
    │   └── export_to_collector() → MetricsCollector
    └── export_metrics() (coordinates export)
         ↓
PerpsBot._run_execution_metrics_export()  (background task, 60s interval)
         ↓
LifecycleService.configure_background_tasks()  (registers task)
         ↓
MetricsCollector (global singleton)
         ↓
Telemetry Export (Prometheus/DataDog)
```

## Exported Metrics

### Order Lifecycle Metrics (Gauges)
- `execution.orders.placed` - Total orders placed
- `execution.orders.filled` - Total orders filled
- `execution.orders.cancelled` - Total orders cancelled
- `execution.orders.rejected` - Total orders rejected
- `execution.orders.post_only_rejected` - Total post-only rejections

### Rejection Reasons (Gauges)
- `execution.orders.rejection.{reason}` - Count per rejection reason
  - `risk` - Risk limit violations
  - `position_sizing` - Position sizing rejections
  - `spec_violation` - Order spec violations
  - `post_only_cross` - Post-only crossed spread
  - `stop_validation` - Stop order validation failures
  - etc.

### Execution State Metrics (Gauges)
- `execution.pending_orders` - Current pending order count
- `execution.stop_triggers` - Total stop triggers registered
- `execution.active_stops` - Currently active stop orders

## Usage

### Automatic Export (Production)

Metrics are exported automatically every 60 seconds via background task:

```python
# No code needed - automatic in production
bot = PerpsBot(config)
await bot.run()  # Metrics exported every 60s
```

### Manual Export (Testing/Development)

```python
from bot_v2.monitoring.metrics_collector import get_metrics_collector

# Get global collector
collector = get_metrics_collector()

# Export from AdvancedExecutionEngine
exec_engine.export_metrics(collector, prefix="execution")

# Get metrics summary
summary = collector.get_metrics_summary()
print(summary["gauges"])
```

### Custom Prefix

```python
# Export with custom prefix
exec_engine.export_metrics(collector, prefix="live_trading")

# Results in metrics like:
# - live_trading.orders.placed
# - live_trading.orders.filled
# etc.
```

## Configuration

### Background Task Interval

Configured in `lifecycle_service.py`:

```python
# Default: 60 seconds
self._task_registry.register(
    lambda: asyncio.create_task(self._run_execution_metrics_export())
)
```

To change interval, modify `_run_execution_metrics_export()` in `perps_bot.py`:

```python
async def _run_execution_metrics_export(self, interval_seconds: int = 60) -> None:
    # Change to 30s:
    # interval_seconds = 30
```

## Prometheus Integration Example

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'gpt-trader'
    static_configs:
      - targets: ['localhost:8000']
    metric_relabel_configs:
      # Collect execution metrics
      - source_labels: [__name__]
        regex: 'execution\\..*'
        action: keep
```

## DataDog Integration Example

```python
from bot_v2.monitoring.metrics_collector import get_metrics_collector
from datadog import statsd

collector = get_metrics_collector()

# Export to DataDog
for name, value in collector.gauges.items():
    if name.startswith("execution."):
        statsd.gauge(name, value, tags=["env:production"])
```

## Testing

### Unit Tests

```bash
# Test OrderMetricsReporter export
pytest tests/unit/bot_v2/features/live_trade/test_order_metrics_telemetry.py -v
```

### Integration Tests

```bash
# Test end-to-end export
pytest tests/integration/test_execution_metrics_export.py -v
```

### Manual Verification

```python
from bot_v2.monitoring.metrics_collector import get_metrics_collector
from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine

# Create engine and collector
engine = AdvancedExecutionEngine(broker=broker)
collector = get_metrics_collector()

# Place some test orders
engine.place_order(...)

# Export and verify
engine.export_metrics(collector)
print(collector.get_metrics_summary())
```

## Monitoring Dashboard Queries

### Grafana/Prometheus

```promql
# Order placement rate (orders/minute)
rate(execution_orders_placed[5m]) * 60

# Rejection rate (%)
(execution_orders_rejected / execution_orders_placed) * 100

# Top rejection reasons
topk(5, execution_orders_rejection)

# Pending order backlog
execution_pending_orders
```

### DataDog

```
# Order placement rate
avg:execution.orders.placed{*}.as_rate()

# Rejection breakdown
sum:execution.orders.rejection.*{*} by {reason}

# Stop order monitoring
avg:execution.active_stops{*}
```

## Implementation Files

### Core Files
- `src/bot_v2/features/live_trade/order_metrics_reporter.py` - Metrics tracking + export
- `src/bot_v2/features/live_trade/advanced_execution.py` - export_metrics() method
- `src/bot_v2/orchestration/perps_bot.py` - _run_execution_metrics_export()
- `src/bot_v2/orchestration/lifecycle_service.py` - Background task registration

### Test Files
- `tests/unit/bot_v2/features/live_trade/test_order_metrics_telemetry.py` - 6 unit tests
- `tests/integration/test_execution_metrics_export.py` - 2 integration tests

## Benefits

1. **Real-time Visibility**: Order metrics surface in dashboards immediately
2. **Alerting**: Set up alerts on rejection rates, pending order backlogs
3. **Debugging**: Rejection reason breakdown helps diagnose issues
4. **Performance**: Non-blocking background export (60s interval)
5. **Zero Config**: Automatic in production, no setup required

## Future Enhancements

- [ ] Export to time-series database (InfluxDB)
- [ ] Per-symbol metrics (e.g., `execution.orders.placed{symbol="BTC-USD"}`)
- [ ] Per-strategy metrics (e.g., `execution.orders.placed{strategy="momentum"}`)
- [ ] Histogram metrics for order latency
- [ ] Custom metric labels/tags
