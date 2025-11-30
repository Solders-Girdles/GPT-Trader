# Monitoring

Observability infrastructure for the trading system.

## Purpose

This package provides comprehensive monitoring capabilities:

- **Alerts**: Severity-based alerting with dispatch to external services
- **Notifications**: Multi-channel notification delivery (Slack, email, webhooks)
- **Guards**: Runtime safety monitoring with automatic remediation
- **Metrics**: Performance and system health telemetry

## Key Components

| Module | Purpose |
|--------|---------|
| `alert_types.py` | Alert/AlertSeverity definitions |
| `notifications/` | Notification service and channels |
| `guards/` | Runtime guard manager |
| `system.py` | System-wide logger and telemetry |

## Alert Severity Levels

| Level | Numeric | Usage |
|-------|---------|-------|
| DEBUG | 10 | Development diagnostics |
| INFO | 20 | Normal operations |
| WARNING | 30 | Recoverable issues |
| ERROR | 40 | Non-recoverable failures |
| CRITICAL | 50 | System-wide failures |

## Integration

```python
from gpt_trader.monitoring.alert_types import Alert, AlertSeverity

alert = Alert(
    severity=AlertSeverity.WARNING,
    title="High Volatility",
    message="BTC-USD volatility exceeds threshold",
)
```

## Related Packages

- `features/live_trade/guard_errors.py` - Risk guard error types
- `persistence/` - Alert storage and history
