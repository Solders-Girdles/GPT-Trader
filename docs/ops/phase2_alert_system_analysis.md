# Phase 2: Alert System Architecture Analysis

**Date:** 2025-01-15
**Scope:** Analyze `alerts.py` vs `alerts_manager.py` for potential consolidation

---

## Executive Summary

**Recommendation: DO NOT CONSOLIDATE**

The two modules serve distinct architectural layers with minimal duplication. Consolidation would mix infrastructure and application concerns, reducing testability and reusability.

---

## Module Comparison

### alerts.py (572 lines) - Infrastructure Layer

**Responsibilities:**
- Core data structures: `Alert`, `AlertSeverity`/`AlertLevel` (enum)
- Channel abstraction: `AlertChannel` base class
- Concrete implementations: `LogChannel`, `SlackChannel`, `PagerDutyChannel`, `EmailChannel`, `WebhookChannel`
- Central dispatcher: `AlertDispatcher`
- Convenience builders: `create_risk_alert()`, `create_execution_alert()`, `create_system_alert()`
- Config-based setup: `AlertDispatcher.from_config()`
- Basic history: `alert_history` (list of alerts)

**Import Patterns (16 files):**
```python
# Most common - fundamental types used across monitoring
from bot_v2.monitoring.alerts import Alert, AlertLevel, AlertSeverity

# Less common - infrastructure usage
from bot_v2.monitoring.alerts import AlertDispatcher, AlertChannel
```

**Key Call Sites:**
- `src/bot_v2/monitoring/__init__.py:10` - Public API exports
- `src/bot_v2/monitoring/alerts_manager.py:18` - Wraps AlertDispatcher
- `src/bot_v2/monitoring/runtime_guards/` - Uses AlertSeverity enum
- `src/bot_v2/features/live_trade/guard_errors.py` - Uses AlertLevel enum
- Tests: `test_alerts_dispatcher.py`, `test_alert_channels.py`

**Test Coverage:**
- `test_alerts_dispatcher.py` - 13 tests covering:
  - Alert serialization
  - Channel threshold filtering
  - Dispatcher multi-channel routing
  - Slack/PagerDuty/Email/Webhook channel implementations
  - Config-based dispatcher construction

---

### alerts_manager.py (455 lines) - Application Layer

**Responsibilities:**
- Wraps `AlertDispatcher` with bot-specific features
- **Deduplication**: `dedup_window_seconds`, `_recent_alerts` tracking
- **Profile-based config**: `_load_dispatcher_config()`, `_resolve_config_path()`, `_PROFILE_FALLBACKS`
- **Env var expansion**: `_expand_env_var()`, `_build_email_config()`
- **Alert creation**: `create_alert()` with deduplication + dispatch
- **Cleanup**: `cleanup_old_alerts()` for retention policy
- **Backward compat**: `get_all_alerts()`, `get_active_alerts()`, `acknowledge_alert()`, `resolve_alert()`
- **Factory methods**: `from_settings()`, `from_profile_yaml()`

**Import Patterns (4 files):**
```python
# Only used in monitoring module
from bot_v2.monitoring.alerts_manager import AlertManager
```

**Key Call Sites:**
- `src/bot_v2/monitoring/system/engine.py:22,67-82` - **Primary consumer**:
  ```python
  # MonitoringSystem creates AlertManager via profile/settings
  self.alert_manager = AlertManager.from_profile_yaml(path=..., profile=...)
  # or
  self.alert_manager = AlertManager.from_settings(alert_settings)
  ```
- `src/bot_v2/monitoring/system/engine.py:161,207-323` - Uses `create_alert()`, `get_active_alerts()`
- Tests: `test_alerts_manager.py`, `test_evaluator_frozen_time.py`

**Test Coverage:**
- `test_alerts_manager.py` - 4 tests covering:
  - Deduplication with clock regression handling
  - Cleanup of old alerts + dedup tracking
  - Recent alerts filtering
  - Dispatch integration
- `test_evaluator_frozen_time.py` - Extensive time-based tests (dedup, cleanup, retention)

---

## Architectural Layering

```
┌─────────────────────────────────────────────────┐
│  Application Layer (alerts_manager.py)          │
│  - AlertManager                                 │
│  - Deduplication policy                         │
│  - YAML profile loading                         │
│  - Bot-specific env var expansion               │
│  - Backward compatibility shims                 │
└────────────────┬────────────────────────────────┘
                 │ HAS-A (composition)
                 ▼
┌─────────────────────────────────────────────────┐
│  Infrastructure Layer (alerts.py)               │
│  - AlertDispatcher                              │
│  - Alert, AlertSeverity (data structures)       │
│  - Channel abstraction (Slack, Email, etc.)     │
│  - Reusable dispatch logic                      │
└─────────────────────────────────────────────────┘
```

**Relationship:**
- **Composition**: `AlertManager` HAS-A `AlertDispatcher` (line 35)
- **Delegation**: Manager delegates dispatch to dispatcher
- **Value-add**: Manager adds dedup, config loading, cleanup on top

---

## Duplication Analysis

### Apparent Duplication

1. **alert_history field:**
   - `AlertDispatcher.alert_history` (alerts.py:377) - stores all dispatched alerts
   - `AlertManager.alert_history` (alerts_manager.py:36) - separate field
   - **Reality**: AlertManager creates alerts and stores them separately, then dispatches via AlertDispatcher. Each maintains history independently.
   - **Verdict**: Not true duplication - different lifecycle tracking

2. **get_recent_alerts() method:**
   - `AlertDispatcher.get_recent_alerts()` (alerts.py:436-447) - filters dispatcher history
   - `AlertManager.get_recent_alerts()` (alerts_manager.py:350-370) - filters manager history with additional filtering
   - **Reality**: AlertManager version adds `severity` and `source` filtering not in dispatcher
   - **Verdict**: Minor overlap, but manager's version has extra functionality

3. **max_history constant:**
   - Both set to 1000 by default
   - **Verdict**: Shared constant, not duplication

### True Duplication: None found

No copy-pasted code blocks or redundant logic.

---

## Why Consolidation Would Be Harmful

### 1. Violates Single Responsibility Principle
- `AlertDispatcher` should only dispatch (infrastructure)
- `AlertManager` should only manage (application policy)
- Merging would create a god class handling both

### 2. Mixes Concerns
Infrastructure concerns (alerts.py):
- How to send alerts to Slack/PagerDuty/Email
- Channel abstraction and routing
- Multi-channel dispatch coordination

Application concerns (alerts_manager.py):
- When to deduplicate alerts (policy)
- How to load bot profiles from YAML
- Where to find config files (`_PROFILE_FALLBACKS`)
- Backward compatibility with old monitoring system

### 3. Reduces Reusability
- `AlertDispatcher` could be used in non-bot contexts (generic alerting library)
- Adding bot-specific logic would pollute the infrastructure layer

### 4. Complicates Testing
Current state:
- `test_alerts_dispatcher.py` - Clean infrastructure tests (channels, routing)
- `test_alerts_manager.py` - Clean application tests (dedup, cleanup, config)

After merge:
- Single massive test file mixing infrastructure and application tests
- Harder to test dispatch logic in isolation

### 5. Breaks Clean Dependency Flow
Current: `MonitoringSystem → AlertManager → AlertDispatcher → Channels`

After merge: `MonitoringSystem → AlertDispatcher+Manager → Channels` (flattened, less clear)

---

## Current Design Strengths

✅ **Clear separation of concerns**: Infrastructure vs application
✅ **Testable**: Each layer tested independently
✅ **Extensible**: Can swap AlertDispatcher implementations
✅ **Reusable**: AlertDispatcher is generic, not bot-specific
✅ **Maintainable**: Changes to dispatch logic don't affect dedup logic

---

## Potential Improvements (If Needed)

If we wanted to clean things up without consolidation:

1. **Clarify alert_history ownership:**
   - Consider whether both need to track history, or if AlertManager should query AlertDispatcher
   - Current design is fine but could be more explicit

2. **Document the layering:**
   - Add module docstrings clarifying "infrastructure" vs "application"
   - Already done in Phase 1 via ARCHITECTURE.md updates ✓

3. **Extract profile loading:**
   - `_load_dispatcher_config()`, `_resolve_config_path()` could move to a separate ConfigLoader
   - Low priority - current design is acceptable

---

## Conclusion

**Status: No action required**

The alert system architecture is **sound**. The apparent duplication is actually proper layering. The two modules serve distinct purposes:

- **alerts.py** = Reusable alerting infrastructure
- **alerts_manager.py** = Bot-specific alert management

Consolidation would reduce code quality, testability, and maintainability.

**Recommendation for Phase 3:** Move to next cleanup target (TBD).
