# TUI Event System Migration Guide

## Overview

This guide explains how to migrate from direct widget queries to the event-based messaging system introduced in the TUI architecture refactoring (Phase 1).

**Why migrate?**
- **Loose coupling**: Managers don't need to know about widget hierarchy
- **Testability**: Events can be tested without mounting widgets
- **Scalability**: Adding new listeners doesn't require changing existing code
- **Reliability**: Widget lifecycle issues (not mounted, unmounted) don't break managers

## Quick Reference

### Before (Direct Widget Access)
```python
# ❌ Anti-pattern: Direct widget query with private attribute access
main_screen = self.app.query_one(MainScreen)
exec_widget = main_screen.query_one("#dash-execution")
trades_widget = exec_widget.query_one(TradesWidget)
if hasattr(trades_widget, "_trade_matcher"):
    trades_widget._trade_matcher.reset()
```

### After (Event-Based)
```python
# ✅ Best practice: Post event
from gpt_trader.tui.events import TradeMatcherResetRequested

self.app.post_message(TradeMatcherResetRequested())
```

## Migration Patterns

### Pattern 1: Replacing Widget Queries

**Scenario**: Manager needs to trigger action on a widget

#### Before
```python
# managers/bot_lifecycle.py
try:
    from gpt_trader.tui.widgets.portfolio import TradesWidget

    main_screen = self.app.query_one(MainScreen)
    exec_widget = main_screen.query_one("#dash-execution")
    trades_widget = exec_widget.query_one(TradesWidget)
    trades_widget._trade_matcher.reset()
    logger.info("Reset trade matcher")
except Exception as e:
    logger.debug(f"Could not reset trade matcher: {e}")
```

**Problems:**
- Fragile: Breaks if widget hierarchy changes
- Tight coupling: Manager knows about widget internal structure
- Private access: Accessing `_trade_matcher` violates encapsulation
- Error-prone: Multiple query_one calls can fail silently

#### After
```python
# managers/bot_lifecycle.py
from gpt_trader.tui.events import TradeMatcherResetRequested

self.app.post_message(TradeMatcherResetRequested())
logger.info("Posted TradeMatcherResetRequested event")
```

```python
# widgets/trades_widget.py
from gpt_trader.tui.events import TradeMatcherResetRequested

class TradesWidget(Static):
    def on_trade_matcher_reset_requested(
        self,
        event: TradeMatcherResetRequested
    ) -> None:
        """Reset trade matcher on bot start/mode switch."""
        if hasattr(self, "_trade_matcher"):
            self._trade_matcher.reset()
            logger.info("Trade matcher reset via event")
```

**Benefits:**
- Decoupled: Manager doesn't know about widget hierarchy
- Encapsulated: Widget controls its own state
- Resilient: Event ignored if widget not mounted (no errors)
- Testable: Can test manager without mounting widgets

### Pattern 2: Broadcasting State Changes

**Scenario**: Notify multiple widgets when state changes

#### Before
```python
# managers/bot_lifecycle.py
async def start_bot(self) -> None:
    # ... start logic ...

    # Manually update multiple widgets
    try:
        status_widget = self.app.query_one(BotStatusWidget)
        status_widget.running = True
    except Exception:
        pass

    try:
        mode_selector = self.app.query_one(ModeSelector)
        mode_selector.enabled = False
    except Exception:
        pass
```

**Problems:**
- Repetitive: Same try/except pattern for each widget
- Incomplete: Easy to forget updating a widget
- Fragile: Widget addition requires manager changes

#### After
```python
# managers/bot_lifecycle.py
from gpt_trader.tui.events import BotStateChanged

async def start_bot(self) -> None:
    # ... start logic ...

    # Broadcast state change - all interested widgets receive it
    self.app.post_message(BotStateChanged(running=True, uptime=0.0))
```

```python
# widgets/status.py
class BotStatusWidget(Static):
    def on_bot_state_changed(self, event: BotStateChanged) -> None:
        """Update display when bot state changes."""
        self.running = event.running
        self.uptime = event.uptime
```

```python
# widgets/mode_selector.py
class ModeSelector(Select):
    def on_bot_state_changed(self, event: BotStateChanged) -> None:
        """Disable mode selector when bot is running."""
        self.enabled = not event.running
```

**Benefits:**
- Broadcast: One event reaches all listeners
- Extensible: Adding new listener doesn't change manager
- Clean: No try/except boilerplate

### Pattern 3: Request-Response with Events

**Scenario**: Manager needs data from widget

#### Before
```python
# Not typically done - direct widget queries are used instead
# But this pattern is needed for state requests
try:
    widget = self.app.query_one(SomeWidget)
    state = widget.get_state()
    # Process state
except Exception:
    state = None
```

#### After
```python
# Component requesting data
from gpt_trader.tui.events import TradeMatcherStateRequest, TradeMatcherStateResponse
import uuid

class MyComponent:
    def __init__(self):
        self._pending_requests = {}

    def request_trade_matcher_state(self) -> None:
        request_id = str(uuid.uuid4())
        self._pending_requests[request_id] = True
        self.app.post_message(TradeMatcherStateRequest(request_id=request_id))

    def on_trade_matcher_state_response(
        self,
        event: TradeMatcherStateResponse
    ) -> None:
        if event.request_id in self._pending_requests:
            # Process response
            state = event.state
            del self._pending_requests[event.request_id]
```

```python
# Widget responding to request
class TradesWidget(Static):
    def on_trade_matcher_state_request(
        self,
        event: TradeMatcherStateRequest
    ) -> None:
        state = {}
        if hasattr(self, "_trade_matcher"):
            state = self._trade_matcher.get_state()

        self.post_message(
            TradeMatcherStateResponse(
                request_id=event.request_id,
                state=state
            )
        )
```

**Benefits:**
- Asynchronous: Doesn't block waiting for response
- Decoupled: Requester doesn't need to know widget location
- Traceable: request_id links request to response

## Using the EventHandlerMixin

For widgets that handle multiple events, use the `EventHandlerMixin`:

### Without Mixin
```python
class MyWidget(Static):
    def on_bot_state_changed(self, event: BotStateChanged) -> None:
        logger.debug(f"Received BotStateChanged: {event.running}")
        # Custom logic

    def on_bot_mode_changed(self, event: BotModeChanged) -> None:
        logger.debug(f"Received BotModeChanged: {event.new_mode}")
        # Custom logic

    def on_state_validation_failed(self, event: StateValidationFailed) -> None:
        logger.warning(f"Validation failed: {len(event.errors)} errors")
        # Custom logic
```

### With Mixin
```python
from gpt_trader.tui.mixins import EventHandlerMixin

class MyWidget(EventHandlerMixin, Static):
    # Mixin provides default logging for all events
    # Override only the ones you need

    def on_bot_state_changed(self, event: BotStateChanged) -> None:
        super().on_bot_state_changed(event)  # Default logging
        # Custom logic here
```

**Benefits:**
- Less boilerplate: Default logging provided
- Consistency: Standard logging format
- Override flexibility: Only override what you need

## Common Migration Scenarios

### Scenario 1: Resetting Trade Matcher

**Files affected:**
- `managers/bot_lifecycle.py` (lines 96-108, 288-300)

**Migration:**
1. Replace widget queries with `TradeMatcherResetRequested` event
2. Add event handler in `TradesWidget`

**Status:** ✅ Identified in Phase 1 plan

### Scenario 2: Updating Mode Selector State

**Files affected:**
- `managers/bot_lifecycle.py` (lines 419-428, 430-439)

**Migration:**
1. Replace widget queries with `BotStateChanged` event
2. Add event handler in `ModeSelector`

**Status:** ✅ Identified in Phase 1 plan

### Scenario 3: Refreshing UI After State Sync

**Files affected:**
- `managers/bot_lifecycle.py` (lines 146, 176, 304, 448)

**Migration:**
1. Replace `_update_main_screen()` calls with `UIRefreshRequested` event
2. Add event handler in `MainScreen`

**Status:** 🔄 Part of Phase 6

## Event Naming Conventions

Follow these conventions when creating new events:

### Request Events
Use `*Requested` suffix for actions that may be denied:
- `BotStartRequested` (may be denied if already running)
- `BotStopRequested` (may be denied if already stopped)
- `ConfigReloadRequested`

### Change Events
Use `*Changed` suffix for completed state changes:
- `BotStateChanged` (state has changed, not a request)
- `BotModeChanged` (mode has changed successfully)
- `ThemeChanged`

### Response Events
Use `*Response` suffix for request-response patterns:
- `TradeMatcherStateResponse` (response to request)

## Testing with Events

### Testing Event Emission

```python
import pytest
from textual.app import App
from gpt_trader.tui.events import BotStateChanged

async def test_bot_start_emits_state_changed():
    """Test that starting bot emits BotStateChanged event."""
    app = App()

    # Capture events
    events_received = []

    def capture_event(event):
        events_received.append(event)

    async with app.run_test() as pilot:
        # Subscribe to events
        app.screen.on(BotStateChanged, capture_event)

        # Trigger action that should emit event
        await app.lifecycle_manager.start_bot()

        # Verify event was emitted
        assert len(events_received) == 1
        assert isinstance(events_received[0], BotStateChanged)
        assert events_received[0].running is True
```

### Testing Event Handling

```python
async def test_widget_handles_bot_state_changed():
    """Test that widget responds to BotStateChanged event."""
    from gpt_trader.tui.widgets.status import BotStatusWidget
    from gpt_trader.tui.events import BotStateChanged

    app = App()
    async with app.run_test() as pilot:
        widget = BotStatusWidget()
        app.screen.mount(widget)

        # Post event
        app.post_message(BotStateChanged(running=True, uptime=10.5))

        # Wait for event to be processed
        await pilot.pause()

        # Verify widget state updated
        assert widget.running is True
        assert widget.uptime == 10.5
```

## Gradual Migration Strategy

Don't migrate everything at once. Follow this phased approach:

### Phase 1: Add Event System (✅ Current)
- Define all events in `events.py`
- Create `EventHandlerMixin`
- Write migration guide (this document)
- Add tests for events

### Phase 2: Critical Path Migration
- Migrate high-impact coupling (trade matcher reset)
- Migrate bot lifecycle events
- Test thoroughly

### Phase 3: UI Coordination Migration
- Migrate responsive state updates
- Migrate theme changes
- Migrate UI refresh requests

### Phase 4: Complete Migration
- Remove all remaining widget queries
- Deprecate direct access patterns
- Update all documentation

### Phase 5: Cleanup
- Remove deprecated code
- Verify all managers use events only
- Code review for any remaining anti-patterns

## Troubleshooting

### Event Not Received

**Symptom**: Posted event not triggering handler

**Possible causes:**
1. Widget not mounted when event posted
2. Handler method name doesn't match event class
3. Event not imported in handler file

**Solutions:**
```python
# 1. Check if widget is mounted
def on_mount(self) -> None:
    logger.info(f"{self.__class__.__name__} mounted and ready for events")

# 2. Verify handler name matches event
# Event: BotStateChanged
# Handler: on_bot_state_changed (lowercase with underscores)

# 3. Import the event class
from gpt_trader.tui.events import BotStateChanged
```

### Event Received Multiple Times

**Symptom**: Handler called multiple times for single event

**Possible causes:**
1. Multiple widgets with same handler
2. Event bubbling not stopped
3. Event posted multiple times in loop

**Solutions:**
```python
# 1. This is expected - multiple widgets can handle same event

# 2. Stop event propagation if needed
def on_bot_state_changed(self, event: BotStateChanged) -> None:
    # Handle event
    event.stop()  # Don't propagate to parent

# 3. Check posting logic
# Use a flag to prevent duplicate posts
if not self._event_posted:
    self.app.post_message(BotStateChanged(...))
    self._event_posted = True
```

## Additional Resources

- [Textual Messages Documentation](https://textual.textualize.io/guide/events/)
- [TUI Events API Reference](../reference/tui_events.md)
- [Architecture Decision Record: Event System](../architecture/adr-001-event-system.md)

## Questions?

If you encounter issues during migration, please:
1. Check this guide for common patterns
2. Review existing event handlers in the codebase
3. Ask in the team chat or create an issue
