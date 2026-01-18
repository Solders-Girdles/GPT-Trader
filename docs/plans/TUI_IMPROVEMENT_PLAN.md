# TUI Architecture Improvement Plan

This document provides concrete implementation steps for addressing the identified technical issues in the GPT-Trader TUI. Each approach includes specific file changes, code patterns, and verification steps.

## Current Architecture Summary

### Data Flow
```
StatusReporter (push via observer)
    ↓
TraderApp._on_status_update()
    ↓
UICoordinator.apply_observer_update()
    ├─→ TuiState.update_from_bot_status()
    └─→ UICoordinator.update_main_screen()
        ├─→ MainScreen.state = state (reactive)
        ├─→ SystemDetailsScreen.state = state (direct)
        └─→ StateRegistry.broadcast(state) → 16 widgets
```

### Key Files
| File | Lines | Purpose |
|------|-------|---------|
| `tui/managers/ui_coordinator.py` | ~555 | Update coordination, heartbeat loop |
| `tui/services/state_registry.py` | ~154 | Observer broadcast pattern |
| `tui/state.py` | ~400+ | Reactive state container (Widget) |
| `tui/screens/main_screen.py` | ~337 | Main dashboard layout |
| `tui/app.py` + 5 mixins | ~200 each | Application + lifecycle management |

---

## Approach A: Unified Observer Registry (Decoupling)

**Goal:** Remove screen-specific queries from `UICoordinator` so it doesn't need to know about `MainScreen` or `SystemDetailsScreen`.

### Implementation Steps

#### Step 1: Extend StateObserver Protocol

**File:** [state_registry.py](../../src/gpt_trader/tui/services/state_registry.py)

Add a priority attribute to allow screens to register with higher priority:

```python
class StateObserver(Protocol):
    """Protocol for widgets that observe TuiState changes."""

    def on_state_updated(self, state: TuiState) -> None:
        """Called when TuiState is updated."""
        ...

    @property
    def observer_priority(self) -> int:
        """Higher priority observers are updated first. Default: 0."""
        return 0
```

#### Step 2: Update StateRegistry Broadcast Order

**File:** [state_registry.py](../../src/gpt_trader/tui/services/state_registry.py)

Sort observers by priority before broadcasting:

```python
def broadcast(self, state: TuiState) -> None:
    """Broadcast state update to all registered observers."""
    observers = sorted(
        self._observers,
        key=lambda o: getattr(o, 'observer_priority', 0),
        reverse=True  # Higher priority first
    )
    # ... rest of broadcast logic
```

#### Step 3: Make MainScreen a StateObserver

**File:** [main_screen.py](../../src/gpt_trader/tui/screens/main_screen.py)

```python
class MainScreen(Screen):
    """Main screen implementing StateObserver for decoupled updates."""

    @property
    def observer_priority(self) -> int:
        return 100  # High priority - update before widgets

    def on_mount(self) -> None:
        # Register with StateRegistry
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)
        # ... existing mount logic

    def on_unmount(self) -> None:
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def on_state_updated(self, state: TuiState) -> None:
        """Receive state updates via registry broadcast."""
        self.state = state
```

#### Step 4: Simplify UICoordinator.update_main_screen()

**File:** [ui_coordinator.py](../../src/gpt_trader/tui/managers/ui_coordinator.py)

Replace direct screen queries with pure broadcast:

```python
def update_main_screen(self) -> None:
    """Broadcast state to all registered observers."""
    if hasattr(self.app, "state_registry"):
        self.app.state_registry.broadcast(self.app.tui_state)

    # Toggle heartbeat animation
    self.app._pulse_heartbeat()
    logger.debug("UI updated via registry broadcast")
```

Remove lines 273-284 (direct MainScreen/SystemDetailsScreen queries).

#### Step 5: Update SystemDetailsScreen

**File:** [system_details_screen.py](../../src/gpt_trader/tui/screens/system_details_screen.py)

Ensure it registers itself on mount (likely already does based on exploration).

### Verification

1. Run `pytest tests/tui/` - existing tests should pass
2. Add test in `tests/tui/test_decoupling.py`:
   ```python
   def test_coordinator_works_without_main_screen():
       """UICoordinator should broadcast without requiring MainScreen."""
       # Mock app with state_registry but no MainScreen mounted
       # Call update_main_screen() - should not raise
   ```
3. Manual: Launch TUI, verify all tiles update correctly

---

## Approach B: Service-Driven Data Flow

**Goal:** Extract data collection from heartbeat loop into dedicated services with cleaner scheduling.

### Implementation Steps

#### Step 1: Create DataCollectionService

**File:** `src/gpt_trader/tui/services/data_collection_service.py` (new)

```python
"""
Background data collection service for TUI.

Handles resilience metrics, validation stats, and market spreads
without blocking the UI heartbeat loop.
"""
import asyncio
from typing import TYPE_CHECKING

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.app import TraderApp

logger = get_logger(__name__, component="tui")


class DataCollectionService:
    """Manages background data collection tasks."""

    def __init__(self, app: TraderApp):
        self.app = app
        self._tasks: dict[str, asyncio.Task] = {}
        self._running = False

    async def start(self) -> None:
        """Start all background collection tasks."""
        self._running = True
        self._tasks["resilience"] = asyncio.create_task(
            self._resilience_loop()
        )
        self._tasks["validation"] = asyncio.create_task(
            self._validation_loop()
        )
        self._tasks["spreads"] = asyncio.create_task(
            self._spread_loop()
        )
        logger.info("DataCollectionService started")

    async def stop(self) -> None:
        """Stop all background tasks."""
        self._running = False
        for name, task in self._tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug(f"{name} collection task cancelled")
        self._tasks.clear()

    async def _resilience_loop(self) -> None:
        """Collect resilience metrics every 6 seconds."""
        while self._running:
            try:
                await asyncio.sleep(6)
                self._collect_resilience()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Resilience collection error: {e}")

    async def _validation_loop(self) -> None:
        """Collect validation metrics every 6 seconds."""
        while self._running:
            try:
                await asyncio.sleep(6)
                self._collect_validation()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Validation collection error: {e}")

    async def _spread_loop(self) -> None:
        """Collect spread data every 10 seconds (non-overlapping)."""
        while self._running:
            try:
                await asyncio.sleep(10)
                await self._collect_spreads()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Spread collection error: {e}")

    def _collect_resilience(self) -> None:
        """Collect resilience metrics from CoinbaseClient."""
        # Move logic from UICoordinator._collect_resilience_metrics()
        ...

    def _collect_validation(self) -> None:
        """Collect validation failure metrics."""
        # Move logic from UICoordinator._collect_validation_metrics()
        ...

    async def _collect_spreads(self) -> None:
        """Collect spread data from order books."""
        # Move logic from UICoordinator._collect_spread_data()
        ...
```

#### Step 2: Integrate with TraderApp Lifecycle

**File:** [app_lifecycle.py](../../src/gpt_trader/tui/app_lifecycle.py)

```python
# In _initialize_with_bot():
from gpt_trader.tui.services.data_collection_service import DataCollectionService

self.data_collection_service = DataCollectionService(self)
await self.data_collection_service.start()

# In on_unmount():
if hasattr(self, "data_collection_service"):
    await self.data_collection_service.stop()
```

#### Step 3: Simplify Heartbeat Loop

**File:** [ui_coordinator.py](../../src/gpt_trader/tui/managers/ui_coordinator.py)

Remove metric collection from `_heartbeat_loop()` - it now only handles:
- Heartbeat animation pulse
- Connection health check
- StatusReporter reconnection check

```python
async def _heartbeat_loop(self) -> None:
    """Heartbeat loop - animation and health only."""
    loop_count = 0
    reconnect_interval = 30

    while True:
        try:
            loop_count += 1

            # Pulse heartbeat when bot running
            if self._should_pulse():
                self.app._pulse_heartbeat()

            # Check connection health (broadcast on transition)
            self._check_connection_health()

            # Check for StatusReporter reconnection
            if loop_count % reconnect_interval == 0:
                await self._check_status_reporter_reconnection()

        except Exception as e:
            logger.debug(f"Heartbeat error: {e}")

        await asyncio.sleep(2)
```

### Verification

1. Run TUI and monitor CPU usage - should be lower
2. Check "Performance" overlay (Ctrl+P) for reduced broadcast timings
3. Verify resilience/validation/spread data still updates in System tile

---

## Approach C: Performance-First State Updates

**Goal:** Reduce reactive overhead by batching state mutations and implementing dirty-checking.

### Implementation Steps

#### Step 1: Add Batch Update Context Manager

**File:** [state.py](../../src/gpt_trader/tui/state.py)

```python
from contextlib import contextmanager

class TuiState(Widget):
    def __init__(self, ...):
        # ...existing init...
        self._batching = False
        self._pending_broadcast = False

    @contextmanager
    def batch_updates(self):
        """Context manager to batch multiple state updates.

        Usage:
            with tui_state.batch_updates():
                tui_state.running = True
                tui_state.uptime = 100
                tui_state.market_data = new_market
            # Single broadcast happens here
        """
        self._batching = True
        self._pending_broadcast = False
        try:
            yield
        finally:
            self._batching = False
            if self._pending_broadcast:
                self._trigger_broadcast()

    def _trigger_broadcast(self) -> None:
        """Request broadcast to observers."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.broadcast(self)
```

#### Step 2: Add Change Detection to Complex Objects

**File:** [types.py](../../src/gpt_trader/tui/types.py)

Add `__eq__` methods to state types for efficient comparison:

```python
@dataclass
class MarketState:
    prices: dict[str, Decimal] = field(default_factory=dict)
    changes_24h: dict[str, Decimal] = field(default_factory=dict)
    volumes: dict[str, Decimal] = field(default_factory=dict)
    spreads: dict[str, Decimal] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MarketState):
            return NotImplemented
        return (
            self.prices == other.prices
            and self.changes_24h == other.changes_24h
            and self.volumes == other.volumes
            and self.spreads == other.spreads
        )

    def signature(self) -> str:
        """Quick hash for change detection."""
        return f"{len(self.prices)}:{sum(self.prices.values()):.2f}"
```

#### Step 3: Use Batch Updates in UICoordinator

**File:** [ui_coordinator.py](../../src/gpt_trader/tui/managers/ui_coordinator.py)

```python
def _apply_status_update(self, status: BotStatus) -> None:
    # ... existing setup ...

    with self.app.tui_state.batch_updates():
        self.app.tui_state.running = self.app.bot.running
        self.app.tui_state.update_from_bot_status(status, runtime_state)
        self.app.tui_state.data_available = True
        self.app.tui_state.last_data_fetch = time.time()

    # Single broadcast happens at end of context manager
    self.update_main_screen()
```

### Verification

1. Add performance test measuring broadcast count before/after
2. Monitor frame timing with Performance overlay
3. Ensure no visual regressions (snapshot tests pass)

---

## Approach D: Responsive Layout Enhancement

**Goal:** Improve Bento Grid adaptivity for small terminals.

### Implementation Steps

#### Step 1: Define Breakpoints

**File:** [responsive_state.py](../../src/gpt_trader/tui/responsive_state.py)

```python
class ResponsiveState(Enum):
    COMPACT = "compact"      # < 80 cols: Stack vertically
    STANDARD = "standard"    # 80-120 cols: 2 columns
    COMFORTABLE = "comfortable"  # 120-160 cols: 3 columns
    WIDE = "wide"            # > 160 cols: Full grid
```

#### Step 2: Add Collapsed Layout CSS

**File:** `src/gpt_trader/tui/styles/layout/responsive.tcss` (new or update)

```css
/* Compact mode: Single column stack */
#bento-grid.compact {
    grid-size: 1;
    grid-rows: auto auto auto auto auto auto;
}

#bento-grid.compact > .bento-tile {
    height: auto;
    min-height: 4;
}

/* Hide less critical tiles in compact mode */
#bento-grid.compact #tile-strategy {
    display: none;
}
```

#### Step 3: Add Tab-Based View for Compact Mode

For very small terminals, consider a tabbed interface:

**File:** `src/gpt_trader/tui/screens/compact_main_screen.py` (new)

```python
class CompactMainScreen(Screen):
    """Tabbed layout for terminals < 80 columns."""

    BINDINGS = [
        Binding("1", "show_position", "Position"),
        Binding("2", "show_market", "Market"),
        Binding("3", "show_system", "System"),
    ]

    def compose(self) -> ComposeResult:
        yield TabPane("Position", id="tab-position")
        yield TabPane("Market", id="tab-market")
        yield TabPane("System", id="tab-system")
```

#### Step 4: Auto-Switch Layout Based on Size

**File:** [app_lifecycle.py](../../src/gpt_trader/tui/app_lifecycle.py)

```python
def on_resize(self, event: events.Resize) -> None:
    """Handle terminal resize events."""
    width = event.size.width

    # Update responsive state
    if width < 80:
        new_state = ResponsiveState.COMPACT
    elif width < 120:
        new_state = ResponsiveState.STANDARD
    elif width < 160:
        new_state = ResponsiveState.COMFORTABLE
    else:
        new_state = ResponsiveState.WIDE

    self.responsive_state = new_state

    # Propagate to MainScreen
    try:
        main_screen = self.query_one(MainScreen)
        main_screen.responsive_state = new_state
    except Exception:
        pass
```

### Verification

1. Test at various terminal sizes: 60, 80, 120, 160, 200 columns
2. Verify tile visibility and readability at each size
3. Test resize during running session

---

## Priority Recommendation

| Approach | Impact | Effort | Priority |
|----------|--------|--------|----------|
| **A: Unified Observer Registry** | High | Medium | P1 |
| **C: Performance-First Updates** | Medium | Low | P2 |
| **B: Service-Driven Data Flow** | Medium | Medium | P3 |
| **D: Responsive Layout** | Low | High | P4 |

**Recommended Order:**
1. **Approach A** first - decoupling is foundational for maintainability
2. **Approach C** second - quick win for performance
3. **Approach B** third - cleaner separation of concerns
4. **Approach D** last - nice-to-have UX improvement

---

## Test Plan Summary

### Unit Tests
```bash
pytest tests/tui/ -v
```

### New Tests Required
| Test File | Purpose |
|-----------|---------|
| `tests/tui/test_decoupling.py` | UICoordinator works without screen dependencies |
| `tests/tui/test_batch_updates.py` | State batching reduces broadcast count |
| `tests/tui/test_data_collection.py` | Background service lifecycle |
| `tests/tui/test_responsive.py` | Layout adaptation at breakpoints |

### Manual Verification
1. Launch TUI: `uv run gpt-trader tui`
2. Check Performance overlay: `Ctrl+P`
3. Test all navigation: arrow keys, Enter, number keys
4. Resize terminal and verify layout adapts
5. Monitor logs for errors: `tail -f logs/gpt_trader.log | grep tui`

---

## CSS Volume Note

The exploration noted large CSS files (>120KB). This is addressed in the existing modular structure under `styles/`. No immediate action needed unless specific performance issues are identified.
