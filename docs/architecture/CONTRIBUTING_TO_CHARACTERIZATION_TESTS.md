# Contributing to Characterization Tests

**Goal**: Collaboratively expand test coverage before refactoring PerpsBot

## What Are Characterization Tests?

**Purpose**: Document current behavior before changing code
- They freeze "what happens now" (not "what should happen")
- They catch regressions during refactoring
- They may test ugly/weird behavior - that's OK!

**Not for**: Testing ideal behavior or adding new features

## Current Status

**Location**: `tests/integration/perps_bot_characterization/` (modular test suite)
**Current**: 59 tests (all active, 0 skipped)
**Structure**: 9 focused test modules + shared fixtures in conftest.py

## How to Contribute

### Quick Start (10 minutes)

1. **Pick an xfail test** from any test module (look for `@pytest.mark.xfail`)
2. **Write a test** documenting current behavior
3. **Run it** to verify it passes
4. **Commit** with message: `test: expand characterization for [area]`

### Example: Adding a Config Change Test

**Step 1: Find an xfail test**
```python
@pytest.mark.xfail(reason="TODO: Test config change removes old symbols from mark_windows")
def test_apply_config_change_removes_old_symbols(self):
    """Placeholder for config change behavior test."""
    pytest.fail("Not implemented")
```

**Step 2: Write the test**
```python
@pytest.mark.asyncio
async def test_apply_config_change_updates_symbols(self, monkeypatch, tmp_path):
    """Document: apply_config_change must update bot.symbols"""
    monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
    monkeypatch.setattr(PerpsBot, "_start_streaming_background", lambda self: None)

    # Initial config
    config = BotConfig(profile=Profile.DEV, symbols=["BTC-USD"], mock_broker=True)
    bot = PerpsBot(config)

    # Mock coordinators to avoid side effects
    bot.execution_coordinator.reset_order_reconciler = Mock()
    bot.config_controller.sync_with_risk_manager = Mock()
    bot.strategy_orchestrator.init_strategy = Mock()
    bot._restart_streaming_if_needed = Mock()

    # Create config change
    from bot_v2.orchestration.config_controller import ConfigChange
    new_config = BotConfig(profile=Profile.DEV, symbols=["ETH-USD", "SOL-USD"], mock_broker=True)
    change = ConfigChange(updated=new_config, diff={"symbols": ["ETH-USD", "SOL-USD"]})

    # Apply change
    bot.apply_config_change(change)

    # Document what happens
    assert bot.symbols == ["ETH-USD", "SOL-USD"]
    assert "ETH-USD" in bot.mark_windows
    assert "SOL-USD" in bot.mark_windows
```

**Step 3: Run the test**
```bash
pytest tests/integration/perps_bot_characterization/test_delegation.py::TestPerpsBotDelegation::test_apply_config_change_updates_symbols -v
```

**Step 4: Commit**
```bash
git add tests/integration/perps_bot_characterization/test_delegation.py
git commit -m "test: characterize config change symbol updates"
```

## Available xfail Tests (20 placeholders)

All TODOs have been converted to `@pytest.mark.xfail` placeholder tests. Find them by searching for `@pytest.mark.xfail` in the test modules:

### By Module:

**test_initialization.py** (5 xfail tests):
- `test_initialization_sets_derivatives_enabled`
- `test_initialization_creates_session_guard`
- `test_initialization_creates_config_controller`
- `test_initialization_verifies_broker_exists`
- `test_initialization_verifies_risk_manager_exists`

**test_update_marks.py** (4 xfail tests):
- `test_concurrent_update_marks_thread_safety`
- `test_update_marks_with_none_quote`
- `test_update_marks_with_invalid_mark_price`
- `test_update_marks_exception_preserves_risk_manager_state`

**test_properties.py** (2 xfail tests):
- `test_property_setters_update_registry`
- `test_properties_work_after_builder_construction`

**test_delegation.py** (3 xfail tests):
- `test_write_health_status_delegation`
- `test_is_reduce_only_mode_delegation`
- `test_set_reduce_only_mode_delegation`

**test_streaming.py** (2 xfail tests):
- `test_concurrent_update_mark_window_calls`
- `test_mark_trimming_is_atomic`

**test_full_cycle.py** (4 xfail tests):
- `test_background_tasks_spawned_non_dry_run`
- `test_all_background_tasks_canceled_on_shutdown`
- `test_shutdown_does_not_hang`
- `test_trading_window_checks`

## Writing Good Characterization Tests

### ✅ DO

**Document behavior, not implementation**
```python
def test_update_marks_updates_risk_manager(self):
    """Document: update_marks MUST update risk_manager.last_mark_update"""
    # Good - tests observable behavior
    await bot.update_marks()
    assert "BTC-USD" in bot.risk_manager.last_mark_update
```

**Use descriptive names**
```python
def test_apply_config_change_removes_old_symbol_marks(self):
    # Good - clear what behavior is tested
```

**Test edge cases**
```python
def test_update_marks_continues_after_symbol_error(self):
    # Good - documents error handling behavior
```

**Keep tests simple**
```python
# Good - one behavior per test
def test_broker_property_raises_when_none(self):
    bot.registry = bot.registry.with_updates(broker=None)
    with pytest.raises(RuntimeError):
        _ = bot.broker
```

### ❌ DON'T

**Test implementation details**
```python
def test_update_marks_calls_asyncio_to_thread(self):
    # Bad - tests HOW, not WHAT
```

**Make tests depend on each other**
```python
def test_step_1(self):
    self.bot = PerpsBot(config)  # Bad - shared state

def test_step_2(self):
    self.bot.update_marks()  # Bad - depends on test_step_1
```

**Test ideal behavior**
```python
def test_update_marks_should_cache_quotes(self):
    # Bad - this is a feature request, not characterization
```

**Overuse mocking**
```python
# Bad - too many mocks obscure behavior
bot.broker = Mock()
bot.risk_manager = Mock()
bot.event_store = Mock()
# ... 10 more mocks
```

## Test Organization

The characterization suite is organized into focused modules:

```
tests/integration/perps_bot_characterization/
├── __init__.py              # Module documentation
├── conftest.py              # Shared fixtures (minimal_config, mock_quote)
├── test_initialization.py   # Service creation and wiring
├── test_update_marks.py     # Mark price updates and window trimming
├── test_properties.py       # Property descriptors (broker, risk_manager, etc.)
├── test_delegation.py       # Method delegation to coordinators
├── test_streaming.py        # Streaming service and lock sharing
├── test_full_cycle.py       # End-to-end lifecycle tests
├── test_feature_toggles.py  # Feature flags and streaming restart
├── test_builder.py          # Builder pattern construction
└── test_strategy_services.py # Strategy orchestrator services
```

Add new tests to the appropriate module based on the behavior being characterized.

## Common Patterns

### Pattern: Testing Error Handling

```python
def test_method_continues_after_error(self):
    """Document: method must not raise, just log error"""
    bot.broker.get_quote = Mock(side_effect=Exception("API error"))

    # Should not raise
    await bot.update_marks()

    # Verify it logged but continued
    assert True  # If we got here, it didn't raise
```

### Pattern: Testing State Changes

```python
def test_method_updates_state(self):
    """Document: method must update specific state"""
    initial_count = len(bot.mark_windows["BTC-USD"])

    await bot.update_marks()

    assert len(bot.mark_windows["BTC-USD"]) == initial_count + 1
```

### Pattern: Testing Thread Safety

```python
def test_concurrent_access_is_safe(self):
    """Document: concurrent calls must not corrupt state"""
    import threading

    def worker():
        bot._update_mark_window("BTC-USD", Decimal("50000"))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify no corruption
    assert all(isinstance(m, Decimal) for m in bot.mark_windows["BTC-USD"])
```

### Pattern: Testing Delegation

```python
def test_method_delegates_to_service(self):
    """Document: method must call specific service method"""
    bot.execution_coordinator.execute_decision = AsyncMock()

    await bot.execute_decision("BTC-USD", decision, mark, product, state)

    bot.execution_coordinator.execute_decision.assert_called_once()
```

## Running Your Tests

### Run single test
```bash
pytest tests/integration/perps_bot_characterization/test_delegation.py::TestPerpsBotDelegation::test_write_health_status_delegation -v
```

### Run single module
```bash
pytest tests/integration/perps_bot_characterization/test_initialization.py -v
```

### Run all characterization tests
```bash
pytest tests/integration/perps_bot_characterization/ -m characterization -v
```

### Run with coverage
```bash
pytest tests/integration/perps_bot_characterization/ --cov=src/bot_v2/orchestration/perps_bot --cov-report=term-missing
```

## Contribution Workflow

1. **Pick an xfail test** - Choose from list above or search for `@pytest.mark.xfail` in test modules
2. **Create branch** - `git checkout -b test/characterize-[area]`
3. **Write test** - Follow patterns above
4. **Verify it passes** - Run the test
5. **Commit** - `git commit -m "test: characterize [specific behavior]"`
6. **Push** - `git push origin test/characterize-[area]`
7. **PR** - Create small PR (1-3 tests per PR is fine)

## PR Guidelines

**Good PR**:
- Title: `test: characterize PerpsBot [area] behavior`
- Description: "Adds characterization tests for [specific behavior]. Expands coverage from TODO line X."
- Changes: 1-5 new tests
- All tests passing ✅

**Example PR Description**:
```markdown
## What

Adds characterization tests for `apply_config_change` behavior.

## Tests Added

- test_apply_config_change_updates_symbols
- test_apply_config_change_removes_old_symbols
- test_apply_config_change_updates_session_guard

## Coverage

Addresses TODOs on lines 138, 139, 141 in test file.

## Checklist

- [x] Tests pass locally
- [x] Tests document behavior, not implementation
- [x] Test names are descriptive
- [x] Followed contribution guide
```

## Questions?

- **Slack**: #bot-refactoring
- **This guide**: `docs/architecture/CONTRIBUTING_TO_CHARACTERIZATION_TESTS.md`
- **Phase 0 Status**: `docs/archive/refactoring-2025-q1/REFACTORING_PHASE_0_STATUS.md`

---

**Goal**: Comprehensive characterization coverage before refactoring
**Current**: 59 tests (all active, 100% completion)
**Progress**: Phase 1 expanded coverage from 38 to 59 active tests (+55%)
**Status**: ✅ COMPLETE - All core behaviors and lifecycle patterns characterized
