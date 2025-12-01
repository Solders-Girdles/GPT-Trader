# TUI Testing Guidelines

This document outlines the standards and best practices for testing the GPT-Trader TUI (Textual User Interface).

## Infrastructure

We use `pytest` along with `textual`'s built-in testing tools.

### Shared Fixtures

Common fixtures are defined in `tests/tui/conftest.py`:

- `mock_bot`: Provides a fully mocked `TradingBot` instance with `StatusReporter` and `StrategyEngine`.
- `mock_app`: Provides a `TraderApp` instance initialized with the `mock_bot`.

### Helper Functions

Helper functions are available in `tests/tui/helpers.py`:

- `wait_for_widget(pilot, widget_id)`: Waits for a widget to appear.
- `assert_widget_visible(app, widget_id)`: Asserts a widget is visible.
- `assert_widget_content(app, widget_id, content)`: Asserts a widget contains specific text.

## Writing Tests

### Integration Tests

Integration tests should simulate user interactions and verify the app's response.

```python
import pytest
from tests.tui.helpers import assert_widget_visible

@pytest.mark.asyncio
async def test_dashboard_load(mock_app):
    async with mock_app.run_test() as pilot:
        await pilot.pause()
        assert_widget_visible(mock_app, "dashboard-grid")
```

### Unit Tests

Unit tests should focus on individual widgets or logic.

```python
from gpt_trader.tui.widgets import MyWidget

def test_my_widget_logic():
    widget = MyWidget()
    # ... test logic ...
```

## Best Practices

1.  **Use `mock_bot`**: Avoid creating ad-hoc mocks for the bot. Use the shared fixture to ensure consistency.
2.  **Async Tests**: Most TUI tests involve async operations. Use `@pytest.mark.asyncio` and `async with app.run_test()`.
3.  **Selectors**: Use CSS selectors (e.g., `#id`, `.class`) to query widgets.
4.  **Stability**: Use `await pilot.pause()` to allow the event loop to process pending events before making assertions.
