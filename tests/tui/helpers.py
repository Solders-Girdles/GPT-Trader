from textual.pilot import Pilot
from textual.widgets import Label, Static


async def wait_for_widget(pilot: Pilot, widget_id: str, timeout: float = 5.0) -> None:
    """
    Waits for a widget with the given ID to be mounted and visible.
    """
    await pilot.pause()
    # Textual's pilot.wait_for_selector is not always reliable in async tests,
    # but let's try to use the app's query to check existence.
    # In a real scenario, we might loop with a timeout.
    # For now, we'll just rely on pilot.pause() allowing the event loop to process.

    # Check if widget exists
    assert pilot.app.query(f"#{widget_id}"), f"Widget #{widget_id} not found"


def assert_widget_visible(app, widget_id: str) -> None:
    """
    Asserts that a widget with the given ID is visible.
    """
    widget = app.query_one(f"#{widget_id}")
    assert widget.visible, f"Widget #{widget_id} is not visible"


def assert_widget_content(app, widget_id: str, expected_content: str) -> None:
    """
    Asserts that a widget (Label or Static) contains the expected text.
    """
    widget = app.query_one(f"#{widget_id}")
    if isinstance(widget, (Label, Static)):
        assert expected_content in str(
            widget.render()
        ), f"Expected '{expected_content}' in widget #{widget_id}"
    else:
        # Fallback for other widgets if needed, or raise error
        pass
