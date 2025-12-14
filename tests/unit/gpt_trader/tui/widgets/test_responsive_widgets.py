from gpt_trader.tui.events import ResponsiveStateChanged
from gpt_trader.tui.responsive_state import ResponsiveState
from gpt_trader.tui.widgets.footer import ContextualFooter
from gpt_trader.tui.widgets.slim_status import SlimStatusWidget
from gpt_trader.tui.widgets.status import BotStatusWidget


class TestResponsiveWidgets:
    def test_contextual_footer_updates_on_responsive_event(self):
        footer = ContextualFooter()
        event = ResponsiveStateChanged(state=ResponsiveState.COMFORTABLE, width=140)
        footer.on_responsive_state_changed(event)
        assert footer.responsive_state == ResponsiveState.COMFORTABLE

    def test_bot_status_updates_on_responsive_event(self):
        status = BotStatusWidget()
        event = ResponsiveStateChanged(state=ResponsiveState.COMPACT, width=100)
        status.on_responsive_state_changed(event)
        assert status.responsive_state == ResponsiveState.COMPACT

    def test_slim_status_updates_on_responsive_event(self):
        slim = SlimStatusWidget()
        event = ResponsiveStateChanged(state=ResponsiveState.WIDE, width=180)
        slim.on_responsive_state_changed(event)
        assert slim.responsive_state == ResponsiveState.WIDE

