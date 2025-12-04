"""
TUI Manager Classes.

Extracted from TraderApp to improve testability and separation of concerns.
"""

from gpt_trader.tui.managers.bot_lifecycle import BotLifecycleManager
from gpt_trader.tui.managers.ui_coordinator import UICoordinator

__all__ = [
    "BotLifecycleManager",
    "UICoordinator",
]
