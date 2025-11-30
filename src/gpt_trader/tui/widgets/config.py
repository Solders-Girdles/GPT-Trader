from typing import Any

from rich.pretty import Pretty
from textual.app import ComposeResult
from textual.containers import Grid
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ConfigModal(ModalScreen):
    """Modal to display current configuration."""

    CSS = """
    ConfigModal {
        align: center middle;
    }

    #dialog {
        grid-size: 2;
        grid-gutter: 1 2;
        grid-rows: 1fr 3;
        padding: 0 1;
        width: 80%;
        height: 80%;
        border: thick $background 80%;
        background: $surface;
    }

    #config-content {
        column-span: 2;
        height: 1fr;
        border: solid $accent;
        overflow: auto;
    }

    #close-btn {
        column-span: 2;
        width: 100%;
    }
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config

    def compose(self) -> ComposeResult:
        yield Grid(
            Static(Pretty(self.config), id="config-content"),
            Button("Close", variant="primary", id="close-btn"),
            id="dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-btn":
            self.dismiss()
