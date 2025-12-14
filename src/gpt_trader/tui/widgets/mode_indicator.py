"""Mode indicator widget for TUI header."""

from textual.reactive import reactive
from textual.widgets import Static


class ModeIndicator(Static):
    """Visual badge showing current bot operating mode."""

    mode = reactive("demo")

    MODE_CONFIG = {
        "demo": {
            "label": "DEMO - Mock Data",
            "style": "bold green",
            "description": "Synthetic prices • 2s updates • No real trading",
        },
        "paper": {
            "label": "PAPER - Real Data",
            "style": "bold yellow",
            "description": "Live Coinbase • 30s updates • Simulated execution",
        },
        "read_only": {
            "label": "OBSERVE - Read Only",
            "style": "bold blue",
            "description": "Live Coinbase • 15s updates • Orders disabled",
        },
        "live": {
            "label": "⚠ LIVE TRADING ⚠",
            "style": "bold white on red",
            "description": "REAL MONEY AT RISK",
        },
    }

    def render(self) -> str:
        """Render mode badge."""
        config = self.MODE_CONFIG.get(self.mode, self.MODE_CONFIG["demo"])
        return f"[{config['style']}]{config['label']}[/]"
