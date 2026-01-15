"""Top-level pytest plugin registration."""

pytest_plugins = [
    "tests.unit.gpt_trader.tui.tui_factories",
    "tests.unit.gpt_trader.tui.tui_pilots",
    "tests.unit.gpt_trader.tui.tui_singletons",
    "tests.unit.gpt_trader.tui.tui_snapshots",
]
