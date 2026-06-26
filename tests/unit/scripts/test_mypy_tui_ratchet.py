from __future__ import annotations

import re
import tomllib
from pathlib import Path


def _mypy_config() -> dict:
    return tomllib.loads(Path("pyproject.toml").read_text())["tool"]["mypy"]


def test_tui_mypy_ratchet_includes_initial_state_and_type_modules() -> None:
    exclude = re.compile(_mypy_config()["exclude"])

    included_paths = [
        "src/gpt_trader/tui/",
        "src/gpt_trader/tui/state_management/",
        "src/gpt_trader/tui/types.py",
        "src/gpt_trader/tui/thresholds.py",
        "src/gpt_trader/tui/state_management/__init__.py",
        "src/gpt_trader/tui/state_management/validators.py",
    ]
    excluded_paths = [
        "src/gpt_trader/tui/app.py",
        "src/gpt_trader/tui/screens/main_screen.py",
        "src/gpt_trader/tui/services/mode_service.py",
        "src/gpt_trader/tui/widgets/dashboard.py",
    ]

    assert all(exclude.search(path) is None for path in included_paths)
    assert all(exclude.search(path) is not None for path in excluded_paths)


def test_tui_mypy_ratchet_disables_ignore_errors_for_included_modules() -> None:
    overrides = _mypy_config()["overrides"]

    ratchet_override = next(
        override
        for override in overrides
        if "gpt_trader.tui.state_management.validators" in override["module"]
    )
    broad_tui_override = next(
        override for override in overrides if override["module"] == "gpt_trader.tui.*"
    )

    assert ratchet_override["ignore_errors"] is False
    assert broad_tui_override["ignore_errors"] is True
