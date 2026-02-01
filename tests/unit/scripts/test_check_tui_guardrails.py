from __future__ import annotations

from pathlib import Path

import scripts.ci.check_tui_guardrails as guardrails


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_check_python_files_blocks_inline_css(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "src" / "widget.py",
        "class Widget:\n" '    CSS = """\n' "#widget {\n" "  width: 1;\n" "}\n" '"""\n',
    )

    errors, warnings = guardrails.check_python_files([path], tmp_path)

    assert warnings == []
    assert any("Contains inline CSS block" in error for error in errors)


def test_check_python_files_blocks_hex_colors(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "src" / "colors.py",
        'BUTTON_COLOR = "#A1B2C3"\n',
    )

    errors, warnings = guardrails.check_python_files([path], tmp_path)

    assert warnings == []
    assert any("Contains hard-coded hex colors" in error for error in errors)


def test_check_python_files_allows_allowlisted_paths(tmp_path: Path) -> None:
    theme_path = _write(
        tmp_path / "src" / "gpt_trader" / "tui" / "theme.py",
        'PRIMARY = "#A1B2C3"\n',
    )
    inline_path = _write(
        tmp_path / "src" / "gpt_trader" / "tui" / "screens" / "watchlist_screen.py",
        "class WatchlistScreen:\n" '    CSS = """\n' "#watchlist {\n" "  width: 1;\n" "}\n" '"""\n',
    )

    errors, warnings = guardrails.check_python_files([theme_path, inline_path], tmp_path)

    assert errors == []
    assert warnings == []


def test_check_css_files_flags_unsupported_patterns(tmp_path: Path) -> None:
    css_path = _write(
        tmp_path / "styles" / "bad.tcss",
        """
Button {
    background: linear-gradient(#fff, #000);
}

Item:first-child {
    color: white;
}
""",
    )

    errors, warnings = guardrails.check_css_files([css_path], tmp_path)

    assert warnings == []
    assert any("linear-gradient() not supported" in error for error in errors)
    assert any(":first-child not supported" in error for error in errors)
    assert all("styles/bad.tcss:" in error for error in errors)


def test_check_required_modules_reports_missing(tmp_path: Path) -> None:
    styles_dir = tmp_path / "styles"
    missing_module = "components/tabs.tcss"

    for module in guardrails.REQUIRED_CSS_MODULES:
        if module == missing_module:
            continue
        _write(styles_dir / module, "/* stub */\n")

    errors, warnings = guardrails.check_required_modules(styles_dir)

    assert warnings == []
    assert errors == [f"Missing required CSS module: {missing_module}"]
