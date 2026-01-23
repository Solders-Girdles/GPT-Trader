#!/usr/bin/env python3
"""Guardrails for keeping TUI styling and layout consistent.

P4 rule set:
- No inline CSS blocks in Textual widgets/screens (e.g., CSS = \"\"\"...\"\"\").
- No hard-coded hex colors in TUI Python files (use TCSS variables or THEME).
- No unsupported CSS features (linear-gradient, first-child, attr selectors).
- Required CSS modules must exist for build.
- Performance budgets must be defined.

This runs in pre-commit on staged TUI Python files.

Usage:
    python scripts/ci/check_tui_guardrails.py [paths...]
    python scripts/ci/check_tui_guardrails.py --strict  # Include warnings
"""

from __future__ import annotations

import argparse
import pathlib
import re
from collections.abc import Sequence

# === Pattern Definitions ===

INLINE_CSS_RE = re.compile(r"^\s*CSS\s*=\s*(\"\"\"|''')", re.MULTILINE)
HEX_COLOR_RE = re.compile(r"#[0-9A-Fa-f]{6}\b")

# Unsupported CSS features in Textual
UNSUPPORTED_CSS_PATTERNS = [
    (re.compile(r"linear-gradient\s*\("), "linear-gradient() not supported"),
    (re.compile(r":first-child\b"), ":first-child not supported, use :first-of-type"),
    (re.compile(r":last-child\b"), ":last-child not supported, use :last-of-type"),
    (re.compile(r":nth-child\b"), ":nth-child not supported"),
    (re.compile(r"\[aria-"), "ARIA attribute selectors not supported, use classes"),
    (re.compile(r"::before\b"), "::before pseudo-element not supported"),
    (re.compile(r"::after\b"), "::after pseudo-element not supported"),
    (re.compile(r"position:\s*absolute"), "position: absolute not supported"),
    (re.compile(r"position:\s*fixed"), "position: fixed not supported"),
    (re.compile(r"transform:"), "transform not supported"),
    (re.compile(r"transition:"), "transition not supported"),
    (re.compile(r"animation:"), "animation not supported"),
    (re.compile(r"@keyframes\b"), "@keyframes not supported"),
    (re.compile(r"@media\b"), "@media queries not supported"),
]

# === Allowlists ===

# Files allowed to contain hex colors (theme definitions).
HEX_ALLOWLIST = {
    pathlib.Path("src/gpt_trader/tui/theme.py"),
}

# Files allowed to contain inline CSS (legacy or intentional).
# P2 detail screens use minimal inline CSS for layout that doesn't warrant separate TCSS files.
INLINE_CSS_ALLOWLIST: set[pathlib.Path] = {
    pathlib.Path("src/gpt_trader/tui/screens/watchlist_screen.py"),
    pathlib.Path("src/gpt_trader/tui/screens/position_detail_screen.py"),
    pathlib.Path("src/gpt_trader/tui/screens/market_detail_screen.py"),
    pathlib.Path("src/gpt_trader/tui/screens/account_detail_screen.py"),
}

# Required CSS modules for build (checked against styles directory).
REQUIRED_CSS_MODULES = [
    "theme/variables.tcss",
    "theme/variables_light.tcss",
    "theme/variables_high_contrast.tcss",
    "layout/screen.tcss",
    "layout/workspace.tcss",
    "components/focus.tcss",
    "components/headers.tcss",
    "components/buttons.tcss",
    "components/tables.tcss",
    "components/tabs.tcss",
    "components/select.tcss",
    "components/interaction_states.tcss",
    "components/polish.tcss",
    "components/accessibility.tcss",
]

# Required performance budget definitions
PERFORMANCE_BUDGET_FILE = pathlib.Path("docs/TUI_STYLE_GUIDE.md")


def check_python_files(
    files: list[pathlib.Path], root: pathlib.Path
) -> tuple[list[str], list[str]]:
    """Check Python files for style violations.

    Returns:
        Tuple of (errors, warnings).
    """
    errors: list[str] = []
    warnings: list[str] = []

    for path in files:
        rel = path.resolve().relative_to(root)
        text = path.read_text(encoding="utf-8")

        # Check for inline CSS
        if rel not in INLINE_CSS_ALLOWLIST and INLINE_CSS_RE.search(text):
            errors.append(f"{rel}: Contains inline CSS block. Move styles into TCSS modules.")

        # Check for hard-coded hex colors
        if rel not in HEX_ALLOWLIST and HEX_COLOR_RE.search(text):
            errors.append(f"{rel}: Contains hard-coded hex colors. Use TCSS variables or THEME.")

    return errors, warnings


def check_css_files(files: list[pathlib.Path], root: pathlib.Path) -> tuple[list[str], list[str]]:
    """Check CSS files for unsupported features.

    Returns:
        Tuple of (errors, warnings).
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Pattern to identify CSS comments
    comment_pattern = re.compile(r"/\*.*?\*/", re.DOTALL)

    for path in files:
        rel = path.resolve().relative_to(root)
        text = path.read_text(encoding="utf-8")

        # Remove comments before checking for unsupported features
        text_no_comments = comment_pattern.sub("", text)

        for pattern, message in UNSUPPORTED_CSS_PATTERNS:
            matches = list(pattern.finditer(text_no_comments))
            if matches:
                for match in matches[:3]:  # Limit to first 3 occurrences
                    # Find approximate line number (may be off due to comment removal)
                    line_num = text_no_comments[: match.start()].count("\n") + 1
                    errors.append(f"{rel}:{line_num}: {message}")

    return errors, warnings


def check_required_modules(styles_dir: pathlib.Path) -> tuple[list[str], list[str]]:
    """Check that required CSS modules exist.

    Returns:
        Tuple of (errors, warnings).
    """
    errors: list[str] = []
    warnings: list[str] = []

    for module in REQUIRED_CSS_MODULES:
        module_path = styles_dir / module
        if not module_path.exists():
            errors.append(f"Missing required CSS module: {module}")

    return errors, warnings


def check_performance_budgets() -> tuple[list[str], list[str]]:
    """Check that performance budgets are documented.

    Returns:
        Tuple of (errors, warnings).
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not PERFORMANCE_BUDGET_FILE.exists():
        warnings.append(f"Performance budget documentation not found: {PERFORMANCE_BUDGET_FILE}")
    else:
        content = PERFORMANCE_BUDGET_FILE.read_text(encoding="utf-8")
        required_sections = [
            "Performance Budgets",
            "Frame Timing",
            "Memory",
            "Update Cadence",
        ]
        for section in required_sections:
            if section not in content:
                warnings.append(
                    f"Performance budget section '{section}' not found in {PERFORMANCE_BUDGET_FILE}"
                )

    return errors, warnings


def scan(paths: Sequence[str], strict: bool = False) -> int:
    """Scan files for TUI guardrail violations.

    Args:
        paths: Files or directories to scan. Empty means scan default TUI paths.
        strict: If True, treat warnings as errors.

    Returns:
        Exit code (0 = success, 1 = errors found).
    """
    root = pathlib.Path.cwd()
    python_files: list[pathlib.Path] = []
    css_files: list[pathlib.Path] = []

    styles_dir = root / "src" / "gpt_trader" / "tui" / "styles"

    if paths:
        for entry in paths:
            p = pathlib.Path(entry)
            if p.is_dir():
                python_files.extend(p.rglob("*.py"))
                css_files.extend(p.rglob("*.tcss"))
            elif p.suffix == ".py":
                python_files.append(p)
            elif p.suffix == ".tcss":
                css_files.append(p)
    else:
        # Default: scan TUI directories
        tui_dir = root / "src" / "gpt_trader" / "tui"
        if tui_dir.exists():
            python_files = list(tui_dir.rglob("*.py"))
            css_files = list(styles_dir.rglob("*.tcss")) if styles_dir.exists() else []

    all_errors: list[str] = []
    all_warnings: list[str] = []

    # Check Python files
    errors, warnings = check_python_files(python_files, root)
    all_errors.extend(errors)
    all_warnings.extend(warnings)

    # Check CSS files
    errors, warnings = check_css_files(css_files, root)
    all_errors.extend(errors)
    all_warnings.extend(warnings)

    # Check required modules (only if scanning default paths)
    if not paths and styles_dir.exists():
        errors, warnings = check_required_modules(styles_dir)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    # Check performance budgets (only if scanning default paths)
    if not paths:
        errors, warnings = check_performance_budgets()
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    # Report results
    if all_errors:
        print("ERRORS:")
        for msg in all_errors:
            print(f"  ❌ {msg}")

    if all_warnings:
        print("WARNINGS:")
        for msg in all_warnings:
            print(f"  ⚠️  {msg}")

    if all_errors:
        print(f"\n{len(all_errors)} error(s) found.")
        return 1

    if strict and all_warnings:
        print(f"\n{len(all_warnings)} warning(s) found (strict mode).")
        return 1

    if not all_errors and not all_warnings:
        print("✓ TUI guardrails passed.")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to scan (default: src/gpt_trader/tui)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    args = parser.parse_args(argv)
    return scan(args.paths, strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
