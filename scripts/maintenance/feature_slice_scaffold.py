#!/usr/bin/env python3
"""
Scaffold a new feature slice with optional tests and documentation.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SLICE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class ScaffoldAction:
    kind: str
    path: Path
    content: str | None = None


@dataclass(frozen=True)
class ScaffoldPlan:
    slice_name: str
    actions: list[ScaffoldAction]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaffold a new feature slice.")
    parser.add_argument(
        "--name",
        required=True,
        help="Slice name in snake_case (letters, numbers, underscores).",
    )
    parser.add_argument(
        "--with-tests",
        action="store_true",
        help="Include a unit test skeleton for the slice.",
    )
    parser.add_argument(
        "--with-readme",
        action="store_true",
        help="Include a README template for the slice.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended actions without writing any files.",
    )
    return parser.parse_args()


def _format_title(name: str) -> str:
    return " ".join(part.capitalize() for part in name.split("_"))


def _init_content(title: str) -> str:
    return f'"""{title} feature slice."""\n'


def _readme_content(title: str, name: str) -> str:
    return (
        f"# {title}\n\n"
        f"Vertical slice for `{name}`.\n\n"
        "## Scope\n\n"
        "- Spot: Describe spot trading coverage.\n"
        "- Perps: Describe perpetuals coverage.\n\n"
        "## Entry Points\n\n"
        "- Describe primary modules and CLI hooks.\n"
    )


def _test_content(name: str) -> str:
    return (
        "from __future__ import annotations\n\n"
        f"def test_{name}_slice_placeholder() -> None:\n"
        "    assert True\n"
    )


def _rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _build_plan(
    name: str,
    *,
    root: Path,
    with_tests: bool,
    with_readme: bool,
) -> ScaffoldPlan:
    title = _format_title(name)
    slice_dir = root / "src" / "gpt_trader" / "features" / name
    init_path = slice_dir / "__init__.py"

    directories = [ScaffoldAction("dir", slice_dir)]
    files = [ScaffoldAction("file", init_path, _init_content(title))]

    if with_readme:
        files.append(ScaffoldAction("file", slice_dir / "README.md", _readme_content(title, name)))

    if with_tests:
        tests_dir = root / "tests" / "unit" / "gpt_trader" / "features" / name
        directories.append(ScaffoldAction("dir", tests_dir))
        files.append(
            ScaffoldAction("file", tests_dir / f"test_{name}_slice.py", _test_content(name))
        )

    return ScaffoldPlan(slice_name=name, actions=directories + files)


def _find_conflicts(actions: list[ScaffoldAction]) -> list[Path]:
    return [action.path for action in actions if action.path.exists()]


def _print_actions(actions: list[ScaffoldAction], *, root: Path, dry_run: bool) -> None:
    for action in actions:
        verb = "mkdir" if action.kind == "dir" else "write"
        if dry_run:
            label = "DRY-RUN"
        else:
            label = "MKDIR" if action.kind == "dir" else "WRITE"
        print(f"[{label}] {verb} {_rel_path(action.path, root)}")


def _apply_actions(actions: list[ScaffoldAction]) -> None:
    for action in actions:
        if action.kind == "dir":
            action.path.mkdir(parents=True, exist_ok=False)

    for action in actions:
        if action.kind == "file" and action.content is not None:
            action.path.write_text(action.content, encoding="utf-8")


def scaffold_slice(
    name: str,
    *,
    root: Path = REPO_ROOT,
    with_tests: bool = False,
    with_readme: bool = False,
    dry_run: bool = False,
) -> int:
    if not SLICE_PATTERN.fullmatch(name):
        print(
            "Slice name must be snake_case using letters, numbers, and underscores.",
            file=sys.stderr,
        )
        return 1

    plan = _build_plan(name, root=root, with_tests=with_tests, with_readme=with_readme)
    conflicts = _find_conflicts(plan.actions)
    if conflicts:
        print("Refusing to overwrite existing paths:", file=sys.stderr)
        for path in conflicts:
            print(f"- {_rel_path(path, root)}", file=sys.stderr)
        return 1

    _print_actions(plan.actions, root=root, dry_run=dry_run)

    if dry_run:
        print("Dry-run complete; no files created.")
        return 0

    _apply_actions(plan.actions)
    print(f"Slice '{name}' scaffolded successfully.")
    return 0


def main() -> int:
    args = parse_args()
    return scaffold_slice(
        args.name,
        root=REPO_ROOT,
        with_tests=args.with_tests,
        with_readme=args.with_readme,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
