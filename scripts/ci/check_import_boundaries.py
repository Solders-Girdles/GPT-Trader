#!/usr/bin/env python3
"""Import boundary guard for architecture slices.

Current rules:
- Feature slices (src/gpt_trader/features) must not import the TUI layer (gpt_trader.tui).

Usage:
    python scripts/ci/check_import_boundaries.py [paths...]

To extend the rule set:
- Add an ImportRule entry to RULES with a descriptive name and source_root.
- Add forbidden_prefixes for the module prefixes to block.
- Optional: use allowlist_files or allowlist_import_prefixes for temporary exceptions.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ImportRule:
    name: str
    description: str
    source_root: Path
    forbidden_prefixes: tuple[str, ...]
    allowlist_files: frozenset[Path] = field(default_factory=frozenset)
    allowlist_import_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImportViolation:
    path: Path
    line: int
    import_path: str
    rule: ImportRule


RULES: tuple[ImportRule, ...] = (
    ImportRule(
        name="features_no_tui_imports",
        description="Feature slices must not import the TUI layer.",
        source_root=REPO_ROOT / "src" / "gpt_trader" / "features",
        forbidden_prefixes=("gpt_trader.tui",),
    ),
)


def _collect_explicit_files(paths: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for entry in paths:
        path = Path(entry)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.is_dir():
            files.extend(path.rglob("*.py"))
        elif path.suffix == ".py":
            files.append(path)
    return files


def _iter_imports(tree: ast.AST) -> Iterable[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                continue
            module = node.module
            for alias in node.names:
                if alias.name == "*":
                    yield node.lineno, module
                else:
                    yield node.lineno, f"{module}.{alias.name}"


def _matches_prefix(import_path: str, prefix: str) -> bool:
    return import_path == prefix or import_path.startswith(f"{prefix}.")


def _is_allowed(import_path: str, rule: ImportRule) -> bool:
    for allowed in rule.allowlist_import_prefixes:
        if _matches_prefix(import_path, allowed):
            return True
    return False


def _scan_rule(rule: ImportRule, explicit_files: list[Path] | None) -> list[ImportViolation]:
    if not rule.source_root.exists():
        return []

    if explicit_files is None:
        candidates = list(rule.source_root.rglob("*.py"))
    else:
        candidates = []
        for path in explicit_files:
            try:
                path.relative_to(rule.source_root)
            except ValueError:
                continue
            candidates.append(path)

    violations: list[ImportViolation] = []
    seen: set[tuple[Path, int, str, str]] = set()

    for path in candidates:
        if path in rule.allowlist_files:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            violations.append(
                ImportViolation(
                    path=path,
                    line=1,
                    import_path=f"<unreadable: {exc}>",
                    rule=rule,
                )
            )
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            violations.append(
                ImportViolation(
                    path=path,
                    line=exc.lineno or 1,
                    import_path="<parse error>",
                    rule=rule,
                )
            )
            continue

        for line, import_path in _iter_imports(tree):
            if _is_allowed(import_path, rule):
                continue
            if any(_matches_prefix(import_path, prefix) for prefix in rule.forbidden_prefixes):
                key = (path, line, import_path, rule.name)
                if key in seen:
                    continue
                seen.add(key)
                violations.append(
                    ImportViolation(
                        path=path,
                        line=line,
                        import_path=import_path,
                        rule=rule,
                    )
                )

    return violations


def scan(paths: Sequence[str] | None = None) -> int:
    explicit_files = _collect_explicit_files(paths or []) if paths else None
    all_violations: list[ImportViolation] = []

    for rule in RULES:
        all_violations.extend(_scan_rule(rule, explicit_files))

    if all_violations:
        print("ERRORS:")
        for violation in sorted(all_violations, key=lambda v: (str(v.path), v.line)):
            rel = violation.path.resolve().relative_to(REPO_ROOT)
            print(
                f"  ERROR {rel}:{violation.line}: imports {violation.import_path} "
                f"(rule: {violation.rule.name})"
            )
            print(f"     {violation.rule.description}")
        print(f"\n{len(all_violations)} violation(s) found.")
        return 1

    print("Import boundary guard passed.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional files/directories to scan (defaults to rule source roots).",
    )
    args = parser.parse_args(argv)
    return scan(args.paths)


if __name__ == "__main__":
    raise SystemExit(main())
