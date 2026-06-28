#!/usr/bin/env python3
"""Generate the decision index in docs/decisions/README.md from frontmatter.

Each decision record under ``docs/decisions/`` carries ``status`` and ``date``
frontmatter plus an H1 title. This script renders a table of those records
between the index markers in ``docs/decisions/README.md`` so the index can never
drift from the files it lists.

Usage:
    uv run python scripts/maintenance/generate_decision_index.py          # rewrite
    uv run python scripts/maintenance/generate_decision_index.py --check  # verify (CI)
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DECISIONS_DIR = REPO_ROOT / "docs" / "decisions"

BEGIN_MARKER = "<!-- BEGIN GENERATED DECISION INDEX -->"
END_MARKER = "<!-- END GENERATED DECISION INDEX -->"

# Files that are not themselves decision records.
EXCLUDED_NAMES = {"README.md"}


@dataclass
class DecisionRecord:
    slug: str  # filename, e.g. "intx-default-derivatives-venue.md"
    title: str
    status: str
    date: str


def _parse_frontmatter(lines: list[str]) -> dict[str, str]:
    """Return key/value pairs from the first ``---`` block within the first lines."""
    fields: dict[str, str] = {}
    limit = min(len(lines), 60)
    start = next((i for i in range(limit) if lines[i].strip() == "---"), None)
    if start is None:
        return fields
    for line in lines[start + 1 :]:
        if line.strip() == "---":
            break
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


def _parse_title(lines: list[str]) -> str | None:
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()
    return None


def parse_decision(path: Path) -> DecisionRecord | None:
    """Parse one decision file, or return None if it is not a usable record."""
    lines = path.read_text(encoding="utf-8").splitlines()
    fields = _parse_frontmatter(lines)
    title = _parse_title(lines)
    status = fields.get("status")
    if title is None or status is None:
        return None
    return DecisionRecord(
        slug=path.name,
        title=title,
        status=status,
        date=fields.get("date", ""),
    )


def collect_decisions(decisions_dir: Path) -> list[DecisionRecord]:
    records: list[DecisionRecord] = []
    for path in sorted(decisions_dir.glob("*.md")):
        if path.name in EXCLUDED_NAMES or path.name.startswith("_"):
            continue
        record = parse_decision(path)
        if record is not None:
            records.append(record)
    # Newest first; undated records sort last, then by title for stability.
    records.sort(key=lambda r: (r.date == "", r.date, r.title), reverse=False)
    records.sort(key=lambda r: r.date, reverse=True)
    return records


def render_index_table(records: list[DecisionRecord]) -> str:
    lines = ["| Date | Decision | Status |", "|------|----------|--------|"]
    if not records:
        lines.append("| — | _No decisions recorded yet._ | — |")
    for record in records:
        date = record.date or "—"
        lines.append(f"| {date} | [{record.title}]({record.slug}) | {record.status} |")
    return "\n".join(lines)


def splice_index(readme_text: str, table: str) -> str:
    pattern = re.compile(
        re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER),
        re.DOTALL,
    )
    replacement = f"{BEGIN_MARKER}\n{table}\n{END_MARKER}"
    if not pattern.search(readme_text):
        raise ValueError(
            f"Index markers not found in README. Expected:\n{BEGIN_MARKER}\n...\n{END_MARKER}"
        )
    return pattern.sub(replacement, readme_text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the docs/decisions index.")
    parser.add_argument("--decisions-dir", type=Path, default=DECISIONS_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the index is current without rewriting (exit 1 if stale).",
    )
    args = parser.parse_args(argv)

    decisions_dir = args.decisions_dir.resolve()
    readme = decisions_dir / "README.md"
    if not readme.exists():
        print(f"Missing decisions index: {readme}", file=sys.stderr)
        return 1

    records = collect_decisions(decisions_dir)
    table = render_index_table(records)
    current = readme.read_text(encoding="utf-8")
    updated = splice_index(current, table)

    if args.check:
        if updated != current:
            print(
                "docs/decisions/README.md index is stale. Run:\n"
                "  uv run python scripts/maintenance/generate_decision_index.py",
                file=sys.stderr,
            )
            return 1
        print(f"✓ Decision index current ({len(records)} records)")
        return 0

    if updated != current:
        readme.write_text(updated, encoding="utf-8")
        print(f"✓ Wrote decision index ({len(records)} records)")
    else:
        print(f"✓ Decision index already current ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
