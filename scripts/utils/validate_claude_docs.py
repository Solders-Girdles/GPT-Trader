#!/usr/bin/env python3
"""
Validation script for Claude docs and agent configuration.

Checks:
- CLAUDE.md path references are valid and no outdated patterns remain
- Agent mapping is single-source in .claude/agents/agent_mapping.yaml
- All .claude/agents/*.md files have basic front-matter and no stray diff markers
"""

from __future__ import annotations

import sys
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def err(msg: str):
    print(f"[ERROR] {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def ok(msg: str):
    print(f"[OK] {msg}")


def check_claude_md() -> int:
    errors = 0
    claude = ROOT / "CLAUDE.md"
    if not claude.exists():
        err("CLAUDE.md not found")
        return 1
    text = claude.read_text(encoding="utf-8", errors="ignore")

    # Required references should exist on disk
    required_paths = [
        ROOT / ".knowledge/STATE.json",
        ROOT / "src/bot_v2/SLICES.md",
        ROOT / "context/active_epics.yaml",
    ]
    for p in required_paths:
        if not p.exists():
            warn(f"Referenced path missing: {p}")

    # Disallow outdated references
    bad_patterns = [
        r"docs/knowledge/",
        r"WEEK_\d+",
        r"test_backtest\.py",
        r"from\s+data_providers\s+import",
    ]
    for pat in bad_patterns:
        if re.search(pat, text):
            err(f"Found outdated reference in CLAUDE.md: pattern '{pat}'")
            errors += 1

    ok("CLAUDE.md checked")
    return errors


def check_mapping_single_source() -> int:
    errors = 0
    mapping_primary = ROOT / ".claude/agents/agent_mapping.yaml"
    mapping_legacy = ROOT / "agents/agent_mapping.yaml"

    if not mapping_primary.exists():
        err("Missing .claude/agents/agent_mapping.yaml")
        errors += 1
    if mapping_legacy.exists():
        err("Found legacy agents/agent_mapping.yaml (should be removed or symlink)")
        errors += 1
    else:
        ok("Single-source mapping confirmed")
    return errors


def check_agents_front_matter() -> int:
    errors = 0
    agents_dir = ROOT / ".claude/agents"
    md_files = sorted(p for p in agents_dir.glob("*.md"))
    if not md_files:
        warn("No agent .md files found in .claude/agents")
        return 0

    for f in md_files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        # Basic checks: YAML fence and name/tools keys
        if "---" not in text.splitlines()[0:5]:
            warn(f"{f}: missing front-matter fence '---'")
        if re.search(r"^\+", text, flags=re.MULTILINE):
            err(f"{f}: contains stray '+' diff markers")
            errors += 1
        if not re.search(r"\bname:\s*\S", text):
            warn(f"{f}: missing 'name:' in front-matter")
        if not re.search(r"\btools:\s*\[", text):
            warn(f"{f}: missing 'tools: [...]' in front-matter")
        # Output section should mention JSON in return format
        if not re.search(r"(?i)\breturn(\s*format)?\b.*json", text):
            warn(f"{f}: outputs should mention JSON in the return format")

    ok("Agent files checked")
    return errors


def main() -> int:
    total_errors = 0
    total_errors += check_claude_md()
    total_errors += check_mapping_single_source()
    total_errors += check_agents_front_matter()
    if total_errors:
        print(f"\nValidation finished with {total_errors} error(s)")
        return 1
    print("\nValidation passed with no errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
