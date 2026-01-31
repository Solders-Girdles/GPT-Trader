#!/usr/bin/env python3
"""
Cleanup stale Codex worktrees.

Targets /tmp/gpt-* worktrees whose branch matches codex/* or issue/* and whose
upstream remote ref is missing. Defaults to dry-run; pass --apply to remove.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BRANCH_PREFIXES = ("codex/", "issue/")
TEMPORARY_PREFIXES = ("/tmp/gpt-", "/private/tmp/gpt-")


@dataclass(frozen=True)
class WorktreeEntry:
    path: Path
    branch_ref: str | None
    locked: bool
    prunable_reason: str | None

    @property
    def branch_name(self) -> str | None:
        if self.branch_ref and self.branch_ref.startswith("refs/heads/"):
            return self.branch_ref[len("refs/heads/") :]
        return None


@dataclass(frozen=True)
class RemovalPlan:
    worktree_path: Path
    branch_name: str
    delete_branch: bool
    skip_reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove stale /tmp/gpt-* worktrees with missing upstreams."
    )
    parser.add_argument(
        "--apply", action="store_true", help="Remove worktrees and delete branches."
    )
    return parser.parse_args()


def run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def parse_worktree_list(output: str) -> list[WorktreeEntry]:
    entries: list[WorktreeEntry] = []
    current: dict[str, str] = {}

    def flush() -> None:
        if not current:
            return
        path_value = current.get("worktree")
        if not path_value:
            return
        entries.append(
            WorktreeEntry(
                path=Path(path_value),
                branch_ref=current.get("branch"),
                locked="locked" in current,
                prunable_reason=current.get("prunable") or None,
            )
        )
        current.clear()

    for line in output.splitlines():
        if not line.strip():
            flush()
            continue
        key, _, value = line.partition(" ")
        current[key] = value

    flush()
    return entries


def is_temporary_worktree(path: Path) -> bool:
    def matches(candidate: Path) -> bool:
        as_text = candidate.as_posix()
        return any(as_text.startswith(prefix) for prefix in TEMPORARY_PREFIXES)

    if matches(path):
        return True
    try:
        return matches(path.resolve(strict=False))
    except RuntimeError:
        return False


def branch_matches(name: str) -> bool:
    return name.startswith(BRANCH_PREFIXES)


def upstream_ref(branch_name: str) -> str | None:
    result = run_git(
        ["for-each-ref", f"refs/heads/{branch_name}", "--format=%(upstream)"],
        check=False,
    )
    if result.returncode != 0:
        return None
    upstream = result.stdout.strip()
    return upstream or None


def upstream_missing(upstream: str) -> bool:
    if not upstream.startswith("refs/remotes/"):
        return False
    result = run_git(["show-ref", "--verify", "--quiet", upstream], check=False)
    return result.returncode != 0


def collect_removals(entries: list[WorktreeEntry]) -> list[RemovalPlan]:
    repo_root = REPO_ROOT.resolve()
    branch_usage: dict[str, int] = {}

    for entry in entries:
        branch_name = entry.branch_name
        if branch_name:
            branch_usage[branch_name] = branch_usage.get(branch_name, 0) + 1

    removals: list[RemovalPlan] = []
    for entry in entries:
        if entry.path.resolve(strict=False) == repo_root:
            continue
        if entry.locked:
            continue
        if not is_temporary_worktree(entry.path):
            continue
        branch_name = entry.branch_name
        if not branch_name or not branch_matches(branch_name):
            continue
        upstream = upstream_ref(branch_name)
        if not upstream or not upstream_missing(upstream):
            continue
        delete_branch = branch_usage.get(branch_name, 0) <= 1
        skip_reason = None if delete_branch else "branch_in_use_by_other_worktree"
        removals.append(
            RemovalPlan(
                worktree_path=entry.path,
                branch_name=branch_name,
                delete_branch=delete_branch,
                skip_reason=skip_reason,
            )
        )
    return removals


def display_plan(removals: list[RemovalPlan], *, apply: bool) -> None:
    if not removals:
        print("No stale /tmp/gpt-* worktrees found.")
        return

    header = "Planned removals (dry-run):" if not apply else "Planned removals:"
    print(header)
    print("Worktrees to remove:")
    for removal in removals:
        print(f"- {removal.worktree_path}")

    branches = [removal.branch_name for removal in removals if removal.delete_branch]
    print("Branches to delete:")
    if branches:
        for branch_name in branches:
            print(f"- {branch_name}")
    else:
        print("- none")

    skipped = [removal for removal in removals if not removal.delete_branch]
    if skipped:
        print("Skipped branch deletions:")
        for removal in skipped:
            print(f"- {removal.branch_name}: {removal.skip_reason}")


def apply_removals(removals: list[RemovalPlan]) -> int:
    failures: list[str] = []

    for removal in removals:
        force_remove = not removal.worktree_path.exists()
        worktree_args = ["worktree", "remove"]
        if force_remove:
            worktree_args.append("--force")
        worktree_args.append(str(removal.worktree_path))
        result = run_git(worktree_args, check=False)
        if result.returncode != 0:
            failures.append(
                f"worktree {removal.worktree_path}: {result.stderr.strip() or 'unknown error'}"
            )
            continue
        if removal.delete_branch:
            branch_result = run_git(["branch", "-D", removal.branch_name], check=False)
            if branch_result.returncode != 0:
                failures.append(
                    f"branch {removal.branch_name}: {branch_result.stderr.strip() or 'unknown error'}"
                )

    if failures:
        print("Some removals failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


def main() -> int:
    args = parse_args()
    result = run_git(["worktree", "list", "--porcelain"])
    entries = parse_worktree_list(result.stdout)
    removals = collect_removals(entries)
    display_plan(removals, apply=args.apply)
    if not args.apply or not removals:
        return 0
    return apply_removals(removals)


if __name__ == "__main__":
    raise SystemExit(main())
