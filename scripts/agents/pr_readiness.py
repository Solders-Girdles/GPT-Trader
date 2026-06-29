#!/usr/bin/env python3
"""Reconcile a pull request's *real* mergeability against "the checks are green".

This is a transparency tool, not a gate. CI being green is necessary but not
sufficient to merge: branch protection can also require the branch to be up to
date (``strict``), require every review conversation resolved
(``required_conversation_resolution``), or require approving reviews. A green PR
can still be ``BLOCKED`` by unresolved bot/human review threads that carry real
findings -- exactly the gap that lets "green-but-unreviewed" work slip through.

``agent-pr-ready`` surfaces that gap:

- Required status checks (from branch protection) and their current state.
- ``mergeStateStatus`` (CLEAN / BLOCKED / BEHIND / DIRTY / ...).
- Unresolved review threads, with parsed severity, author, and location.
- An artifact-freshness advisory when the diff touches files that feed the
  generated ``var/agents/`` context.

It prints a verdict and a markdown receipt suitable for a PR body. By default it
always exits 0 (report, do not block); pass ``--exit-on-not-ready`` to opt into
an advisory non-zero exit for scripts that *want* a gate.

For change/test selection, use ``agent-impact`` -- this tool intentionally does
not duplicate that analysis.

Usage:
    uv run agent-pr-ready                      # auto-detect PR for current branch
    uv run agent-pr-ready --pr 1056
    uv run agent-pr-ready --format markdown    # receipt for the PR body
    uv run agent-pr-ready --format json
    uv run agent-pr-ready --no-github          # local artifact advisory only
    uv run agent-pr-ready --exit-on-not-ready  # opt-in advisory gate
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Changes under these roots can change generated var/agents/ context, so the
# freshness check (agent-regenerate --verify) is worth running before merge.
_ARTIFACT_SOURCE_PREFIXES = ("src/", "tests/", "scripts/", "config/")
_ARTIFACT_SOURCE_FILES = (
    "pyproject.toml",
    "pytest.ini",
    "config/environments/.env.template",
)
_ARTIFACT_SOURCE_SUFFIXES = (".py", ".tcss", ".yaml", ".yml")

# Severity tokens emitted by review bots (CodeRabbit, Codex) in comment bodies,
# ordered most-severe first so the highest match wins.
_SEVERITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("critical", re.compile(r"\bP0\b|\bcritical\b", re.IGNORECASE)),
    ("high", re.compile(r"\bP1\b|\bmajor\b", re.IGNORECASE)),
    ("medium", re.compile(r"\bP2\b|\bmoderate\b", re.IGNORECASE)),
    ("low", re.compile(r"\bP3\b|\bminor\b|\bnit\b", re.IGNORECASE)),
)

# mergeStateStatus values that mean "not mergeable right now", with guidance.
_BLOCKING_MERGE_STATES = {
    "BEHIND": "Branch is behind the base; update/rebase so it is current.",
    "DIRTY": "Branch has merge conflicts with the base.",
    "BLOCKED": "Branch protection is blocking the merge (see findings below).",
    "DRAFT": "Pull request is a draft; mark it ready for review before merging.",
    "UNKNOWN": "GitHub has not determined mergeability yet; retry after merge state settles.",
}

_PASSING_CHECK_STATES = {"SUCCESS", "SKIPPED", "NEUTRAL"}


@dataclass(frozen=True)
class CheckStatus:
    name: str
    state: str  # SUCCESS / PENDING / FAILURE / ERROR / ...
    required: bool


@dataclass(frozen=True)
class ReviewThread:
    author: str
    resolved: bool
    severity: str | None
    path: str | None
    line: int | None


@dataclass(frozen=True)
class BranchProtection:
    strict: bool
    conversation_resolution: bool
    required_review_count: int
    required_checks: tuple[str, ...]


@dataclass(frozen=True)
class PullRequestState:
    number: int
    head_oid: str
    merge_state_status: str
    review_decision: str
    checks: tuple[CheckStatus, ...]
    threads: tuple[ReviewThread, ...]
    protection: BranchProtection
    base_ref_name: str = "main"


@dataclass(frozen=True)
class Finding:
    severity: str  # blocker / warning / info
    message: str


@dataclass
class ReadinessReport:
    ready: bool
    findings: list[Finding] = field(default_factory=list)
    artifact_advisory: str | None = None


# --------------------------------------------------------------------------- #
# Pure logic (unit-tested): no git, no gh, no I/O.
# --------------------------------------------------------------------------- #
def affects_agent_artifacts(paths: list[str]) -> bool:
    """True when any changed path can make generated var/agents/ context stale."""
    for raw in paths:
        path = raw.strip()
        if not path:
            continue
        if path in _ARTIFACT_SOURCE_FILES:
            return True
        if path.startswith(_ARTIFACT_SOURCE_PREFIXES) and path.endswith(_ARTIFACT_SOURCE_SUFFIXES):
            return True
    return False


def parse_severity(body: str) -> str | None:
    """Extract a normalized severity from a review comment body, if present."""
    for label, pattern in _SEVERITY_PATTERNS:
        if pattern.search(body):
            return label
    return None


def parse_branch_protection(raw: dict[str, Any] | None) -> BranchProtection:
    """Build BranchProtection from the GitHub protection API payload."""
    raw = raw or {}
    checks_block = raw.get("required_status_checks") or {}
    modern_checks = [
        item.get("context", "") for item in checks_block.get("checks") or [] if item.get("context")
    ]
    legacy_contexts = [
        context for context in checks_block.get("contexts") or [] if isinstance(context, str)
    ]
    required = tuple(sorted({*modern_checks, *legacy_contexts}))
    reviews = raw.get("required_pull_request_reviews") or {}
    return BranchProtection(
        strict=bool(checks_block.get("strict", False)),
        conversation_resolution=bool(
            (raw.get("required_conversation_resolution") or {}).get("enabled", False)
        ),
        required_review_count=int(reviews.get("required_approving_review_count", 0) or 0),
        required_checks=required,
    )


def parse_pr_state(
    pr_json: dict[str, Any],
    threads_json: list[dict[str, Any]],
    protection: BranchProtection,
) -> PullRequestState:
    """Build PullRequestState from gh payloads + parsed branch protection."""
    required_names = set(protection.required_checks)
    checks: list[CheckStatus] = []
    for node in pr_json.get("statusCheckRollup") or []:
        name = node.get("name") or node.get("context") or ""
        if not name:
            continue
        # CheckRun uses status/conclusion; StatusContext uses state.
        state = node.get("state")
        if not state:
            if node.get("status") == "COMPLETED":
                state = node.get("conclusion") or "UNKNOWN"
            else:
                state = node.get("status") or "UNKNOWN"
        checks.append(
            CheckStatus(name=name, state=str(state).upper(), required=name in required_names)
        )

    threads: list[ReviewThread] = []
    for node in threads_json:
        comments = (node.get("comments") or {}).get("nodes") or []
        first = comments[0] if comments else {}
        author = ((first.get("author") or {}).get("login")) or "unknown"
        body = " ".join(c.get("body", "") for c in comments)
        threads.append(
            ReviewThread(
                author=author,
                resolved=bool(node.get("isResolved", False)),
                severity=parse_severity(body),
                path=node.get("path"),
                line=node.get("line"),
            )
        )

    return PullRequestState(
        number=int(pr_json.get("number", 0)),
        head_oid=str(pr_json.get("headRefOid", "")),
        base_ref_name=str(pr_json.get("baseRefName", "main") or "main"),
        merge_state_status=str(pr_json.get("mergeStateStatus", "UNKNOWN")).upper(),
        review_decision=str(pr_json.get("reviewDecision", "") or ""),
        checks=tuple(checks),
        threads=tuple(threads),
        protection=protection,
    )


def assess_readiness(state: PullRequestState) -> ReadinessReport:
    """Reconcile checks + protection + threads into a non-blocking verdict."""
    findings: list[Finding] = []

    observed_check_names = {check.name for check in state.checks}
    for name in sorted(set(state.protection.required_checks) - observed_check_names):
        findings.append(Finding("blocker", f"Required check '{name}' is missing."))

    failing_required = [
        check for check in state.checks if check.required and not _is_passing_check(check)
    ]
    for check in failing_required:
        findings.append(Finding("blocker", f"Required check '{check.name}' is {check.state}."))

    failing_other = [
        check
        for check in state.checks
        if not check.required and check.state in {"FAILURE", "ERROR", "CANCELLED", "TIMED_OUT"}
    ]
    for check in failing_other:
        findings.append(Finding("warning", f"Non-required check '{check.name}' is {check.state}."))

    unresolved = [thread for thread in state.threads if not thread.resolved]
    if unresolved:
        severity = "blocker" if state.protection.conversation_resolution else "warning"
        findings.append(
            Finding(
                severity,
                _describe_unresolved(unresolved, state.protection.conversation_resolution),
            )
        )

    guidance = _BLOCKING_MERGE_STATES.get(state.merge_state_status)
    if guidance:
        if state.merge_state_status == "BLOCKED":
            has_specific_blocker = any(finding.severity == "blocker" for finding in findings)
            if not has_specific_blocker:
                findings.append(Finding("blocker", guidance))
        else:
            findings.append(Finding("blocker", guidance))

    if state.protection.required_review_count > 0 and state.review_decision != "APPROVED":
        findings.append(
            Finding(
                "blocker",
                f"Branch protection requires {state.protection.required_review_count} "
                f"approving review(s); current decision is "
                f"'{state.review_decision or 'none'}'.",
            )
        )

    ready = not any(finding.severity == "blocker" for finding in findings)
    if ready and not findings:
        findings.append(Finding("info", "No blockers detected; PR is mergeable."))
    return ReadinessReport(ready=ready, findings=findings)


def _is_passing_check(check: CheckStatus) -> bool:
    return check.state in _PASSING_CHECK_STATES


def _required_check_names(state: PullRequestState) -> set[str]:
    names = set(state.protection.required_checks)
    if names:
        return names
    return {check.name for check in state.checks if check.required}


def _describe_unresolved(unresolved: list[ReviewThread], enforced: bool) -> str:
    by_severity: dict[str, int] = {}
    for thread in unresolved:
        key = thread.severity or "unspecified"
        by_severity[key] = by_severity.get(key, 0) + 1
    breakdown = ", ".join(f"{count} {label}" for label, count in sorted(by_severity.items()))
    authors = ", ".join(sorted({thread.author for thread in unresolved}))
    enforced_note = (
        " Branch protection requires conversation resolution, so these block the merge."
        if enforced
        else " Conversation resolution is not enforced, but review them before merging."
    )
    return (
        f"{len(unresolved)} unresolved review thread(s) [{breakdown}] from {authors}."
        + enforced_note
    )


# --------------------------------------------------------------------------- #
# Receipt / output formatting (pure).
# --------------------------------------------------------------------------- #
def _verdict(report: ReadinessReport) -> str:
    return "READY" if report.ready else "NOT READY"


_SEVERITY_GLYPH = {"blocker": "x", "warning": "!", "info": "+"}


def format_text(state: PullRequestState | None, report: ReadinessReport) -> str:
    lines: list[str] = []
    if state is not None:
        lines.append(f"PR #{state.number} @ {state.head_oid[:8]}")
        lines.append(f"base: {state.base_ref_name}")
        lines.append(f"merge state: {state.merge_state_status}")
        required_names = _required_check_names(state)
        passing = sum(1 for c in state.checks if c.name in required_names and _is_passing_check(c))
        lines.append(f"required checks: {passing}/{len(required_names)} passing")
        unresolved = sum(1 for t in state.threads if not t.resolved)
        lines.append(f"review threads: {unresolved} unresolved / {len(state.threads)} total")
    lines.append(f"verdict: {_verdict(report)}")
    if report.artifact_advisory:
        lines.append(f"artifacts: {report.artifact_advisory}")
    lines.append("findings:")
    for finding in report.findings:
        glyph = _SEVERITY_GLYPH.get(finding.severity, "-")
        lines.append(f"  {glyph} [{finding.severity}] {finding.message}")
    return "\n".join(lines)


def format_markdown(state: PullRequestState | None, report: ReadinessReport) -> str:
    lines = ["## PR readiness receipt", ""]
    lines.append(f"**Verdict:** {_verdict(report)}")
    lines.append("")
    if state is not None:
        required_names = _required_check_names(state)
        passing = sum(1 for c in state.checks if c.name in required_names and _is_passing_check(c))
        unresolved = sum(1 for t in state.threads if not t.resolved)
        lines.append(f"- PR `#{state.number}` at `{state.head_oid[:8]}`")
        lines.append(f"- Base: `{state.base_ref_name}`")
        lines.append(f"- Merge state: `{state.merge_state_status}`")
        lines.append(f"- Required checks: {passing}/{len(required_names)} passing")
        lines.append(f"- Review threads: {unresolved} unresolved / {len(state.threads)} total")
        if state.protection.conversation_resolution:
            lines.append("- Conversation resolution: **required** by branch protection")
    if report.artifact_advisory:
        lines.append(f"- Artifacts: {report.artifact_advisory}")
    lines.append("")
    lines.append("### Findings")
    for finding in report.findings:
        glyph = _SEVERITY_GLYPH.get(finding.severity, "-")
        lines.append(f"- {glyph} **{finding.severity}** - {finding.message}")
    return "\n".join(lines)


def format_json(state: PullRequestState | None, report: ReadinessReport) -> str:
    payload: dict[str, Any] = {
        "ready": report.ready,
        "verdict": _verdict(report),
        "artifact_advisory": report.artifact_advisory,
        "findings": [asdict(finding) for finding in report.findings],
    }
    if state is not None:
        payload["pull_request"] = {
            "number": state.number,
            "head_oid": state.head_oid,
            "base_ref_name": state.base_ref_name,
            "merge_state_status": state.merge_state_status,
            "review_decision": state.review_decision,
            "checks": [asdict(check) for check in state.checks],
            "threads": [asdict(thread) for thread in state.threads],
            "protection": asdict(state.protection),
        }
    return json.dumps(payload, indent=2)


# --------------------------------------------------------------------------- #
# I/O boundary: git + gh. Thin wrappers, not unit-tested.
# --------------------------------------------------------------------------- #
def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, capture_output=True, text=True, cwd=PROJECT_ROOT, check=False)
    except OSError as error:
        return subprocess.CompletedProcess(args, 127, "", str(error))


def changed_paths(base: str) -> list[str]:
    """Changed files for the current branch vs base (committed + working tree)."""
    seen: list[str] = []
    base_specs = [f"{base}...HEAD"]
    if not base.startswith("origin/"):
        base_specs.append(f"origin/{base}...HEAD")
    commands = [["git", "diff", "--name-only", spec] for spec in base_specs] + [
        ["git", "diff", "--name-only"],
        ["git", "diff", "--name-only", "--cached"],
    ]
    for args in commands:
        result = _run(args)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                path = line.strip()
                if path and path not in seen:
                    seen.append(path)
    return seen


def _gh_json(args: list[str]) -> Any:
    result = _run(["gh", *args])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "gh command failed")
    return json.loads(result.stdout)


def detect_repo() -> str:
    return str(_gh_json(["repo", "view", "--json", "nameWithOwner"])["nameWithOwner"])


def detect_pr_number(repo: str) -> int | None:
    try:
        data = _gh_json(["pr", "view", "--repo", repo, "--json", "number"])
    except RuntimeError as error:
        if _is_no_pull_request_error(error):
            return None
        raise
    return int(data["number"]) if data.get("number") else None


def _is_no_pull_request_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "no pull requests found" in message or "no open pull requests" in message


def fetch_branch_protection(repo: str, branch: str = "main") -> dict[str, Any] | None:
    try:
        encoded_branch = quote(branch, safe="")
        payload = _gh_json(["api", f"repos/{repo}/branches/{encoded_branch}/protection"])
    except RuntimeError as error:
        if _is_missing_branch_protection(error):
            return None
        raise
    return payload if isinstance(payload, dict) else None


def _is_missing_branch_protection(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "http 404" in message or "branch not protected" in message


def fetch_pr_payload(repo: str, pr: int) -> dict[str, Any]:
    payload = _gh_json(
        [
            "pr",
            "view",
            str(pr),
            "--repo",
            repo,
            "--json",
            "number,headRefOid,baseRefName,mergeStateStatus,reviewDecision,statusCheckRollup",
        ]
    )
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected gh pr view payload")
    return payload


def fetch_pr_changed_paths(repo: str, pr: int) -> list[str]:
    result = _run(["gh", "pr", "diff", str(pr), "--repo", repo, "--name-only"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Could not load PR #{pr} diff")
    seen: list[str] = []
    for line in result.stdout.splitlines():
        path = line.strip()
        if path and path not in seen:
            seen.append(path)
    return seen


def fetch_review_threads(repo: str, pr: int) -> list[dict[str, Any]]:
    if repo.count("/") != 1:
        raise RuntimeError("--repo must use owner/name format")
    owner, name = repo.split("/", 1)
    if not owner or not name:
        raise RuntimeError("--repo must use owner/name format")
    query = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        nodes {
          isResolved
          path
          line
          comments(first: 5) {
            nodes {
              author { login }
              body
            }
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}
"""
    nodes: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        gh_args = [
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-F",
            f"owner={owner}",
            "-F",
            f"name={name}",
            "-F",
            f"number={pr}",
        ]
        if cursor:
            gh_args.extend(["-F", f"cursor={cursor}"])
        data = _gh_json(gh_args)
        page = (
            data.get("data", {})
            .get("repository", {})
            .get("pullRequest", {})
            .get("reviewThreads", {})
        )
        page_nodes = page.get("nodes", [])
        if isinstance(page_nodes, list):
            nodes.extend(page_nodes)
        page_info = page.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        next_cursor = page_info.get("endCursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            break
        cursor = next_cursor
    return nodes


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _artifact_advisory(paths: list[str]) -> str | None:
    if affects_agent_artifacts(paths):
        return (
            "diff touches generated-context sources; run "
            "`uv run agent-regenerate --verify` (and commit if stale)."
        )
    return None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile a PR's real mergeability against green CI (transparency, not a gate)."
    )
    parser.add_argument(
        "--pr", type=int, default=None, help="PR number (auto-detected if omitted)."
    )
    parser.add_argument("--repo", default=None, help="owner/name (auto-detected if omitted).")
    parser.add_argument(
        "--base",
        default=None,
        help="Base branch override. Defaults to the PR base when GitHub is available, otherwise main.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
        help="Output format. 'markdown' is suitable for a PR body.",
    )
    parser.add_argument(
        "--no-github",
        action="store_true",
        help="Skip gh calls; report only the local artifact-freshness advisory.",
    )
    parser.add_argument(
        "--exit-on-not-ready",
        action="store_true",
        help="Opt-in advisory gate: exit 1 when NOT READY (default exits 0).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.no_github:
        advisory = _artifact_advisory(changed_paths(args.base or "main"))
        report = ReadinessReport(
            ready=True,
            findings=[Finding("info", "GitHub checks skipped (--no-github).")],
            artifact_advisory=advisory,
        )
        print(_render(args.format, None, report))
        return 0

    try:
        repo = args.repo or detect_repo()
        pr = args.pr or detect_pr_number(repo)
    except RuntimeError as error:
        return _github_failure(args, f"Could not reach GitHub via gh: {error}")

    if pr is None:
        advisory = _artifact_advisory(changed_paths(args.base or "main"))
        report = ReadinessReport(
            ready=True,
            findings=[Finding("info", "No open PR for the current branch.")],
            artifact_advisory=advisory,
        )
        print(_render(args.format, None, report))
        return 0

    try:
        pr_payload = fetch_pr_payload(repo, pr)
        base_ref_name = str(args.base or pr_payload.get("baseRefName") or "main")
        advisory = _artifact_advisory(fetch_pr_changed_paths(repo, pr))
        protection = parse_branch_protection(fetch_branch_protection(repo, base_ref_name))
        state = parse_pr_state(pr_payload, fetch_review_threads(repo, pr), protection)
    except RuntimeError as error:
        return _github_failure(args, f"Could not load PR #{pr}: {error}")

    report = assess_readiness(state)
    report.artifact_advisory = advisory
    print(_render(args.format, state, report))
    return 1 if (args.exit_on_not_ready and not report.ready) else 0


def _render(fmt: str, state: PullRequestState | None, report: ReadinessReport) -> str:
    if fmt == "json":
        return format_json(state, report)
    if fmt == "markdown":
        return format_markdown(state, report)
    return format_text(state, report)


def _github_failure(args: argparse.Namespace, message: str) -> int:
    advisory = _artifact_advisory(changed_paths(args.base or "main"))
    report = ReadinessReport(
        ready=False,
        findings=[Finding("blocker", message)],
        artifact_advisory=advisory,
    )
    print(_render(args.format, None, report))
    return 1 if args.exit_on_not_ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
