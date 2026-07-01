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
- Current-head review/reaction signals, including the repo's Codex connector
  ``+1`` convention.
- Generated ``var/agents/`` context freshness via ``agent-regenerate --verify``
  when the diff touches files that feed those artifacts.

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
    uv run agent-pr-ready --no-github          # local artifact freshness only
    uv run agent-pr-ready --skip-artifact-verify
    uv run agent-pr-ready --require-current-head-review-signal
    uv run agent-pr-ready --exit-on-not-ready  # opt-in advisory gate
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
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
_DEFAULT_REVIEW_BOT = "chatgpt-codex-connector[bot]"
_DEFAULT_ACCEPTANCE_MARKER = "+1"
_THUMBS_UP_MARKERS = {"+1", "thumbs_up", "thumbsup", "thumbs up"}
_THUMBS_UP_REACTIONS = {"+1", "THUMBS_UP"}


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
class ReviewSignal:
    kind: str  # review / pr_reaction
    author: str
    state: str
    current_head: bool
    created_at: str | None = None
    url: str | None = None


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
    head_committed_at: str | None = None
    review_signals: tuple[ReviewSignal, ...] = ()


@dataclass(frozen=True)
class Finding:
    severity: str  # blocker / warning / info
    message: str


@dataclass(frozen=True)
class ArtifactFreshness:
    required: bool
    checked: bool
    fresh: bool | None
    command: str | None
    summary: str


@dataclass
class ReadinessReport:
    ready: bool
    findings: list[Finding] = field(default_factory=list)
    artifact_advisory: str | None = None
    artifact_freshness: ArtifactFreshness | None = None


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
    reactions_json: list[dict[str, Any]] | None = None,
    *,
    review_bot: str | None = _DEFAULT_REVIEW_BOT,
    acceptance_marker: str | None = _DEFAULT_ACCEPTANCE_MARKER,
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

    head_oid = str(pr_json.get("headRefOid", ""))
    head_committed_at = _current_head_committed_at(pr_json, head_oid)

    return PullRequestState(
        number=int(pr_json.get("number", 0)),
        head_oid=head_oid,
        base_ref_name=str(pr_json.get("baseRefName", "main") or "main"),
        merge_state_status=str(pr_json.get("mergeStateStatus", "UNKNOWN")).upper(),
        review_decision=str(pr_json.get("reviewDecision", "") or ""),
        checks=tuple(checks),
        threads=tuple(threads),
        protection=protection,
        head_committed_at=head_committed_at,
        review_signals=tuple(
            _parse_review_signals(
                pr_json,
                reactions_json or [],
                head_oid,
                head_committed_at,
                review_bot=review_bot,
                acceptance_marker=acceptance_marker,
            )
        ),
    )


def assess_readiness(
    state: PullRequestState,
    *,
    require_current_head_review_signal: bool = False,
) -> ReadinessReport:
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

    if not _has_current_head_review_signal(state):
        severity = "blocker" if require_current_head_review_signal else "warning"
        findings.append(
            Finding(
                severity,
                "No current-head review or acceptance reaction was found. This is visibility "
                "evidence by default, not a universal PR gate; pass "
                "`--require-current-head-review-signal` for pipeline routes that require it.",
            )
        )

    ready = not any(finding.severity == "blocker" for finding in findings)
    if ready and not findings:
        findings.append(
            Finding(
                "info",
                "No blocking readiness issues detected; merge still requires explicit route/approval.",
            )
        )
    return ReadinessReport(ready=ready, findings=findings)


def _parse_review_signals(
    pr_json: dict[str, Any],
    reactions_json: list[dict[str, Any]],
    head_oid: str,
    head_committed_at: str | None,
    *,
    review_bot: str | None,
    acceptance_marker: str | None,
) -> list[ReviewSignal]:
    signals: list[ReviewSignal] = []
    for review in pr_json.get("reviews") or []:
        if not isinstance(review, dict):
            continue
        author = _login(review.get("author"))
        if review_bot and not _author_matches_review_bot(author, review_bot):
            continue
        state = str(review.get("state") or "UNKNOWN").upper()
        commit_oid = _review_commit_oid(review)
        body = str(review.get("body") or "")
        current_head = _head_matches(commit_oid, head_oid) or _body_mentions_head(body, head_oid)
        signals.append(
            ReviewSignal(
                kind="review",
                author=author or "unknown",
                state=state,
                current_head=current_head,
                created_at=review.get("submittedAt") or review.get("createdAt"),
                url=review.get("url"),
            )
        )

    if acceptance_marker:
        for reaction in reactions_json:
            if not isinstance(reaction, dict):
                continue
            if not _reaction_matches(reaction, acceptance_marker, review_bot):
                continue
            created_at = reaction.get("created_at") or reaction.get("createdAt")
            current_head = _timestamp_at_or_after(created_at, head_committed_at)
            signals.append(
                ReviewSignal(
                    kind="pr_reaction",
                    author=_reaction_actor(reaction) or "unknown",
                    state=str(reaction.get("content") or ""),
                    current_head=current_head,
                    created_at=created_at,
                    url=None,
                )
            )
    return signals


def _current_head_committed_at(pr_json: dict[str, Any], head_oid: str) -> str | None:
    for commit in pr_json.get("commits") or []:
        if not isinstance(commit, dict):
            continue
        if _head_matches(str(commit.get("oid") or ""), head_oid):
            value = commit.get("committedDate") or commit.get("authoredDate")
            return str(value) if value else None
    return None


def _review_commit_oid(review: dict[str, Any]) -> str:
    commit = review.get("commit")
    if isinstance(commit, dict):
        return str(commit.get("oid") or "")
    return str(review.get("commitOID") or review.get("commitOid") or "")


def _login(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("login") or "")
    return str(value or "")


def _canonical_login(login: str | None) -> str:
    return str(login or "").removeprefix("app/").removesuffix("[bot]").lower()


def _author_matches_review_bot(author: str, review_bot: str | None) -> bool:
    return not review_bot or _canonical_login(author) == _canonical_login(review_bot)


def _head_matches(candidate: str | None, head: str) -> bool:
    candidate = str(candidate or "").strip().lower()
    head = head.lower()
    return len(candidate) >= 7 and head.startswith(candidate)


def _body_mentions_head(body: str, head: str) -> bool:
    return any(
        _head_matches(match.group(0), head) for match in re.finditer(r"\b[0-9a-f]{7,40}\b", body)
    )


def _acceptance_reaction_content(acceptance_marker: str) -> set[str]:
    marker = acceptance_marker.strip().lower()
    if marker in _THUMBS_UP_MARKERS:
        return _THUMBS_UP_REACTIONS
    return {acceptance_marker, acceptance_marker.upper()}


def _reaction_actor(reaction: dict[str, Any]) -> str:
    user = reaction.get("user")
    if isinstance(user, dict):
        return str(user.get("login") or "")
    return str(reaction.get("user_login") or reaction.get("author") or "")


def _reaction_matches(
    reaction: dict[str, Any],
    acceptance_marker: str,
    review_bot: str | None,
) -> bool:
    content = str(reaction.get("content") or "")
    if content not in _acceptance_reaction_content(acceptance_marker):
        return False
    return _author_matches_review_bot(_reaction_actor(reaction), review_bot)


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _timestamp_at_or_after(value: Any, baseline: Any) -> bool:
    timestamp = _parse_timestamp(value)
    baseline_timestamp = _parse_timestamp(baseline)
    if timestamp is None or baseline_timestamp is None:
        return False
    return timestamp >= baseline_timestamp


def _has_current_head_review_signal(state: PullRequestState) -> bool:
    return any(signal.current_head for signal in state.review_signals)


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


def _review_signal_summary(state: PullRequestState) -> str:
    current = [signal for signal in state.review_signals if signal.current_head]
    reviews = [signal for signal in current if signal.kind == "review"]
    reactions = [signal for signal in current if signal.kind == "pr_reaction"]
    total_reviews = sum(1 for signal in state.review_signals if signal.kind == "review")
    total_reactions = sum(1 for signal in state.review_signals if signal.kind == "pr_reaction")
    return (
        f"{len(current)} current-head "
        f"({len(reviews)} review(s), {len(reactions)} PR reaction(s)); "
        f"{total_reviews} review(s) and {total_reactions} PR reaction(s) read"
    )


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
        lines.append(f"current-head review signals: {_review_signal_summary(state)}")
    lines.append(f"verdict: {_verdict(report)}")
    if report.artifact_freshness:
        lines.append(f"artifacts: {report.artifact_freshness.summary}")
    elif report.artifact_advisory:
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
        lines.append(f"- Current-head review signals: {_review_signal_summary(state)}")
        if state.protection.conversation_resolution:
            lines.append("- Conversation resolution: **required** by branch protection")
    if report.artifact_freshness:
        lines.append(f"- Artifacts: {report.artifact_freshness.summary}")
    elif report.artifact_advisory:
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
            "head_committed_at": state.head_committed_at,
            "checks": [asdict(check) for check in state.checks],
            "threads": [asdict(thread) for thread in state.threads],
            "review_signals": [asdict(signal) for signal in state.review_signals],
            "protection": asdict(state.protection),
        }
    if report.artifact_freshness is not None:
        payload["artifact_freshness"] = asdict(report.artifact_freshness)
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
            "number,headRefOid,baseRefName,mergeStateStatus,reviewDecision,statusCheckRollup,reviews,commits",
        ]
    )
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected gh pr view payload")
    return payload


def fetch_pr_reactions(repo: str, pr: int) -> list[dict[str, Any]]:
    payload = _gh_json(
        [
            "api",
            "-H",
            "Accept: application/vnd.github+json",
            f"repos/{repo}/issues/{pr}/reactions",
            "--paginate",
        ]
    )
    return payload if isinstance(payload, list) else []


def fetch_pr_changed_paths(repo: str, pr: int) -> list[str]:
    result = _run(["gh", "pr", "diff", str(pr), "--repo", repo, "--name-only"])
    if result.returncode != 0:
        message = result.stderr.strip() or f"Could not load PR #{pr} diff"
        if _is_pr_diff_too_large(message):
            return fetch_pr_changed_paths_from_files_api(repo, pr)
        raise RuntimeError(message)
    return _unique_lines(result.stdout)


def _is_pr_diff_too_large(message: str) -> bool:
    normalized = message.lower()
    return "http 406" in normalized and "diff exceeded" in normalized


def fetch_pr_changed_paths_from_files_api(repo: str, pr: int) -> list[str]:
    """Changed files for large PRs where GitHub refuses the raw diff endpoint."""
    if repo.count("/") != 1:
        raise RuntimeError("--repo must use owner/name format")
    result = _run(
        [
            "gh",
            "api",
            "--paginate",
            f"repos/{repo}/pulls/{pr}/files",
            "--jq",
            ".[].filename",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Could not load PR #{pr} files")
    return _unique_lines(result.stdout)


def _unique_lines(output: str) -> list[str]:
    seen: list[str] = []
    for line in output.splitlines():
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
def check_artifact_freshness(paths: list[str], *, verify: bool = True) -> ArtifactFreshness:
    """Verify generated agent artifacts when changed paths can affect them."""
    if not affects_agent_artifacts(paths):
        return ArtifactFreshness(
            required=False,
            checked=False,
            fresh=True,
            command=None,
            summary="not required (diff does not touch generated-context sources).",
        )
    command = "uv run agent-regenerate --verify"
    if not verify:
        return ArtifactFreshness(
            required=True,
            checked=False,
            fresh=None,
            command=command,
            summary=f"not checked; run `{command}` before merge.",
        )
    result = _run(["uv", "run", "agent-regenerate", "--verify"])
    if result.returncode == 0:
        return ArtifactFreshness(
            required=True,
            checked=True,
            fresh=True,
            command=command,
            summary=f"verified fresh via `{command}`.",
        )
    detail = _process_summary(result)
    return ArtifactFreshness(
        required=True,
        checked=True,
        fresh=False,
        command=command,
        summary=f"stale or unverifiable; `{command}` exited {result.returncode}. {detail}",
    )


def apply_artifact_freshness(
    report: ReadinessReport,
    freshness: ArtifactFreshness,
) -> None:
    report.artifact_freshness = freshness
    report.artifact_advisory = freshness.summary if freshness.required else None
    if freshness.required and freshness.checked and freshness.fresh is False:
        _remove_clean_mergeable_info(report.findings)
        report.findings.append(Finding("blocker", freshness.summary))
    elif freshness.required and not freshness.checked:
        _remove_clean_mergeable_info(report.findings)
        report.findings.append(Finding("warning", freshness.summary))
    report.ready = not any(finding.severity == "blocker" for finding in report.findings)


def _process_summary(result: subprocess.CompletedProcess[str]) -> str:
    text = "\n".join(part for part in (result.stdout, result.stderr) if part)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return " ".join(lines[-3:])


def _remove_clean_mergeable_info(findings: list[Finding]) -> None:
    findings[:] = [
        finding
        for finding in findings
        if finding.message
        != "No blocking readiness issues detected; merge still requires explicit route/approval."
    ]


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
        help="Skip gh calls; report only the local artifact-freshness status.",
    )
    parser.add_argument(
        "--skip-artifact-verify",
        action="store_true",
        help="Do not run agent-regenerate --verify; report artifact freshness as unchecked.",
    )
    parser.add_argument(
        "--review-bot",
        default=_DEFAULT_REVIEW_BOT,
        help=(
            "Bot login required for current-head review/reaction signals "
            f"(default: {_DEFAULT_REVIEW_BOT}). Use an empty value to accept any reviewer."
        ),
    )
    parser.add_argument(
        "--acceptance-marker",
        default=_DEFAULT_ACCEPTANCE_MARKER,
        help=(
            "PR-level reaction marker that can satisfy current-head review evidence "
            f"(default: {_DEFAULT_ACCEPTANCE_MARKER}). Use an empty value to disable reaction matching."
        ),
    )
    parser.add_argument(
        "--require-current-head-review-signal",
        action="store_true",
        help="Treat a missing current-head review/reaction signal as NOT READY.",
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
        freshness = check_artifact_freshness(
            changed_paths(args.base or "main"),
            verify=not args.skip_artifact_verify,
        )
        report = ReadinessReport(
            ready=True,
            findings=[Finding("info", "GitHub checks skipped (--no-github).")],
        )
        apply_artifact_freshness(report, freshness)
        print(_render(args.format, None, report))
        return 1 if (args.exit_on_not_ready and not report.ready) else 0

    try:
        repo = args.repo or detect_repo()
        pr = args.pr or detect_pr_number(repo)
    except RuntimeError as error:
        return _github_failure(args, f"Could not reach GitHub via gh: {error}")

    if pr is None:
        freshness = check_artifact_freshness(
            changed_paths(args.base or "main"),
            verify=not args.skip_artifact_verify,
        )
        report = ReadinessReport(
            ready=True,
            findings=[Finding("info", "No open PR for the current branch.")],
        )
        apply_artifact_freshness(report, freshness)
        print(_render(args.format, None, report))
        return 1 if (args.exit_on_not_ready and not report.ready) else 0

    try:
        pr_payload = fetch_pr_payload(repo, pr)
        base_ref_name = str(args.base or pr_payload.get("baseRefName") or "main")
        paths = fetch_pr_changed_paths(repo, pr)
        freshness = check_artifact_freshness(paths, verify=not args.skip_artifact_verify)
        protection = parse_branch_protection(fetch_branch_protection(repo, base_ref_name))
        state = parse_pr_state(
            pr_payload,
            fetch_review_threads(repo, pr),
            protection,
            fetch_pr_reactions(repo, pr),
            review_bot=args.review_bot or None,
            acceptance_marker=args.acceptance_marker or None,
        )
    except RuntimeError as error:
        return _github_failure(args, f"Could not load PR #{pr}: {error}")

    report = assess_readiness(
        state,
        require_current_head_review_signal=args.require_current_head_review_signal,
    )
    apply_artifact_freshness(report, freshness)
    print(_render(args.format, state, report))
    return 1 if (args.exit_on_not_ready and not report.ready) else 0


def _render(fmt: str, state: PullRequestState | None, report: ReadinessReport) -> str:
    if fmt == "json":
        return format_json(state, report)
    if fmt == "markdown":
        return format_markdown(state, report)
    return format_text(state, report)


def _github_failure(args: argparse.Namespace, message: str) -> int:
    freshness = check_artifact_freshness(
        changed_paths(args.base or "main"),
        verify=not args.skip_artifact_verify,
    )
    report = ReadinessReport(
        ready=False,
        findings=[Finding("blocker", message)],
    )
    apply_artifact_freshness(report, freshness)
    print(_render(args.format, None, report))
    return 1 if args.exit_on_not_ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
