from __future__ import annotations

import json

import scripts.agents.pr_readiness as pr_readiness
from scripts.agents.pr_readiness import (
    BranchProtection,
    CheckStatus,
    PullRequestState,
    ReviewSignal,
    ReviewThread,
    assess_readiness,
    format_json,
    format_markdown,
    parse_branch_protection,
    parse_pr_state,
    parse_severity,
)


def _protection(
    *,
    conversation_resolution: bool = False,
    required_checks: tuple[str, ...] = ("Unit Tests (Core)",),
    required_review_count: int = 0,
    strict: bool = True,
) -> BranchProtection:
    return BranchProtection(
        strict=strict,
        conversation_resolution=conversation_resolution,
        required_review_count=required_review_count,
        required_checks=required_checks,
    )


# --------------------------------------------------------------------------- #
# parse_severity
# --------------------------------------------------------------------------- #
def test_parse_severity_recognizes_bot_tokens() -> None:
    assert parse_severity("P2 Badge: preserve sub-cent levels") == "medium"
    assert parse_severity("Major data integrity issue") == "high"
    assert parse_severity("P0 critical") == "critical"
    assert parse_severity("minor nit") == "low"


def test_parse_severity_returns_none_without_token() -> None:
    assert parse_severity("just a normal comment") is None


# --------------------------------------------------------------------------- #
# parse_branch_protection
# --------------------------------------------------------------------------- #
def test_parse_branch_protection_extracts_flags() -> None:
    raw = {
        "required_status_checks": {
            "strict": True,
            "checks": [{"context": "Type Check"}, {"context": "Unit Tests (Core)"}],
            "contexts": ["CodeQL"],
        },
        "required_pull_request_reviews": {"required_approving_review_count": 0},
        "required_conversation_resolution": {"enabled": True},
    }
    protection = parse_branch_protection(raw)
    assert protection.strict is True
    assert protection.conversation_resolution is True
    assert protection.required_review_count == 0
    assert protection.required_checks == ("CodeQL", "Type Check", "Unit Tests (Core)")


def test_parse_branch_protection_handles_missing_payload() -> None:
    protection = parse_branch_protection(None)
    assert protection.required_checks == ()
    assert protection.conversation_resolution is False


# --------------------------------------------------------------------------- #
# parse_pr_state
# --------------------------------------------------------------------------- #
def test_parse_pr_state_maps_checks_and_threads() -> None:
    pr_json = {
        "number": 1056,
        "headRefOid": "a367f3c6deadbeef",
        "baseRefName": "release/2026-06",
        "mergeStateStatus": "blocked",
        "reviewDecision": "",
        "statusCheckRollup": [
            {"name": "Unit Tests (Core)", "status": "COMPLETED", "conclusion": "SUCCESS"},
            {"name": "CodeRabbit", "state": "SUCCESS"},
        ],
    }
    threads_json = [
        {
            "isResolved": False,
            "path": "src/x.py",
            "line": 152,
            "comments": {"nodes": [{"author": {"login": "coderabbitai"}, "body": "Major bug"}]},
        }
    ]
    state = parse_pr_state(pr_json, threads_json, _protection())

    assert state.number == 1056
    assert state.base_ref_name == "release/2026-06"
    assert state.merge_state_status == "BLOCKED"
    core = next(check for check in state.checks if check.name == "Unit Tests (Core)")
    assert core.required is True
    assert core.state == "SUCCESS"
    coderabbit = next(check for check in state.checks if check.name == "CodeRabbit")
    assert coderabbit.required is False
    assert state.threads[0].severity == "high"
    assert state.threads[0].author == "coderabbitai"


def test_parse_pr_state_marks_current_head_pr_reaction_signal() -> None:
    head = "a367f3c6deadbeef"
    pr_json = {
        "number": 1056,
        "headRefOid": head,
        "baseRefName": "main",
        "mergeStateStatus": "CLEAN",
        "reviewDecision": "",
        "statusCheckRollup": [],
        "commits": [{"oid": head, "committedDate": "2026-06-30T16:46:28Z"}],
    }
    reactions = [
        {
            "content": "+1",
            "created_at": "2026-06-30T16:49:26Z",
            "user": {"login": "chatgpt-codex-connector[bot]"},
        }
    ]

    state = parse_pr_state(pr_json, [], _protection(required_checks=()), reactions)

    assert state.head_committed_at == "2026-06-30T16:46:28Z"
    assert state.head_updated_at == "2026-06-30T16:46:28Z"
    assert state.review_signals == (
        ReviewSignal(
            kind="pr_reaction",
            author="chatgpt-codex-connector[bot]",
            state="+1",
            current_head=True,
            created_at="2026-06-30T16:49:26Z",
            url=None,
        ),
    )


def test_parse_pr_state_rejects_stale_pr_reaction_signal() -> None:
    head = "a367f3c6deadbeef"
    pr_json = {
        "number": 1056,
        "headRefOid": head,
        "baseRefName": "main",
        "mergeStateStatus": "CLEAN",
        "reviewDecision": "",
        "statusCheckRollup": [],
        "commits": [{"oid": head, "committedDate": "2026-06-30T16:46:28Z"}],
    }
    reactions = [
        {
            "content": "+1",
            "created_at": "2026-06-30T16:40:00Z",
            "user": {"login": "chatgpt-codex-connector[bot]"},
        }
    ]

    state = parse_pr_state(pr_json, [], _protection(required_checks=()), reactions)

    assert len(state.review_signals) == 1
    assert state.review_signals[0].current_head is False


def test_parse_pr_state_ignores_generic_pr_update_for_reaction_freshness() -> None:
    head = "a367f3c6deadbeef"
    pr_json = {
        "number": 1056,
        "headRefOid": head,
        "baseRefName": "main",
        "mergeStateStatus": "CLEAN",
        "reviewDecision": "",
        "statusCheckRollup": [],
        "commits": [{"oid": head, "committedDate": "2026-06-30T16:30:00Z"}],
        "updatedAt": "2026-06-30T17:00:00Z",
    }
    reactions = [
        {
            "content": "+1",
            "created_at": "2026-06-30T16:45:00Z",
            "user": {"login": "chatgpt-codex-connector[bot]"},
        }
    ]

    state = parse_pr_state(pr_json, [], _protection(required_checks=()), reactions)

    assert state.head_updated_at == "2026-06-30T16:30:00Z"
    assert len(state.review_signals) == 1
    assert state.review_signals[0].current_head is True


def test_parse_pr_state_rejects_reaction_before_force_push_to_head() -> None:
    head = "a367f3c6deadbeef"
    pr_json = {
        "number": 1056,
        "headRefOid": head,
        "baseRefName": "main",
        "mergeStateStatus": "CLEAN",
        "reviewDecision": "",
        "statusCheckRollup": [],
        "commits": [{"oid": head, "committedDate": "2026-06-30T16:30:00Z"}],
    }
    force_push_events = [
        {
            "createdAt": "2026-06-30T16:50:00Z",
            "afterCommit": {"oid": head},
            "beforeCommit": {"oid": "oldhead"},
        }
    ]
    reactions = [
        {
            "content": "+1",
            "created_at": "2026-06-30T16:45:00Z",
            "user": {"login": "chatgpt-codex-connector[bot]"},
        }
    ]

    state = parse_pr_state(
        pr_json,
        [],
        _protection(required_checks=()),
        reactions,
        force_push_events,
    )

    assert state.head_updated_at == "2026-06-30T16:50:00Z"
    assert len(state.review_signals) == 1
    assert state.review_signals[0].current_head is False


def test_parse_pr_state_rejects_reaction_before_fast_forward_push_to_head() -> None:
    # A fast-forward push emits no HeadRefForcePushedEvent; the head commit's
    # pushedDate (16:50) is the only evidence it landed after the 16:45 reaction.
    head = "a367f3c6deadbeef"
    pr_json = {
        "number": 1056,
        "headRefOid": head,
        "baseRefName": "main",
        "mergeStateStatus": "CLEAN",
        "reviewDecision": "",
        "statusCheckRollup": [],
        "commits": [{"oid": head, "committedDate": "2026-06-30T16:30:00Z"}],
    }
    reactions = [
        {
            "content": "+1",
            "created_at": "2026-06-30T16:45:00Z",
            "user": {"login": "chatgpt-codex-connector[bot]"},
        }
    ]

    state = parse_pr_state(
        pr_json,
        [],
        _protection(required_checks=()),
        reactions,
        [],  # no force-push events for an ordinary push
        "2026-06-30T16:50:00Z",  # head commit pushedDate
    )

    assert state.head_updated_at == "2026-06-30T16:50:00Z"
    assert len(state.review_signals) == 1
    assert state.review_signals[0].current_head is False


def test_parse_pr_state_accepts_reaction_after_fast_forward_push_to_head() -> None:
    # The floor works both ways: a reaction after the push counts as current.
    head = "a367f3c6deadbeef"
    pr_json = {
        "number": 1056,
        "headRefOid": head,
        "baseRefName": "main",
        "mergeStateStatus": "CLEAN",
        "reviewDecision": "",
        "statusCheckRollup": [],
        "commits": [{"oid": head, "committedDate": "2026-06-30T16:30:00Z"}],
    }
    reactions = [
        {
            "content": "+1",
            "created_at": "2026-06-30T16:55:00Z",
            "user": {"login": "chatgpt-codex-connector[bot]"},
        }
    ]

    state = parse_pr_state(
        pr_json,
        [],
        _protection(required_checks=()),
        reactions,
        [],
        "2026-06-30T16:50:00Z",
    )

    assert state.head_updated_at == "2026-06-30T16:50:00Z"
    assert state.review_signals[0].current_head is True


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def test_format_json_is_machine_readable() -> None:
    report = assess_readiness(
        PullRequestState(
            number=9,
            head_oid="abc1234567",
            merge_state_status="CLEAN",
            review_decision="",
            checks=(CheckStatus("Unit Tests (Core)", "SUCCESS", required=True),),
            threads=(),
            protection=_protection(),
        )
    )
    payload = json.loads(format_json(None, report))
    assert payload["ready"] is True
    assert payload["verdict"] == "READY"


def test_format_markdown_includes_verdict_and_findings() -> None:
    state = PullRequestState(
        number=10,
        head_oid="abcdef1234",
        merge_state_status="BLOCKED",
        review_decision="",
        checks=(CheckStatus("Unit Tests (Core)", "SUCCESS", required=True),),
        threads=(ReviewThread("coderabbitai", False, "high", "src/x.py", 1),),
        protection=_protection(conversation_resolution=True),
    )
    markdown = format_markdown(state, assess_readiness(state))
    assert "## PR readiness receipt" in markdown
    assert "NOT READY" in markdown
    assert "Conversation resolution" in markdown


def test_module_exposes_main() -> None:
    assert callable(pr_readiness.main)
