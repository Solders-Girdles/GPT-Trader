from __future__ import annotations

import json

import scripts.agents.pr_readiness as pr_readiness
from scripts.agents.pr_readiness import (
    BranchProtection,
    CheckStatus,
    PullRequestState,
    ReviewThread,
    affects_agent_artifacts,
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
# affects_agent_artifacts
# --------------------------------------------------------------------------- #
def test_artifact_advisory_triggers_for_source_and_tests() -> None:
    assert affects_agent_artifacts(["src/gpt_trader/features/x.py"]) is True
    assert affects_agent_artifacts(["tests/unit/x_test.py"]) is True
    assert affects_agent_artifacts(["pyproject.toml"]) is True
    assert affects_agent_artifacts(["pytest.ini"]) is True
    assert affects_agent_artifacts(["config/environments/.env.template"]) is True
    assert affects_agent_artifacts(["config/agents/flows/default.yaml"]) is True
    assert affects_agent_artifacts(["src/gpt_trader/tui/styles/a.tcss"]) is True


def test_artifact_advisory_skips_unrelated_changes() -> None:
    assert affects_agent_artifacts(["docs/STATUS.md", "README.md"]) is False
    assert affects_agent_artifacts([""]) is False
    assert affects_agent_artifacts([]) is False


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


# --------------------------------------------------------------------------- #
# assess_readiness - the scenario that bit us
# --------------------------------------------------------------------------- #
def test_green_checks_but_unresolved_threads_is_not_ready() -> None:
    """All required checks SUCCESS, but enforced conversation resolution + open
    threads must still report NOT READY (the PR #1056 situation)."""
    state = PullRequestState(
        number=1056,
        head_oid="c6f5c2c4",
        merge_state_status="BLOCKED",
        review_decision="",
        checks=(CheckStatus("Unit Tests (Core)", "SUCCESS", required=True),),
        threads=(
            ReviewThread("chatgpt-codex-connector", False, "medium", "src/x.py", 152),
            ReviewThread("coderabbitai", False, "high", "src/x.py", 236),
        ),
        protection=_protection(conversation_resolution=True),
    )
    report = assess_readiness(state)

    assert report.ready is False
    blockers = [finding for finding in report.findings if finding.severity == "blocker"]
    assert len(blockers) == 1
    assert "unresolved review thread" in blockers[0].message
    assert "block the merge" in blockers[0].message


def test_unresolved_threads_without_enforcement_is_warning_not_blocker() -> None:
    state = PullRequestState(
        number=1,
        head_oid="abc",
        merge_state_status="CLEAN",
        review_decision="",
        checks=(CheckStatus("Unit Tests (Core)", "SUCCESS", required=True),),
        threads=(ReviewThread("coderabbitai", False, "low", "docs/x.md", 1),),
        protection=_protection(conversation_resolution=False),
    )
    report = assess_readiness(state)

    assert report.ready is True
    assert any(finding.severity == "warning" for finding in report.findings)


def test_failing_required_check_is_blocker() -> None:
    state = PullRequestState(
        number=2,
        head_oid="abc",
        merge_state_status="BLOCKED",
        review_decision="",
        checks=(
            CheckStatus("Unit Tests (Core)", "FAILURE", required=True),
            CheckStatus("CodeRabbit", "SUCCESS", required=False),
        ),
        threads=(),
        protection=_protection(),
    )
    report = assess_readiness(state)

    assert report.ready is False
    assert any("Unit Tests (Core)" in finding.message for finding in report.findings)


def test_missing_required_check_is_blocker() -> None:
    state = PullRequestState(
        number=20,
        head_oid="abc",
        merge_state_status="CLEAN",
        review_decision="",
        checks=(),
        threads=(),
        protection=_protection(required_checks=("Unit Tests (Core)",)),
    )
    report = assess_readiness(state)

    assert report.ready is False
    assert any("missing" in finding.message for finding in report.findings)


def test_skipped_and_neutral_required_checks_are_passing() -> None:
    state = PullRequestState(
        number=21,
        head_oid="abc",
        merge_state_status="CLEAN",
        review_decision="",
        checks=(
            CheckStatus("Docs Link Audit", "SKIPPED", required=True),
            CheckStatus("Analyze Python", "NEUTRAL", required=True),
        ),
        threads=(),
        protection=_protection(required_checks=("Docs Link Audit", "Analyze Python")),
    )
    report = assess_readiness(state)

    assert report.ready is True


def test_blocked_state_without_specific_finding_is_not_ready() -> None:
    state = PullRequestState(
        number=22,
        head_oid="abc",
        merge_state_status="BLOCKED",
        review_decision="",
        checks=(CheckStatus("Unit Tests (Core)", "SUCCESS", required=True),),
        threads=(ReviewThread("coderabbitai", False, "low", "docs/x.md", 1),),
        protection=_protection(conversation_resolution=False),
    )
    report = assess_readiness(state)

    assert report.ready is False
    assert any("Branch protection" in finding.message for finding in report.findings)


def test_draft_state_is_not_ready() -> None:
    state = PullRequestState(
        number=23,
        head_oid="abc",
        merge_state_status="DRAFT",
        review_decision="",
        checks=(CheckStatus("Unit Tests (Core)", "SUCCESS", required=True),),
        threads=(),
        protection=_protection(),
    )
    report = assess_readiness(state)

    assert report.ready is False
    assert any("draft" in finding.message.lower() for finding in report.findings)


def test_unknown_merge_state_is_not_ready() -> None:
    state = PullRequestState(
        number=24,
        head_oid="abc",
        merge_state_status="UNKNOWN",
        review_decision="",
        checks=(CheckStatus("Unit Tests (Core)", "SUCCESS", required=True),),
        threads=(),
        protection=_protection(),
    )
    report = assess_readiness(state)

    assert report.ready is False
    assert any("mergeability" in finding.message for finding in report.findings)


def test_clean_pr_with_no_open_threads_is_ready() -> None:
    state = PullRequestState(
        number=3,
        head_oid="abc",
        merge_state_status="CLEAN",
        review_decision="",
        checks=(CheckStatus("Unit Tests (Core)", "SUCCESS", required=True),),
        threads=(ReviewThread("coderabbitai", True, "high", "src/x.py", 1),),
        protection=_protection(conversation_resolution=True),
    )
    report = assess_readiness(state)

    assert report.ready is True
    assert all(finding.severity == "info" for finding in report.findings)


def test_behind_base_is_blocker() -> None:
    state = PullRequestState(
        number=4,
        head_oid="abc",
        merge_state_status="BEHIND",
        review_decision="",
        checks=(CheckStatus("Unit Tests (Core)", "SUCCESS", required=True),),
        threads=(),
        protection=_protection(),
    )
    report = assess_readiness(state)

    assert report.ready is False
    assert any("behind" in finding.message.lower() for finding in report.findings)


def test_required_review_missing_is_blocker() -> None:
    state = PullRequestState(
        number=5,
        head_oid="abc",
        merge_state_status="BLOCKED",
        review_decision="REVIEW_REQUIRED",
        checks=(CheckStatus("Unit Tests (Core)", "SUCCESS", required=True),),
        threads=(),
        protection=_protection(required_review_count=1),
    )
    report = assess_readiness(state)

    assert report.ready is False
    assert any("approving review" in finding.message for finding in report.findings)


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
