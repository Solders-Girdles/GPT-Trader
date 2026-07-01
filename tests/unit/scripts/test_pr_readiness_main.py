from __future__ import annotations

import json
from typing import Any

import scripts.agents.pr_readiness as pr_readiness


def test_main_uses_pr_base_for_advisory_and_branch_protection(monkeypatch) -> None:
    bases: list[str] = []

    monkeypatch.setattr(
        pr_readiness, "changed_paths", lambda base: bases.append(f"diff:{base}") or []
    )
    monkeypatch.setattr(
        pr_readiness,
        "fetch_pr_changed_paths",
        lambda repo, pr: bases.append(f"pr-diff:{pr}") or [],
    )
    monkeypatch.setattr(pr_readiness, "fetch_review_threads", lambda repo, pr: [])
    monkeypatch.setattr(pr_readiness, "fetch_pr_reactions", lambda repo, pr: [])
    monkeypatch.setattr(pr_readiness, "fetch_head_update_events", lambda repo, pr: [])
    monkeypatch.setattr(pr_readiness, "fetch_head_commit_pushed_at", lambda repo, pr: None)
    monkeypatch.setattr(
        pr_readiness,
        "check_artifact_freshness",
        lambda paths, verify=True: pr_readiness.ArtifactFreshness(
            required=False,
            checked=False,
            fresh=True,
            command=None,
            summary="not required",
        ),
    )
    monkeypatch.setattr(
        pr_readiness,
        "fetch_pr_payload",
        lambda repo, pr: {
            "number": 12,
            "headRefOid": "abcdef1234",
            "baseRefName": "release/2026-06",
            "mergeStateStatus": "CLEAN",
            "reviewDecision": "",
            "statusCheckRollup": [],
        },
    )

    def fake_protection(repo: str, branch: str) -> dict[str, Any]:
        bases.append(branch)
        return {"required_status_checks": {"checks": []}}

    monkeypatch.setattr(pr_readiness, "fetch_branch_protection", fake_protection)

    assert pr_readiness.main(["--repo", "owner/repo", "--pr", "12"]) == 0
    assert "pr-diff:12" in bases
    assert "release/2026-06" in bases


def test_main_pr_mode_uses_pr_diff_for_artifact_advisory(monkeypatch, capsys) -> None:
    monkeypatch.setattr(pr_readiness, "fetch_review_threads", lambda repo, pr: [])
    monkeypatch.setattr(pr_readiness, "fetch_pr_reactions", lambda repo, pr: [])
    monkeypatch.setattr(pr_readiness, "fetch_head_update_events", lambda repo, pr: [])
    monkeypatch.setattr(pr_readiness, "fetch_head_commit_pushed_at", lambda repo, pr: None)
    monkeypatch.setattr(pr_readiness, "fetch_branch_protection", lambda repo, branch: {})
    monkeypatch.setattr(pr_readiness, "fetch_pr_changed_paths", lambda repo, pr: ["pytest.ini"])
    monkeypatch.setattr(
        pr_readiness,
        "check_artifact_freshness",
        lambda paths, verify=True: pr_readiness.ArtifactFreshness(
            required=True,
            checked=True,
            fresh=True,
            command="uv run agent-regenerate --verify",
            summary="verified fresh via `uv run agent-regenerate --verify`.",
        ),
    )
    monkeypatch.setattr(
        pr_readiness,
        "fetch_pr_payload",
        lambda repo, pr: {
            "number": 12,
            "headRefOid": "abcdef1234",
            "baseRefName": "main",
            "mergeStateStatus": "CLEAN",
            "reviewDecision": "",
            "statusCheckRollup": [],
        },
    )

    assert pr_readiness.main(["--repo", "owner/repo", "--pr", "12"]) == 0
    assert "agent-regenerate --verify" in capsys.readouterr().out


def test_main_reports_missing_current_head_review_signal_by_default(monkeypatch, capsys) -> None:
    monkeypatch.setattr(pr_readiness, "fetch_review_threads", lambda repo, pr: [])
    monkeypatch.setattr(pr_readiness, "fetch_pr_reactions", lambda repo, pr: [])
    monkeypatch.setattr(pr_readiness, "fetch_head_update_events", lambda repo, pr: [])
    monkeypatch.setattr(pr_readiness, "fetch_head_commit_pushed_at", lambda repo, pr: None)
    monkeypatch.setattr(pr_readiness, "fetch_branch_protection", lambda repo, branch: {})
    monkeypatch.setattr(pr_readiness, "fetch_pr_changed_paths", lambda repo, pr: [])
    monkeypatch.setattr(
        pr_readiness,
        "check_artifact_freshness",
        lambda paths, verify=True: pr_readiness.ArtifactFreshness(
            required=False,
            checked=False,
            fresh=True,
            command=None,
            summary="not required",
        ),
    )
    monkeypatch.setattr(
        pr_readiness,
        "fetch_pr_payload",
        lambda repo, pr: {
            "number": 12,
            "headRefOid": "abcdef1234",
            "baseRefName": "main",
            "mergeStateStatus": "CLEAN",
            "reviewDecision": "",
            "statusCheckRollup": [],
        },
    )

    assert pr_readiness.main(["--repo", "owner/repo", "--pr", "12"]) == 0
    output = capsys.readouterr().out
    assert "verdict: READY" in output
    assert "[warning]" in output
    assert "No current-head review" in output


def test_main_requires_current_head_review_signal_when_requested(monkeypatch, capsys) -> None:
    monkeypatch.setattr(pr_readiness, "fetch_review_threads", lambda repo, pr: [])
    monkeypatch.setattr(pr_readiness, "fetch_pr_reactions", lambda repo, pr: [])
    monkeypatch.setattr(pr_readiness, "fetch_head_update_events", lambda repo, pr: [])
    monkeypatch.setattr(pr_readiness, "fetch_head_commit_pushed_at", lambda repo, pr: None)
    monkeypatch.setattr(pr_readiness, "fetch_branch_protection", lambda repo, branch: {})
    monkeypatch.setattr(pr_readiness, "fetch_pr_changed_paths", lambda repo, pr: [])
    monkeypatch.setattr(
        pr_readiness,
        "check_artifact_freshness",
        lambda paths, verify=True: pr_readiness.ArtifactFreshness(
            required=False,
            checked=False,
            fresh=True,
            command=None,
            summary="not required",
        ),
    )
    monkeypatch.setattr(
        pr_readiness,
        "fetch_pr_payload",
        lambda repo, pr: {
            "number": 12,
            "headRefOid": "abcdef1234",
            "baseRefName": "main",
            "mergeStateStatus": "CLEAN",
            "reviewDecision": "",
            "statusCheckRollup": [],
        },
    )

    assert (
        pr_readiness.main(
            ["--repo", "owner/repo", "--pr", "12", "--require-current-head-review-signal"]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "verdict: NOT READY" in output
    assert "[blocker]" in output


def test_main_exit_on_not_ready_fails_when_github_unavailable(monkeypatch) -> None:
    def raise_unavailable() -> str:
        raise RuntimeError("gh missing")

    monkeypatch.setattr(pr_readiness, "detect_repo", raise_unavailable)
    monkeypatch.setattr(
        pr_readiness,
        "check_artifact_freshness",
        lambda paths, verify=True: pr_readiness.ArtifactFreshness(
            required=False,
            checked=False,
            fresh=True,
            command=None,
            summary="not required",
        ),
    )

    assert pr_readiness.main(["--exit-on-not-ready"]) == 1


def test_main_json_format_survives_github_unavailable(monkeypatch, capsys) -> None:
    def raise_unavailable() -> str:
        raise RuntimeError("gh missing")

    monkeypatch.setattr(pr_readiness, "detect_repo", raise_unavailable)
    monkeypatch.setattr(
        pr_readiness,
        "check_artifact_freshness",
        lambda paths, verify=True: pr_readiness.ArtifactFreshness(
            required=False,
            checked=False,
            fresh=True,
            command=None,
            summary="not required",
        ),
    )

    assert pr_readiness.main(["--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["verdict"] == "NOT READY"
