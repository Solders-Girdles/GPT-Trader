from __future__ import annotations

import json
import subprocess
from typing import Any

import scripts.agents.pr_readiness as pr_readiness


def test_run_converts_missing_executable_to_failed_process(monkeypatch) -> None:
    def raise_missing(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("missing gh")

    monkeypatch.setattr(pr_readiness.subprocess, "run", raise_missing)

    result = pr_readiness._run(["gh", "version"])

    assert result.returncode == 127
    assert "missing gh" in result.stderr


def test_changed_paths_retries_origin_base_when_local_base_is_missing(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args == ["git", "diff", "--name-only", "release/1.2...HEAD"]:
            return subprocess.CompletedProcess(args, 128, "", "unknown revision")
        if args == ["git", "diff", "--name-only", "origin/release/1.2...HEAD"]:
            return subprocess.CompletedProcess(args, 0, "scripts/agents/pr_readiness.py\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(pr_readiness, "_run", fake_run)

    assert pr_readiness.changed_paths("release/1.2") == ["scripts/agents/pr_readiness.py"]
    assert ["git", "diff", "--name-only", "origin/release/1.2...HEAD"] in calls


def test_detect_pr_number_returns_none_only_for_no_pr(monkeypatch) -> None:
    def fake_gh_json(args: list[str]) -> dict[str, Any]:
        raise RuntimeError("no pull requests found for branch")

    monkeypatch.setattr(pr_readiness, "_gh_json", fake_gh_json)

    assert pr_readiness.detect_pr_number("owner/repo") is None


def test_detect_pr_number_reraises_api_failures(monkeypatch) -> None:
    def fake_gh_json(args: list[str]) -> dict[str, Any]:
        raise RuntimeError("gh: API rate limit exceeded (HTTP 403)")

    monkeypatch.setattr(pr_readiness, "_gh_json", fake_gh_json)

    try:
        pr_readiness.detect_pr_number("owner/repo")
    except RuntimeError as error:
        assert "rate limit" in str(error)
    else:
        raise AssertionError("expected RuntimeError")


def test_fetch_review_threads_paginates(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_gh_json(args: list[str]) -> dict[str, Any]:
        calls.append(args)
        if len(calls) == 1:
            return {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "reviewThreads": {
                                "nodes": [{"line": 1}],
                                "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
                            }
                        }
                    }
                }
            }
        return {
            "data": {
                "repository": {
                    "pullRequest": {
                        "reviewThreads": {
                            "nodes": [{"line": 2}],
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        }
                    }
                }
            }
        }

    monkeypatch.setattr(pr_readiness, "_gh_json", fake_gh_json)

    nodes = pr_readiness.fetch_review_threads("owner/repo", 9)

    assert [node["line"] for node in nodes] == [1, 2]
    assert any(part == "cursor=cursor-1" for part in calls[1])


def test_fetch_review_threads_rejects_malformed_repo() -> None:
    try:
        pr_readiness.fetch_review_threads("not-a-repo", 9)
    except RuntimeError as error:
        assert "owner/name" in str(error)
    else:
        raise AssertionError("expected RuntimeError")


def test_fetch_branch_protection_encodes_branch_segments(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_gh_json(args: list[str]) -> dict[str, Any]:
        calls.append(args)
        return {"required_status_checks": {"checks": []}}

    monkeypatch.setattr(pr_readiness, "_gh_json", fake_gh_json)

    assert pr_readiness.fetch_branch_protection("owner/repo", "release/1.2") == {
        "required_status_checks": {"checks": []}
    }
    assert calls == [["api", "repos/owner/repo/branches/release%2F1.2/protection"]]


def test_fetch_branch_protection_returns_none_for_missing_protection(monkeypatch) -> None:
    def fake_gh_json(args: list[str]) -> dict[str, Any]:
        raise RuntimeError("gh: Branch not protected (HTTP 404)")

    monkeypatch.setattr(pr_readiness, "_gh_json", fake_gh_json)

    assert pr_readiness.fetch_branch_protection("owner/repo", "main") is None


def test_fetch_branch_protection_reraises_api_failures(monkeypatch) -> None:
    def fake_gh_json(args: list[str]) -> dict[str, Any]:
        raise RuntimeError("gh: API rate limit exceeded (HTTP 403)")

    monkeypatch.setattr(pr_readiness, "_gh_json", fake_gh_json)

    try:
        pr_readiness.fetch_branch_protection("owner/repo", "main")
    except RuntimeError as error:
        assert "rate limit" in str(error)
    else:
        raise AssertionError("expected RuntimeError")


def test_fetch_pr_changed_paths_uses_requested_pr_diff(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "pytest.ini\npytest.ini\n", "")

    monkeypatch.setattr(pr_readiness, "_run", fake_run)

    assert pr_readiness.fetch_pr_changed_paths("owner/repo", 123) == ["pytest.ini"]
    assert calls == [["gh", "pr", "diff", "123", "--repo", "owner/repo", "--name-only"]]


def test_fetch_pr_changed_paths_falls_back_for_large_pr_diff(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[:3] == ["gh", "pr", "diff"]:
            return subprocess.CompletedProcess(
                args,
                1,
                "",
                "HTTP 406: Sorry, the diff exceeded the maximum number of files (300).",
            )
        return subprocess.CompletedProcess(args, 0, "src/a.py\nsrc/a.py\nsrc/b.py\n", "")

    monkeypatch.setattr(pr_readiness, "_run", fake_run)

    assert pr_readiness.fetch_pr_changed_paths("owner/repo", 123) == [
        "src/a.py",
        "src/b.py",
    ]
    assert calls == [
        ["gh", "pr", "diff", "123", "--repo", "owner/repo", "--name-only"],
        [
            "gh",
            "api",
            "--paginate",
            "repos/owner/repo/pulls/123/files",
            "--jq",
            ".[].filename",
        ],
    ]


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
    monkeypatch.setattr(pr_readiness, "fetch_branch_protection", lambda repo, branch: {})
    monkeypatch.setattr(pr_readiness, "fetch_pr_changed_paths", lambda repo, pr: ["pytest.ini"])
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


def test_main_exit_on_not_ready_fails_when_github_unavailable(monkeypatch) -> None:
    def raise_unavailable() -> str:
        raise RuntimeError("gh missing")

    monkeypatch.setattr(pr_readiness, "detect_repo", raise_unavailable)

    assert pr_readiness.main(["--exit-on-not-ready"]) == 1


def test_main_json_format_survives_github_unavailable(monkeypatch, capsys) -> None:
    def raise_unavailable() -> str:
        raise RuntimeError("gh missing")

    monkeypatch.setattr(pr_readiness, "detect_repo", raise_unavailable)

    assert pr_readiness.main(["--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["verdict"] == "NOT READY"
