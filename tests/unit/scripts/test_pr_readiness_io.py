from __future__ import annotations

import subprocess
from typing import Any

import pytest
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


def test_fetch_head_update_events_paginates(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_gh_json(args: list[str]) -> dict[str, Any]:
        calls.append(args)
        if len(calls) == 1:
            return {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "timelineItems": {
                                "nodes": [{"createdAt": "2026-06-30T16:00:00Z"}],
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
                        "timelineItems": {
                            "nodes": [{"createdAt": "2026-06-30T17:00:00Z"}],
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        }
                    }
                }
            }
        }

    monkeypatch.setattr(pr_readiness, "_gh_json", fake_gh_json)

    nodes = pr_readiness.fetch_head_update_events("owner/repo", 9)

    assert [node["createdAt"] for node in nodes] == [
        "2026-06-30T16:00:00Z",
        "2026-06-30T17:00:00Z",
    ]
    assert any(part == "cursor=cursor-1" for part in calls[1])


def test_fetch_pr_reactions_flattens_slurped_pages(monkeypatch) -> None:
    calls: list[list[str]] = []

    # `gh api --paginate --slurp` returns an outer array with one inner array
    # per page; fetch_pr_reactions must flatten that into a single reaction list.
    def fake_gh_json(args: list[str]) -> Any:
        calls.append(args)
        return [
            [{"id": 1, "content": "+1"}, {"id": 2, "content": "eyes"}],
            [{"id": 3, "content": "rocket"}],
        ]

    monkeypatch.setattr(pr_readiness, "_gh_json", fake_gh_json)

    reactions = pr_readiness.fetch_pr_reactions("owner/repo", 9)

    assert [r["id"] for r in reactions] == [1, 2, 3]
    # The multi-page path must be requested with --slurp so _gh_json sees one
    # valid JSON document instead of concatenated per-page arrays.
    assert "--slurp" in calls[0]


def test_fetch_pr_reactions_handles_non_list_payload(monkeypatch) -> None:
    monkeypatch.setattr(pr_readiness, "_gh_json", lambda args: {"message": "Not Found"})
    assert pr_readiness.fetch_pr_reactions("owner/repo", 9) == []


def test_fetch_review_threads_rejects_malformed_repo() -> None:
    try:
        pr_readiness.fetch_review_threads("not-a-repo", 9)
    except RuntimeError as error:
        assert "owner/name" in str(error)
    else:
        raise AssertionError("expected RuntimeError")


def test_fetch_head_update_events_rejects_malformed_repo() -> None:
    try:
        pr_readiness.fetch_head_update_events("not-a-repo", 9)
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


def test_fetch_pr_changed_paths_raises_on_non_size_diff_error(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, "", "fatal: not authenticated")

    monkeypatch.setattr(pr_readiness, "_run", fake_run)

    with pytest.raises(RuntimeError, match="not authenticated"):
        pr_readiness.fetch_pr_changed_paths("owner/repo", 123)

    # No files-API fallback for an error that is not the raw-diff size limit.
    assert calls == [["gh", "pr", "diff", "123", "--repo", "owner/repo", "--name-only"]]


def test_fetch_pr_reactions_reads_issue_reactions(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_gh_json(args: list[str]) -> list[Any]:
        calls.append(args)
        # --slurp wraps the single page in an outer array.
        return [[{"content": "+1"}]]

    monkeypatch.setattr(pr_readiness, "_gh_json", fake_gh_json)

    assert pr_readiness.fetch_pr_reactions("owner/repo", 123) == [{"content": "+1"}]
    assert calls == [
        [
            "api",
            "-H",
            "Accept: application/vnd.github+json",
            "repos/owner/repo/issues/123/reactions",
            "--paginate",
            "--slurp",
        ]
    ]
