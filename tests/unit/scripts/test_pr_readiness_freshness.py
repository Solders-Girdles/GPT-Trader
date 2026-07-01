from __future__ import annotations

import scripts.agents.pr_readiness as pr_readiness
from scripts.agents.pr_readiness import (
    ReadinessReport,
    affects_agent_artifacts,
    apply_artifact_freshness,
    check_artifact_freshness,
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
    assert affects_agent_artifacts(["src/gpt_trader/features/x.yaml"]) is True


def test_artifact_advisory_skips_unrelated_changes() -> None:
    assert affects_agent_artifacts(["docs/STATUS.md", "README.md"]) is False
    assert affects_agent_artifacts([""]) is False
    assert affects_agent_artifacts([]) is False


# --------------------------------------------------------------------------- #
# check_artifact_freshness / apply_artifact_freshness
# --------------------------------------------------------------------------- #
def test_artifact_freshness_blocks_when_verify_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        pr_readiness,
        "_run",
        lambda args: pr_readiness.subprocess.CompletedProcess(
            args,
            1,
            "var/agents/testing/index.json stale\n",
            "",
        ),
    )
    report = ReadinessReport(ready=True, findings=[])

    freshness = check_artifact_freshness(["tests/unit/x_test.py"])
    apply_artifact_freshness(report, freshness)

    assert report.ready is False
    assert freshness.checked is True
    assert freshness.fresh is False
    assert any("agent-regenerate --verify" in finding.message for finding in report.findings)


def test_artifact_freshness_can_be_marked_unchecked() -> None:
    freshness = check_artifact_freshness(["tests/unit/x_test.py"], verify=False)

    assert freshness.required is True
    assert freshness.checked is False
    assert freshness.fresh is None


def test_artifact_freshness_skips_verify_when_checkout_differs_from_pr_head(monkeypatch) -> None:
    # Running --pr N from an unrelated checkout must not attribute the local
    # tree's freshness to the PR; the verify is skipped with an explanatory note.
    monkeypatch.setattr(pr_readiness, "_local_head_oid", lambda: "a" * 40)

    def fail_if_run(args: list[str]) -> pr_readiness.subprocess.CompletedProcess[str]:
        raise AssertionError(f"agent-regenerate should not run on mismatch: {args}")

    monkeypatch.setattr(pr_readiness, "_run", fail_if_run)

    freshness = check_artifact_freshness(["tests/unit/x_test.py"], head_oid="b" * 40)

    assert freshness.checked is False
    assert freshness.fresh is None
    assert "differs from PR head" in freshness.summary


def test_artifact_freshness_verifies_when_checkout_matches_pr_head(monkeypatch) -> None:
    head = "c" * 40
    monkeypatch.setattr(pr_readiness, "_local_head_oid", lambda: head)
    monkeypatch.setattr(
        pr_readiness,
        "_run",
        lambda args: pr_readiness.subprocess.CompletedProcess(args, 0, "", ""),
    )

    freshness = check_artifact_freshness(["tests/unit/x_test.py"], head_oid=head)

    assert freshness.checked is True
    assert freshness.fresh is True
