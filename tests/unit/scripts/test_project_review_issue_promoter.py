from __future__ import annotations

import json
from pathlib import Path

from scripts.maintenance import project_review_issue_promoter as promoter


def test_example_packet_is_valid() -> None:
    assert promoter.validate_packet(promoter.example_packet()) == []


def test_trading_execution_findings_require_human_decision() -> None:
    packet = promoter.example_packet()
    packet["scope"]["touches_trading_execution"] = True
    packet["routing"]["needs_human_decision"] = False

    errors = promoter.validate_packet(packet)

    assert (
        "scope.touches_trading_execution=true requires routing.needs_human_decision=true" in errors
    )


def test_dry_run_renders_issue_body(tmp_path: Path, capsys) -> None:
    packet_path = tmp_path / "finding.json"
    packet_path.write_text(json.dumps(promoter.example_packet()), encoding="utf-8")

    exit_code = promoter.main(["--packet", str(packet_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "gpt-trader-agent-finding-id: agent-artifacts-stale-example" in captured.out
    assert "## Acceptance Criteria" in captured.out
    assert "agent-ready" in captured.out


def test_invalid_candidate_type_is_reported() -> None:
    packet = promoter.example_packet()
    packet["routing"]["candidate_for"] = [{"not": "a string"}]

    errors = promoter.validate_packet(packet)

    assert "routing.candidate_for must contain only strings" in errors


def test_evidence_requires_command_path_or_url_anchor() -> None:
    packet = promoter.example_packet()
    del packet["evidence"][0]["command"]

    errors = promoter.validate_packet(packet)

    assert "evidence[1] must include at least one anchor: command, path, url" in errors


def test_human_gated_packet_is_not_agent_ready() -> None:
    packet = promoter.example_packet()
    packet["routing"]["needs_human_decision"] = True
    packet["routing"]["blocked_by"] = ["RJ venue decision"]

    labels = promoter.packet_labels(packet)

    assert "agent-ready" not in labels
    assert "needs-human-decision" in labels
