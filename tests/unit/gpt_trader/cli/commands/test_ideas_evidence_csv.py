from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from tests.unit.gpt_trader.cli.commands.test_ideas_evidence_exports import (
    _idea,
    _root_args,
    _run_json,
    _run_text,
    _seed_evidence_store,
)


def test_audit_list_filters_paginates_and_exports_csv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "ideas"
    _seed_evidence_store(root)

    exit_code, response = _run_json(
        capsys,
        [
            "ideas",
            "audit",
            "list",
            *_root_args(root),
            "--actor",
            "rj",
            "--action",
            "approved",
            "--state",
            "approved",
            "--limit",
            "1",
            "--offset",
            "0",
        ],
    )

    assert exit_code == 0
    data = response["data"]
    assert data["schema_version"] == "gpt-trader.trade_ideas.audit_export.v1"
    assert data["pagination"] == {
        "total_count": 1,
        "returned_count": 1,
        "limit": 1,
        "offset": 0,
        "next_offset": None,
    }
    assert data["events"][0]["decision_id"] == "trade-evidence-filled"
    assert data["events"][0]["actor_id"] == "rj"

    exit_code, csv_output = _run_text(
        capsys,
        [
            "ideas",
            "audit",
            "export",
            *_root_args(root, output_format="csv"),
            "--decision-id",
            "trade-evidence-filled",
            "--action",
            "filled",
        ],
    )

    assert exit_code == 0
    rows = list(csv.DictReader(io.StringIO(csv_output)))
    assert [row["action"] for row in rows] == ["filled"]
    assert rows[0]["decision_id"] == "trade-evidence-filled"

    exit_code, csv_output = _run_text(
        capsys,
        [
            "ideas",
            "audit",
            "export",
            *_root_args(root, output_format="csv"),
            "--decision-id",
            "trade-evidence-filled",
            "--action",
            "proposed",
        ],
    )

    assert exit_code == 0
    rows = list(csv.DictReader(io.StringIO(csv_output)))
    assert len(rows) == 1
    assert rows[0]["reason"] == "'=formula-style proposal reason"
    assert json.loads(rows[0]["evidence"]) == [
        "'+proposal evidence",
        "'-proposal evidence",
        "'@proposal evidence",
        "'\tproposal evidence",
        "'\rproposal evidence",
    ]


def test_audit_csv_output_dir_names_include_content_digest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "ideas"
    service = _seed_evidence_store(root)
    output_dir = tmp_path / "artifacts"

    exit_code, _csv_output = _run_text(
        capsys,
        [
            "ideas",
            "audit",
            "export",
            *_root_args(root, output_format="csv"),
            "--actor",
            "generator-a",
            "--action",
            "proposed",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert exit_code == 0
    first_paths = sorted(output_dir.glob("trade-idea-audit-audit-*.csv"))
    assert len(first_paths) == 1
    first_path = first_paths[0]
    first_content = first_path.read_text(encoding="utf-8")

    service.propose(
        _idea("trade-evidence-second"),
        actor_id="generator-a",
        reason="Second matching proposal",
    )

    exit_code, _csv_output = _run_text(
        capsys,
        [
            "ideas",
            "audit",
            "export",
            *_root_args(root, output_format="csv"),
            "--actor",
            "generator-a",
            "--action",
            "proposed",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert exit_code == 0
    paths = sorted(output_dir.glob("trade-idea-audit-audit-*.csv"))
    assert len(paths) == 2
    assert first_path in paths
    assert first_path.read_text(encoding="utf-8") == first_content
    assert any(
        path != first_path and path.read_text(encoding="utf-8") != first_content for path in paths
    )
