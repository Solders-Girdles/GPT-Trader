from __future__ import annotations

from typing import Any

from scripts.agents import quality_gate


def _passing_result(name: str) -> quality_gate.CheckResult:
    return quality_gate.CheckResult(
        name=name,
        passed=True,
        duration_seconds=0.01,
        exit_code=0,
        summary=f"{name} passed",
    )


def test_parse_import_boundary_output() -> None:
    output = "\n".join(
        [
            "ERRORS:",
            "  ERROR src/gpt_trader/features/example.py:24: imports "
            "gpt_trader.cli.widgets.Widget (rule: features_no_entrypoint_imports)",
            "     Feature slices must not import entrypoint layers or the DI container.",
            "",
            "1 violation(s) found.",
        ]
    )

    findings = quality_gate.parse_import_boundary_output(output, "")

    assert findings == [
        {
            "file": "src/gpt_trader/features/example.py",
            "line": 24,
            "message": (
                "imports gpt_trader.cli.widgets.Widget " "(rule: features_no_entrypoint_imports)"
            ),
            "severity": "error",
        }
    ]


def test_default_quality_gate_runs_boundary_check(monkeypatch) -> None:
    calls: list[tuple[str, Any]] = []

    def _record(name: str):
        def _runner(*args: Any, **kwargs: Any) -> quality_gate.CheckResult:
            calls.append((name, args or kwargs))
            return _passing_result(name)

        return _runner

    monkeypatch.setattr(quality_gate, "run_lint_check", _record("lint"))
    monkeypatch.setattr(quality_gate, "run_format_check", _record("format"))
    monkeypatch.setattr(quality_gate, "run_type_check", _record("types"))
    monkeypatch.setattr(quality_gate, "run_import_boundary_check", _record("boundaries"))
    monkeypatch.setattr(quality_gate, "run_test_check", _record("tests"))

    report = quality_gate.run_quality_gate()

    assert report["success"] is True
    assert [result["name"] for result in report["results"]] == [
        "lint",
        "format",
        "types",
        "boundaries",
        "tests",
    ]
    assert [name for name, _ in calls] == ["lint", "format", "types", "boundaries", "tests"]
