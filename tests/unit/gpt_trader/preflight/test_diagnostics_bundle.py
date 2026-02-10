from __future__ import annotations

from gpt_trader.preflight.diagnostics_bundle import _format_readiness_payload


def _make_result(
    status: str, message: str, details: dict[str, object] | None = None
) -> dict[str, object]:
    return {"status": status, "message": message, "details": details or {}}


def test_readiness_summary_empty() -> None:
    payload = _format_readiness_payload([])

    assert payload["status"] == "UNKNOWN"
    assert payload["counts"]["total"] == 0
    assert payload["checks"] == []


def test_readiness_summary_partial() -> None:
    payload = _format_readiness_payload(
        [
            _make_result("warn", "slow response"),
            _make_result("fail", "api down", {"errors": ["timeout"]}),
        ]
    )

    assert payload["status"] == "NOT READY"
    assert payload["counts"]["warn"] == 1
    assert payload["counts"]["fail"] == 1
    assert payload["counts"]["total"] == 2
    assert payload["checks"]


def test_readiness_summary_all_passes() -> None:
    payload = _format_readiness_payload(
        [_make_result("pass", "ok"), _make_result("pass", "still ok")]
    )

    assert payload["status"] == "READY"
    assert payload["counts"]["pass"] == 2
    assert payload["counts"]["total"] == 2
    assert payload["message"].startswith("System is READY")
