"""Edge-case tests for configuration guardian detector."""

from __future__ import annotations

from unittest.mock import patch

from gpt_trader.monitoring.configuration_guardian.detector import (
    ConfigurationGuardianDetector,
)
from gpt_trader.monitoring.configuration_guardian.models import ConfigurationIssue


class _Validator:
    def __init__(self, issues: list[ConfigurationIssue] | None = None, *, raise_exc: bool = False):
        self._issues = issues or []
        self._raise_exc = raise_exc
        self.name = "stub_validator"

    def validate(self) -> list[ConfigurationIssue]:
        if self._raise_exc:
            raise RuntimeError("validator failed")
        return self._issues


def test_configuration_issue_serialization_and_repr() -> None:
    issue = ConfigurationIssue(
        category="env",
        severity="high",
        recommendation="restart",
        message="Missing required env var",
        details={"field": "COINBASE_ENABLE_DERIVATIVES"},
    )

    payload = issue.to_dict()

    assert payload["category"] == "env"
    assert payload["severity"] == "high"
    assert payload["recommendation"] == "restart"
    assert "ConfigurationIssue" in repr(issue)
    assert "category='env'" in repr(issue)


def test_detector_returns_empty_when_no_validators_fail() -> None:
    detector = ConfigurationGuardianDetector(validators=[_Validator()])

    assert detector.detect() == []


def test_detector_continues_when_validator_raises() -> None:
    issue = ConfigurationIssue(
        category="config",
        severity="warning",
        recommendation="review",
        message="Config mismatch",
    )
    detector = ConfigurationGuardianDetector(
        validators=[_Validator(raise_exc=True), _Validator([issue])]
    )

    with patch("gpt_trader.monitoring.configuration_guardian.detector.logger") as mock_logger:
        issues = detector.detect()

    assert issues == [issue]
    mock_logger.warning.assert_called_once()


def test_response_payload_includes_required_fields() -> None:
    issue = ConfigurationIssue(
        category="runtime",
        severity="critical",
        recommendation="rollback",
    )
    detector = ConfigurationGuardianDetector()

    payload = detector.build_response([issue])

    issue_payload = payload["issues"][0]
    assert "category" in issue_payload
    assert "severity" in issue_payload
    assert "recommendation" in issue_payload
