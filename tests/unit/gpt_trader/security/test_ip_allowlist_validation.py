"""Tests for IPAllowlistEnforcer validation and logging."""

from __future__ import annotations

import pytest

from gpt_trader.security.ip_allowlist_enforcer import IPAllowlistEnforcer


def test_validate_ip_exact_match(enforcer: IPAllowlistEnforcer) -> None:
    """Test IP validation with exact match."""
    enforcer.add_rule("test_service", ["192.168.1.1", "192.168.1.2"])

    result = enforcer.validate_ip("192.168.1.1", "test_service")

    assert result.is_allowed
    assert result.matched_rule == "192.168.1.1"


def test_validate_ip_cidr_match(enforcer: IPAllowlistEnforcer) -> None:
    """Test IP validation with CIDR match."""
    enforcer.add_rule("test_service", ["10.0.0.0/24"])

    result = enforcer.validate_ip("10.0.0.50", "test_service")

    assert result.is_allowed
    assert result.matched_rule == "10.0.0.0/24"


def test_validate_ip_not_in_allowlist(enforcer: IPAllowlistEnforcer) -> None:
    """Test IP validation rejection."""
    enforcer.add_rule("test_service", ["192.168.1.1"])

    result = enforcer.validate_ip("10.0.0.1", "test_service")

    assert not result.is_allowed
    assert result.matched_rule is None
    assert "not in allowlist" in result.reason


def test_validate_ip_no_rule(enforcer: IPAllowlistEnforcer) -> None:
    """Test IP validation with no rule configured."""
    result = enforcer.validate_ip("192.168.1.1", "nonexistent_service")

    assert not result.is_allowed
    assert "No IP allowlist rule" in result.reason


def test_validate_ip_disabled_rule(enforcer: IPAllowlistEnforcer) -> None:
    """Test IP validation with disabled rule."""
    enforcer.add_rule("test_service", ["192.168.1.1"])
    enforcer.disable_rule("test_service")

    result = enforcer.validate_ip("192.168.1.1", "test_service")

    assert not result.is_allowed
    assert "disabled" in result.reason


def test_enforcement_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test with enforcement disabled."""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "0")

    enforcer = IPAllowlistEnforcer(enable_enforcement=False)

    result = enforcer.validate_ip("any-ip", "any-service")
    assert result.is_allowed
    assert "disabled" in result.reason


def test_validate_ip_without_logging(enforcer: IPAllowlistEnforcer) -> None:
    """Test validation with log_validation=False."""
    enforcer.add_rule("test_service", ["192.168.1.1"])
    enforcer.clear_validation_log()

    result = enforcer.validate_ip("192.168.1.1", "test_service", log_validation=False)

    assert result.is_allowed
    log = enforcer.get_validation_log()
    assert len(log) == 0


def test_validate_ip_no_rule_without_logging(enforcer: IPAllowlistEnforcer) -> None:
    """Test validation with no rule and log_validation=False."""
    enforcer.clear_validation_log()

    result = enforcer.validate_ip("192.168.1.1", "unknown_service", log_validation=False)

    assert not result.is_allowed
    log = enforcer.get_validation_log()
    assert len(log) == 0


def test_validate_ip_disabled_rule_without_logging(enforcer: IPAllowlistEnforcer) -> None:
    """Test validation with disabled rule and log_validation=False."""
    enforcer.add_rule("test_service", ["192.168.1.1"])
    enforcer.disable_rule("test_service")
    enforcer.clear_validation_log()

    result = enforcer.validate_ip("192.168.1.1", "test_service", log_validation=False)

    assert not result.is_allowed
    log = enforcer.get_validation_log()
    assert len(log) == 0


def test_validation_logging(enforcer: IPAllowlistEnforcer) -> None:
    """Test validation logging."""
    enforcer.add_rule("test_service", ["192.168.1.1"])

    enforcer.validate_ip("192.168.1.1", "test_service")
    enforcer.validate_ip("10.0.0.1", "test_service")

    log = enforcer.get_validation_log(limit=10)
    assert len(log) == 2

    assert log[0]["client_ip"] == "192.168.1.1"
    assert log[0]["is_allowed"] is True

    assert log[1]["client_ip"] == "10.0.0.1"
    assert log[1]["is_allowed"] is False


def test_clear_validation_log(enforcer: IPAllowlistEnforcer) -> None:
    """Test clearing validation log."""
    enforcer.add_rule("test_service", ["192.168.1.1"])
    enforcer.validate_ip("192.168.1.1", "test_service")

    enforcer.clear_validation_log()

    log = enforcer.get_validation_log()
    assert len(log) == 0


def test_validation_log_trimming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that validation log is trimmed when exceeding max size."""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    enforcer = IPAllowlistEnforcer(enable_enforcement=True)
    enforcer._max_log_size = 5  # Small size for testing
    enforcer.add_rule("test_service", ["192.168.1.1"])

    for i in range(10):
        enforcer.validate_ip(f"192.168.{i}.1", "test_service")

    log = enforcer.get_validation_log(limit=100)
    assert len(log) <= 5
