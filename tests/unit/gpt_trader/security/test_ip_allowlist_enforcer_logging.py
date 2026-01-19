"""Tests for IPAllowlistEnforcer validation logging."""

import pytest

from gpt_trader.security.ip_allowlist_enforcer import IPAllowlistEnforcer


@pytest.fixture
def enforcer(monkeypatch):
    """IP Allowlist Enforcer instance."""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    monkeypatch.setenv("IP_ALLOWLIST_COINBASE_INTX", "192.168.1.1,10.0.0.0/24")
    return IPAllowlistEnforcer(enable_enforcement=True)


def test_validation_logging(enforcer):
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


def test_clear_validation_log(enforcer):
    """Test clearing validation log."""
    enforcer.add_rule("test_service", ["192.168.1.1"])
    enforcer.validate_ip("192.168.1.1", "test_service")

    enforcer.clear_validation_log()

    log = enforcer.get_validation_log()
    assert len(log) == 0


def test_validation_log_trimming(monkeypatch):
    """Test that validation log is trimmed when exceeding max size."""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    enforcer = IPAllowlistEnforcer(enable_enforcement=True)
    enforcer._max_log_size = 5  # Small size for testing
    enforcer.add_rule("test_service", ["192.168.1.1"])

    for i in range(10):
        enforcer.validate_ip(f"192.168.{i}.1", "test_service")

    log = enforcer.get_validation_log(limit=100)
    assert len(log) <= 5
