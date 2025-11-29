"""Tests for IP Allowlist Enforcer"""

import pytest

from gpt_trader.security.ip_allowlist_enforcer import (
    IPAllowlistEnforcer,
)


@pytest.fixture
def enforcer(monkeypatch):
    """IP Allowlist Enforcer instance"""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    monkeypatch.setenv("IP_ALLOWLIST_COINBASE_INTX", "192.168.1.1,10.0.0.0/24")
    return IPAllowlistEnforcer(enable_enforcement=True)


def test_load_rules_from_environment(enforcer):
    """Test loading rules from environment variables"""
    rules = enforcer.list_rules()
    assert len(rules) > 0

    coinbase_rule = enforcer.get_rule("coinbase_intx")
    assert coinbase_rule is not None
    assert "192.168.1.1" in coinbase_rule.allowed_ips
    assert "10.0.0.0/24" in coinbase_rule.allowed_ips


def test_add_rule(enforcer):
    """Test adding IP allowlist rule"""
    success = enforcer.add_rule(
        "coinbase_production",
        ["192.168.1.100", "10.0.0.0/8"],
        description="Production API server IPs",
    )

    assert success

    rule = enforcer.get_rule("coinbase_production")
    assert rule is not None
    assert rule.service_name == "coinbase_production"
    assert "192.168.1.100" in rule.allowed_ips
    assert "10.0.0.0/8" in rule.allowed_ips
    assert rule.description == "Production API server IPs"


def test_add_rule_invalid_ip(enforcer):
    """Test adding rule with invalid IP"""
    success = enforcer.add_rule(
        "test_service",
        ["192.168.1.1", "invalid-ip"],
    )

    assert not success


def test_validate_ip_exact_match(enforcer):
    """Test IP validation with exact match"""
    enforcer.add_rule("test_service", ["192.168.1.1", "192.168.1.2"])

    result = enforcer.validate_ip("192.168.1.1", "test_service")

    assert result.is_allowed
    assert result.matched_rule == "192.168.1.1"


def test_validate_ip_cidr_match(enforcer):
    """Test IP validation with CIDR match"""
    enforcer.add_rule("test_service", ["10.0.0.0/24"])

    result = enforcer.validate_ip("10.0.0.50", "test_service")

    assert result.is_allowed
    assert result.matched_rule == "10.0.0.0/24"


def test_validate_ip_not_in_allowlist(enforcer):
    """Test IP validation rejection"""
    enforcer.add_rule("test_service", ["192.168.1.1"])

    result = enforcer.validate_ip("10.0.0.1", "test_service")

    assert not result.is_allowed
    assert result.matched_rule is None
    assert "not in allowlist" in result.reason


def test_validate_ip_no_rule(enforcer):
    """Test IP validation with no rule configured"""
    result = enforcer.validate_ip("192.168.1.1", "nonexistent_service")

    assert not result.is_allowed
    assert "No IP allowlist rule" in result.reason


def test_validate_ip_disabled_rule(enforcer):
    """Test IP validation with disabled rule"""
    enforcer.add_rule("test_service", ["192.168.1.1"])
    enforcer.disable_rule("test_service")

    result = enforcer.validate_ip("192.168.1.1", "test_service")

    assert not result.is_allowed
    assert "disabled" in result.reason


def test_enable_disable_rule(enforcer):
    """Test enabling and disabling rules"""
    enforcer.add_rule("test_service", ["192.168.1.1"])

    # Disable
    success = enforcer.disable_rule("test_service")
    assert success

    rule = enforcer.get_rule("test_service")
    assert not rule.enabled

    # Enable
    success = enforcer.enable_rule("test_service")
    assert success

    rule = enforcer.get_rule("test_service")
    assert rule.enabled


def test_remove_rule(enforcer):
    """Test removing rule"""
    enforcer.add_rule("test_service", ["192.168.1.1"])

    success = enforcer.remove_rule("test_service")
    assert success

    rule = enforcer.get_rule("test_service")
    assert rule is None


def test_enforcement_disabled(monkeypatch):
    """Test with enforcement disabled"""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "0")

    enforcer = IPAllowlistEnforcer(enable_enforcement=False)

    # Should allow all IPs when enforcement is disabled
    result = enforcer.validate_ip("any-ip", "any-service")
    assert result.is_allowed
    assert "disabled" in result.reason


def test_validation_logging(enforcer):
    """Test validation logging"""
    enforcer.add_rule("test_service", ["192.168.1.1"])

    # Generate some validations
    enforcer.validate_ip("192.168.1.1", "test_service")
    enforcer.validate_ip("10.0.0.1", "test_service")

    log = enforcer.get_validation_log(limit=10)
    assert len(log) == 2

    # Check log entries
    assert log[0]["client_ip"] == "192.168.1.1"
    assert log[0]["is_allowed"] is True

    assert log[1]["client_ip"] == "10.0.0.1"
    assert log[1]["is_allowed"] is False


def test_clear_validation_log(enforcer):
    """Test clearing validation log"""
    enforcer.add_rule("test_service", ["192.168.1.1"])
    enforcer.validate_ip("192.168.1.1", "test_service")

    enforcer.clear_validation_log()

    log = enforcer.get_validation_log()
    assert len(log) == 0


def test_check_ip_cidr_edge_cases(enforcer):
    """Test CIDR edge cases"""
    # /32 CIDR (single IP)
    assert enforcer._check_ip_in_allowlist("192.168.1.1", ["192.168.1.1/32"]) == "192.168.1.1/32"

    # /0 CIDR (all IPs)
    assert enforcer._check_ip_in_allowlist("192.168.1.1", ["0.0.0.0/0"]) == "0.0.0.0/0"

    # Invalid IP
    assert enforcer._check_ip_in_allowlist("invalid-ip", ["192.168.1.0/24"]) is None


def test_list_rules(enforcer):
    """Test listing all rules"""
    enforcer.add_rule("service1", ["192.168.1.1"])
    enforcer.add_rule("service2", ["10.0.0.0/24"])

    rules = enforcer.list_rules()
    assert len(rules) >= 2

    service_names = {r.service_name for r in rules}
    assert "service1" in service_names
    assert "service2" in service_names


# ============================================================
# Additional tests for coverage gaps
# ============================================================


def test_update_existing_rule_with_description(enforcer):
    """Test updating existing rule updates description"""
    enforcer.add_rule("test_service", ["192.168.1.1"], description="Original")

    # Update with new IPs and description
    success = enforcer.add_rule("test_service", ["192.168.1.2"], description="Updated description")
    assert success

    rule = enforcer.get_rule("test_service")
    assert rule.description == "Updated description"
    assert "192.168.1.2" in rule.allowed_ips
    assert "192.168.1.1" not in rule.allowed_ips


def test_update_existing_rule_without_description(enforcer):
    """Test updating existing rule preserves description if not provided"""
    enforcer.add_rule("test_service", ["192.168.1.1"], description="Original")

    # Update without description
    success = enforcer.add_rule("test_service", ["192.168.1.2"])
    assert success

    rule = enforcer.get_rule("test_service")
    assert rule.description == "Original"  # Should be preserved


def test_enable_rule_nonexistent(monkeypatch):
    """Test enabling nonexistent rule returns False"""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    enforcer = IPAllowlistEnforcer(enable_enforcement=True)

    success = enforcer.enable_rule("nonexistent_service")
    assert not success


def test_disable_rule_nonexistent(monkeypatch):
    """Test disabling nonexistent rule returns False"""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    enforcer = IPAllowlistEnforcer(enable_enforcement=True)

    success = enforcer.disable_rule("nonexistent_service")
    assert not success


def test_remove_rule_nonexistent(monkeypatch):
    """Test removing nonexistent rule returns False"""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    enforcer = IPAllowlistEnforcer(enable_enforcement=True)

    success = enforcer.remove_rule("nonexistent_service")
    assert not success


def test_validation_log_trimming(monkeypatch):
    """Test that validation log is trimmed when exceeding max size"""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    enforcer = IPAllowlistEnforcer(enable_enforcement=True)
    enforcer._max_log_size = 5  # Small size for testing
    enforcer.add_rule("test_service", ["192.168.1.1"])

    # Generate more validations than max log size
    for i in range(10):
        enforcer.validate_ip(f"192.168.{i}.1", "test_service")

    log = enforcer.get_validation_log(limit=100)
    assert len(log) <= 5  # Should be trimmed


def test_validate_ip_without_logging(enforcer):
    """Test validation with log_validation=False"""
    enforcer.add_rule("test_service", ["192.168.1.1"])
    enforcer.clear_validation_log()

    # Validate with logging disabled
    result = enforcer.validate_ip("192.168.1.1", "test_service", log_validation=False)

    assert result.is_allowed
    log = enforcer.get_validation_log()
    assert len(log) == 0  # No log entry should be created


def test_validate_ip_no_rule_without_logging(enforcer):
    """Test validation with no rule and log_validation=False"""
    enforcer.clear_validation_log()

    result = enforcer.validate_ip("192.168.1.1", "unknown_service", log_validation=False)

    assert not result.is_allowed
    log = enforcer.get_validation_log()
    assert len(log) == 0


def test_validate_ip_disabled_rule_without_logging(enforcer):
    """Test validation with disabled rule and log_validation=False"""
    enforcer.add_rule("test_service", ["192.168.1.1"])
    enforcer.disable_rule("test_service")
    enforcer.clear_validation_log()

    result = enforcer.validate_ip("192.168.1.1", "test_service", log_validation=False)

    assert not result.is_allowed
    log = enforcer.get_validation_log()
    assert len(log) == 0


def test_check_ip_invalid_cidr_fallback(enforcer):
    """Test IP check with invalid CIDR that falls back to IP comparison"""
    # This tests the fallback path when a value is not a valid CIDR
    # but might still be an IP address
    result = enforcer._check_ip_in_allowlist("192.168.1.1", ["not-a-cidr", "192.168.1.1"])
    assert result == "192.168.1.1"


def test_check_ip_completely_invalid_entry(enforcer):
    """Test IP check skips completely invalid entries"""
    result = enforcer._check_ip_in_allowlist("192.168.1.1", ["completely-invalid", "also-invalid"])
    assert result is None


def test_environment_empty_allowlist(monkeypatch):
    """Test that empty IP list from environment is skipped"""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    monkeypatch.setenv("IP_ALLOWLIST_EMPTY_SERVICE", "")  # Empty value
    monkeypatch.setenv("IP_ALLOWLIST_VALID_SERVICE", "192.168.1.1")
    enforcer = IPAllowlistEnforcer(enable_enforcement=True)

    # Empty service should not create a rule
    assert enforcer.get_rule("empty_service") is None
    # Valid service should have a rule
    assert enforcer.get_rule("valid_service") is not None


def test_global_singleton_and_convenience_functions():
    """Test global singleton and convenience functions"""
    from gpt_trader.security.ip_allowlist_enforcer import (
        add_ip_allowlist_rule,
        get_ip_allowlist_enforcer,
        validate_ip,
    )

    # Get enforcer (creates singleton if needed)
    enforcer = get_ip_allowlist_enforcer()
    assert enforcer is not None

    # Add rule via convenience function
    success = add_ip_allowlist_rule("convenience_test", ["192.168.1.1"])
    assert success

    # Validate via convenience function
    result = validate_ip("192.168.1.1", "convenience_test")
    assert result is not None
