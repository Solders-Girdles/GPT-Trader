"""
IP Allowlist Enforcer for GPT-Trader

Enforces IP allowlisting for API keys to prevent unauthorized access.
Critical for INTX (Coinbase International Exchange) and production trading.
"""

from __future__ import annotations

import ipaddress
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="ip_allowlist")


@dataclass
class IPAllowlistRule:
    """IP allowlist rule with metadata"""

    service_name: str  # e.g., 'coinbase_intx', 'coinbase_production'
    allowed_ips: list[str]  # IP addresses or CIDR blocks
    created_at: datetime
    updated_at: datetime
    enabled: bool = True
    description: str | None = None


@dataclass
class IPValidationResult:
    """Result of IP validation"""

    is_allowed: bool
    client_ip: str
    service_name: str
    matched_rule: str | None = None  # The rule that matched (IP or CIDR)
    reason: str | None = None  # Reason for rejection if not allowed


class IPAllowlistEnforcer:
    """
    Enforces IP allowlisting for API keys.

    Features:
    - Per-service IP allowlisting
    - CIDR notation support
    - Thread-safe rule management
    - Audit logging of all validation attempts
    - Environment variable configuration
    """

    def __init__(
        self,
        *,
        enable_enforcement: bool = True,
    ) -> None:
        self._enable_enforcement = enable_enforcement
        self._lock = threading.RLock()
        self._rules: dict[str, IPAllowlistRule] = {}
        self._validation_log: list[dict[str, Any]] = []
        self._max_log_size = 1000

        # Load rules from environment if available
        self._load_rules_from_environment()

    def _load_rules_from_environment(self) -> None:
        """Load IP allowlist rules from environment variables"""
        # Check if enforcement is enabled
        enforcement_enabled = os.environ.get("IP_ALLOWLIST_ENABLED", "1")
        self._enable_enforcement = enforcement_enabled.lower() in ("1", "true", "yes")

        # Load service-specific allowlists
        # Format: IP_ALLOWLIST_<SERVICE>=ip1,ip2,cidr1,cidr2
        for key, value in os.environ.items():
            if key.startswith("IP_ALLOWLIST_") and key != "IP_ALLOWLIST_ENABLED":
                service_name = key.replace("IP_ALLOWLIST_", "").lower()
                allowed_ips = [ip.strip() for ip in value.split(",") if ip.strip()]

                if allowed_ips:
                    now = datetime.now(UTC)
                    rule = IPAllowlistRule(
                        service_name=service_name,
                        allowed_ips=allowed_ips,
                        created_at=now,
                        updated_at=now,
                        enabled=True,
                        description=f"Loaded from environment: {key}",
                    )
                    self._rules[service_name] = rule

                    logger.info(
                        f"Loaded IP allowlist for {service_name} from environment",
                        operation="load_rules",
                        service=service_name,
                        ip_count=len(allowed_ips),
                    )

    def add_rule(
        self,
        service_name: str,
        allowed_ips: list[str],
        *,
        description: str | None = None,
    ) -> bool:
        """
        Add or update IP allowlist rule.

        Args:
            service_name: Service identifier (e.g., 'coinbase_intx', 'coinbase_production')
            allowed_ips: List of allowed IP addresses or CIDR blocks
            description: Optional description of the rule

        Returns:
            Success status
        """
        with self._lock:
            # Validate all IPs/CIDR blocks
            for ip_or_cidr in allowed_ips:
                if not self._validate_ip_or_cidr(ip_or_cidr):
                    logger.error(
                        f"Invalid IP or CIDR: {ip_or_cidr}",
                        operation="add_rule",
                        service=service_name,
                        invalid_value=ip_or_cidr,
                    )
                    return False

            now = datetime.now(UTC)

            # Update existing rule or create new one
            if service_name in self._rules:
                rule = self._rules[service_name]
                rule.allowed_ips = allowed_ips
                rule.updated_at = now
                if description:
                    rule.description = description
            else:
                rule = IPAllowlistRule(
                    service_name=service_name,
                    allowed_ips=allowed_ips,
                    created_at=now,
                    updated_at=now,
                    enabled=True,
                    description=description,
                )
                self._rules[service_name] = rule

            logger.info(
                f"Added/updated IP allowlist rule for {service_name}",
                operation="add_rule",
                service=service_name,
                ip_count=len(allowed_ips),
            )

            return True

    def validate_ip(
        self,
        client_ip: str,
        service_name: str,
        *,
        log_validation: bool = True,
    ) -> IPValidationResult:
        """
        Validate if client IP is allowed for service.

        Args:
            client_ip: Client IP address to validate
            service_name: Service identifier
            log_validation: Whether to log this validation attempt

        Returns:
            IPValidationResult with validation outcome
        """
        with self._lock:
            # If enforcement is disabled, allow all
            if not self._enable_enforcement:
                return IPValidationResult(
                    is_allowed=True,
                    client_ip=client_ip,
                    service_name=service_name,
                    reason="IP allowlist enforcement is disabled",
                )

            # Check if rule exists for service
            rule = self._rules.get(service_name)
            if not rule:
                # No rule = deny by default in production
                result = IPValidationResult(
                    is_allowed=False,
                    client_ip=client_ip,
                    service_name=service_name,
                    reason=f"No IP allowlist rule configured for service: {service_name}",
                )

                if log_validation:
                    self._log_validation(result, rule_exists=False)

                logger.warning(
                    f"IP validation failed - no rule for {service_name}",
                    operation="validate_ip",
                    service=service_name,
                    client_ip=client_ip,
                )

                return result

            # Check if rule is enabled
            if not rule.enabled:
                result = IPValidationResult(
                    is_allowed=False,
                    client_ip=client_ip,
                    service_name=service_name,
                    reason=f"IP allowlist rule for {service_name} is disabled",
                )

                if log_validation:
                    self._log_validation(result, rule_exists=True, rule_enabled=False)

                return result

            # Validate IP against allowlist
            matched_rule = self._check_ip_in_allowlist(client_ip, rule.allowed_ips)

            if matched_rule:
                result = IPValidationResult(
                    is_allowed=True,
                    client_ip=client_ip,
                    service_name=service_name,
                    matched_rule=matched_rule,
                )

                logger.debug(
                    f"IP {client_ip} allowed for {service_name} (matched: {matched_rule})",
                    operation="validate_ip",
                    service=service_name,
                    client_ip=client_ip,
                    matched_rule=matched_rule,
                )
            else:
                result = IPValidationResult(
                    is_allowed=False,
                    client_ip=client_ip,
                    service_name=service_name,
                    reason=f"IP {client_ip} not in allowlist for {service_name}",
                )

                logger.warning(
                    f"IP {client_ip} REJECTED for {service_name} - not in allowlist",
                    operation="validate_ip",
                    service=service_name,
                    client_ip=client_ip,
                    allowed_ips=rule.allowed_ips,
                )

            if log_validation:
                self._log_validation(result, rule_exists=True, rule_enabled=True)

            return result

    def _check_ip_in_allowlist(self, client_ip: str, allowed_ips: list[str]) -> str | None:
        """
        Check if IP is in allowlist.

        Returns:
            The matched rule (IP or CIDR) if found, None otherwise
        """
        try:
            client_addr = ipaddress.ip_address(client_ip)

            for allowed in allowed_ips:
                # Try exact match first
                if allowed == client_ip:
                    return allowed

                # Try CIDR notation
                try:
                    network = ipaddress.ip_network(allowed, strict=False)
                    if client_addr in network:
                        return allowed
                except ValueError:
                    # Not a valid CIDR, try as plain IP
                    try:
                        allowed_addr = ipaddress.ip_address(allowed)
                        if client_addr == allowed_addr:
                            return allowed
                    except ValueError:
                        # Invalid format, skip
                        continue

            return None

        except ValueError:
            logger.warning(
                f"Invalid IP address format: {client_ip}",
                operation="check_ip",
                client_ip=client_ip,
            )
            return None

    def _validate_ip_or_cidr(self, value: str) -> bool:
        """Validate if value is a valid IP address or CIDR block"""
        try:
            # Try as IP address
            ipaddress.ip_address(value)
            return True
        except ValueError:
            pass

        try:
            # Try as CIDR network
            ipaddress.ip_network(value, strict=False)
            return True
        except ValueError:
            pass

        return False

    def _log_validation(
        self,
        result: IPValidationResult,
        *,
        rule_exists: bool = True,
        rule_enabled: bool = True,
    ) -> None:
        """Log validation attempt for audit trail"""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "client_ip": result.client_ip,
            "service_name": result.service_name,
            "is_allowed": result.is_allowed,
            "matched_rule": result.matched_rule,
            "reason": result.reason,
            "rule_exists": rule_exists,
            "rule_enabled": rule_enabled,
        }

        self._validation_log.append(log_entry)

        # Trim log if too large
        if len(self._validation_log) > self._max_log_size:
            self._validation_log = self._validation_log[-self._max_log_size :]

    def get_rule(self, service_name: str) -> IPAllowlistRule | None:
        """Get IP allowlist rule for service"""
        with self._lock:
            return self._rules.get(service_name)

    def list_rules(self) -> list[IPAllowlistRule]:
        """List all IP allowlist rules"""
        with self._lock:
            return list(self._rules.values())

    def enable_rule(self, service_name: str) -> bool:
        """Enable IP allowlist rule"""
        with self._lock:
            rule = self._rules.get(service_name)
            if not rule:
                return False

            rule.enabled = True
            rule.updated_at = datetime.now(UTC)

            logger.info(
                f"Enabled IP allowlist rule for {service_name}",
                operation="enable_rule",
                service=service_name,
            )

            return True

    def disable_rule(self, service_name: str) -> bool:
        """Disable IP allowlist rule"""
        with self._lock:
            rule = self._rules.get(service_name)
            if not rule:
                return False

            rule.enabled = False
            rule.updated_at = datetime.now(UTC)

            logger.warning(
                f"Disabled IP allowlist rule for {service_name}",
                operation="disable_rule",
                service=service_name,
            )

            return True

    def remove_rule(self, service_name: str) -> bool:
        """Remove IP allowlist rule"""
        with self._lock:
            if service_name not in self._rules:
                return False

            del self._rules[service_name]

            logger.warning(
                f"Removed IP allowlist rule for {service_name}",
                operation="remove_rule",
                service=service_name,
            )

            return True

    def get_validation_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent validation log entries"""
        with self._lock:
            return self._validation_log[-limit:]

    def clear_validation_log(self) -> None:
        """Clear validation log"""
        with self._lock:
            self._validation_log.clear()


# Global instance
_ip_allowlist_enforcer: IPAllowlistEnforcer | None = None


def get_ip_allowlist_enforcer() -> IPAllowlistEnforcer:
    """Get the global IP allowlist enforcer instance"""
    global _ip_allowlist_enforcer
    if _ip_allowlist_enforcer is None:
        _ip_allowlist_enforcer = IPAllowlistEnforcer()
    return _ip_allowlist_enforcer


# Convenience functions
def validate_ip(client_ip: str, service_name: str) -> IPValidationResult:
    """Validate if client IP is allowed for service"""
    return get_ip_allowlist_enforcer().validate_ip(client_ip, service_name)


def add_ip_allowlist_rule(
    service_name: str,
    allowed_ips: list[str],
    *,
    description: str | None = None,
) -> bool:
    """Add or update IP allowlist rule"""
    return get_ip_allowlist_enforcer().add_rule(service_name, allowed_ips, description=description)
