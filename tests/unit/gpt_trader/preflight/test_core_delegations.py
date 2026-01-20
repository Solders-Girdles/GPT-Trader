"""Tests for PreflightCheck method delegation wiring."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

import gpt_trader.preflight.core as preflight_core_module
from gpt_trader.preflight.core import PreflightCheck


class TestPreflightCheckDelegations:
    """Test that check methods delegate to module functions."""

    @pytest.mark.parametrize(
        ("method_name", "function_name", "return_value"),
        [
            ("check_python_version", "check_python_version", True),
            ("check_dependencies", "check_dependencies", True),
            ("check_environment_variables", "check_environment_variables", False),
            ("check_api_connectivity", "check_api_connectivity", True),
            ("check_key_permissions", "check_key_permissions", True),
            ("check_risk_configuration", "check_risk_configuration", True),
            ("check_pretrade_diagnostics", "check_pretrade_diagnostics", True),
            ("check_readiness_report", "check_readiness_report", True),
            ("check_event_store_redaction", "check_event_store_redaction", True),
            ("check_test_suite", "check_test_suite", True),
            ("check_profile_configuration", "check_profile_configuration", True),
            ("check_system_time", "check_system_time", True),
            ("check_disk_space", "check_disk_space", True),
            ("simulate_dry_run", "simulate_dry_run", True),
            ("generate_report", "generate_report", (True, "READY")),
        ],
    )
    def test_delegations_call_module_function(
        self,
        method_name: str,
        function_name: str,
        return_value: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        check = PreflightCheck()

        mock = Mock(return_value=return_value)
        monkeypatch.setattr(preflight_core_module, function_name, mock)
        result = getattr(check, method_name)()

        mock.assert_called_once_with(check)
        assert result == return_value
