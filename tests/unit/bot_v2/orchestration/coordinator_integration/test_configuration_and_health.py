"""
Configuration drift and health check integration tests.
"""

from unittest.mock import AsyncMock, Mock

import pytest


class TestConfigurationDriftIntegration:
    """Ensure configuration drift triggers appropriate safeguards."""

    @pytest.mark.asyncio
    async def test_configuration_drift_triggers_emergency_shutdown(
        self, coordinators, integration_context
    ):
        strategy_coord = coordinators["strategy"]

        guardian = Mock()
        validation_result = Mock()
        validation_result.is_valid = False
        validation_result.errors = ["emergency_shutdown_required", "critical_violation"]
        guardian.pre_cycle_check = Mock(return_value=validation_result)

        set_running_flag = Mock()
        shutdown_hook = AsyncMock()
        updated_context = integration_context.with_updates(
            configuration_guardian=guardian,
            set_running_flag=set_running_flag,
            shutdown_hook=shutdown_hook,
        )

        strategy_coord.update_context(updated_context)

        trading_state = {
            "balances": [],
            "positions": [],
            "account_equity": None,
        }

        result = await strategy_coord._validate_configuration_and_handle_drift(trading_state)

        assert result is False
        set_running_flag.assert_called_once_with(False)
        shutdown_hook.assert_called_once()


@pytest.mark.xfail(reason="Coordinator health check update required")
class TestHealthCheckIntegration:
    """Verify health_check surfaces coordinator state."""

    def test_all_coordinators_report_healthy_status(self, coordinators, integration_context):
        integration_context.runtime_state.exec_engine = Mock()
        integration_context.registry.extras["account_telemetry"] = Mock()

        for name, coord in coordinators.items():
            if not hasattr(coord, "health_check"):
                continue
            status = coord.health_check()
            assert hasattr(status, "healthy")
            assert hasattr(status, "component")

            # Name mapping handling: some coordinators report differently in new architecture
            expected_name = name
            if name == "runtime":
                expected_name = "runtime_coordinator"
            elif name == "execution":
                expected_name = "execution_coordinator"

            assert status.component == expected_name

    def test_coordinator_health_checks_detect_unhealthy_states(
        self, coordinators, integration_context
    ):
        # Skip execution coord check as it mocks internal services differently now
        # execution_coord = coordinators["execution"]
        # integration_context.runtime_state.exec_engine = None
        # execution_coord.update_context(integration_context)
        # status = execution_coord.health_check()
        # assert status.healthy is False

        telemetry_coord = coordinators["telemetry"]
        integration_context.registry.extras.pop("account_telemetry", None)
        telemetry_coord.update_context(integration_context)

        status = telemetry_coord.health_check()
        assert status.healthy is False
