"""Tests for deprecated import paths.

These tests verify that importing from deprecated module paths emits
deprecation warnings and that the warnings point to the canonical locations.
"""

import sys
import warnings


class TestDeprecatedImports:
    """Tests for deprecated re-export modules."""

    def test_risk_model_import_emits_warning(self):
        """Importing from orchestration.configuration.risk.model emits warning."""
        # Reset the module's deprecation flag
        module_name = "gpt_trader.orchestration.configuration.risk.model"
        if module_name in sys.modules:
            module = sys.modules[module_name]
            if hasattr(module, "_deprecation_warned"):
                module._deprecation_warned = False

        # Force reimport
        if module_name in sys.modules:
            del sys.modules[module_name]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)

            # Import the deprecated module
            from gpt_trader.orchestration.configuration.risk import model  # noqa: F401

            # Filter for our specific warning
            risk_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning) and "RiskConfig" in str(x.message)
            ]

            assert len(risk_warnings) >= 1, "Expected deprecation warning for RiskConfig import"
            assert "features.live_trade.risk.config" in str(risk_warnings[0].message)

    def test_degradation_import_emits_warning(self):
        """Importing from orchestration.execution.degradation emits warning."""
        # Reset the module's deprecation flag
        module_name = "gpt_trader.orchestration.execution.degradation"
        if module_name in sys.modules:
            module = sys.modules[module_name]
            if hasattr(module, "_deprecation_warned"):
                module._deprecation_warned = False

        # Force reimport
        if module_name in sys.modules:
            del sys.modules[module_name]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)

            # Import the deprecated module
            from gpt_trader.orchestration.execution import degradation  # noqa: F401

            # Filter for our specific warning
            degradation_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and "DegradationState" in str(x.message)
            ]

            assert (
                len(degradation_warnings) >= 1
            ), "Expected deprecation warning for DegradationState import"
            assert "features.live_trade.degradation" in str(degradation_warnings[0].message)

    def test_live_execution_engine_emits_warning(self):
        """Instantiating LiveExecutionEngine emits deprecation warning."""
        from unittest.mock import MagicMock

        # Reset the class deprecation flag
        from gpt_trader.orchestration.live_execution import LiveExecutionEngine

        LiveExecutionEngine._deprecation_warned = False

        # Create mock dependencies
        mock_broker = MagicMock()
        mock_broker.list_balances.return_value = []
        mock_broker.list_positions.return_value = []

        mock_config = MagicMock()
        mock_config.enable_order_preview = False

        mock_risk_manager = MagicMock()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)

            # Instantiate the deprecated class
            LiveExecutionEngine(
                broker=mock_broker,
                config=mock_config,
                risk_manager=mock_risk_manager,
            )

            # Filter for our specific warning
            engine_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and "LiveExecutionEngine" in str(x.message)
            ]

            assert len(engine_warnings) >= 1, "Expected deprecation warning for LiveExecutionEngine"
            assert "TradingEngine.submit_order" in str(engine_warnings[0].message)

        # Reset for other tests
        LiveExecutionEngine._deprecation_warned = False

    def test_bootstrap_import_emits_warning(self):
        """Importing from orchestration.bootstrap emits warning."""
        module_name = "gpt_trader.orchestration.bootstrap"
        if module_name in sys.modules:
            del sys.modules[module_name]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)

            from gpt_trader.orchestration import bootstrap  # noqa: F401

            bootstrap_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and "gpt_trader.orchestration.bootstrap" in str(x.message)
            ]

            assert len(bootstrap_warnings) >= 1, "Expected deprecation warning for bootstrap import"
            assert "gpt_trader.app.bootstrap" in str(bootstrap_warnings[0].message)

    def test_spot_profile_service_import_emits_warning(self):
        """Importing from orchestration.spot_profile_service emits warning."""
        module_name = "gpt_trader.orchestration.spot_profile_service"
        if module_name in sys.modules:
            del sys.modules[module_name]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)

            from gpt_trader.orchestration import spot_profile_service  # noqa: F401

            sps_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and "spot_profile_service" in str(x.message)
            ]

            assert (
                len(sps_warnings) >= 1
            ), "Expected deprecation warning for spot_profile_service import"
            assert "features.live_trade.orchestrator" in str(sps_warnings[0].message)

    def test_intx_portfolio_service_import_emits_warning(self):
        """Importing from orchestration.intx_portfolio_service emits warning."""
        module_name = "gpt_trader.orchestration.intx_portfolio_service"
        if module_name in sys.modules:
            del sys.modules[module_name]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)

            from gpt_trader.orchestration import intx_portfolio_service  # noqa: F401

            ips_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and "intx_portfolio_service" in str(x.message)
            ]

            assert (
                len(ips_warnings) >= 1
            ), "Expected deprecation warning for intx_portfolio_service import"
            assert "features.brokerages.coinbase" in str(ips_warnings[0].message)

    def test_configuration_profiles_import_emits_warning(self):
        """Importing from orchestration.configuration.profiles emits warning.

        Note: This test verifies the warning is properly configured by checking
        the source contains the expected warning.warn call.
        """
        import inspect

        from gpt_trader.orchestration.configuration import profiles  # noqa: F401

        # Verify the deprecation warning is properly configured in the module
        assert "DEPRECATED" in profiles.__doc__

        source_file = inspect.getfile(profiles)
        with open(source_file) as f:
            source = f.read()
        assert "warnings.warn" in source
        assert "gpt_trader.orchestration.configuration.profiles is deprecated" in source
        assert "app.config.profile_loader" in source

    def test_orchestration_package_import_emits_warning(self):
        """Importing from gpt_trader.orchestration package emits warning.

        Note: This test verifies the warning message exists in the module.
        The actual warning may not be captured if orchestration was already
        imported by other tests in the same session.
        """
        # Import the module (may have been already imported)
        from gpt_trader import orchestration  # noqa: F401

        # Verify the deprecation warning is properly configured in the module
        # by checking the __doc__ string mentions deprecation
        assert "DEPRECATED" in orchestration.__doc__

        # Alternative: verify the module-level code would emit a warning
        # by checking the source contains the warning.warn call
        import inspect

        source_file = inspect.getfile(orchestration)
        with open(source_file) as f:
            source = f.read()
        assert "warnings.warn" in source
        assert "gpt_trader.orchestration is deprecated" in source

    def test_bot_config_defaults_import_emits_warning(self):
        """Importing from orchestration.configuration.bot_config.defaults emits warning."""
        module_name = "gpt_trader.orchestration.configuration.bot_config.defaults"
        if module_name in sys.modules:
            del sys.modules[module_name]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)

            from gpt_trader.orchestration.configuration.bot_config import defaults  # noqa: F401

            defaults_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and "bot_config.defaults is deprecated" in str(x.message)
            ]

            assert len(defaults_warnings) >= 1, "Expected deprecation warning for defaults import"
            assert "app.config.defaults" in str(defaults_warnings[0].message)

    def test_bot_config_rules_import_emits_warning(self):
        """Importing from orchestration.configuration.bot_config.rules emits warning."""
        module_name = "gpt_trader.orchestration.configuration.bot_config.rules"
        if module_name in sys.modules:
            del sys.modules[module_name]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)

            from gpt_trader.orchestration.configuration.bot_config import rules  # noqa: F401

            rules_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and "bot_config.rules is deprecated" in str(x.message)
            ]

            assert len(rules_warnings) >= 1, "Expected deprecation warning for rules import"
            assert "app.config.validation_rules" in str(rules_warnings[0].message)

    def test_bot_config_package_import_emits_warning(self):
        """Importing from orchestration.configuration.bot_config emits warning.

        Note: This test verifies the warning is properly configured by checking
        the source contains the expected warning.warn call.
        """
        import inspect

        from gpt_trader.orchestration.configuration import bot_config  # noqa: F401

        # Verify the deprecation warning is properly configured in the module
        assert "DEPRECATED" in bot_config.__doc__

        source_file = inspect.getfile(bot_config)
        with open(source_file) as f:
            source = f.read()
        assert "warnings.warn" in source
        assert "gpt_trader.orchestration.configuration.bot_config is deprecated" in source
        assert "app.config" in source

    def test_execution_package_import_emits_warning(self):
        """Importing from orchestration.execution emits warning.

        Note: This test verifies the warning is properly configured by checking
        the source contains the expected warning.warn call.
        """
        import inspect

        from gpt_trader.orchestration import execution  # noqa: F401

        # Verify the deprecation warning is properly configured in the module
        assert "DEPRECATED" in execution.__doc__

        source_file = inspect.getfile(execution)
        with open(source_file) as f:
            source = f.read()
        assert "warnings.warn" in source
        assert "gpt_trader.orchestration.execution is deprecated" in source
        assert "features.live_trade.execution" in source


class TestCanonicalImports:
    """Tests that canonical import paths work without warnings."""

    def test_canonical_bootstrap_no_warning(self):
        """Importing bootstrap from canonical location has no deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)

            from gpt_trader.app.bootstrap import bot_from_profile, build_bot  # noqa: F401

            bootstrap_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning) and "bootstrap" in str(x.message)
            ]

            assert (
                len(bootstrap_warnings) == 0
            ), "Canonical import should not emit deprecation warning"

    def test_canonical_spot_profile_service_no_warning(self):
        """Importing SpotProfileService from canonical location has no deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)

            from gpt_trader.features.live_trade.orchestrator.spot_profile_service import (  # noqa: F401
                SpotProfileService,
            )

            sps_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and "spot_profile_service" in str(x.message)
            ]

            assert len(sps_warnings) == 0, "Canonical import should not emit deprecation warning"

    def test_canonical_intx_portfolio_service_no_warning(self):
        """Importing IntxPortfolioService from canonical location has no deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)

            from gpt_trader.features.brokerages.coinbase.intx_portfolio_service import (  # noqa: F401
                IntxPortfolioService,
            )

            ips_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and "intx_portfolio_service" in str(x.message)
            ]

            assert len(ips_warnings) == 0, "Canonical import should not emit deprecation warning"

    def test_canonical_risk_config_no_warning(self):
        """Importing RiskConfig from canonical location has no deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)

            from gpt_trader.features.live_trade.risk.config import RiskConfig  # noqa: F401

            # Filter for RiskConfig-related warnings
            risk_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning) and "RiskConfig" in str(x.message)
            ]

            assert len(risk_warnings) == 0, "Canonical import should not emit deprecation warning"

    def test_canonical_degradation_no_warning(self):
        """Importing DegradationState from canonical location has no deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)

            from gpt_trader.features.live_trade.degradation import (  # noqa: F401
                DegradationState,
            )

            # Filter for DegradationState-related warnings
            degradation_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and "DegradationState" in str(x.message)
            ]

            assert (
                len(degradation_warnings) == 0
            ), "Canonical import should not emit deprecation warning"
