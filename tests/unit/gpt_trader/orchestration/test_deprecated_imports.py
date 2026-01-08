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


class TestCanonicalImports:
    """Tests that canonical import paths work without warnings."""

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
