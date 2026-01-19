"""Integration tests for validation failure escalation metrics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.app.container import (
    ApplicationContainer,
    clear_application_container,
    set_application_container,
)
from gpt_trader.features.live_trade.engines.base import CoordinatorContext
from gpt_trader.features.live_trade.engines.strategy import TradingEngine
from gpt_trader.features.live_trade.risk import LiveRiskManager

pytestmark = pytest.mark.integration


class TestValidationEscalationWithMetrics:
    """Integration tests verifying metrics are recorded during escalation."""

    @pytest.fixture
    def mock_broker(self) -> MagicMock:
        """Create a mock broker."""
        broker = MagicMock()
        broker.list_balances.return_value = []
        broker.get_product.return_value = MagicMock()
        return broker

    @pytest.fixture
    def bot_config(self) -> BotConfig:
        """Create a basic bot config."""
        return BotConfig(symbols=["BTC-USD"])

    @patch("gpt_trader.features.live_trade.execution.validation.record_counter")
    def test_metrics_recorded_during_escalation(
        self,
        mock_record_counter: MagicMock,
        mock_broker: MagicMock,
        bot_config: BotConfig,
    ) -> None:
        """Test that metrics are recorded when escalation occurs."""
        risk_manager = LiveRiskManager()
        risk_manager.set_reduce_only_mode(False, reason="test_setup")

        container = ApplicationContainer(bot_config)
        set_application_container(container)
        try:
            context = CoordinatorContext(
                config=bot_config,
                container=container,
                broker=mock_broker,
                risk_manager=risk_manager,
                event_store=container.event_store,
                bot_id="validation_escalation",
            )
            engine = TradingEngine(context)
            validator = engine._order_validator
            assert validator is not None

            with patch.object(
                risk_manager,
                "check_mark_staleness",
                side_effect=RuntimeError("API failure"),
            ):
                for _ in range(5):
                    validator.ensure_mark_is_fresh("BTC-USD")
        finally:
            clear_application_container()

        mock_record_counter.assert_called()
