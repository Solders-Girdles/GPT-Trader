"""Tests for TradingBot delegation to broker and engine components."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

import gpt_trader.features.live_trade.bot as bot_module
from gpt_trader.features.live_trade.bot import TradingBot


@pytest.fixture
def mock_engine_class(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock_engine_class = Mock()
    monkeypatch.setattr(bot_module, "TradingEngine", mock_engine_class)
    return mock_engine_class


class TestTradingBotExecuteDecision:
    """Test TradingBot execute_decision method."""

    @pytest.fixture
    def bot(self, mock_engine_class: Mock) -> TradingBot:
        """Create a TradingBot with mocked engine."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60
        mock_container = Mock()

        engine = Mock()
        engine.execute_decision = Mock(return_value="order-123")
        mock_engine_class.return_value = engine
        bot = TradingBot(config=config, container=mock_container)

        return bot

    def test_execute_decision_delegates_to_engine(self, bot: TradingBot) -> None:
        """Test that execute_decision delegates to engine."""
        decision = Mock()
        mark = Mock()
        product = Mock()
        position_state = Mock()

        result = bot.execute_decision(
            symbol="BTC-PERP-USDC",
            decision=decision,
            mark=mark,
            product=product,
            position_state=position_state,
        )

        assert result == "order-123"
        bot.engine.execute_decision.assert_called_once_with(
            "BTC-PERP-USDC", decision, mark, product, position_state
        )

    def test_execute_decision_returns_none_when_engine_lacks_method(
        self,
        mock_engine_class: Mock,
    ) -> None:
        """Test execute_decision returns None if engine lacks the method."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60
        mock_container = Mock()

        engine = Mock(spec=[])  # No execute_decision
        mock_engine_class.return_value = engine
        bot = TradingBot(config=config, container=mock_container)

        result = bot.execute_decision(
            symbol="BTC-PERP-USDC",
            decision=Mock(),
        )

        assert result is None

    def test_execute_decision_with_minimal_args(self, bot: TradingBot) -> None:
        """Test execute_decision with minimal arguments."""
        decision = Mock()

        result = bot.execute_decision(
            symbol="BTC-PERP-USDC",
            decision=decision,
        )

        assert result == "order-123"
        bot.engine.execute_decision.assert_called_once_with(
            "BTC-PERP-USDC", decision, None, None, None
        )


class TestTradingBotGetProduct:
    """Test TradingBot get_product method."""

    def test_get_product_with_broker(self, mock_engine_class: Mock) -> None:
        """Test get_product when broker is available."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60
        mock_container = Mock()

        mock_product = Mock()
        mock_container.broker = Mock()
        mock_container.broker.get_product = Mock(return_value=mock_product)
        mock_container.risk_manager = Mock()
        mock_container.event_store = Mock()
        mock_container.notification_service = Mock()

        mock_engine_class.return_value = Mock()
        bot = TradingBot(config=config, container=mock_container)

        result = bot.get_product("BTC-PERP-USDC")

        assert result is mock_product
        mock_container.broker.get_product.assert_called_once_with("BTC-PERP-USDC")

    def test_get_product_without_broker(self, mock_engine_class: Mock) -> None:
        """Test get_product when broker is None."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60
        mock_container = Mock()

        # Set container.broker to None
        mock_container.broker = None
        mock_container.risk_manager = Mock()
        mock_container.event_store = Mock()
        mock_container.notification_service = Mock()

        mock_engine_class.return_value = Mock()
        bot = TradingBot(config=config, container=mock_container)

        result = bot.get_product("BTC-PERP-USDC")

        assert result is None

    def test_get_product_broker_without_method(self, mock_engine_class: Mock) -> None:
        """Test get_product when broker lacks get_product method."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60
        mock_container = Mock()

        mock_container.broker = Mock(spec=[])  # No get_product
        mock_container.risk_manager = Mock()
        mock_container.event_store = Mock()
        mock_container.notification_service = Mock()

        mock_engine_class.return_value = Mock()
        bot = TradingBot(config=config, container=mock_container)

        result = bot.get_product("BTC-PERP-USDC")

        assert result is None
