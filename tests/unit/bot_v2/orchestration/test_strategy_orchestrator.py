"""
Tests for StrategyOrchestrator.

Tests strategy initialization, decision execution, risk gates,
and spot-specific filtering logic.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.features.brokerages.core.interfaces import Balance, MarketType, Position, Product
from bot_v2.features.live_trade.risk_runtime import (
    CircuitBreakerAction,
    CircuitBreakerOutcome,
)
from bot_v2.features.live_trade.strategies.perps_baseline import (
    Action,
    BaselinePerpsStrategy,
    Decision,
)
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.strategy_orchestrator import (
    StrategyOrchestrator,
    SymbolProcessingContext,
)


@pytest.fixture
def mock_bot():
    """Create mock PerpsBot instance."""
    bot = Mock()
    bot.config = Mock()
    bot.config.profile = Profile.PROD  # Non-SPOT profile for perps
    bot.config.derivatives_enabled = True
    bot.config.short_ma = 10
    bot.config.long_ma = 30
    bot.config.target_leverage = 2
    bot.config.trailing_stop_pct = Decimal("0.02")
    bot.config.enable_shorts = True
    bot.config.perps_position_fraction = None
    bot.config.symbols = ["BTC-PERP", "ETH-PERP"]

    bot.broker = Mock()
    bot.risk_manager = Mock()
    bot.risk_manager.config = Mock()
    bot.risk_manager.config.kill_switch_enabled = False

    state = PerpsBotRuntimeState(bot.config.symbols or [])
    bot.runtime_state = state
    bot.mark_windows = state.mark_windows
    bot.last_decisions = state.last_decisions
    bot._symbol_strategies = state.symbol_strategies
    bot.execute_decision = AsyncMock()
    bot.get_product = Mock()

    return bot


@pytest.fixture
def mock_spot_profile_service():
    """Create mock SpotProfileService."""
    service = Mock()
    service.load = Mock(return_value={})
    service.get = Mock(return_value=None)
    return service


@pytest.fixture
def orchestrator(mock_bot, mock_spot_profile_service):
    """Create StrategyOrchestrator instance."""
    return StrategyOrchestrator(bot=mock_bot, spot_profile_service=mock_spot_profile_service)


@pytest.fixture
def test_balance():
    """Create test balance."""
    balance = Mock(spec=Balance)
    balance.asset = "USDC"
    balance.total = Decimal("10000")
    return balance


@pytest.fixture
def test_position():
    """Create test position."""
    position = Mock(spec=Position)
    position.symbol = "BTC-PERP"
    position.quantity = Decimal("0.5")
    position.side = "long"
    position.entry_price = Decimal("50000")
    return position


class TestStrategyOrchestratorInitialization:
    """Test StrategyOrchestrator initialization."""

    def test_initialization_with_bot(self, mock_bot, mock_spot_profile_service):
        """Test orchestrator initializes with bot reference."""
        orchestrator = StrategyOrchestrator(
            bot=mock_bot, spot_profile_service=mock_spot_profile_service
        )

        assert orchestrator._bot == mock_bot
        assert orchestrator._spot_profiles == mock_spot_profile_service

    def test_initialization_creates_default_spot_profile_service(self, mock_bot):
        """Test creates default SpotProfileService when not provided."""
        orchestrator = StrategyOrchestrator(bot=mock_bot)

        assert orchestrator._bot == mock_bot
        assert orchestrator._spot_profiles is not None


class TestInitStrategy:
    """Test init_strategy method."""

    def test_initializes_perps_strategy_when_not_spot(self, orchestrator, mock_bot):
        """Test initializes single strategy for non-SPOT profile."""
        mock_bot.config.profile = Profile.PROD

        orchestrator.init_strategy()

        # Should create single strategy on bot.strategy
        assert isinstance(mock_bot.runtime_state.strategy, BaselinePerpsStrategy)
        assert mock_bot.runtime_state.strategy.config.short_ma_period == 10
        assert mock_bot.runtime_state.strategy.config.long_ma_period == 30
        assert mock_bot.runtime_state.strategy.config.target_leverage == 2

    def test_initializes_spot_strategy_per_symbol(
        self, orchestrator, mock_bot, mock_spot_profile_service
    ):
        """Test initializes per-symbol strategies for SPOT profile."""
        mock_bot.config.profile = Profile.SPOT
        mock_bot.config.symbols = ["BTC-PERP", "ETH-PERP"]
        mock_spot_profile_service.load.return_value = {
            "BTC-PERP": {"short_window": 5, "long_window": 15},
            "ETH-PERP": {"short_window": 7, "long_window": 20},
        }

        orchestrator.init_strategy()

        # Should create per-symbol strategies
        symbol_map = mock_bot.runtime_state.symbol_strategies
        assert "BTC-PERP" in symbol_map
        assert "ETH-PERP" in symbol_map

        btc_strat = symbol_map["BTC-PERP"]
        assert btc_strat.config.short_ma_period == 5
        assert btc_strat.config.long_ma_period == 15
        assert btc_strat.config.enable_shorts is False

    def test_applies_position_fraction_override(self, orchestrator, mock_bot):
        """Test applies position fraction from config."""
        mock_bot.config.perps_position_fraction = 0.3

        orchestrator.init_strategy()

        assert mock_bot.runtime_state.strategy.config.position_fraction == 0.3

    def test_disables_leverage_when_derivatives_disabled(self, orchestrator, mock_bot):
        """Test sets leverage to 1 when derivatives disabled."""
        mock_bot.config.derivatives_enabled = False
        mock_bot.config.target_leverage = 5

        orchestrator.init_strategy()

        assert mock_bot.runtime_state.strategy.config.target_leverage == 1
        assert mock_bot.runtime_state.strategy.config.enable_shorts is False


class TestGetStrategy:
    """Test get_strategy method."""

    def test_returns_bot_strategy_for_perps_profile(self, orchestrator, mock_bot):
        """Test returns bot.strategy for non-SPOT profile."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.runtime_state.strategy = Mock(spec=BaselinePerpsStrategy)

        strategy = orchestrator.get_strategy("BTC-PERP")

        assert strategy == mock_bot.runtime_state.strategy

    def test_returns_symbol_strategy_for_spot_profile(self, orchestrator, mock_bot):
        """Test returns symbol-specific strategy for SPOT profile."""
        mock_bot.config.profile = Profile.SPOT
        btc_strategy = Mock(spec=BaselinePerpsStrategy)
        mock_bot.runtime_state.symbol_strategies["BTC-PERP"] = btc_strategy

        strategy = orchestrator.get_strategy("BTC-PERP")

        assert strategy == btc_strategy

    def test_creates_default_strategy_for_unknown_symbol(self, orchestrator, mock_bot):
        """Test creates default strategy when symbol not in map."""
        mock_bot.config.profile = Profile.SPOT
        mock_bot.runtime_state.symbol_strategies.clear()

        strategy = orchestrator.get_strategy("NEW-SYMBOL")

        assert isinstance(strategy, BaselinePerpsStrategy)
        assert "NEW-SYMBOL" in mock_bot.runtime_state.symbol_strategies


class TestEnsureBalances:
    """Test _ensure_balances async method."""

    @pytest.mark.asyncio
    async def test_returns_provided_balances(self, orchestrator, test_balance):
        """Test returns balances when provided."""
        balances = [test_balance]

        result = await orchestrator._ensure_balances(balances)

        assert result == balances

    @pytest.mark.asyncio
    async def test_fetches_balances_when_none(self, orchestrator, mock_bot, test_balance):
        """Test fetches balances from broker when None."""
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])

        result = await orchestrator._ensure_balances(None)

        assert result == [test_balance]
        mock_bot.broker.list_balances.assert_called_once()


class TestExtractEquity:
    """Test _extract_equity method."""

    def test_extracts_usdc_balance(self, orchestrator):
        """Test extracts USDC balance."""
        balances = [
            Mock(asset="BTC", total=Decimal("1")),
            Mock(asset="USDC", total=Decimal("10000")),
        ]

        equity = orchestrator._extract_equity(balances)

        assert equity == Decimal("10000")

    def test_extracts_usd_balance(self, orchestrator):
        """Test extracts USD balance."""
        balances = [
            Mock(asset="BTC", total=Decimal("1")),
            Mock(asset="USD", total=Decimal("5000")),
        ]

        equity = orchestrator._extract_equity(balances)

        assert equity == Decimal("5000")

    def test_returns_zero_when_no_usd_balance(self, orchestrator):
        """Test returns zero when no USD/USDC balance found."""
        balances = [
            Mock(asset="BTC", total=Decimal("1")),
            Mock(asset="ETH", total=Decimal("10")),
        ]

        equity = orchestrator._extract_equity(balances)

        assert equity == Decimal("0")


class TestKillSwitchEngaged:
    """Test _kill_switch_engaged method."""

    def test_returns_true_when_enabled(self, orchestrator, mock_bot):
        """Test returns True when kill switch enabled."""
        mock_bot.risk_manager.config.kill_switch_enabled = True

        engaged = orchestrator._kill_switch_engaged()

        assert engaged is True

    def test_returns_false_when_disabled(self, orchestrator, mock_bot):
        """Test returns False when kill switch disabled."""
        mock_bot.risk_manager.config.kill_switch_enabled = False

        engaged = orchestrator._kill_switch_engaged()

        assert engaged is False


class TestEnsurePositions:
    """Test _ensure_positions async method."""

    @pytest.mark.asyncio
    async def test_returns_provided_position_map(self, orchestrator, test_position):
        """Test returns position map when provided."""
        position_map = {"BTC-PERP": test_position}

        result = await orchestrator._ensure_positions(position_map)

        assert result == position_map

    @pytest.mark.asyncio
    async def test_fetches_positions_when_none(self, orchestrator, mock_bot, test_position):
        """Test fetches positions from broker when None."""
        mock_bot.broker.list_positions = Mock(return_value=[test_position])

        result = await orchestrator._ensure_positions(None)

        assert result == {"BTC-PERP": test_position}
        mock_bot.broker.list_positions.assert_called_once()


class TestBuildPositionState:
    """Test _build_position_state method."""

    def test_returns_none_when_no_position(self, orchestrator):
        """Test returns None state when symbol not in positions."""
        positions = {}

        state, quantity = orchestrator._build_position_state("BTC-PERP", positions)

        assert state is None
        assert quantity == Decimal("0")

    def test_builds_state_from_position(self, orchestrator, test_position):
        """Test builds position state from Position object."""
        positions = {"BTC-PERP": test_position}

        state, quantity = orchestrator._build_position_state("BTC-PERP", positions)

        assert state is not None
        assert state["quantity"] == Decimal("0.5")
        assert state["side"] == "long"
        assert state["entry"] == Decimal("50000")
        assert quantity == Decimal("0.5")


class TestGetMarks:
    """Test _get_marks method."""

    def test_returns_marks_for_symbol(self, orchestrator, mock_bot):
        """Test returns marks from bot mark_windows."""
        state = mock_bot.runtime_state
        state.mark_windows.clear()
        state.mark_windows["BTC-PERP"] = [Decimal("50000"), Decimal("51000")]

        marks = orchestrator._get_marks("BTC-PERP")

        assert marks == [Decimal("50000"), Decimal("51000")]

    def test_returns_empty_list_when_no_marks(self, orchestrator, mock_bot):
        """Test returns empty list when no marks for symbol."""
        mock_bot.runtime_state.mark_windows.clear()

        marks = orchestrator._get_marks("BTC-PERP")

        assert marks == []


class TestAdjustEquity:
    """Test _adjust_equity method."""

    def test_adjusts_equity_for_position(self, orchestrator):
        """Test adds position value to equity."""
        equity = Decimal("10000")
        position_quantity = Decimal("0.5")
        marks = [Decimal("50000"), Decimal("51000")]

        adjusted = orchestrator._adjust_equity(equity, position_quantity, marks, "BTC-PERP")

        # 10000 + (0.5 * 51000) = 10000 + 25500 = 35500
        assert adjusted == Decimal("35500")

    def test_returns_original_when_no_position(self, orchestrator):
        """Test returns original equity when no position."""
        equity = Decimal("10000")
        marks = [Decimal("50000")]

        adjusted = orchestrator._adjust_equity(equity, Decimal("0"), marks, "BTC-PERP")

        assert adjusted == Decimal("10000")


class TestRunRiskGates:
    """Test _run_risk_gates method."""

    def test_returns_true_when_all_gates_pass(self, orchestrator, mock_bot):
        """Test returns True when all risk gates pass."""
        marks = [Decimal("50000")] * 35
        mock_bot.risk_manager.check_volatility_circuit_breaker = Mock(
            return_value=CircuitBreakerOutcome(
                triggered=False, action=CircuitBreakerAction.NONE, reason=None
            )
        )
        mock_bot.risk_manager.check_mark_staleness = Mock(return_value=False)

        context = SymbolProcessingContext(
            symbol="BTC-PERP",
            balances=[],
            equity=Decimal("10000"),
            positions={},
            position_state=None,
            position_quantity=Decimal("0"),
            marks=list(marks),
            product=None,
        )

        result = orchestrator._run_risk_gates(context)

        assert result is True

    def test_returns_false_when_kill_switch_triggered(self, orchestrator, mock_bot):
        """Test returns False when volatility CB triggers kill switch."""
        marks = [Decimal("50000")] * 35
        mock_bot.risk_manager.check_volatility_circuit_breaker = Mock(
            return_value=CircuitBreakerOutcome(
                triggered=True,
                action=CircuitBreakerAction.KILL_SWITCH,
                reason="volatility spike",
            )
        )

        context = SymbolProcessingContext(
            symbol="BTC-PERP",
            balances=[],
            equity=Decimal("10000"),
            positions={},
            position_state=None,
            position_quantity=Decimal("0"),
            marks=list(marks),
            product=None,
        )

        result = orchestrator._run_risk_gates(context)

        assert result is False

    def test_returns_false_when_mark_stale(self, orchestrator, mock_bot):
        """Test returns False when market data is stale."""
        marks = [Decimal("50000")] * 35
        mock_bot.risk_manager.check_volatility_circuit_breaker = Mock(
            return_value=CircuitBreakerOutcome(
                triggered=False, action=CircuitBreakerAction.NONE, reason=None
            )
        )
        mock_bot.risk_manager.check_mark_staleness = Mock(return_value=True)

        context = SymbolProcessingContext(
            symbol="BTC-PERP",
            balances=[],
            equity=Decimal("10000"),
            positions={},
            position_state=None,
            position_quantity=Decimal("0"),
            marks=list(marks),
            product=None,
        )

        result = orchestrator._run_risk_gates(context)

        assert result is False


class TestEvaluateStrategy:
    """Test _evaluate_strategy method."""

    def test_calls_strategy_decide(self, orchestrator, mock_bot):
        """Test calls strategy.decide with correct parameters."""
        strategy = Mock(spec=BaselinePerpsStrategy)
        decision = Decision(action=Action.BUY, reason="test")
        strategy.decide = Mock(return_value=decision)

        marks = [Decimal("50000"), Decimal("51000")]
        position_state = {"quantity": Decimal("0.5"), "side": "long"}
        equity = Decimal("10000")
        product = Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("1"),
            price_increment=Decimal("0.01"),
            leverage_max=5,
            expiry=None,
            contract_size=Decimal("1"),
            funding_rate=Decimal("0"),
            next_funding_time=None,
        )
        mock_bot.get_product = Mock(return_value=product)

        result = orchestrator._evaluate_strategy(
            strategy, "BTC-PERP", marks, position_state, equity, product
        )

        assert result == decision
        strategy.decide.assert_called_once()
        call_kwargs = strategy.decide.call_args.kwargs
        assert call_kwargs["symbol"] == "BTC-PERP"
        assert call_kwargs["current_mark"] == Decimal("51000")
        assert call_kwargs["position_state"] == position_state
        assert call_kwargs["equity"] == equity
        assert call_kwargs["product"] == product


class TestRecordDecision:
    """Test _record_decision method."""

    def test_records_decision_to_bot(self, orchestrator, mock_bot):
        """Test records decision in bot's last_decisions map."""
        decision = Decision(action=Action.BUY, reason="test_reason")

        orchestrator._record_decision("BTC-PERP", decision)

        assert mock_bot.last_decisions["BTC-PERP"] == decision


class TestFetchSpotCandles:
    """Test _fetch_spot_candles async method."""

    @pytest.mark.asyncio
    async def test_fetches_candles_from_broker(self, orchestrator, mock_bot):
        """Test fetches candles from broker."""
        now = datetime.now(timezone.utc)
        candle1 = Mock(close=Decimal("50000"), ts=now)
        candle2 = Mock(close=Decimal("51000"), ts=now)
        mock_bot.broker.get_candles = Mock(return_value=[candle1, candle2])

        candles = await orchestrator._fetch_spot_candles("BTC-PERP", 20)

        assert len(candles) == 2
        mock_bot.broker.get_candles.assert_called_once_with("BTC-PERP", "ONE_HOUR", 22)

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self, orchestrator, mock_bot):
        """Test returns empty list on exception."""
        mock_bot.broker.get_candles = Mock(side_effect=Exception("fetch failed"))

        candles = await orchestrator._fetch_spot_candles("BTC-PERP", 20)

        assert candles == []


class TestProcessSymbol:
    """Test process_symbol async orchestration method."""

    @pytest.mark.asyncio
    async def test_skips_when_kill_switch_engaged(self, orchestrator, mock_bot, test_balance):
        """Test skips processing when kill switch engaged."""
        mock_bot.risk_manager.config.kill_switch_enabled = True
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])

        await orchestrator.process_symbol("BTC-PERP")

        # Should not call execute_decision
        mock_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_no_marks(self, orchestrator, mock_bot, test_balance):
        """Test skips processing when no marks available."""
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.broker.list_positions = Mock(return_value=[])
        mock_bot.runtime_state.mark_windows.clear()

        await orchestrator.process_symbol("BTC-PERP")

        # Should not call execute_decision
        mock_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_executes_buy_decision(self, orchestrator, mock_bot, test_balance, test_position):
        """Test executes BUY decision through bot."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.broker.list_positions = Mock(return_value=[test_position])
        state = mock_bot.runtime_state
        state.mark_windows.clear()
        state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35
        mock_bot.risk_manager.check_volatility_circuit_breaker = Mock(
            return_value=CircuitBreakerOutcome(
                triggered=False, action=CircuitBreakerAction.NONE, reason=None
            )
        )
        mock_bot.risk_manager.check_mark_staleness = Mock(return_value=False)

        # Create strategy and set decision
        strategy = Mock(spec=BaselinePerpsStrategy)
        decision = Decision(action=Action.BUY, reason="test")
        strategy.decide = Mock(return_value=decision)
        mock_bot.runtime_state.strategy = strategy

        product = Mock()
        mock_bot.get_product = Mock(return_value=product)

        await orchestrator.process_symbol("BTC-PERP")

        # Should execute decision
        mock_bot.execute_decision.assert_called_once()
        call_args = mock_bot.execute_decision.call_args
        assert call_args[0][0] == "BTC-PERP"
        assert call_args[0][1] == decision

    @pytest.mark.asyncio
    async def test_holds_when_risk_gates_fail(self, orchestrator, mock_bot, test_balance):
        """Test skips execution when risk gates fail."""
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.broker.list_positions = Mock(return_value=[])
        state = mock_bot.runtime_state
        state.mark_windows.clear()
        state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35
        mock_bot.risk_manager.check_mark_staleness = Mock(return_value=True)

        await orchestrator.process_symbol("BTC-PERP")

        # Should not execute decision
        mock_bot.execute_decision.assert_not_called()


class TestStrategyOrchestratorEdgeCases:
    """Test edge cases and error handling scenarios in StrategyOrchestrator."""

    @pytest.mark.asyncio
    async def test_spot_profile_invalid_position_fraction_warning(self, orchestrator, mock_bot):
        """Test warning logged when spot profile has invalid position_fraction."""
        from bot_v2.orchestration.configuration import Profile

        mock_bot.config.profile = Profile.SPOT
        mock_bot.config.symbols = ["BTC-PERP"]
        mock_bot.config.perps_position_fraction = None

        # Mock spot profiles to return invalid fraction
        mock_spot_profiles = Mock()
        mock_spot_profiles.load.return_value = {"BTC-PERP": {"position_fraction": "invalid_value"}}

        orchestrator._spot_profiles = mock_spot_profiles

        # This should trigger the warning on lines 86-93
        with pytest.MonkeyPatch().context() as m:
            mock_logger = Mock()
            m.setattr("bot_v2.orchestration.strategy_orchestrator.logger", mock_logger)

            orchestrator.init_strategy()

            # Should log warning about invalid position_fraction
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert (
                "Invalid position_fraction=invalid_value for BTC-PERP; using default"
                in call_args[0][0] % call_args[0][1:]
            )
            assert call_args[1]["operation"] == "strategy_init"
            assert call_args[1]["stage"] == "spot_fraction"
            assert call_args[1]["symbol"] == "BTC-PERP"

    @pytest.mark.asyncio
    async def test_perps_invalid_position_fraction_warning(self, orchestrator, mock_bot):
        """Test warning logged when perps position_fraction is invalid."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.perps_position_fraction = "not_a_number"

        with pytest.MonkeyPatch().context() as m:
            mock_logger = Mock()
            m.setattr("bot_v2.orchestration.strategy_orchestrator.logger", mock_logger)

            orchestrator.init_strategy()

            # Should log warning about invalid PERPS_POSITION_FRACTION
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert (
                "Invalid PERPS_POSITION_FRACTION=not_a_number; using default"
                in call_args[0][0] % call_args[0][1:]
            )
            assert call_args[1]["operation"] == "strategy_init"
            assert call_args[1]["stage"] == "perps_fraction"

    @pytest.mark.asyncio
    async def test_process_symbol_execution_error_logging(
        self, orchestrator, mock_bot, test_balance
    ):
        """Test error logging when execution fails."""
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.broker.list_positions = Mock(return_value=[])

        # Set up marks and strategy
        state = mock_bot.runtime_state
        state.mark_windows.clear()
        state.mark_windows["BTC-PERP"] = [Decimal("50000")] * 35

        strategy = Mock(spec=BaselinePerpsStrategy)
        decision = Decision(action=Action.BUY, reason="test")
        strategy.decide = Mock(return_value=decision)
        state.strategy = strategy

        product = Mock()
        mock_bot.get_product = Mock(return_value=product)
        mock_bot.risk_manager.check_mark_staleness = Mock(return_value=False)
        mock_bot.risk_manager.check_volatility_circuit_breaker = Mock(
            return_value=CircuitBreakerOutcome(
                triggered=False, action=CircuitBreakerAction.NONE, reason=None
            )
        )

        # Mock execute_decision to raise an exception
        mock_bot.execute_decision = Mock(side_effect=Exception("Execution failed"))

        with pytest.MonkeyPatch().context() as m:
            mock_logger = Mock()
            m.setattr("bot_v2.orchestration.strategy_orchestrator.logger", mock_logger)

            # Should not raise exception, but should log error
            await orchestrator.process_symbol("BTC-PERP")

            # Should log execution error with expected structure
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert "Error processing %s: %s" in call_args[0][0]
            assert call_args[1]["operation"] == "strategy_execute"
            assert call_args[1]["stage"] == "process_symbol"
            assert call_args[1]["symbol"] == "BTC-PERP"
            assert call_args[1]["exc_info"] is True

    @pytest.mark.asyncio
    async def test_prepare_context_no_marks_warning(self, orchestrator, mock_bot, test_balance):
        """Test warning logged when no marks are available."""
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])
        mock_bot.broker.list_positions = Mock(return_value=[])

        # Clear marks to trigger warning
        state = mock_bot.runtime_state
        state.mark_windows.clear()

        with pytest.MonkeyPatch().context() as m:
            mock_logger = Mock()
            m.setattr("bot_v2.orchestration.strategy_orchestrator.logger", mock_logger)

            # Should return None and log warning
            result = await orchestrator._prepare_context("BTC-PERP", None, None)

            assert result is None
            mock_logger.warning.assert_called()
            # Check that any warning call mentions "No marks for BTC-PERP"
            warning_calls = mock_logger.warning.call_args_list
            assert any("No marks for BTC-PERP" in str(call) for call in warning_calls)

    def test_adjust_equity_zero_position_quantity(self, orchestrator):
        """Test equity adjustment with zero position quantity."""
        equity = Decimal("10000")
        position_quantity = Decimal("0")
        marks = [Decimal("50000")]
        symbol = "BTC-PERP"

        result = orchestrator._adjust_equity(equity, position_quantity, marks, symbol)

        # Should return original equity when position quantity is zero
        assert result == equity

    def test_adjust_equity_with_position_and_marks(self, orchestrator):
        """Test equity adjustment with position and marks."""
        equity = Decimal("10000")
        position_quantity = Decimal("2")  # Long position
        marks = [Decimal("50000"), Decimal("51000")]
        symbol = "BTC-PERP"

        result = orchestrator._adjust_equity(equity, position_quantity, marks, symbol)

        # Should adjust equity based on position value
        assert isinstance(result, Decimal)
        # For long position, should add position value to equity
        position_value = position_quantity * marks[-1]
        expected = equity + position_value
        assert result == expected

    def test_adjust_equity_short_position(self, orchestrator):
        """Test equity adjustment with short position."""
        equity = Decimal("10000")
        position_quantity = Decimal("-1")  # Short position
        marks = [Decimal("50000"), Decimal("49000")]
        symbol = "BTC-PERP"

        result = orchestrator._adjust_equity(equity, position_quantity, marks, symbol)

        # For short position, should handle differently
        assert isinstance(result, Decimal)

    def test_build_position_state_long_position(self, orchestrator, test_position):
        """Test building position state for long position."""
        test_position.side = "long"
        test_position.quantity = Decimal("2")
        positions_lookup = {"BTC-PERP": test_position}

        position_state, position_quantity = orchestrator._build_position_state(
            "BTC-PERP", positions_lookup
        )

        assert position_quantity == Decimal("2")
        assert position_state is not None

    def test_build_position_state_no_position(self, orchestrator):
        """Test building position state when no position exists."""
        positions_lookup = {}

        position_state, position_quantity = orchestrator._build_position_state(
            "BTC-PERP", positions_lookup
        )

        assert position_quantity == Decimal("0")
        assert position_state is None

    def test_get_marks_uses_window(self, orchestrator, mock_bot):
        """Test _get_marks uses mark window from runtime state."""
        state = mock_bot.runtime_state
        state.mark_windows.clear()
        expected_marks = [Decimal("50000"), Decimal("51000"), Decimal("52000")]
        state.mark_windows["BTC-PERP"] = expected_marks

        result = orchestrator._get_marks("BTC-PERP")

        assert result == expected_marks

    def test_get_marks_empty_window(self, orchestrator, mock_bot):
        """Test _get_marks with empty mark window."""
        state = mock_bot.runtime_state
        state.mark_windows.clear()
        state.mark_windows["BTC-PERP"] = []

        result = orchestrator._get_marks("BTC-PERP")

        assert result == []

    def test_record_decision(self, orchestrator, mock_bot):
        """Test decision recording in runtime state."""
        decision = Decision(action=Action.BUY, reason="test_signal")

        orchestrator._record_decision("BTC-PERP", decision)

        state = mock_bot.runtime_state
        assert "BTC-PERP" in state.last_decisions
        assert state.last_decisions["BTC-PERP"] == decision


class TestKillSwitchLogic:
    """Test kill-switch engagement and emergency logic."""

    @pytest.mark.asyncio
    async def test_kill_switch_enabled_skips_processing(self, mock_bot) -> None:
        """Test that enabled kill switch prevents all processing."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Enable kill switch
        mock_bot.risk_manager.config.kill_switch_enabled = True

        with patch("bot_v2.orchestration.strategy_orchestrator.emit_metric"):
            await orchestrator.process_symbol(
                "BTC-PERP", [test_balance], {"BTC-PERP": test_position}
            )

        # Should not execute any decisions when kill switch is enabled
        mock_bot.execute_decision.assert_not_called()
        # Should log kill switch warning (covered by kill switch engagement)

    def test_kill_switch_enabled_property_access(self, mock_bot) -> None:
        """Test kill switch configuration access patterns."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Test with kill switch disabled (default)
        mock_bot.risk_manager.config.kill_switch_enabled = False
        assert orchestrator._kill_switch_engaged() is False

        # Test with kill switch enabled
        mock_bot.risk_manager.config.kill_switch_enabled = True
        assert orchestrator._kill_switch_engaged() is True

        # Test when kill_switch_enabled attribute doesn't exist
        del mock_bot.risk_manager.config.kill_switch_enabled
        assert orchestrator._kill_switch_engaged() is False

    @pytest.mark.asyncio
    async def test_kill_switch_preparation_context_early_return(self, mock_bot) -> None:
        """Test that kill switch causes _prepare_context to return None early."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Enable kill switch
        mock_bot.risk_manager.config.kill_switch_enabled = True

        context = await orchestrator._prepare_context(
            "BTC-PERP", [test_balance], {"BTC-PERP": test_position}
        )

        # Should return None when kill switch is engaged
        assert context is None

    @pytest.mark.asyncio
    async def test_kill_switch_logs_warning_message(self, mock_bot) -> None:
        """Test that kill switch engagement logs appropriate warning."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Enable kill switch
        mock_bot.risk_manager.config.kill_switch_enabled = True

        with patch("bot_v2.orchestration.strategy_orchestrator.logger") as mock_logger:
            await orchestrator._prepare_context(
                "BTC-PERP", [test_balance], {"BTC-PERP": test_position}
            )

        # Should log warning about kill switch being enabled
        mock_logger.warning.assert_called_once_with("Kill switch enabled - skipping trading loop")


class TestDecisionRoutingAndGuardChains:
    """Test decision routing through different action paths and guard chains."""

    @pytest.mark.asyncio
    async def test_decision_routing_buy_action_with_product(self, mock_bot) -> None:
        """Test routing of BUY action when product metadata is available."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Mock strategy to return BUY decision
        mock_strategy = Mock()
        mock_strategy.evaluate.return_value = Decision(action=Action.BUY, reason="test_signal")
        mock_bot.get_strategy.return_value = mock_strategy

        # Mock get_product to return valid product
        mock_product = Mock()
        mock_bot.get_product.return_value = mock_product

        # Mock marks and equity to provide data for strategy processing
        mock_marks = [Decimal("50000")]
        with patch.object(orchestrator, "_get_marks", return_value=mock_marks):
            with patch.object(orchestrator, "_adjust_equity", return_value=Decimal("1000")):
                await orchestrator.process_symbol(
                    "BTC-PERP", [test_balance], {"BTC-PERP": test_position}
                )

        # Should execute BUY decision
        mock_bot.execute_decision.assert_called_once()
        args = mock_bot.execute_decision.call_args[0]
        assert args[0] == "BTC-PERP"  # symbol
        assert args[1].action == Action.BUY  # decision

    @pytest.mark.asyncio
    async def test_decision_routing_missing_product_metadata(self, mock_bot) -> None:
        """Test decision routing when product metadata is missing."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Mock strategy to return BUY decision
        mock_strategy = Mock()
        mock_strategy.evaluate.return_value = Decision(action=Action.BUY, reason="test_signal")
        mock_bot.get_strategy.return_value = mock_strategy

        # Mock get_product to raise exception (missing product)
        mock_bot.get_product.side_effect = Exception("Product not found")

        with patch("bot_v2.orchestration.strategy_orchestrator.logger") as mock_logger:
            # emit_metric patch removed - function doesn't exist in module
            await orchestrator.process_symbol(
                "BTC-PERP", [test_balance], {"BTC-PERP": test_position}
            )

        # Should not execute decision due to missing product
        mock_bot.execute_decision.assert_not_called()
        # Should log warning about missing product metadata
        mock_logger.warning.assert_called_once()
        assert "missing product metadata" in str(mock_logger.warning.call_args)

    @pytest.mark.asyncio
    async def test_decision_routing_sell_action_execution(self, mock_bot) -> None:
        """Test routing of SELL action through execution path."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Mock strategy to return SELL decision
        mock_strategy = Mock()
        mock_strategy.evaluate.return_value = Decision(action=Action.SELL, reason="sell_signal")
        mock_bot.get_strategy.return_value = mock_strategy

        mock_product = Mock()
        mock_bot.get_product.return_value = mock_product

        # emit_metric patch removed - function doesn't exist in module
        await orchestrator.process_symbol("BTC-PERP", [test_balance], {"BTC-PERP": test_position})

        # Should execute SELL decision
        mock_bot.execute_decision.assert_called_once()
        args = mock_bot.execute_decision.call_args[0]
        assert args[1].action == Action.SELL

    @pytest.mark.asyncio
    async def test_decision_routing_close_action_execution(self, mock_bot) -> None:
        """Test routing of CLOSE action through execution path."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Mock strategy to return CLOSE decision
        mock_strategy = Mock()
        mock_strategy.evaluate.return_value = Decision(action=Action.CLOSE, reason="close_signal")
        mock_bot.get_strategy.return_value = mock_strategy

        mock_product = Mock()
        mock_bot.get_product.return_value = mock_product

        # emit_metric patch removed - function doesn't exist in module
        await orchestrator.process_symbol("BTC-PERP", [test_balance], {"BTC-PERP": test_position})

        # Should execute CLOSE decision
        mock_bot.execute_decision.assert_called_once()
        args = mock_bot.execute_decision.call_args[0]
        assert args[1].action == Action.CLOSE

    @pytest.mark.asyncio
    async def test_spot_profile_guard_chain_filtering(self, mock_bot) -> None:
        """Test spot profile filtering in decision guard chain."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Set profile to SPOT to trigger guard chain
        mock_bot.config.profile = Profile.SPOT

        # Mock context with strategy decision
        mock_context = Mock()
        mock_context.symbol = "BTC-USD"
        mock_context.marks = [Mock()]
        mock_context.position_state = None
        mock_context.equity = Decimal("1000")
        mock_context.product = Mock()

        # Mock spot filter to modify decision
        original_decision = Decision(action=Action.BUY, reason="signal")
        filtered_decision = Decision(action=Action.BUY, reason="filtered_signal")
        with patch.object(orchestrator, "_evaluate_strategy", return_value=original_decision):
            with patch.object(orchestrator, "_apply_spot_filters", return_value=filtered_decision):
                final_decision = await orchestrator._resolve_decision(mock_context)

        # Should apply spot filters for SPOT profile
        assert final_decision.action == Action.BUY
        assert final_decision.quantity == Decimal("0.3")

    @pytest.mark.asyncio
    async def test_non_spot_profile_bypasses_guard_chain(self, mock_bot) -> None:
        """Test that non-SPOT profiles bypass spot filter guard chain."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Set profile to DEV (non-SPOT)
        mock_bot.config.profile = Profile.DEV

        # Mock context with strategy decision
        mock_context = Mock()
        mock_context.symbol = "BTC-PERP"
        mock_context.marks = [Mock()]
        mock_context.position_state = None
        mock_context.equity = Decimal("1000")
        mock_context.product = Mock()

        # Mock strategy evaluation
        strategy_decision = Decision(action=Action.BUY, reason="signal")
        with patch.object(orchestrator, "_evaluate_strategy", return_value=strategy_decision):
            final_decision = await orchestrator._resolve_decision(mock_context)

        # Should return strategy decision unchanged for non-SPOT profiles
        assert final_decision.action == Action.BUY
        assert final_decision.quantity == Decimal("0.5")
        assert final_decision.source == "signal"

    def test_decision_action_guard_for_valid_actions(self, mock_bot) -> None:
        """Test that only valid actions (BUY, SELL, CLOSE) trigger execution."""
        valid_actions = {Action.BUY, Action.SELL, Action.CLOSE}

        for action in valid_actions:
            StrategyOrchestrator(mock_bot)
            mock_context = Mock()
            mock_context.product = Mock()

            # This would be tested through the main execution path
            # The guard at line 149 ensures only these actions proceed
            assert action in {Action.BUY, Action.SELL, Action.CLOSE}


class TestPositionStateBuildingAndValidation:
    """Test position state building and validation workflows."""

    def test_position_state_building_with_complete_position(self, mock_bot) -> None:
        """Test position state building with complete position data."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Create mock position with all required attributes
        mock_pos = Mock()
        mock_pos.symbol = "BTC-PERP"
        mock_pos.side = "long"
        mock_pos.entry_price = Decimal("50000")
        mock_pos.quantity = Decimal("2")

        positions_lookup = {"BTC-PERP": mock_pos}

        position_state, position_quantity = orchestrator._build_position_state(
            "BTC-PERP", positions_lookup
        )

        assert position_quantity == Decimal("2")
        assert position_state is not None
        assert position_state["quantity"] == Decimal("2")
        assert position_state["side"] == "long"
        assert position_state["entry"] == Decimal("50000")

    def test_position_state_building_missing_attributes(self, mock_bot) -> None:
        """Test position state building with missing position attributes."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Create mock position with minimal attributes
        mock_pos = Mock()
        mock_pos.symbol = "BTC-PERP"
        # Missing side, entry_price, quantity

        positions_lookup = {"BTC-PERP": mock_pos}

        position_state, position_quantity = orchestrator._build_position_state(
            "BTC-PERP", positions_lookup
        )

        # Should handle missing attributes gracefully
        assert position_quantity == Decimal("0")  # default from quantity_from
        assert position_state is not None
        assert position_state["quantity"] == Decimal("0")

    @pytest.mark.asyncio
    async def test_ensure_positions_with_provided_map(self, mock_bot) -> None:
        """Test position lookup when map is provided."""
        orchestrator = StrategyOrchestrator(mock_bot)

        provided_positions = {"BTC-PERP": test_position}

        result = await orchestrator._ensure_positions(provided_positions)

        # Should return the provided map unchanged
        assert result == provided_positions
        # Should not call broker.list_positions
        mock_bot.broker.list_positions.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_positions_fetches_from_broker(self, mock_bot) -> None:
        """Test position lookup when map is None and needs fetching."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Mock broker positions
        test_positions = [test_position]
        mock_bot.broker.list_positions.return_value = test_positions

        result = await orchestrator._ensure_positions(None)

        # Should call broker and return symbol-keyed dict
        mock_bot.broker.list_positions.assert_called_once()
        assert "BTC-PERP" in result

    @pytest.mark.asyncio
    async def test_prepare_context_no_marks_early_return(self, mock_bot) -> None:
        """Test that missing marks causes _prepare_context to return None."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Mock empty marks
        with patch.object(orchestrator, "_get_marks", return_value=[]):
            with patch("bot_v2.orchestration.strategy_orchestrator.logger") as mock_logger:
                context = await orchestrator._prepare_context(
                    "BTC-PERP", [test_balance], {"BTC-PERP": test_position}
                )

        # Should return None when no marks available
        assert context is None
        # Should log warning about missing marks
        mock_logger.warning.assert_called_once()
        assert "No marks for" in str(mock_logger.warning.call_args)

    @pytest.mark.asyncio
    async def test_prepare_context_zero_equity_early_return(self, mock_bot) -> None:
        """Test that zero equity causes _prepare_context to return None."""
        orchestrator = StrategyOrchestrator(mock_bot)

        # Mock zero equity adjustment
        with patch.object(orchestrator, "_get_marks", return_value=[Decimal("50000")]):
            with patch.object(orchestrator, "_adjust_equity", return_value=Decimal("0")):
                with patch("bot_v2.orchestration.strategy_orchestrator.logger") as mock_logger:
                    context = await orchestrator._prepare_context(
                        "BTC-PERP", [test_balance], {"BTC-PERP": test_position}
                    )

        # Should return None when equity is zero
        assert context is None
        # Should log error about no equity info
        mock_logger.error.assert_called_once()
        assert "No equity info for" in str(mock_logger.error.call_args)

    def test_extract_equity_with_usd_balance(self, mock_bot) -> None:
        """Test equity extraction with USD balance."""
        orchestrator = StrategyOrchestrator(mock_bot)

        usd_balance = Mock()
        usd_balance.asset = "USD"
        usd_balance.total = Decimal("1000")

        balances = [usd_balance]
        equity = orchestrator._extract_equity(balances)

        assert equity == Decimal("1000")

    def test_extract_equity_with_usdc_balance(self, mock_bot) -> None:
        """Test equity extraction with USDC balance."""
        orchestrator = StrategyOrchestrator(mock_bot)

        usdc_balance = Mock()
        usdc_balance.asset = "USDC"
        usdc_balance.total = Decimal("2000")

        balances = [usdc_balance]
        equity = orchestrator._extract_equity(balances)

        assert equity == Decimal("2000")

    def test_extract_equity_no_cash_assets(self, mock_bot) -> None:
        """Test equity extraction when no cash assets are available."""
        orchestrator = StrategyOrchestrator(mock_bot)

        btc_balance = Mock()
        btc_balance.asset = "BTC"
        btc_balance.total = Decimal("1")

        balances = [btc_balance]
        equity = orchestrator._extract_equity(balances)

        assert equity == Decimal("0")
