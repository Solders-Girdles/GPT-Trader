"""
Tests for StrategyOrchestrator.

Tests strategy initialization, decision execution, risk gates,
and spot-specific filtering logic.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, MagicMock, AsyncMock, patch

import pytest

from bot_v2.orchestration.strategy_orchestrator import (
    StrategyOrchestrator,
    SymbolProcessingContext,
)
from bot_v2.orchestration.configuration import Profile
from bot_v2.features.brokerages.core.interfaces import Balance, Position, Product, MarketType
from bot_v2.features.live_trade.strategies.perps_baseline import (
    Action,
    Decision,
    BaselinePerpsStrategy,
)
from bot_v2.features.live_trade.risk_runtime import (
    CircuitBreakerOutcome,
    CircuitBreakerAction,
)
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState


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
