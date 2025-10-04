"""Tests for strategy orchestrator"""

import pytest
import time as _time
from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from bot_v2.features.brokerages.core.interfaces import Balance, Position, Product
from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator
from bot_v2.orchestration.configuration import Profile


@pytest.fixture
def strategy_types():
    """Provide strategy types without module-level import"""
    from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

    return {"Action": Action, "Decision": Decision}


@pytest.fixture
def mock_bot():
    """Mock PerpsBot instance"""
    bot = Mock()
    bot.config = Mock()
    bot.config.profile = Profile.PROD
    bot.config.symbols = ["BTC-USD", "ETH-USD"]
    bot.config.short_ma = 10
    bot.config.long_ma = 30
    bot.config.target_leverage = 2
    bot.config.trailing_stop_pct = 0.02
    bot.config.enable_shorts = True
    bot.config.derivatives_enabled = True
    bot.config.perps_position_fraction = 0.5
    bot.broker = Mock()
    bot.broker.list_balances = Mock()
    bot.broker.list_positions = Mock()
    bot.broker.get_quote = Mock()
    bot.broker.get_candles = Mock()
    bot.risk_manager = Mock()
    bot.risk_manager.config = Mock()
    bot.risk_manager.config.kill_switch_enabled = False
    bot.risk_manager.check_volatility_circuit_breaker = Mock()
    bot.risk_manager.check_mark_staleness = Mock(return_value=False)
    bot.strategy = None
    bot._symbol_strategies = {}
    bot._product_map = {}
    bot.mark_windows = {}
    bot.last_decisions = {}
    bot.execute_decision = AsyncMock()
    bot.get_product = Mock()
    return bot


@pytest.fixture
def orchestrator(mock_bot):
    """Create StrategyOrchestrator instance"""
    return StrategyOrchestrator(mock_bot)


@pytest.fixture
def sample_balances():
    """Create sample balances"""
    balance = Mock(spec=Balance)
    balance.asset = "USD"
    balance.total = Decimal("10000")
    return [balance]


@pytest.fixture
def sample_positions():
    """Create sample positions"""
    position = Mock(spec=Position)
    position.symbol = "BTC-USD"
    position.quantity = Decimal("0.5")
    position.side = "long"
    position.entry_price = Decimal("48000")
    return [position]


@pytest.fixture
def sample_product():
    """Create sample product"""
    product = Mock(spec=Product)
    product.symbol = "BTC-USD"
    product.base_increment = Decimal("0.00000001")
    product.quote_increment = Decimal("0.01")
    return product


class TestStrategyOrchestrator:
    """Test suite for StrategyOrchestrator"""

    def test_initialization(self, orchestrator, mock_bot):
        """Test orchestrator initialization"""
        assert orchestrator._bot == mock_bot
        assert orchestrator._spot_profiles is not None

    def test_init_strategy_perps_profile(self, orchestrator, mock_bot):
        """Test strategy initialization for perpetuals"""
        orchestrator.init_strategy()

        assert mock_bot.strategy is not None

    def test_init_strategy_spot_profile(self, orchestrator, mock_bot):
        """Test strategy initialization for spot profile"""
        mock_bot.config.profile = Profile.SPOT
        mock_bot.config.symbols = ["BTC-USD", "ETH-USD"]

        orchestrator.init_strategy()

        # Should create per-symbol strategies
        assert "BTC-USD" in mock_bot._symbol_strategies
        assert "ETH-USD" in mock_bot._symbol_strategies

    def test_get_strategy_perps_profile(self, orchestrator, mock_bot):
        """Test getting strategy in perps profile"""
        # Initialize strategy first
        orchestrator.init_strategy()

        strategy = orchestrator.get_strategy("BTC-USD")

        assert strategy is not None
        assert strategy == mock_bot.strategy

    def test_get_strategy_spot_profile_existing(self, orchestrator, mock_bot):
        """Test getting existing strategy in spot profile"""
        mock_bot.config.profile = Profile.SPOT
        mock_bot.config.symbols = ["BTC-USD"]

        # Initialize strategies
        orchestrator.init_strategy()

        # Get the initialized strategy
        strategy = orchestrator.get_strategy("BTC-USD")

        assert strategy is not None
        assert "BTC-USD" in mock_bot._symbol_strategies
        assert strategy == mock_bot._symbol_strategies["BTC-USD"]

    def test_get_strategy_spot_profile_creates_new(self, orchestrator, mock_bot):
        """Test creating new strategy for symbol in spot profile (lazy creation)"""
        mock_bot.config.profile = Profile.SPOT
        mock_bot.config.symbols = ["BTC-USD"]  # Only BTC initially

        # Initialize with just BTC
        orchestrator.init_strategy()

        # Request strategy for NEW-USD (not in initial config)
        strategy = orchestrator.get_strategy("NEW-USD")

        assert strategy is not None
        # Lazy creation happens in registry, check it's there
        assert "NEW-USD" in orchestrator.strategy_registry.symbol_strategies

    @pytest.mark.asyncio
    async def test_process_symbol_success(
        self, orchestrator, mock_bot, sample_balances, sample_product
    ):
        """Test successful symbol processing"""
        mock_bot.mark_windows["BTC-USD"] = [
            Decimal("49000"),
            Decimal("49500"),
            Decimal("50000"),
        ]
        mock_bot.get_product.return_value = sample_product
        mock_bot.broker.list_balances.return_value = sample_balances
        mock_bot.broker.list_positions.return_value = []

        from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

        orchestrator.get_strategy = Mock()
        mock_strategy = Mock()
        mock_strategy.decide = Mock(
            return_value=Decision(
                action=Action.BUY,
                reason="test_signal",
                target_notional=Decimal("1000"),
                leverage=2,
            )
        )
        orchestrator.get_strategy.return_value = mock_strategy

        await orchestrator.process_symbol("BTC-USD")

        mock_bot.execute_decision.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_symbol_no_marks(self, orchestrator, mock_bot, sample_balances):
        """Test symbol processing with no market data"""
        mock_bot.mark_windows["BTC-USD"] = []
        mock_bot.broker.list_balances.return_value = sample_balances

        await orchestrator.process_symbol("BTC-USD")

        # Should not execute decision without marks
        mock_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_kill_switch(self, orchestrator, mock_bot, sample_balances):
        """Test symbol processing with kill switch enabled"""
        mock_bot.risk_manager.config.kill_switch_enabled = True
        mock_bot.broker.list_balances.return_value = sample_balances

        await orchestrator.process_symbol("BTC-USD")

        # Should not process when kill switch is enabled
        mock_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_stale_data(self, orchestrator, mock_bot, sample_balances):
        """Test symbol processing with stale market data"""
        mock_bot.mark_windows["BTC-USD"] = [Decimal("50000")]
        mock_bot.broker.list_balances.return_value = sample_balances
        mock_bot.risk_manager.check_mark_staleness.return_value = True

        await orchestrator.process_symbol("BTC-USD")

        # Should not process with stale data
        mock_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_symbol_with_existing_position(
        self, orchestrator, mock_bot, sample_balances, sample_positions, sample_product
    ):
        """Test symbol processing with existing position"""
        mock_bot.mark_windows["BTC-USD"] = [
            Decimal("49000"),
            Decimal("50000"),
        ]
        mock_bot.get_product.return_value = sample_product
        mock_bot.broker.list_balances.return_value = sample_balances
        mock_bot.broker.list_positions.return_value = sample_positions

        from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

        orchestrator.get_strategy = Mock()
        mock_strategy = Mock()
        mock_strategy.decide = Mock(
            return_value=Decision(
                action=Action.CLOSE,
                reason="take_profit",
                leverage=2,
                reduce_only=True,
            )
        )
        orchestrator.get_strategy.return_value = mock_strategy

        await orchestrator.process_symbol("BTC-USD")

        mock_bot.execute_decision.assert_called_once()
        # Verify position state was passed
        call_args = mock_bot.execute_decision.call_args[0]
        decision = call_args[1]
        assert decision.action == Action.CLOSE

    @pytest.mark.asyncio
    async def test_ensure_balances_provided(self, orchestrator, sample_balances):
        """Test balance ensuring when provided"""
        result = await orchestrator._ensure_balances(sample_balances)

        assert result == sample_balances

    @pytest.mark.asyncio
    async def test_ensure_balances_fetch(self, orchestrator, mock_bot, sample_balances):
        """Test balance fetching when not provided"""
        mock_bot.broker.list_balances.return_value = sample_balances

        result = await orchestrator._ensure_balances(None)

        assert result == sample_balances
        mock_bot.broker.list_balances.assert_called_once()

    def test_extract_equity_usd(self, orchestrator, sample_balances):
        """Test equity extraction with USD balance"""
        equity = orchestrator.equity_calculator.extract_cash_balance(sample_balances)

        assert equity == Decimal("10000")

    def test_extract_equity_usdc(self, orchestrator):
        """Test equity extraction with USDC balance"""
        balance = Mock(spec=Balance)
        balance.asset = "USDC"
        balance.total = Decimal("5000")

        equity = orchestrator.equity_calculator.extract_cash_balance([balance])

        assert equity == Decimal("5000")

    def test_extract_equity_no_cash(self, orchestrator):
        """Test equity extraction with no cash balance"""
        balance = Mock(spec=Balance)
        balance.asset = "BTC"
        balance.total = Decimal("1.5")

        equity = orchestrator.equity_calculator.extract_cash_balance([balance])

        assert equity == Decimal("0")

    def test_kill_switch_engaged(self, orchestrator, mock_bot):
        """Test kill switch detection"""
        mock_bot.risk_manager.config.kill_switch_enabled = True

        result = orchestrator._kill_switch_engaged()

        assert result is True

    def test_kill_switch_not_engaged(self, orchestrator, mock_bot):
        """Test kill switch not engaged"""
        mock_bot.risk_manager.config.kill_switch_enabled = False

        result = orchestrator._kill_switch_engaged()

        assert result is False

    @pytest.mark.asyncio
    async def test_ensure_positions_provided(self, orchestrator, sample_positions):
        """Test position ensuring when provided"""
        position_map = {p.symbol: p for p in sample_positions}

        result = await orchestrator._ensure_positions(position_map)

        assert result == position_map

    @pytest.mark.asyncio
    async def test_ensure_positions_fetch(self, orchestrator, mock_bot, sample_positions):
        """Test position fetching when not provided"""
        mock_bot.broker.list_positions.return_value = sample_positions

        result = await orchestrator._ensure_positions(None)

        assert "BTC-USD" in result
        mock_bot.broker.list_positions.assert_called_once()

    def test_build_position_state_with_position(self, orchestrator, sample_positions):
        """Test building position state with existing position"""
        positions_lookup = {p.symbol: p for p in sample_positions}

        state, quantity = orchestrator._build_position_state("BTC-USD", positions_lookup)

        assert state is not None
        assert state["quantity"] == Decimal("0.5")
        assert state["side"] == "long"
        assert quantity == Decimal("0.5")

    def test_build_position_state_no_position(self, orchestrator):
        """Test building position state with no position"""
        state, quantity = orchestrator._build_position_state("BTC-USD", {})

        assert state is None
        assert quantity == Decimal("0")

    def test_get_marks(self, orchestrator, mock_bot):
        """Test getting mark prices"""
        marks = [Decimal("50000"), Decimal("50100")]
        mock_bot.mark_windows["BTC-USD"] = marks

        result = orchestrator._get_marks("BTC-USD")

        assert result == marks

    def test_get_marks_missing(self, orchestrator, mock_bot):
        """Test getting marks for missing symbol"""
        result = orchestrator._get_marks("MISSING-USD")

        assert result == []

    def test_adjust_equity_with_position(self, orchestrator):
        """Test equity calculation with position"""
        balances = [Mock(spec=Balance, asset="USD", total=Decimal("10000"))]
        position_quantity = Decimal("0.5")
        current_mark = Decimal("50000")

        equity = orchestrator.equity_calculator.calculate(
            balances=balances,
            position_quantity=position_quantity,
            current_mark=current_mark,
            symbol="BTC-USD",
        )

        # Should add position value: 10000 + (0.5 * 50000) = 35000
        assert equity == Decimal("35000")

    def test_adjust_equity_no_position(self, orchestrator):
        """Test equity calculation without position"""
        balances = [Mock(spec=Balance, asset="USD", total=Decimal("10000"))]

        equity = orchestrator.equity_calculator.calculate(
            balances=balances,
            position_quantity=Decimal("0"),
            current_mark=None,
            symbol="BTC-USD",
        )

        assert equity == Decimal("10000")

    def test_run_risk_gates_pass(self, orchestrator, mock_bot):
        """Test risk gates passing"""
        marks = [Decimal(f"{50000 + i*100}") for i in range(30)]
        mock_bot.risk_manager.check_volatility_circuit_breaker.return_value = Mock(triggered=False)
        mock_bot.risk_manager.check_mark_staleness.return_value = False

        result = orchestrator.risk_gate_validator.validate_gates(
            "BTC-USD", marks, lookback_window=20
        )

        assert result is True

    def test_run_risk_gates_circuit_breaker(self, orchestrator, mock_bot):
        """Test risk gates with circuit breaker"""
        from bot_v2.features.live_trade.risk_runtime import CircuitBreakerAction

        marks = [Decimal("50000")]
        mock_bot.risk_manager.check_volatility_circuit_breaker.return_value = Mock(
            triggered=True,
            action=CircuitBreakerAction.KILL_SWITCH,
        )

        result = orchestrator.risk_gate_validator.validate_gates(
            "BTC-USD", marks, lookback_window=20
        )

        assert result is False

    def test_record_decision(self, orchestrator, mock_bot):
        """Test decision recording via StrategyExecutor"""
        from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

        decision = Decision(action=Action.BUY, reason="test", leverage=2)

        orchestrator.strategy_executor.record_decision("BTC-USD", decision)

        assert mock_bot.last_decisions["BTC-USD"] == decision

    @pytest.mark.asyncio
    async def test_process_symbol_exception_handling(self, orchestrator, mock_bot, sample_balances):
        """Test exception handling during symbol processing"""
        mock_bot.broker.list_balances.side_effect = Exception("API error")

        # Should not raise, just log error
        await orchestrator.process_symbol("BTC-USD")

        mock_bot.execute_decision.assert_not_called()

    def test_init_strategy_spot_invalid_position_fraction(self, orchestrator, mock_bot):
        """Test spot strategy initialization with invalid position_fraction"""
        mock_bot.config.profile = Profile.SPOT
        mock_bot.config.symbols = ["BTC-USD"]

        # Mock spot profiles to return invalid position_fraction
        orchestrator._spot_profiles.load = Mock(
            return_value={"BTC-USD": {"position_fraction": "invalid"}}
        )

        orchestrator.init_strategy()

        # Should still create strategy, just log warning
        assert "BTC-USD" in mock_bot._symbol_strategies

    def test_init_strategy_perps_invalid_position_fraction(self, orchestrator, mock_bot):
        """Test perps strategy initialization with invalid position_fraction"""
        mock_bot.config.perps_position_fraction = "invalid"

        orchestrator.init_strategy()

        # Should still create strategy, just log warning
        assert mock_bot.strategy is not None

    def test_adjust_equity_exception_handling(self, orchestrator):
        """Test equity calculation with exception in position value calculation"""
        balances = [Mock(spec=Balance, asset="USD", total=Decimal("10000"))]
        # Create a bad mark that raises exception when multiplied
        bad_mark = Mock()
        bad_mark.__mul__ = Mock(side_effect=ValueError("Invalid mark"))

        equity = orchestrator.equity_calculator.calculate(
            balances=balances,
            position_quantity=Decimal("0.5"),
            current_mark=bad_mark,
            symbol="BTC-USD",
        )

        # Should return cash-only equity on exception
        assert equity == Decimal("10000")

    def test_run_risk_gates_volatility_check_exception(self, orchestrator, mock_bot):
        """Test risk gates when volatility check raises exception"""
        marks = [Decimal("50000")]
        mock_bot.risk_manager.check_volatility_circuit_breaker.side_effect = Exception(
            "Calculation error"
        )
        mock_bot.risk_manager.check_mark_staleness.return_value = False

        result = orchestrator.risk_gate_validator.validate_gates(
            "BTC-USD", marks, lookback_window=20
        )

        # Should continue despite exception
        assert result is True

    def test_run_risk_gates_staleness_check_exception(self, orchestrator, mock_bot):
        """Test risk gates when staleness check raises exception"""
        marks = [Decimal("50000")]
        mock_bot.risk_manager.check_volatility_circuit_breaker.return_value = Mock(triggered=False)
        mock_bot.risk_manager.check_mark_staleness.side_effect = Exception("Check error")

        result = orchestrator.risk_gate_validator.validate_gates(
            "BTC-USD", marks, lookback_window=20
        )

        # Should continue despite exception
        assert result is True

    def test_evaluate_strategy(self, orchestrator, mock_bot, sample_product):
        """Test strategy evaluation via StrategyExecutor"""
        from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

        strategy = Mock()
        strategy.decide = Mock(return_value=Decision(action=Action.BUY, reason="test", leverage=2))
        marks = [Decimal("50000"), Decimal("50100")]
        mock_bot.get_product.return_value = sample_product

        decision = orchestrator.strategy_executor.evaluate(
            strategy, "BTC-USD", marks, None, Decimal("10000")
        )

        assert decision.action == Action.BUY
        strategy.decide.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_symbol_spot_profile_with_filters(
        self, orchestrator, mock_bot, sample_balances, sample_product
    ):
        """Test spot profile symbol processing with filters applied"""
        mock_bot.config.profile = Profile.SPOT
        mock_bot.mark_windows["BTC-USD"] = [Decimal("50000"), Decimal("50100")]
        mock_bot.get_product.return_value = sample_product
        mock_bot.broker.list_balances.return_value = sample_balances
        mock_bot.broker.list_positions.return_value = []

        # Mock spot profiles with filters
        orchestrator._spot_profiles.get = Mock(
            return_value={"volatility_filter": {"window": 20, "min_vol": 0.01, "max_vol": 0.05}}
        )

        from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

        orchestrator.get_strategy = Mock()
        mock_strategy = Mock()
        mock_strategy.decide = Mock(
            return_value=Decision(action=Action.BUY, reason="test", leverage=1)
        )
        orchestrator.get_strategy.return_value = mock_strategy

        # Mock candle fetch to return insufficient data
        orchestrator._fetch_spot_candles = AsyncMock(return_value=[])

        await orchestrator.process_symbol("BTC-USD")

        # Should not execute due to insufficient candle data
        mock_bot.execute_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_spot_filters_no_rules(self, orchestrator):
        """Test spot filters with no rules configured"""
        from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

        orchestrator._spot_profiles.get = Mock(return_value=None)
        decision = Decision(action=Action.BUY, reason="test", leverage=1)

        result = await orchestrator._apply_spot_filters("BTC-USD", decision, None)

        assert result == decision

    @pytest.mark.asyncio
    async def test_apply_spot_filters_non_buy_action(self, orchestrator):
        """Test spot filters with non-BUY action"""
        from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

        orchestrator._spot_profiles.get = Mock(return_value={})
        decision = Decision(action=Action.SELL, reason="test", leverage=1)

        result = await orchestrator._apply_spot_filters("BTC-USD", decision, None)

        assert result == decision

    @pytest.mark.asyncio
    async def test_apply_spot_filters_existing_position(self, orchestrator):
        """Test spot filters with existing position"""
        from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

        orchestrator._spot_profiles.get = Mock(return_value={})
        decision = Decision(action=Action.BUY, reason="test", leverage=1)
        position_state = {"quantity": Decimal("0.5")}

        result = await orchestrator._apply_spot_filters("BTC-USD", decision, position_state)

        assert result == decision

    @pytest.mark.asyncio
    async def test_apply_spot_filters_volume_filter(self, orchestrator):
        """Test spot filters with volume filter"""
        from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

        orchestrator._spot_profiles.get = Mock(
            return_value={"volume_filter": {"window": 5, "multiplier": 1.5}}
        )
        decision = Decision(action=Action.BUY, reason="test", leverage=1)

        # Mock candles with insufficient volume
        mock_candles = []
        for i in range(10):
            candle = Mock()
            candle.close = 50000 + i * 100
            candle.volume = 1000  # All same volume, won't pass multiplier check
            candle.high = 51000
            candle.low = 49000
            mock_candles.append(candle)

        orchestrator._fetch_spot_candles = AsyncMock(return_value=mock_candles)

        result = await orchestrator._apply_spot_filters("BTC-USD", decision, None)

        # Should block entry due to volume filter
        assert result.action == Action.HOLD
        assert "volume_filter" in result.reason

    @pytest.mark.asyncio
    async def test_apply_spot_filters_momentum_filter(self, orchestrator):
        """Test spot filters with momentum (RSI) filter"""
        from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

        orchestrator._spot_profiles.get = Mock(
            return_value={"momentum_filter": {"window": 14, "oversold": 30, "overbought": 70}}
        )
        decision = Decision(action=Action.BUY, reason="test", leverage=1)

        # Mock candles with increasing prices (RSI will be high)
        mock_candles = []
        for i in range(20):
            candle = Mock()
            candle.close = 50000 + i * 500  # Strong uptrend
            candle.volume = 1000
            candle.high = candle.close + 100
            candle.low = candle.close - 100
            mock_candles.append(candle)

        orchestrator._fetch_spot_candles = AsyncMock(return_value=mock_candles)

        result = await orchestrator._apply_spot_filters("BTC-USD", decision, None)

        # Should block entry if RSI > oversold threshold
        assert result.action == Action.HOLD
        assert "momentum_filter" in result.reason

    @pytest.mark.asyncio
    async def test_apply_spot_filters_trend_filter(self, orchestrator):
        """Test spot filters with trend filter"""
        from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

        orchestrator._spot_profiles.get = Mock(
            return_value={"trend_filter": {"window": 10, "min_slope": 100}}
        )
        decision = Decision(action=Action.BUY, reason="test", leverage=1)

        # Mock candles with flat trend
        mock_candles = []
        for i in range(15):
            candle = Mock()
            candle.close = 50000  # Flat, no slope
            candle.volume = 1000
            candle.high = 50100
            candle.low = 49900
            mock_candles.append(candle)

        orchestrator._fetch_spot_candles = AsyncMock(return_value=mock_candles)

        result = await orchestrator._apply_spot_filters("BTC-USD", decision, None)

        # Should block entry due to insufficient slope
        assert result.action == Action.HOLD
        assert "trend_filter" in result.reason

    @pytest.mark.asyncio
    async def test_apply_spot_filters_volatility_filter(self, orchestrator):
        """Test spot filters with volatility (ATR) filter"""
        from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

        orchestrator._spot_profiles.get = Mock(
            return_value={"volatility_filter": {"window": 10, "min_vol": 0.05, "max_vol": 0.10}}
        )
        decision = Decision(action=Action.BUY, reason="test", leverage=1)

        # Mock candles with low volatility
        mock_candles = []
        for i in range(15):
            candle = Mock()
            candle.close = 50000
            candle.volume = 1000
            candle.high = 50010  # Very small range
            candle.low = 49990
            mock_candles.append(candle)

        orchestrator._fetch_spot_candles = AsyncMock(return_value=mock_candles)

        result = await orchestrator._apply_spot_filters("BTC-USD", decision, None)

        # Should block entry due to low volatility
        assert result.action == Action.HOLD
        assert "volatility_filter" in result.reason

    @pytest.mark.asyncio
    async def test_apply_spot_filters_all_pass(self, orchestrator):
        """Test spot filters when all conditions pass"""
        from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

        orchestrator._spot_profiles.get = Mock(
            return_value={
                "volume_filter": {"window": 5, "multiplier": 1.2},
                "trend_filter": {"window": 5, "min_slope": 1},
            }
        )
        decision = Decision(action=Action.BUY, reason="test", leverage=1)

        # Mock candles with good conditions
        mock_candles = []
        for i in range(10):
            candle = Mock()
            candle.close = 50000 + i * 100  # Uptrend
            candle.volume = 1000 + i * 200  # Increasing volume
            candle.high = candle.close + 100
            candle.low = candle.close - 100
            mock_candles.append(candle)

        orchestrator._fetch_spot_candles = AsyncMock(return_value=mock_candles)

        result = await orchestrator._apply_spot_filters("BTC-USD", decision, None)

        # Should pass all filters
        assert result.action == Action.BUY

    @pytest.mark.asyncio
    async def test_fetch_spot_candles_success(self, orchestrator, mock_bot):
        """Test fetching spot candles successfully"""
        mock_candle = Mock()
        mock_candle.ts = _time.time()
        mock_bot.broker.get_candles.return_value = [mock_candle]

        candles = await orchestrator._fetch_spot_candles("BTC-USD", 20)

        assert len(candles) == 1
        mock_bot.broker.get_candles.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_spot_candles_exception(self, orchestrator, mock_bot):
        """Test fetching spot candles with exception"""
        mock_bot.broker.get_candles.side_effect = Exception("API error")

        candles = await orchestrator._fetch_spot_candles("BTC-USD", 20)

        assert candles == []

    @pytest.mark.asyncio
    async def test_process_symbol_zero_equity(self, orchestrator, mock_bot, sample_balances):
        """Test symbol processing with zero equity"""
        mock_bot.mark_windows["BTC-USD"] = [Decimal("50000")]
        balance = Mock(spec=Balance)
        balance.asset = "BTC"  # No cash balance
        balance.total = Decimal("1.5")
        mock_bot.broker.list_balances.return_value = [balance]
        mock_bot.broker.list_positions.return_value = []

        await orchestrator.process_symbol("BTC-USD")

        # Should not execute with zero equity
        mock_bot.execute_decision.assert_not_called()

    def test_build_position_state_quantity_conversion_error(self, orchestrator):
        """Test position state building with quantity conversion error"""
        position = Mock()
        position.symbol = "BTC-USD"
        position.quantity = "invalid"
        position.side = "long"
        positions_lookup = {"BTC-USD": position}

        state, quantity = orchestrator._build_position_state("BTC-USD", positions_lookup)

        # Should handle conversion error gracefully
        assert state is not None
        assert quantity == Decimal("0")
