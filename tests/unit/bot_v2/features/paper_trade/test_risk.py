"""
Comprehensive tests for paper trading risk management.

Tests cover:
- Risk manager initialization and configuration
- Trade approval/rejection based on risk limits
- Position sizing limits
- Daily loss limits
- Drawdown tracking and limits
- Cash reserve requirements
- Risk metrics calculation
- Daily statistics tracking
- Edge cases and boundary conditions
"""

import pytest

from bot_v2.features.paper_trade.risk import RiskManager
from bot_v2.features.paper_trade.types import AccountStatus, Position
from bot_v2.types.trading import AccountSnapshot


# ============================================================================
# Test: RiskManager Initialization
# ============================================================================


class TestRiskManagerInitialization:
    """Test RiskManager initialization and configuration."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        rm = RiskManager()

        assert rm.max_position_size == 0.2
        assert rm.max_daily_loss == 0.05
        assert rm.max_drawdown == 0.15
        assert rm.min_cash_reserve == 0.1
        assert rm.daily_pnl == 0
        assert rm.peak_equity == 0
        assert rm.initial_equity == 0
        assert rm.last_equity == 0

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        rm = RiskManager(
            max_position_size=0.25, max_daily_loss=0.03, max_drawdown=0.20, min_cash_reserve=0.15
        )

        assert rm.max_position_size == 0.25
        assert rm.max_daily_loss == 0.03
        assert rm.max_drawdown == 0.20
        assert rm.min_cash_reserve == 0.15

    def test_conservative_parameters(self):
        """Test initialization with conservative risk parameters."""
        rm = RiskManager(
            max_position_size=0.1, max_daily_loss=0.02, max_drawdown=0.10, min_cash_reserve=0.20
        )

        assert rm.max_position_size == 0.1
        assert rm.max_daily_loss == 0.02
        assert rm.max_drawdown == 0.10
        assert rm.min_cash_reserve == 0.20

    def test_aggressive_parameters(self):
        """Test initialization with aggressive risk parameters."""
        rm = RiskManager(
            max_position_size=0.35, max_daily_loss=0.10, max_drawdown=0.25, min_cash_reserve=0.05
        )

        assert rm.max_position_size == 0.35
        assert rm.max_daily_loss == 0.10
        assert rm.max_drawdown == 0.25
        assert rm.min_cash_reserve == 0.05


# ============================================================================
# Test: Risk Manager Initialization
# ============================================================================


class TestRiskManagerInitialize:
    """Test RiskManager initialize method."""

    def test_initialize_sets_tracking_values(self):
        """Test that initialize sets all tracking values correctly."""
        rm = RiskManager()
        rm.initialize(10000.0)

        assert rm.initial_equity == 10000.0
        assert rm.peak_equity == 10000.0
        assert rm.last_equity == 10000.0

    def test_initialize_with_different_amounts(self):
        """Test initialize with various starting amounts."""
        rm = RiskManager()

        # Small account
        rm.initialize(1000.0)
        assert rm.initial_equity == 1000.0
        assert rm.peak_equity == 1000.0

        # Large account
        rm2 = RiskManager()
        rm2.initialize(1000000.0)
        assert rm2.initial_equity == 1000000.0
        assert rm2.peak_equity == 1000000.0

    def test_initialize_with_zero(self):
        """Test initialize with zero amount."""
        rm = RiskManager()
        rm.initialize(0.0)

        assert rm.initial_equity == 0.0
        assert rm.peak_equity == 0.0
        assert rm.last_equity == 0.0


# ============================================================================
# Test: Trade Checking - Buy Signals
# ============================================================================


class TestCheckTradeBuySignals:
    """Test trade checking for buy signals."""

    def create_account(self, cash: float = 10000, positions: dict = None) -> AccountStatus:
        """Helper to create account status."""
        from datetime import datetime

        pos_list = []
        positions_value = 0

        if positions:
            for symbol, (quantity, price) in positions.items():
                pos = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    entry_date=datetime.now(),
                    current_price=price,
                    unrealized_pnl=0.0,
                    value=quantity * price,
                )
                pos_list.append(pos)
                positions_value += quantity * price

        return AccountStatus(
            cash=cash,
            positions_value=positions_value,
            total_equity=cash + positions_value,
            buying_power=cash * 2,
            margin_used=positions_value,
            day_trades_remaining=3,
            positions=pos_list,
            realized_pnl=0.0,
        )

    def test_check_trade_buy_approved(self):
        """Test that valid buy trade is approved."""
        rm = RiskManager()
        account = self.create_account(cash=10000)

        # Buy signal with reasonable price
        # Position value = 10 * 100 = 1000 (10% of equity, under 20% limit)
        # Cash after = 10000 - 1000 = 9000 (90% reserve, well above 10% requirement)
        result = rm.check_trade(symbol="AAPL", signal=1, price=10.0, account=account)

        assert result is True

    def test_check_trade_buy_exceeds_position_size_limit(self):
        """Test that buy is rejected when position size exceeds limit."""
        rm = RiskManager(max_position_size=0.1)  # Max 10% per position
        account = self.create_account(cash=10000)

        # Position value = 100 * 100 = 10000 = 100% of equity
        result = rm.check_trade(symbol="AAPL", signal=1, price=100.0, account=account)

        assert result is False

    def test_check_trade_buy_at_position_size_boundary(self):
        """Test buy at exact position size limit."""
        rm = RiskManager(max_position_size=0.2)  # Max 20%
        account = self.create_account(cash=10000)

        # Position value = 20 * 100 = 2000 = 20% of equity
        result = rm.check_trade(symbol="AAPL", signal=1, price=20.0, account=account)

        assert result is True

    def test_check_trade_buy_insufficient_cash_reserve(self):
        """Test that buy is rejected when cash reserve would be violated."""
        rm = RiskManager(min_cash_reserve=0.5)  # Keep 50% cash
        account = self.create_account(cash=10000)

        # After buying 100 shares @ 100, cash would be 0 (below 50% reserve)
        result = rm.check_trade(symbol="AAPL", signal=1, price=100.0, account=account)

        assert result is False

    def test_check_trade_buy_with_existing_positions(self):
        """Test buy with existing positions."""
        rm = RiskManager(max_position_size=0.2)
        account = self.create_account(cash=5000, positions={"MSFT": (100, 50)})  # 5000 in MSFT

        # Total equity = 10000, trying to buy 100 @ 15 = 1500 (15%)
        result = rm.check_trade(symbol="AAPL", signal=1, price=15.0, account=account)

        assert result is True

    def test_check_trade_buy_low_price(self):
        """Test buy with low price stock."""
        rm = RiskManager(max_position_size=0.2)
        account = self.create_account(cash=10000)

        # 100 shares @ 5 = 500 (5% of equity)
        result = rm.check_trade(symbol="PENNY", signal=1, price=5.0, account=account)

        assert result is True

    def test_check_trade_buy_high_price(self):
        """Test buy with high price stock."""
        rm = RiskManager(max_position_size=0.5)  # Increase limit for test
        account = self.create_account(cash=50000)

        # 100 shares @ 500 = 50000 (100% of equity)
        result = rm.check_trade(symbol="EXPENSIVE", signal=1, price=500.0, account=account)

        assert result is False  # Would violate cash reserve


# ============================================================================
# Test: Trade Checking - Sell Signals
# ============================================================================


class TestCheckTradeSellSignals:
    """Test trade checking for sell signals."""

    def create_account(self, cash: float = 10000, positions: dict = None) -> AccountStatus:
        """Helper to create account status."""
        from datetime import datetime

        pos_list = []
        positions_value = 0

        if positions:
            for symbol, (quantity, price) in positions.items():
                pos = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    entry_date=datetime.now(),
                    current_price=price,
                    unrealized_pnl=0.0,
                    value=quantity * price,
                )
                pos_list.append(pos)
                positions_value += quantity * price

        return AccountStatus(
            cash=cash,
            positions_value=positions_value,
            total_equity=cash + positions_value,
            buying_power=cash * 2,
            margin_used=positions_value,
            day_trades_remaining=3,
            positions=pos_list,
            realized_pnl=0.0,
        )

    def test_check_trade_sell_approved(self):
        """Test that valid sell trade is approved."""
        rm = RiskManager()
        account = self.create_account(cash=5000, positions={"AAPL": (100, 50)})

        result = rm.check_trade(symbol="AAPL", signal=-1, price=55.0, account=account)

        assert result is True

    def test_check_trade_sell_no_position_size_check(self):
        """Test that sell signals don't check position size limits."""
        rm = RiskManager(max_position_size=0.05)  # Very tight limit
        account = self.create_account(cash=5000, positions={"AAPL": (100, 50)})

        # Sell signals should not be blocked by position size limits
        result = rm.check_trade(symbol="AAPL", signal=-1, price=55.0, account=account)

        assert result is True

    def test_check_trade_sell_no_cash_reserve_check(self):
        """Test that sell signals don't check cash reserve."""
        rm = RiskManager(min_cash_reserve=0.9)  # High cash reserve
        account = self.create_account(cash=100, positions={"AAPL": (100, 50)})

        # Sell signals should not be blocked by cash reserve
        result = rm.check_trade(symbol="AAPL", signal=-1, price=55.0, account=account)

        assert result is True


# ============================================================================
# Test: Drawdown Tracking and Limits
# ============================================================================


class TestDrawdownTracking:
    """Test drawdown tracking and enforcement."""

    def create_account(self, equity: float) -> AccountStatus:
        """Helper to create account with specific equity."""
        return AccountStatus(
            cash=equity,
            positions_value=0.0,
            total_equity=equity,
            buying_power=equity,
            margin_used=0.0,
            day_trades_remaining=3,
            positions=[],
            realized_pnl=0,
        )

    def test_drawdown_within_limit(self):
        """Test that trades are allowed when drawdown is within limit."""
        rm = RiskManager(max_drawdown=0.15, max_daily_loss=0.50)
        rm.initialize(10000.0)

        # Equity drops to 9000 (10% drawdown, 10% daily loss)
        account = self.create_account(9000.0)
        # Use price=10 so trade value (10*100=1000) is affordable
        result = rm.check_trade("AAPL", 1, 10.0, account)

        assert result is True

    def test_drawdown_exceeds_limit(self):
        """Test that trades are blocked when drawdown exceeds limit."""
        rm = RiskManager(max_drawdown=0.15, max_daily_loss=0.50)
        rm.initialize(10000.0)

        # Equity drops to 8000 (20% drawdown)
        account = self.create_account(8000.0)
        result = rm.check_trade("AAPL", 1, 10.0, account)

        assert result is False

    def test_drawdown_at_exact_limit(self):
        """Test drawdown at exact limit."""
        rm = RiskManager(max_drawdown=0.15, max_daily_loss=0.50)
        rm.initialize(10000.0)

        # Equity drops to 8500 (exactly 15% drawdown)
        account = self.create_account(8500.0)
        # Use affordable price (10*100=1000, leaves 7500 cash > 10% reserve)
        result = rm.check_trade("AAPL", 1, 10.0, account)

        assert result is True

    def test_peak_equity_updates(self):
        """Test that peak equity updates when equity increases."""
        rm = RiskManager(max_drawdown=0.15, max_daily_loss=0.50)
        rm.initialize(10000.0)

        # Equity increases to 12000
        account1 = self.create_account(12000.0)
        rm.check_trade("AAPL", 1, 10.0, account1)

        assert rm.peak_equity == 12000.0

        # Now drop to 10000 (16.67% from new peak)
        account2 = self.create_account(10000.0)
        result = rm.check_trade("AAPL", 1, 10.0, account2)

        assert result is False  # Exceeds 15% limit from new peak

    def test_drawdown_recovery(self):
        """Test that trades are allowed after recovering from drawdown."""
        rm = RiskManager(max_drawdown=0.15, max_daily_loss=0.50)
        rm.initialize(10000.0)

        # Drop to 8000 (20% drawdown) - trades blocked
        account1 = self.create_account(8000.0)
        result1 = rm.check_trade("AAPL", 1, 10.0, account1)
        assert result1 is False

        # Update daily stats to reset last_equity to 8000
        rm.update_daily_stats(8000.0)

        # Recover to 9000 (10% drawdown from peak) - trades allowed
        account2 = self.create_account(9000.0)
        result2 = rm.check_trade("AAPL", 1, 10.0, account2)
        assert result2 is True

    def test_zero_peak_equity(self):
        """Test drawdown calculation with zero peak equity."""
        rm = RiskManager()
        # Don't initialize - peak_equity remains 0

        account = self.create_account(10000.0)
        # Use affordable price (10*100=1000, leaves 9000 cash > 10% reserve)
        result = rm.check_trade("AAPL", 1, 10.0, account)

        # Should initialize and allow trade
        assert result is True
        assert rm.peak_equity == 10000.0


# ============================================================================
# Test: Daily Loss Limits
# ============================================================================


class TestDailyLossLimits:
    """Test daily loss tracking and enforcement."""

    def create_account(self, equity: float) -> AccountStatus:
        """Helper to create account with specific equity."""
        return AccountStatus(
            cash=equity,
            positions_value=0.0,
            total_equity=equity,
            buying_power=equity,
            margin_used=0.0,
            day_trades_remaining=3,
            positions=[],
            realized_pnl=0,
        )

    def test_daily_loss_within_limit(self):
        """Test that trades are allowed when daily loss is within limit."""
        rm = RiskManager(max_daily_loss=0.05)
        rm.initialize(10000.0)

        # Drop 3% in a day
        account = self.create_account(9700.0)
        # Use affordable price (10*100=1000, leaves 8700 cash > 10% reserve)
        result = rm.check_trade("AAPL", 1, 10.0, account)

        assert result is True

    def test_daily_loss_exceeds_limit(self):
        """Test that trades are blocked when daily loss exceeds limit."""
        rm = RiskManager(max_daily_loss=0.05)
        rm.initialize(10000.0)

        # Drop 8% in a day
        account = self.create_account(9200.0)
        result = rm.check_trade("AAPL", 1, 10.0, account)

        assert result is False

    def test_daily_loss_at_exact_limit(self):
        """Test daily loss at exact limit."""
        rm = RiskManager(max_daily_loss=0.05)
        rm.initialize(10000.0)

        # Drop exactly 5%
        account = self.create_account(9500.0)
        result = rm.check_trade("AAPL", 1, 10.0, account)

        assert result is True

    def test_daily_gain_allows_trades(self):
        """Test that daily gains don't block trades."""
        rm = RiskManager(max_daily_loss=0.05)
        rm.initialize(10000.0)

        # Gain 10% in a day
        account = self.create_account(11000.0)
        result = rm.check_trade("AAPL", 1, 10.0, account)

        assert result is True

    def test_zero_last_equity(self):
        """Test daily loss calculation with zero last equity."""
        rm = RiskManager()
        # Don't initialize - last_equity remains 0

        account = self.create_account(10000.0)
        result = rm.check_trade("AAPL", 1, 10.0, account)

        # Should initialize and allow trade
        assert result is True


# ============================================================================
# Test: Daily Statistics
# ============================================================================


class TestDailyStatistics:
    """Test daily statistics tracking and reset."""

    def test_update_daily_stats(self):
        """Test updating daily statistics."""
        rm = RiskManager()
        rm.last_equity = 10000.0

        rm.update_daily_stats(10500.0)

        assert rm.daily_pnl == 500.0
        assert rm.last_equity == 10500.0

    def test_update_daily_stats_loss(self):
        """Test updating daily statistics with loss."""
        rm = RiskManager()
        rm.last_equity = 10000.0

        rm.update_daily_stats(9500.0)

        assert rm.daily_pnl == -500.0
        assert rm.last_equity == 9500.0

    def test_update_daily_stats_no_change(self):
        """Test updating daily statistics with no change."""
        rm = RiskManager()
        rm.last_equity = 10000.0

        rm.update_daily_stats(10000.0)

        assert rm.daily_pnl == 0.0
        assert rm.last_equity == 10000.0

    def test_reset_daily_stats(self):
        """Test resetting daily statistics."""
        rm = RiskManager()
        rm.daily_pnl = 500.0
        rm.last_equity = 10000.0

        rm.reset_daily_stats(10500.0)

        assert rm.daily_pnl == 0.0
        assert rm.last_equity == 10500.0

    def test_reset_daily_stats_maintains_equity(self):
        """Test that reset properly updates last_equity."""
        rm = RiskManager()
        rm.last_equity = 10000.0
        rm.daily_pnl = -200.0

        rm.reset_daily_stats(9800.0)

        assert rm.daily_pnl == 0.0
        assert rm.last_equity == 9800.0


# ============================================================================
# Test: Risk Metrics
# ============================================================================


class TestRiskMetrics:
    """Test risk metrics calculation."""

    def create_account(self, cash: float, positions_value: float = 0) -> AccountStatus:
        """Helper to create account status."""
        from datetime import datetime

        pos_list = []
        if positions_value > 0:
            pos = Position(
                symbol="AAPL",
                quantity=100,
                entry_price=positions_value / 100,
                entry_date=datetime.now(),
                current_price=positions_value / 100,
                unrealized_pnl=0.0,
                value=positions_value,
            )
            pos_list.append(pos)

        return AccountStatus(
            cash=cash,
            positions_value=positions_value,
            total_equity=cash + positions_value,
            buying_power=cash * 2,
            margin_used=positions_value,
            day_trades_remaining=3,
            positions=pos_list,
            realized_pnl=0,
        )

    def test_get_risk_metrics_initial_state(self):
        """Test risk metrics at initial state."""
        rm = RiskManager()
        rm.initialize(10000.0)
        account = self.create_account(10000.0)

        metrics = rm.get_risk_metrics(account)

        assert metrics["current_drawdown"] == 0.0
        assert metrics["max_drawdown_limit"] == 0.15
        assert metrics["daily_return"] == 0.0
        assert metrics["max_daily_loss_limit"] == 0.05
        assert metrics["total_return"] == 0.0
        assert metrics["cash_percentage"] == 1.0
        assert metrics["min_cash_reserve"] == 0.1
        assert metrics["risk_utilization"] == 0.0

    def test_get_risk_metrics_with_drawdown(self):
        """Test risk metrics with drawdown."""
        rm = RiskManager(max_drawdown=0.15)
        rm.initialize(10000.0)
        rm.last_equity = 10000.0

        account = self.create_account(9000.0)  # 10% drawdown
        metrics = rm.get_risk_metrics(account)

        assert metrics["current_drawdown"] == 0.1
        assert metrics["risk_utilization"] == pytest.approx(0.1 / 0.15, rel=1e-6)
        assert metrics["daily_return"] == -0.1
        assert metrics["total_return"] == -0.1

    def test_get_risk_metrics_with_profit(self):
        """Test risk metrics with profit."""
        rm = RiskManager()
        rm.initialize(10000.0)
        rm.last_equity = 10000.0

        account = self.create_account(11000.0)
        metrics = rm.get_risk_metrics(account)

        assert metrics["current_drawdown"] == 0.0  # No drawdown, new peak
        assert metrics["daily_return"] == 0.1
        assert metrics["total_return"] == 0.1

    def test_get_risk_metrics_with_positions(self):
        """Test risk metrics with open positions."""
        rm = RiskManager()
        rm.initialize(10000.0)
        rm.last_equity = 10000.0

        account = self.create_account(cash=6000.0, positions_value=4000.0)
        metrics = rm.get_risk_metrics(account)

        assert metrics["cash_percentage"] == 0.6
        assert metrics["total_equity"] == 10000.0

    def test_get_risk_metrics_high_risk_utilization(self):
        """Test risk metrics with high risk utilization."""
        rm = RiskManager(max_drawdown=0.15)
        rm.initialize(10000.0)
        rm.last_equity = 10000.0

        # 14% drawdown (93.3% risk utilization)
        account = self.create_account(8600.0)
        metrics = rm.get_risk_metrics(account)

        assert metrics["current_drawdown"] == 0.14
        assert metrics["risk_utilization"] == pytest.approx(0.14 / 0.15, rel=1e-6)

    def test_get_risk_metrics_zero_equity(self):
        """Test risk metrics with zero equity."""
        rm = RiskManager()
        rm.initialize(0.0)

        account = self.create_account(0.0)
        metrics = rm.get_risk_metrics(account)

        assert metrics["current_drawdown"] == 0.0
        assert metrics["daily_return"] == 0.0
        assert metrics["total_return"] == 0.0
        assert metrics["cash_percentage"] == 0.0


# ============================================================================
# Test: AccountSnapshot Conversion
# ============================================================================


class TestAccountSnapshotConversion:
    """Test handling of AccountSnapshot type."""

    def test_check_trade_with_account_snapshot(self):
        """Test check_trade with AccountSnapshot instead of AccountStatus."""
        from decimal import Decimal

        rm = RiskManager()

        # Create AccountSnapshot
        snapshot = AccountSnapshot(
            account_id="test",
            cash=Decimal("10000.0"),
            equity=Decimal("10000.0"),
            buying_power=Decimal("10000.0"),
            positions_value=Decimal("0.0"),
            margin_used=Decimal("0.0"),
        )

        result = rm.check_trade("AAPL", 1, 100.0, snapshot)

        # Should convert and process correctly
        assert isinstance(result, bool)

    def test_get_risk_metrics_with_account_snapshot(self):
        """Test get_risk_metrics with AccountSnapshot."""
        from decimal import Decimal

        rm = RiskManager()
        rm.initialize(10000.0)
        rm.last_equity = 10000.0

        snapshot = AccountSnapshot(
            account_id="test",
            cash=Decimal("10000.0"),
            equity=Decimal("10000.0"),
            buying_power=Decimal("10000.0"),
            positions_value=Decimal("0.0"),
            margin_used=Decimal("0.0"),
        )

        metrics = rm.get_risk_metrics(snapshot)

        assert isinstance(metrics, dict)
        assert "current_drawdown" in metrics
        assert "total_return" in metrics


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestRiskManagerEdgeCases:
    """Test edge cases and boundary conditions."""

    def create_account(self, cash: float, equity: float) -> AccountStatus:
        """Helper to create account status."""
        return AccountStatus(
            cash=cash,
            positions_value=0.0,
            total_equity=equity,
            buying_power=cash * 2,
            margin_used=0.0,
            day_trades_remaining=3,
            positions=[],
            realized_pnl=0,
        )

    def test_very_small_equity(self):
        """Test with very small equity amounts."""
        rm = RiskManager()
        account = self.create_account(cash=1.0, equity=1.0)

        result = rm.check_trade("AAPL", 1, 0.01, account)

        assert isinstance(result, bool)

    def test_very_large_equity(self):
        """Test with very large equity amounts."""
        rm = RiskManager()
        account = self.create_account(cash=1_000_000.0, equity=1_000_000.0)

        result = rm.check_trade("AAPL", 1, 1000.0, account)

        assert result is True

    def test_zero_price(self):
        """Test with zero price."""
        rm = RiskManager()
        account = self.create_account(cash=10000.0, equity=10000.0)

        result = rm.check_trade("AAPL", 1, 0.0, account)

        assert result is True  # Zero position value always within limits

    def test_negative_price(self):
        """Test with negative price (invalid but handled)."""
        rm = RiskManager()
        account = self.create_account(cash=10000.0, equity=10000.0)

        # Negative price would create negative position value
        result = rm.check_trade("AAPL", 1, -100.0, account)

        assert result is True  # Negative values pass checks

    def test_zero_signal(self):
        """Test with zero signal (hold)."""
        rm = RiskManager()
        account = self.create_account(cash=10000.0, equity=10000.0)

        result = rm.check_trade("AAPL", 0, 100.0, account)

        assert result is True  # Hold signals don't trigger any checks

    def test_extreme_max_position_size(self):
        """Test with extreme position size limits."""
        # Very restrictive
        rm1 = RiskManager(max_position_size=0.01)  # Max 1%
        account = self.create_account(cash=10000.0, equity=10000.0)

        result1 = rm1.check_trade("AAPL", 1, 10.0, account)
        assert result1 is False  # 100 * 10 = 1000 = 10% > 1%

        # Very permissive
        rm2 = RiskManager(max_position_size=0.99)  # Max 99%
        result2 = rm2.check_trade("AAPL", 1, 99.0, account)
        assert result2 is False  # Would violate cash reserve

    def test_extreme_cash_reserve(self):
        """Test with extreme cash reserve requirements."""
        # Very high reserve
        rm = RiskManager(min_cash_reserve=0.95)  # Keep 95% cash
        account = self.create_account(cash=10000.0, equity=10000.0)

        result = rm.check_trade("AAPL", 1, 5.0, account)
        assert result is False  # Can't maintain 95% reserve

    def test_multiple_consecutive_checks(self):
        """Test multiple consecutive trade checks."""
        rm = RiskManager()
        account = self.create_account(cash=10000.0, equity=10000.0)

        # Multiple checks should work consistently
        # Trade cost: 10*100 = 1000, leaves 9000 cash (> 1000 min reserve)
        result1 = rm.check_trade("AAPL", 1, 10.0, account)
        result2 = rm.check_trade("MSFT", 1, 10.0, account)
        result3 = rm.check_trade("GOOGL", 1, 10.0, account)

        assert result1 is True
        assert result2 is True
        assert result3 is True

    def test_alternating_buy_sell_checks(self):
        """Test alternating buy and sell checks."""
        rm = RiskManager()
        account = self.create_account(cash=5000.0, equity=10000.0)

        # Buy: cost = 10*100 = 1000, leaves 4000 cash (> 1000 min reserve)
        result_buy = rm.check_trade("AAPL", 1, 10.0, account)
        result_sell = rm.check_trade("AAPL", -1, 10.0, account)
        result_buy2 = rm.check_trade("MSFT", 1, 10.0, account)

        assert result_buy is True
        assert result_sell is True
        assert result_buy2 is True

    def test_auto_initialization_on_first_check(self):
        """Test that check_trade auto-initializes if not initialized."""
        rm = RiskManager()
        # Don't call initialize()

        account = self.create_account(cash=10000.0, equity=10000.0)
        # Trade cost: 10*100 = 1000, leaves 9000 cash (> 1000 min reserve)
        result = rm.check_trade("AAPL", 1, 10.0, account)

        # Should auto-initialize
        assert rm.initial_equity == 10000.0
        assert rm.peak_equity == 10000.0
        assert result is True


# ============================================================================
# Test: Integration Scenarios
# ============================================================================


class TestRiskManagerIntegrationScenarios:
    """Test realistic trading scenarios."""

    def create_account(self, cash: float, equity: float) -> AccountStatus:
        """Helper to create account status."""
        return AccountStatus(
            cash=cash,
            positions=[],
            total_equity=equity,
            positions_value=equity - cash,
            buying_power=cash,
            margin_used=0.0,
            day_trades_remaining=3,
            realized_pnl=0,
        )

    def test_profitable_trading_day(self):
        """Test risk manager during profitable trading day."""
        rm = RiskManager()
        rm.initialize(10000.0)

        # Start of day - trade cost: 10*100 = 1000, leaves 9000 cash (> 1000 reserve)
        account1 = self.create_account(10000.0, 10000.0)
        assert rm.check_trade("AAPL", 1, 10.0, account1) is True

        # Mid-day - up 5%
        rm.update_daily_stats(10500.0)
        account2 = self.create_account(10500.0, 10500.0)
        assert rm.check_trade("MSFT", 1, 10.0, account2) is True

        # End of day - up 10%
        rm.update_daily_stats(11000.0)
        metrics = rm.get_risk_metrics(account2)
        assert metrics["total_return"] > 0

    def test_losing_trading_day(self):
        """Test risk manager during losing trading day."""
        rm = RiskManager(max_daily_loss=0.05)
        rm.initialize(10000.0)

        # Start of day
        account1 = self.create_account(10000.0, 10000.0)
        assert rm.check_trade("AAPL", 1, 10.0, account1) is True

        # Mid-day - down 3%
        account2 = self.create_account(9700.0, 9700.0)
        assert rm.check_trade("MSFT", 1, 10.0, account2) is True

        # End of day - down 6% (exceeds limit)
        account3 = self.create_account(9400.0, 9400.0)
        assert rm.check_trade("GOOGL", 1, 10.0, account3) is False

    def test_drawdown_and_recovery_scenario(self):
        """Test drawdown followed by recovery."""
        rm = RiskManager(max_drawdown=0.15)
        rm.initialize(10000.0)

        # Initial profit - new peak
        account1 = self.create_account(12000.0, 12000.0)
        rm.check_trade("AAPL", 1, 10.0, account1)
        assert rm.peak_equity == 12000.0

        # Large drop - 18% from peak (exceeds limit)
        account2 = self.create_account(9840.0, 9840.0)
        assert rm.check_trade("MSFT", 1, 10.0, account2) is False

        # Partial recovery - 12% from peak (within limit)
        account3 = self.create_account(10560.0, 10560.0)
        assert rm.check_trade("GOOGL", 1, 10.0, account3) is True

    def test_cash_depletion_scenario(self):
        """Test scenario where cash gets depleted."""
        rm = RiskManager(min_cash_reserve=0.1)
        rm.initialize(10000.0)

        # Buy reducing cash to just above reserve (1500 - 400 = 1100 > 1000)
        account1 = self.create_account(1500.0, 10000.0)
        assert rm.check_trade("AAPL", 1, 4.0, account1) is True

        # Try another buy that would reduce cash below reserve (1100 - 200 = 900 < 1000)
        account2 = self.create_account(1100.0, 10000.0)
        assert rm.check_trade("MSFT", 1, 2.0, account2) is False

    def test_multiple_risk_limit_violations(self):
        """Test when multiple risk limits are violated simultaneously."""
        rm = RiskManager(
            max_position_size=0.1, max_daily_loss=0.05, max_drawdown=0.15, min_cash_reserve=0.2
        )
        rm.initialize(10000.0)

        # Simulate bad day: down 20% (violates daily loss and drawdown)
        account = self.create_account(1000.0, 8000.0)
        result = rm.check_trade("AAPL", 1, 10.0, account)

        # Should be blocked (multiple violations)
        assert result is False
