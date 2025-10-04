"""
Unit tests for MarginCalculator.

Comprehensive edge-case coverage for margin/leverage calculations
with property-based assertions for liquidation safety.
"""

from decimal import Decimal

import pytest

from bot_v2.features.live_trade.margin_calculator import MarginCalculator, MarginMetrics


class TestBasicMarginCalculation:
    """Tests for basic margin calculations."""

    def test_standard_margin_calculation(self):
        """Test standard margin calculation with 10x leverage."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("100000"),  # $100k position
            cash_balance=Decimal("10000"),  # $10k cash
        )

        # Initial margin = 100k / 10 = 10k
        assert metrics.initial_margin_required == Decimal("10000")
        # Maintenance margin = 100k * 0.05 = 5k
        assert metrics.maintenance_margin_required == Decimal("5000")
        # Leverage = 100k / 10k = 10x
        assert metrics.leverage == Decimal("10")
        # Margin available = 10k - 5k = 5k
        assert metrics.margin_available == Decimal("5000")

    def test_margin_with_unrealized_pnl(self):
        """Test margin calculation includes unrealized PnL in equity."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("100000"),
            cash_balance=Decimal("10000"),
            unrealized_pnl=Decimal("2000"),  # $2k unrealized profit
        )

        # Equity = 10k + 2k = 12k
        assert metrics.equity == Decimal("12000")
        # Leverage = 100k / 12k = 8.33x
        assert metrics.leverage == Decimal("100000") / Decimal("12000")
        # Margin available = 12k - 5k = 7k
        assert metrics.margin_available == Decimal("7000")

    def test_margin_with_unrealized_loss(self):
        """Test margin calculation with unrealized loss."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("100000"),
            cash_balance=Decimal("10000"),
            unrealized_pnl=Decimal("-3000"),  # $3k unrealized loss
        )

        # Equity = 10k - 3k = 7k
        assert metrics.equity == Decimal("7000")
        # Leverage = 100k / 7k = 14.29x
        expected_leverage = Decimal("100000") / Decimal("7000")
        assert metrics.leverage == expected_leverage
        # Margin available = 7k - 5k = 2k
        assert metrics.margin_available == Decimal("2000")


class TestZeroAndNegativeEquity:
    """Tests for zero and negative equity edge cases."""

    def test_zero_equity_liquidation_risk(self):
        """Test zero equity triggers liquidation risk."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("50000"),
            cash_balance=Decimal("0"),
        )

        assert metrics.margin_health == "liquidation_risk"
        assert metrics.leverage == Decimal("999")  # Effectively infinite
        assert metrics.margin_available == Decimal("0")
        assert len(metrics.warnings) > 0
        assert "liquidation" in metrics.warnings[0].lower()

    def test_negative_equity_liquidation_risk(self):
        """Test negative equity triggers liquidation risk."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("50000"),
            cash_balance=Decimal("5000"),
            unrealized_pnl=Decimal("-10000"),  # Loss exceeds cash
        )

        assert metrics.equity == Decimal("-5000")
        assert metrics.margin_health == "liquidation_risk"
        assert "liquidation" in metrics.warnings[0].lower()

    def test_small_positive_equity_still_liquidation_risk(self):
        """Test tiny positive equity with large position is liquidation risk."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("100000"),
            cash_balance=Decimal("100"),  # Very small cash
        )

        # Maintenance margin = 100k * 0.05 = 5k
        # Margin available = 100 - 5000 = -4900 (negative)
        assert metrics.margin_available < 0
        assert metrics.margin_health == "liquidation_risk"


class TestZeroPositions:
    """Tests for zero positions edge case."""

    def test_zero_positions_healthy(self):
        """Test zero positions with positive cash is healthy."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("0"),
            cash_balance=Decimal("10000"),
        )

        assert metrics.positions_value == Decimal("0")
        assert metrics.leverage == Decimal("0")
        assert metrics.margin_available == Decimal("10000")
        assert metrics.margin_health == "healthy"
        assert metrics.margin_buffer_pct == Decimal("100")
        assert len(metrics.warnings) == 0

    def test_zero_positions_with_unrealized_pnl(self):
        """Test zero positions with unrealized PnL."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("0"),
            cash_balance=Decimal("5000"),
            unrealized_pnl=Decimal("1000"),
        )

        assert metrics.equity == Decimal("6000")
        assert metrics.margin_available == Decimal("6000")
        assert metrics.margin_health == "healthy"


class TestHighLeverageScenarios:
    """Tests for high leverage scenarios."""

    def test_leverage_exactly_at_max(self):
        """Test leverage exactly at max (10x) is healthy."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("100000"),
            cash_balance=Decimal("10000"),
        )

        assert metrics.leverage == Decimal("10")
        assert metrics.margin_health == "healthy"  # At limit but healthy
        assert len(metrics.warnings) == 0

    def test_leverage_exceeds_max(self):
        """Test leverage exceeding max triggers warning."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("150000"),
            cash_balance=Decimal("10000"),
        )

        # Leverage = 150k / 10k = 15x (exceeds 10x max)
        assert metrics.leverage == Decimal("15")
        assert metrics.margin_health == "warning"
        assert any("leverage" in w.lower() for w in metrics.warnings)

    def test_extreme_leverage(self):
        """Test extreme leverage scenario."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("200000"),
            cash_balance=Decimal("5000"),
        )

        # Leverage = 200k / 5k = 40x
        assert metrics.leverage == Decimal("40")
        # Maintenance margin = 200k * 0.05 = 10k
        # Margin available = 5k - 10k = -5k
        assert metrics.margin_available < 0
        assert metrics.margin_health == "liquidation_risk"


class TestWarningThresholds:
    """Tests for margin health warning thresholds."""

    def test_healthy_margin_no_warnings(self):
        """Test healthy margin (>20% buffer) has no warnings."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("40000"),
            cash_balance=Decimal("10000"),
        )

        # Maintenance margin = 40k * 0.05 = 2k
        # Margin available = 10k - 2k = 8k
        # Buffer % = 8k / 10k = 80%
        assert metrics.margin_buffer_pct == Decimal("80")
        assert metrics.margin_health == "healthy"
        assert len(metrics.warnings) == 0

    def test_warning_threshold_triggers(self):
        """Test warning threshold (15% buffer) triggers warning."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("170000"),
            cash_balance=Decimal("10000"),
        )

        # Maintenance margin = 170k * 0.05 = 8.5k
        # Margin available = 10k - 8.5k = 1.5k
        # Buffer % = 1.5k / 10k = 15%
        assert metrics.margin_buffer_pct == Decimal("15")
        assert metrics.margin_health == "warning"
        assert "WARNING" in metrics.warnings[0]

    def test_critical_threshold_triggers(self):
        """Test critical threshold (8% buffer) triggers critical."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("184000"),
            cash_balance=Decimal("10000"),
        )

        # Maintenance margin = 184k * 0.05 = 9.2k
        # Margin available = 10k - 9.2k = 0.8k
        # Buffer % = 0.8k / 10k = 8%
        assert metrics.margin_buffer_pct == Decimal("8")
        assert metrics.margin_health == "critical"
        assert "CRITICAL" in metrics.warnings[0]

    def test_healthy_above_warning_boundary(self):
        """Test healthy margin well above warning boundary."""
        # With 10x max leverage and 5% maintenance margin:
        # At max leverage, buffer = (equity - positions * 0.05) / equity
        #                         = (equity - equity * 10 * 0.05) / equity = 50%
        # So staying within leverage limits means buffer >= 50%

        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("80000"),
            cash_balance=Decimal("10000"),
        )

        # Leverage = 80k / 10k = 8x (within limits)
        # Maintenance margin = 80k * 0.05 = 4k
        # Margin available = 10k - 4k = 6k
        # Buffer % = 6k / 10k = 60%
        assert metrics.margin_buffer_pct == Decimal("60")
        assert metrics.margin_health == "healthy"
        assert len(metrics.warnings) == 0


class TestVaryingMaintenanceRequirements:
    """Tests for different maintenance margin rates."""

    def test_lower_maintenance_margin(self):
        """Test lower maintenance margin (3%) allows more leverage."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("100000"),
            cash_balance=Decimal("10000"),
            maintenance_margin_rate=Decimal("0.03"),  # 3% instead of 5%
        )

        # Maintenance margin = 100k * 0.03 = 3k
        assert metrics.maintenance_margin_required == Decimal("3000")
        # Margin available = 10k - 3k = 7k
        assert metrics.margin_available == Decimal("7000")
        # Buffer % = 7k / 10k = 70%
        assert metrics.margin_buffer_pct == Decimal("70")
        assert metrics.margin_health == "healthy"

    def test_higher_maintenance_margin(self):
        """Test higher maintenance margin (10%) reduces available margin."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("50000"),
            cash_balance=Decimal("10000"),
            maintenance_margin_rate=Decimal("0.10"),  # 10% instead of 5%
        )

        # Maintenance margin = 50k * 0.10 = 5k
        assert metrics.maintenance_margin_required == Decimal("5000")
        # Margin available = 10k - 5k = 5k
        assert metrics.margin_available == Decimal("5000")
        # Buffer % = 5k / 10k = 50%
        assert metrics.margin_buffer_pct == Decimal("50")

    def test_custom_max_leverage(self):
        """Test custom max leverage (20x)."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("200000"),
            cash_balance=Decimal("10000"),
            max_leverage=Decimal("20"),  # 20x instead of 10x
        )

        # Initial margin = 200k / 20 = 10k
        assert metrics.initial_margin_required == Decimal("10000")
        # Leverage = 200k / 10k = 20x
        assert metrics.leverage == Decimal("20")
        # Should not warn about leverage (at max)
        assert not any("exceeds max" in w for w in metrics.warnings)


class TestPropertyBasedAssertions:
    """Property-based tests ensuring invariants hold."""

    def test_margin_available_invariant(self):
        """Test margin_available = equity - maintenance_margin (when positive)."""
        test_cases = [
            (Decimal("100000"), Decimal("20000")),
            (Decimal("50000"), Decimal("10000")),
            (Decimal("25000"), Decimal("5000")),
        ]

        for positions_value, cash_balance in test_cases:
            metrics = MarginCalculator.calculate_margin_metrics(
                positions_value=positions_value,
                cash_balance=cash_balance,
            )

            if metrics.equity > 0 and metrics.margin_available > 0:
                expected = metrics.equity - metrics.maintenance_margin_required
                assert metrics.margin_available == expected

    def test_leverage_invariant(self):
        """Test leverage = positions_value / equity (when equity > 0)."""
        test_cases = [
            (Decimal("100000"), Decimal("10000")),
            (Decimal("75000"), Decimal("15000")),
            (Decimal("50000"), Decimal("20000")),
        ]

        for positions_value, cash_balance in test_cases:
            metrics = MarginCalculator.calculate_margin_metrics(
                positions_value=positions_value,
                cash_balance=cash_balance,
            )

            if metrics.equity > 0 and positions_value > 0:
                expected_leverage = positions_value / metrics.equity
                assert metrics.leverage == expected_leverage

    def test_margin_buffer_percentage_invariant(self):
        """Test margin_buffer_pct = (margin_available / equity) * 100."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("80000"),
            cash_balance=Decimal("10000"),
        )

        if metrics.equity > 0:
            expected_pct = (metrics.margin_available / metrics.equity) * Decimal("100")
            assert metrics.margin_buffer_pct == expected_pct

    def test_warnings_trigger_for_unhealthy_states(self):
        """Test warnings are always present for non-healthy states."""
        unhealthy_cases = [
            # Critical case
            (Decimal("184000"), Decimal("10000"), Decimal("0")),
            # Warning case
            (Decimal("170000"), Decimal("10000"), Decimal("0")),
            # Liquidation risk case
            (Decimal("200000"), Decimal("5000"), Decimal("0")),
        ]

        for positions_value, cash_balance, unrealized_pnl in unhealthy_cases:
            metrics = MarginCalculator.calculate_margin_metrics(
                positions_value=positions_value,
                cash_balance=cash_balance,
                unrealized_pnl=unrealized_pnl,
            )

            if metrics.margin_health != "healthy":
                assert len(metrics.warnings) > 0, f"Expected warnings for {metrics.margin_health}"


class TestMaxPositionSizeCalculation:
    """Tests for maximum position size calculation."""

    def test_max_position_size_standard(self):
        """Test max position size with 10x leverage."""
        max_size = MarginCalculator.calculate_max_position_size(
            equity=Decimal("10000"),
            price=Decimal("50000"),
        )

        # Max notional = 10k * 10 = 100k
        # Max quantity = 100k / 50k = 2.0
        assert max_size == Decimal("2.0")

    def test_max_position_size_higher_leverage(self):
        """Test max position size with 20x leverage."""
        max_size = MarginCalculator.calculate_max_position_size(
            equity=Decimal("5000"),
            price=Decimal("50000"),
            max_leverage=Decimal("20"),
        )

        # Max notional = 5k * 20 = 100k
        # Max quantity = 100k / 50k = 2.0
        assert max_size == Decimal("2.0")

    def test_max_position_size_zero_equity(self):
        """Test max position size with zero equity returns zero."""
        max_size = MarginCalculator.calculate_max_position_size(
            equity=Decimal("0"),
            price=Decimal("50000"),
        )

        assert max_size == Decimal("0")

    def test_max_position_size_zero_price(self):
        """Test max position size with zero price returns zero."""
        max_size = MarginCalculator.calculate_max_position_size(
            equity=Decimal("10000"),
            price=Decimal("0"),
        )

        assert max_size == Decimal("0")


class TestMarginMetricsSerialization:
    """Tests for MarginMetrics serialization."""

    def test_to_dict_serialization(self):
        """Test MarginMetrics serializes to dict correctly."""
        metrics = MarginCalculator.calculate_margin_metrics(
            positions_value=Decimal("100000"),
            cash_balance=Decimal("10000"),
        )

        result = metrics.to_dict()

        assert result["positions_value"] == 100000.0
        assert result["equity"] == 10000.0
        assert result["leverage"] == 10.0
        assert result["margin_health"] == "healthy"
        assert isinstance(result["warnings"], list)
