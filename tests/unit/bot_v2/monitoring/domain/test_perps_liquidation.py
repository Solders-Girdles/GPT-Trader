"""
Unit tests for liquidation distance monitoring.
"""

from decimal import Decimal

from bot_v2.monitoring.domain.perps.liquidation import (
    LiquidationMonitor,
    create_test_margin_info,
)


class TestLiquidationMonitor:
    """Test liquidation distance calculation and risk assessment."""

    def setup_method(self):
        """Set up liquidation monitor."""
        self.monitor = LiquidationMonitor(
            warning_buffer_pct=20.0,
            critical_buffer_pct=15.0,
            enable_reduce_only_guard=True,
            enable_entry_rejection=True,
        )

    def test_liquidation_price_calculation_long(self):
        """Test liquidation price calculation for long position."""
        margin_info = create_test_margin_info(
            position_side="long",
            entry_price=50000.0,
            leverage=5.0,  # 5x leverage
            maintenance_margin_rate=0.05,  # 5%
        )

        liq_price = self.monitor.calculate_liquidation_price(margin_info)

        # Long: liq = 50000 * (1 - 1/5 + 0.05) = 50000 * (1 - 0.2 + 0.05) = 50000 * 0.85 = 42500
        expected = Decimal("42500.0")
        assert liq_price is not None
        assert abs(liq_price - expected) < Decimal("0.01")

    def test_liquidation_price_calculation_short(self):
        """Test liquidation price calculation for short position."""
        margin_info = create_test_margin_info(
            position_side="short",
            entry_price=50000.0,
            leverage=5.0,  # 5x leverage
            maintenance_margin_rate=0.05,  # 5%
        )

        liq_price = self.monitor.calculate_liquidation_price(margin_info)

        # Short: liq = 50000 * (1 + 1/5 - 0.05) = 50000 * (1 + 0.2 - 0.05) = 50000 * 1.15 = 57500
        expected = Decimal("57500.0")
        assert liq_price is not None
        assert abs(liq_price - expected) < Decimal("0.01")

    def test_distance_calculation_long_safe(self):
        """Test distance calculation for long position in safe zone."""
        current_price = Decimal("50000")
        liq_price = Decimal("40000")  # 20% below current

        distance_pct, distance_bps = self.monitor.calculate_distance_to_liquidation(
            current_price, liq_price, "long"
        )

        # Long distance = (current - liq) / current = (50000 - 40000) / 50000 = 0.2 = 20%
        assert abs(distance_pct - 20.0) < 0.01
        assert abs(distance_bps - 2000.0) < 0.01

    def test_distance_calculation_short_safe(self):
        """Test distance calculation for short position in safe zone."""
        current_price = Decimal("50000")
        liq_price = Decimal("60000")  # 20% above current

        distance_pct, distance_bps = self.monitor.calculate_distance_to_liquidation(
            current_price, liq_price, "short"
        )

        # Short distance = (liq - current) / current = (60000 - 50000) / 50000 = 0.2 = 20%
        assert abs(distance_pct - 20.0) < 0.01
        assert abs(distance_bps - 2000.0) < 0.01

    def test_risk_assessment_safe(self):
        """Test risk assessment for safe position."""
        margin_info = create_test_margin_info(
            current_price=50000.0,
            leverage=2.0,  # Same as entry  # Low leverage
        )

        risk = self.monitor.assess_liquidation_risk(margin_info)

        assert risk.risk_level == "safe"
        assert not risk.should_reduce_only
        assert not risk.should_reject_entry
        assert risk.distance_pct > 25  # Should be well above warning buffer

    def test_risk_assessment_warning_or_critical(self):
        """Test risk assessment for warning/critical level functionality."""
        # Create position close to liquidation
        margin_info = create_test_margin_info(
            position_side="long",
            entry_price=50000.0,
            current_price=42000.0,  # Price dropped significantly
            leverage=4.0,  # Higher leverage
        )

        risk = self.monitor.assess_liquidation_risk(margin_info)

        # Should be in warning or critical zone (distance <= 20%)
        assert risk.risk_level in ["warning", "critical"]
        assert risk.should_reject_entry  # Should reject new entries
        assert risk.distance_pct <= 20.0  # Close to liquidation

    def test_risk_assessment_critical(self):
        """Test risk assessment for critical level."""
        # Create position very close to liquidation
        margin_info = create_test_margin_info(
            position_side="long",
            entry_price=50000.0,
            current_price=43500.0,  # Very close to liq price ~42500
            leverage=5.0,
        )

        risk = self.monitor.assess_liquidation_risk(margin_info)

        # Should be in critical zone (distance <= 15%)
        assert risk.risk_level == "critical"
        assert risk.should_reduce_only
        assert risk.should_reject_entry
        assert risk.distance_pct <= 15.0

    def test_portfolio_level_blocking(self):
        """Test portfolio-level entry blocking due to critical positions."""
        # Create critical position in BTC-PERP
        critical_position = create_test_margin_info(
            symbol="BTC-PERP",
            position_side="long",
            current_price=43000.0,  # Near liquidation
            leverage=5.0,
        )

        positions = {"BTC-PERP": critical_position}

        # Should block new ETH-PERP entry due to BTC-PERP risk
        should_block, reason = self.monitor.should_block_new_position("ETH-PERP", positions)

        assert should_block
        assert "Portfolio liquidation risk" in reason
        assert "BTC-PERP" in reason

    def test_no_position_safe(self):
        """Test risk assessment with no position."""
        margin_info = create_test_margin_info(position_size=0.0)

        risk = self.monitor.assess_liquidation_risk(margin_info)

        assert risk.risk_level == "safe"
        assert risk.reason == "No position"
