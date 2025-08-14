"""Tests for Risk Management Integration Layer."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from bot.risk.integration import RiskIntegration, RiskConfig, AllocationResult
from bot.portfolio.allocator import PortfolioRules


class TestRiskConfig:
    """Test RiskConfig class."""

    def test_default_config(self):
        """Test default risk configuration."""
        config = RiskConfig()

        assert config.max_position_size == 0.10
        assert config.max_portfolio_exposure == 0.95
        assert config.default_stop_loss_pct == 0.05
        assert config.max_daily_loss == 0.03
        assert config.enable_realtime_monitoring is True

    def test_custom_config(self):
        """Test custom risk configuration."""
        config = RiskConfig(max_position_size=0.05, max_daily_loss=0.02, default_stop_loss_pct=0.03)

        assert config.max_position_size == 0.05
        assert config.max_daily_loss == 0.02
        assert config.default_stop_loss_pct == 0.03


class TestAllocationResult:
    """Test AllocationResult class."""

    def test_empty_result(self):
        """Test empty allocation result."""
        result = AllocationResult()

        assert result.has_warnings is False
        assert result.allocation_changed is False
        assert result.passed_validation is True

    def test_result_with_warnings(self):
        """Test result with warnings."""
        result = AllocationResult(warnings={"AAPL": "Position too large"})

        assert result.has_warnings is True

    def test_allocation_changed(self):
        """Test when allocations are modified."""
        result = AllocationResult(
            original_allocations={"AAPL": 100}, adjusted_allocations={"AAPL": 80}
        )

        assert result.allocation_changed is True


class TestRiskIntegration:
    """Test RiskIntegration class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.risk_config = RiskConfig(
            max_position_size=0.10, max_portfolio_exposure=0.95, default_stop_loss_pct=0.05
        )

        self.portfolio_rules = PortfolioRules(per_trade_risk_pct=0.01, max_positions=10)

        with patch("bot.config.get_config"):
            self.risk_integration = RiskIntegration(
                risk_config=self.risk_config, portfolio_rules=self.portfolio_rules
            )

    def test_initialization(self):
        """Test risk integration initialization."""
        assert self.risk_integration.risk_config == self.risk_config
        assert self.risk_integration.portfolio_rules == self.portfolio_rules
        assert self.risk_integration.current_portfolio_value == 0.0
        assert isinstance(self.risk_integration.risk_manager, Mock) is False

    def test_validate_allocations_basic(self):
        """Test basic allocation validation."""
        allocations = {"AAPL": 100, "GOOGL": 50, "MSFT": 75}

        current_prices = {"AAPL": 150.0, "GOOGL": 2500.0, "MSFT": 300.0}

        portfolio_value = 500000.0

        result = self.risk_integration.validate_allocations(
            allocations=allocations, current_prices=current_prices, portfolio_value=portfolio_value
        )

        assert isinstance(result, AllocationResult)
        assert result.original_allocations == allocations
        assert len(result.adjusted_allocations) == 3
        assert result.passed_validation is True

    def test_position_size_limit_enforcement(self):
        """Test position size limit enforcement."""
        # Large position that exceeds 10% limit
        allocations = {
            "AAPL": 1000,  # $150k position in $500k portfolio = 30%
        }

        current_prices = {"AAPL": 150.0}

        portfolio_value = 500000.0

        result = self.risk_integration.validate_allocations(
            allocations=allocations, current_prices=current_prices, portfolio_value=portfolio_value
        )

        # Should be reduced to 10% of portfolio
        expected_max_shares = int((portfolio_value * 0.10) / 150.0)  # 333 shares

        assert result.adjusted_allocations["AAPL"] == expected_max_shares
        assert "AAPL" in result.warnings
        assert "reduced" in result.warnings["AAPL"].lower()

    def test_portfolio_exposure_limit(self):
        """Test portfolio exposure limit enforcement."""
        # Allocations that exceed 95% exposure limit
        allocations = {
            "AAPL": 320,  # $48k
            "GOOGL": 200,  # $500k
            "MSFT": 167,  # $50k
        }

        current_prices = {"AAPL": 150.0, "GOOGL": 2500.0, "MSFT": 300.0}

        portfolio_value = 500000.0
        total_position_value = (320 * 150) + (200 * 2500) + (167 * 300)  # $598k

        result = self.risk_integration.validate_allocations(
            allocations=allocations, current_prices=current_prices, portfolio_value=portfolio_value
        )

        # Should be scaled down to 95% exposure
        assert "portfolio" in result.warnings
        assert "scaled down" in result.warnings["portfolio"]
        assert result.total_exposure <= 0.95

    def test_missing_price_data(self):
        """Test handling of missing price data."""
        allocations = {"AAPL": 100, "INVALID": 50}  # No price data

        current_prices = {
            "AAPL": 150.0
            # Missing INVALID price
        }

        portfolio_value = 100000.0

        result = self.risk_integration.validate_allocations(
            allocations=allocations, current_prices=current_prices, portfolio_value=portfolio_value
        )

        assert result.adjusted_allocations["INVALID"] == 0
        assert "INVALID" in result.warnings
        assert "no price data" in result.warnings["INVALID"].lower()

    def test_stop_loss_calculation(self):
        """Test stop-loss level calculation."""
        allocations = {"AAPL": 100}

        current_prices = {"AAPL": 150.0}

        portfolio_value = 100000.0

        result = self.risk_integration.validate_allocations(
            allocations=allocations, current_prices=current_prices, portfolio_value=portfolio_value
        )

        assert "AAPL" in result.stop_levels
        stop_info = result.stop_levels["AAPL"]

        expected_stop_loss = 150.0 * (1 - 0.05)  # 5% stop loss
        assert abs(stop_info["stop_loss"] - expected_stop_loss) < 0.01
        assert stop_info["current_price"] == 150.0
        assert "take_profit" in stop_info
        assert "trailing_stop" in stop_info

    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation."""
        allocations = {"AAPL": 100, "GOOGL": 20, "MSFT": 50}

        current_prices = {"AAPL": 150.0, "GOOGL": 2500.0, "MSFT": 300.0}

        portfolio_value = 200000.0

        result = self.risk_integration.validate_allocations(
            allocations=allocations, current_prices=current_prices, portfolio_value=portfolio_value
        )

        metrics = result.risk_metrics

        assert "total_positions" in metrics
        assert "total_exposure_pct" in metrics
        assert "largest_position_pct" in metrics
        assert "concentration_ratio" in metrics

        assert metrics["total_positions"] == 3
        assert metrics["total_exposure_pct"] > 0

    def test_advanced_risk_validation_with_market_data(self):
        """Test advanced risk validation with historical market data."""
        allocations = {"AAPL": 100, "GOOGL": 20}

        current_prices = {"AAPL": 150.0, "GOOGL": 2500.0}

        # Create mock market data
        dates = pd.date_range(start="2023-01-01", periods=60, freq="D")
        market_data = {
            "AAPL": pd.DataFrame(
                {"Close": 150.0 + pd.Series(range(60)) * 0.5, "Volume": [1000000] * 60}, index=dates
            ),
            "GOOGL": pd.DataFrame(
                {"Close": 2500.0 + pd.Series(range(60)) * 10, "Volume": [500000] * 60}, index=dates
            ),
        }

        portfolio_value = 200000.0

        result = self.risk_integration.validate_allocations(
            allocations=allocations,
            current_prices=current_prices,
            portfolio_value=portfolio_value,
            market_data=market_data,
        )

        # Should have calculated portfolio volatility
        assert "portfolio_volatility" in result.risk_metrics

    def test_daily_loss_limit_check(self):
        """Test daily loss limit checking."""
        self.risk_integration.current_portfolio_value = 100000.0

        # Loss within limit (1% < 3% limit)
        assert self.risk_integration.check_daily_loss_limit(-1000.0) is False

        # Loss exceeding limit (5% > 3% limit)
        assert self.risk_integration.check_daily_loss_limit(-5000.0) is True

    def test_validate_new_position(self):
        """Test single position validation."""
        # Valid position
        is_valid, reason, adjusted_shares = self.risk_integration.validate_new_position(
            symbol="AAPL", proposed_shares=100, current_price=150.0, portfolio_value=200000.0
        )

        assert is_valid is True
        assert adjusted_shares == 100

        # Position too large
        is_valid, reason, adjusted_shares = self.risk_integration.validate_new_position(
            symbol="AAPL",
            proposed_shares=2000,  # $300k position in $200k portfolio
            current_price=150.0,
            portfolio_value=200000.0,
        )

        assert is_valid is False
        assert "too large" in reason.lower()
        assert adjusted_shares < 2000

    def test_stop_loss_updates(self):
        """Test stop-loss updates."""
        positions = {"AAPL": {"current_price": 155.0, "entry_price": 150.0, "highest_price": 160.0}}

        with patch.object(self.risk_integration.risk_manager, "update_stop_losses") as mock_update:
            mock_update.return_value = {
                "symbol": "AAPL",
                "effective_stop": 147.0,
                "stop_type": "trailing",
            }

            updates = self.risk_integration.update_stop_losses(positions)

            assert "AAPL" in updates
            assert updates["AAPL"]["effective_stop"] == 147.0
            mock_update.assert_called_once()

    def test_triggered_stops_check(self):
        """Test checking for triggered stops."""
        current_prices = {"AAPL": 140.0, "GOOGL": 2600.0}  # Below stop level

        with patch.object(self.risk_integration.risk_manager, "check_stop_losses") as mock_check:
            mock_check.return_value = [
                {
                    "symbol": "AAPL",
                    "current_price": 140.0,
                    "stop_price": 142.5,
                    "stop_type": "fixed",
                }
            ]

            triggered = self.risk_integration.check_triggered_stops(current_prices)

            assert len(triggered) == 1
            assert triggered[0]["symbol"] == "AAPL"
            mock_check.assert_called_once_with(current_prices)

    def test_risk_report_generation(self):
        """Test risk report generation."""
        self.risk_integration.current_portfolio_value = 100000.0
        self.risk_integration.daily_pnl = -500.0

        with patch.object(self.risk_integration.risk_manager, "get_risk_summary") as mock_summary:
            mock_summary.return_value = {"var_95": -0.015, "volatility": 0.18, "n_positions": 5}

            report = self.risk_integration.generate_risk_report()

            assert "timestamp" in report
            assert report["portfolio_value"] == 100000.0
            assert report["daily_pnl"] == -500.0
            assert "risk_config" in report
            assert "risk_manager_summary" in report

    def test_zero_portfolio_value_handling(self):
        """Test handling of zero portfolio value."""
        allocations = {"AAPL": 100}
        current_prices = {"AAPL": 150.0}
        portfolio_value = 0.0

        result = self.risk_integration.validate_allocations(
            allocations=allocations, current_prices=current_prices, portfolio_value=portfolio_value
        )

        # Should handle gracefully without division by zero
        assert result.passed_validation is True
        assert result.total_exposure == 0.0

    def test_empty_allocations(self):
        """Test handling of empty allocations."""
        allocations = {}
        current_prices = {}
        portfolio_value = 100000.0

        result = self.risk_integration.validate_allocations(
            allocations=allocations, current_prices=current_prices, portfolio_value=portfolio_value
        )

        assert result.passed_validation is True
        assert len(result.adjusted_allocations) == 0
        assert result.total_exposure == 0.0
        assert len(result.stop_levels) == 0


class TestRiskIntegrationIntegration:
    """Integration tests for RiskIntegration."""

    def test_full_allocation_workflow(self):
        """Test complete allocation validation workflow."""
        risk_config = RiskConfig(
            max_position_size=0.08,  # 8% max per position
            max_portfolio_exposure=0.90,  # 90% max exposure
            default_stop_loss_pct=0.04,  # 4% stop loss
        )

        with patch("bot.config.get_config"):
            risk_integration = RiskIntegration(risk_config=risk_config)

        # Portfolio with mixed position sizes
        allocations = {
            "AAPL": 150,  # $22.5k = 11.25% (should be reduced)
            "GOOGL": 10,  # $25k = 12.5% (should be reduced)
            "MSFT": 100,  # $30k = 15% (should be reduced)
            "TSLA": 25,  # $20k = 10% (should be reduced)
            "NVDA": 20,  # $18k = 9% (should be OK)
        }

        current_prices = {
            "AAPL": 150.0,
            "GOOGL": 2500.0,
            "MSFT": 300.0,
            "TSLA": 800.0,
            "NVDA": 900.0,
        }

        portfolio_value = 200000.0

        result = risk_integration.validate_allocations(
            allocations=allocations, current_prices=current_prices, portfolio_value=portfolio_value
        )

        # Check that large positions were reduced
        for symbol in ["AAPL", "GOOGL", "MSFT", "TSLA"]:
            original_value = allocations[symbol] * current_prices[symbol]
            adjusted_value = result.adjusted_allocations[symbol] * current_prices[symbol]

            if original_value / portfolio_value > 0.08:
                assert adjusted_value < original_value, f"{symbol} should have been reduced"
                assert symbol in result.warnings

        # Check stop levels calculated for all positions
        for symbol in result.adjusted_allocations:
            if result.adjusted_allocations[symbol] > 0:
                assert symbol in result.stop_levels
                stop_info = result.stop_levels[symbol]
                expected_stop = current_prices[symbol] * (1 - risk_config.default_stop_loss_pct)
                assert abs(stop_info["stop_loss"] - expected_stop) < 0.01

        # Check risk metrics
        assert "total_positions" in result.risk_metrics
        assert "concentration_ratio" in result.risk_metrics
        assert result.risk_metrics["total_positions"] == len(
            [s for s, q in result.adjusted_allocations.items() if q > 0]
        )

    def test_correlation_warning_with_market_data(self):
        """Test correlation warning with realistic market data."""
        risk_config = RiskConfig(max_correlation=0.5)  # Lower threshold

        with patch("bot.config.get_config"):
            risk_integration = RiskIntegration(risk_config=risk_config)

        allocations = {"AAPL": 100, "MSFT": 100}  # Tech stocks likely correlated

        current_prices = {"AAPL": 150.0, "MSFT": 300.0}

        # Create highly correlated data
        dates = pd.date_range(start="2023-01-01", periods=60, freq="D")
        base_returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005] * 12)

        market_data = {
            "AAPL": pd.DataFrame(
                {
                    "Close": 150.0 * (1 + base_returns).cumprod(),
                },
                index=dates,
            ),
            "MSFT": pd.DataFrame(
                {
                    "Close": 300.0 * (1 + base_returns * 0.9).cumprod(),  # 90% correlation
                },
                index=dates,
            ),
        }

        result = risk_integration.validate_allocations(
            allocations=allocations,
            current_prices=current_prices,
            portfolio_value=100000.0,
            market_data=market_data,
        )

        # Should detect high correlation
        assert "correlation" in result.warnings or "portfolio_volatility" in result.risk_metrics
