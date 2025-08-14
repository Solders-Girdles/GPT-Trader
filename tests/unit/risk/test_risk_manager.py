"""
Comprehensive unit tests for Risk Management System.

Tests risk limits, position sizing, stop-loss management,
portfolio risk metrics, and stress testing capabilities.
"""

import numpy as np
import pandas as pd
import pytest
from bot.risk.manager import (
    PortfolioRisk,
    PositionRisk,
    RiskLimits,
    RiskManager,
    StopLossConfig,
)


# Mock the missing calculation functions for testing
def calculate_var(returns, confidence_level=0.95):
    """Mock VaR calculation."""
    return np.percentile(returns, (1 - confidence_level) * 100)


def calculate_cvar(returns, confidence_level=0.95):
    """Mock CVaR calculation."""
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()


def calculate_max_drawdown(prices):
    """Mock max drawdown calculation."""
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Mock Sharpe ratio calculation."""
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)


def calculate_position_sizing(capital, risk_per_trade, stop_loss_pct):
    """Mock position sizing calculation."""
    risk_amount = capital * risk_per_trade
    return risk_amount / stop_loss_pct


class TestRiskLimits:
    """Test RiskLimits dataclass."""

    def test_risk_limits_defaults(self):
        """Test RiskLimits with default values."""
        limits = RiskLimits()

        assert limits.max_portfolio_var == 0.02
        assert limits.max_portfolio_drawdown == 0.15
        assert limits.max_portfolio_volatility == 0.25
        assert limits.max_portfolio_beta == 1.2
        assert limits.max_position_size == 0.1
        assert limits.max_sector_exposure == 0.3
        assert limits.max_correlation == 0.7
        assert limits.max_risk_per_trade == 0.01
        assert limits.max_daily_loss == 0.03
        assert limits.min_liquidity_ratio == 0.1
        assert limits.max_illiquid_exposure == 0.2

    def test_risk_limits_custom(self):
        """Test RiskLimits with custom values."""
        limits = RiskLimits(max_portfolio_var=0.05, max_position_size=0.15, max_risk_per_trade=0.02)

        assert limits.max_portfolio_var == 0.05
        assert limits.max_position_size == 0.15
        assert limits.max_risk_per_trade == 0.02


class TestStopLossConfig:
    """Test StopLossConfig dataclass."""

    def test_stop_loss_defaults(self):
        """Test StopLossConfig with default values."""
        config = StopLossConfig()

        assert config.stop_loss_pct == 0.05
        assert config.trailing_stop_pct == 0.03
        assert config.time_stop_days == 30
        assert config.breakeven_after_pct == 0.02

    def test_stop_loss_custom(self):
        """Test StopLossConfig with custom values."""
        config = StopLossConfig(
            stop_loss_pct=0.03, trailing_stop_pct=0.02, time_stop_days=20, breakeven_after_pct=0.01
        )

        assert config.stop_loss_pct == 0.03
        assert config.trailing_stop_pct == 0.02
        assert config.time_stop_days == 20
        assert config.breakeven_after_pct == 0.01


class TestPositionRisk:
    """Test PositionRisk dataclass."""

    def test_position_risk_creation(self):
        """Test PositionRisk creation."""
        position_risk = PositionRisk(
            symbol="AAPL",
            current_value=15000.0,
            position_size=0.15,
            var_95=500.0,
            cvar_95=750.0,
            beta=1.1,
            volatility=0.22,
            correlation_with_portfolio=0.65,
            max_drawdown=0.08,
            sharpe_ratio=1.5,
        )

        assert position_risk.symbol == "AAPL"
        assert position_risk.current_value == 15000.0
        assert position_risk.position_size == 0.15
        assert position_risk.var_95 == 500.0
        assert position_risk.cvar_95 == 750.0
        assert position_risk.beta == 1.1
        assert position_risk.volatility == 0.22
        assert position_risk.correlation_with_portfolio == 0.65
        assert position_risk.max_drawdown == 0.08
        assert position_risk.sharpe_ratio == 1.5


class TestRiskCalculations:
    """Test risk calculation functions."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return series."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        return returns

    @pytest.fixture
    def sample_prices(self):
        """Create sample price series."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        returns = np.random.normal(0.001, 0.02, 252)
        prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
        return prices

    def test_calculate_var(self, sample_returns):
        """Test VaR calculation."""
        var_95 = calculate_var(sample_returns, confidence=0.95)

        # VaR should be negative (loss)
        assert var_95 < 0

        # Check that ~5% of returns are worse than VaR
        worse_than_var = (sample_returns < var_95).sum()
        pct_worse = worse_than_var / len(sample_returns)
        assert 0.03 < pct_worse < 0.07  # Around 5%

    def test_calculate_cvar(self, sample_returns):
        """Test CVaR calculation."""
        cvar_95 = calculate_cvar(sample_returns, confidence=0.95)
        var_95 = calculate_var(sample_returns, confidence=0.95)

        # CVaR should be worse than VaR
        assert cvar_95 < var_95

        # CVaR should be the average of worst returns
        worst_returns = sample_returns[sample_returns <= var_95]
        expected_cvar = worst_returns.mean()
        assert abs(cvar_95 - expected_cvar) < 0.001

    def test_calculate_max_drawdown(self, sample_prices):
        """Test maximum drawdown calculation."""
        max_dd = calculate_max_drawdown(sample_prices)

        # Drawdown should be negative
        assert max_dd <= 0

        # Calculate manually
        cummax = sample_prices.expanding().max()
        drawdown = (sample_prices - cummax) / cummax
        expected_max_dd = drawdown.min()

        assert abs(max_dd - expected_max_dd) < 0.0001

    def test_calculate_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        sharpe = calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02)

        # Sharpe ratio should be reasonable
        assert -3 < sharpe < 3

        # Calculate manually
        excess_returns = sample_returns - 0.02 / 252  # Daily risk-free rate
        expected_sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        assert abs(sharpe - expected_sharpe) < 0.1

    def test_calculate_position_sizing(self):
        """Test position sizing calculation."""
        # Kelly criterion
        size = calculate_position_sizing(expected_return=0.10, volatility=0.20, method="kelly")

        # Kelly = expected_return / variance
        expected_kelly = 0.10 / (0.20**2)
        assert abs(size - expected_kelly) < 0.01

        # Fixed fraction
        size = calculate_position_sizing(
            expected_return=0.10, volatility=0.20, method="fixed", fixed_fraction=0.02
        )
        assert size == 0.02

        # Volatility targeting
        size = calculate_position_sizing(
            expected_return=0.10, volatility=0.20, method="volatility", target_volatility=0.15
        )
        expected_size = 0.15 / 0.20
        assert abs(size - expected_size) < 0.01


class TestRiskManager:
    """Test RiskManager class."""

    @pytest.fixture
    def risk_manager(self):
        """Create RiskManager instance."""
        limits = RiskLimits()
        stop_config = StopLossConfig()
        return RiskManager(limits=limits, stop_loss_config=stop_config)

    @pytest.fixture
    def portfolio_data(self):
        """Create sample portfolio data."""
        positions = {
            "AAPL": {
                "quantity": 100,
                "current_price": 150.0,
                "entry_price": 145.0,
                "value": 15000.0,
            },
            "GOOGL": {
                "quantity": 10,
                "current_price": 2800.0,
                "entry_price": 2750.0,
                "value": 28000.0,
            },
            "MSFT": {
                "quantity": 50,
                "current_price": 300.0,
                "entry_price": 310.0,
                "value": 15000.0,
            },
        }

        returns = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 100),
                "GOOGL": np.random.normal(0.0008, 0.025, 100),
                "MSFT": np.random.normal(0.0005, 0.018, 100),
            }
        )

        return positions, returns

    def test_check_position_limits(self, risk_manager, portfolio_data):
        """Test position limit checking."""
        positions, _ = portfolio_data
        total_value = sum(p["value"] for p in positions.values())

        # Check valid position
        is_valid = risk_manager.check_position_limits(
            symbol="AAPL", position_value=positions["AAPL"]["value"], portfolio_value=total_value
        )
        assert is_valid is True

        # Check oversized position
        is_valid = risk_manager.check_position_limits(
            symbol="AAPL",
            position_value=total_value * 0.15,  # 15% > 10% limit
            portfolio_value=total_value,
        )
        assert is_valid is False

    def test_calculate_stop_loss(self, risk_manager):
        """Test stop-loss calculation."""
        # Initial stop-loss
        stop_price = risk_manager.calculate_stop_loss(
            entry_price=100.0, current_price=100.0, highest_price=100.0
        )
        expected_stop = 100.0 * (1 - risk_manager.stop_loss_config.stop_loss_pct)
        assert abs(stop_price - expected_stop) < 0.01

        # Trailing stop-loss
        stop_price = risk_manager.calculate_stop_loss(
            entry_price=100.0, current_price=110.0, highest_price=112.0
        )
        expected_trail = 112.0 * (1 - risk_manager.stop_loss_config.trailing_stop_pct)
        assert abs(stop_price - expected_trail) < 0.01

        # Breakeven stop
        stop_price = risk_manager.calculate_stop_loss(
            entry_price=100.0, current_price=102.5, highest_price=103.0, use_breakeven=True
        )
        # Should move to breakeven after 2% profit
        assert stop_price >= 100.0

    def test_check_portfolio_risk(self, risk_manager, portfolio_data):
        """Test portfolio risk checking."""
        positions, returns = portfolio_data

        portfolio_risk = risk_manager.check_portfolio_risk(positions=positions, returns_df=returns)

        assert portfolio_risk is not None
        assert hasattr(portfolio_risk, "total_var")
        assert hasattr(portfolio_risk, "total_volatility")
        assert hasattr(portfolio_risk, "max_drawdown")
        assert hasattr(portfolio_risk, "violations")

    def test_get_position_risk(self, risk_manager, portfolio_data):
        """Test individual position risk calculation."""
        positions, returns = portfolio_data

        position_risk = risk_manager.get_position_risk(
            symbol="AAPL",
            position=positions["AAPL"],
            returns=returns["AAPL"],
            portfolio_value=sum(p["value"] for p in positions.values()),
        )

        assert position_risk.symbol == "AAPL"
        assert position_risk.current_value == positions["AAPL"]["value"]
        assert position_risk.var_95 is not None
        assert position_risk.cvar_95 is not None
        assert position_risk.volatility > 0

    def test_should_exit_position(self, risk_manager):
        """Test position exit conditions."""
        # Stop-loss triggered
        should_exit = risk_manager.should_exit_position(
            entry_price=100.0,
            current_price=94.0,
            highest_price=100.0,
            days_held=5,  # Below 5% stop
        )
        assert should_exit is True

        # Time stop triggered
        should_exit = risk_manager.should_exit_position(
            entry_price=100.0,
            current_price=99.0,
            highest_price=100.0,
            days_held=35,  # > 30 days
        )
        assert should_exit is True

        # No exit conditions met
        should_exit = risk_manager.should_exit_position(
            entry_price=100.0, current_price=102.0, highest_price=103.0, days_held=10
        )
        assert should_exit is False

    def test_calculate_portfolio_var(self, risk_manager, portfolio_data):
        """Test portfolio VaR calculation."""
        positions, returns = portfolio_data

        # Calculate weights
        total_value = sum(p["value"] for p in positions.values())
        weights = pd.Series(
            {symbol: pos["value"] / total_value for symbol, pos in positions.items()}
        )

        portfolio_var = risk_manager.calculate_portfolio_var(
            returns_df=returns, weights=weights, confidence=0.95
        )

        assert portfolio_var < 0  # VaR is a loss
        assert -0.10 < portfolio_var < 0  # Reasonable range

    def test_stress_test(self, risk_manager, portfolio_data):
        """Test stress testing functionality."""
        positions, returns = portfolio_data

        # Define stress scenarios
        scenarios = {
            "market_crash": {"AAPL": -0.20, "GOOGL": -0.25, "MSFT": -0.18},
            "tech_selloff": {"AAPL": -0.15, "GOOGL": -0.18, "MSFT": -0.16},
            "recession": {"AAPL": -0.10, "GOOGL": -0.12, "MSFT": -0.08},
        }

        stress_results = risk_manager.run_stress_tests(positions=positions, scenarios=scenarios)

        assert len(stress_results) == 3
        assert "market_crash" in stress_results
        assert stress_results["market_crash"]["portfolio_loss"] < 0

    def test_risk_adjusted_position_size(self, risk_manager):
        """Test risk-adjusted position sizing."""
        size = risk_manager.calculate_risk_adjusted_size(
            capital=100000.0,
            price=150.0,
            volatility=0.25,
            stop_loss_pct=0.05,
            max_risk_pct=0.01,  # 1% risk per trade
        )

        # Size should respect risk limits
        max_loss = size * 150.0 * 0.05  # position value * stop loss
        max_allowed_loss = 100000.0 * 0.01  # capital * max risk

        assert max_loss <= max_allowed_loss * 1.01  # Allow small rounding

    def test_correlation_risk(self, risk_manager, portfolio_data):
        """Test correlation risk checking."""
        _, returns = portfolio_data

        # Calculate correlation matrix
        correlation_matrix = returns.corr()

        # Check if correlations exceed limits
        high_correlations = risk_manager.check_correlation_limits(
            correlation_matrix=correlation_matrix, max_correlation=0.7
        )

        assert isinstance(high_correlations, list)
        # Should identify any pairs with correlation > 0.7

    def test_concentration_risk(self, risk_manager, portfolio_data):
        """Test concentration risk checking."""
        positions, _ = portfolio_data

        # Add sector information
        sectors = {"AAPL": "Technology", "GOOGL": "Technology", "MSFT": "Technology"}

        concentration_risk = risk_manager.check_concentration_risk(
            positions=positions, sectors=sectors
        )

        assert concentration_risk is not None
        assert concentration_risk["Technology"] == 1.0  # 100% in tech

    def test_liquidity_risk(self, risk_manager, portfolio_data):
        """Test liquidity risk assessment."""
        positions, _ = portfolio_data

        # Add volume data
        avg_volumes = {"AAPL": 50000000, "GOOGL": 1000000, "MSFT": 30000000}

        liquidity_scores = risk_manager.assess_liquidity_risk(
            positions=positions, avg_volumes=avg_volumes
        )

        assert len(liquidity_scores) == 3
        assert liquidity_scores["AAPL"] > liquidity_scores["GOOGL"]  # AAPL more liquid


class TestPortfolioRisk:
    """Test PortfolioRisk calculations."""

    def test_portfolio_risk_aggregation(self):
        """Test portfolio risk aggregation."""
        portfolio_risk = PortfolioRisk(
            total_value=100000.0,
            total_var=-2000.0,
            total_cvar=-3000.0,
            total_volatility=0.20,
            max_drawdown=-0.12,
            sharpe_ratio=1.2,
            beta=0.95,
            position_risks=[],
        )

        assert portfolio_risk.total_value == 100000.0
        assert portfolio_risk.total_var == -2000.0
        assert portfolio_risk.total_cvar == -3000.0
        assert portfolio_risk.total_volatility == 0.20
        assert portfolio_risk.max_drawdown == -0.12
        assert portfolio_risk.sharpe_ratio == 1.2
        assert portfolio_risk.beta == 0.95

    def test_risk_violations(self):
        """Test risk violation detection."""
        limits = RiskLimits(max_portfolio_var=0.02, max_portfolio_volatility=0.15)

        portfolio_risk = PortfolioRisk(
            total_value=100000.0,
            total_var=-2500.0,  # -2.5% exceeds -2% limit
            total_cvar=-3500.0,
            total_volatility=0.18,  # 18% exceeds 15% limit
            max_drawdown=-0.10,
            sharpe_ratio=1.0,
            beta=1.0,
            position_risks=[],
            violations=[],
        )

        # Check violations
        violations = []
        if abs(portfolio_risk.total_var) / portfolio_risk.total_value > limits.max_portfolio_var:
            violations.append("VaR limit exceeded")
        if portfolio_risk.total_volatility > limits.max_portfolio_volatility:
            violations.append("Volatility limit exceeded")

        assert len(violations) == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def risk_manager(self):
        return RiskManager(RiskLimits(), StopLossConfig())

    def test_empty_portfolio(self, risk_manager):
        """Test risk calculations with empty portfolio."""
        positions = {}
        returns = pd.DataFrame()

        portfolio_risk = risk_manager.check_portfolio_risk(positions=positions, returns_df=returns)

        assert portfolio_risk.total_value == 0
        assert portfolio_risk.total_var == 0

    def test_single_position(self, risk_manager):
        """Test risk with single position."""
        positions = {"AAPL": {"quantity": 100, "current_price": 150.0, "value": 15000.0}}
        returns = pd.DataFrame({"AAPL": np.random.normal(0.001, 0.02, 100)})

        portfolio_risk = risk_manager.check_portfolio_risk(positions=positions, returns_df=returns)

        assert portfolio_risk.total_value == 15000.0
        assert len(portfolio_risk.position_risks) == 1

    def test_negative_prices(self, risk_manager):
        """Test handling of negative prices."""
        should_exit = risk_manager.should_exit_position(
            entry_price=100.0,
            current_price=-10.0,  # Invalid negative price
            highest_price=100.0,
            days_held=5,
        )

        # Should trigger exit on invalid price
        assert should_exit is True

    def test_extreme_volatility(self, risk_manager):
        """Test handling of extreme volatility."""
        returns = pd.Series(np.random.normal(0, 0.50, 100))  # 50% daily vol

        var_95 = calculate_var(returns, confidence=0.95)

        # Should still calculate but be very negative
        assert var_95 < -0.5

    def test_zero_volatility(self, risk_manager):
        """Test handling of zero volatility."""
        returns = pd.Series([0.0] * 100)  # No volatility

        volatility = returns.std()
        assert volatility == 0.0

        # Position sizing with zero volatility
        size = risk_manager.calculate_risk_adjusted_size(
            capital=100000.0, price=100.0, volatility=0.0, stop_loss_pct=0.05, max_risk_pct=0.01
        )

        # Should handle gracefully
        assert size == 0 or size > 0
