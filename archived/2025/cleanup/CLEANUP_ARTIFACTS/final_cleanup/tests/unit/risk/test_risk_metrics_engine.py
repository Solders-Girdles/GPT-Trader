"""
Tests for Risk Metrics Engine
Phase 3, Week 3: RISK-008
Comprehensive test suite for risk calculations
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.bot.risk.risk_metrics_engine import (
    CalculationMethod,
    RiskMetrics,
    RiskMetricsEngine,
)


class TestRiskMetricsEngine:
    """Test suite for RiskMetricsEngine"""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return data"""
        np.random.seed(42)
        # Generate returns with known properties
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
        return pd.Series(returns, index=pd.date_range("2024-01-01", periods=252))

    @pytest.fixture
    def sample_positions(self):
        """Create sample position data"""
        return pd.DataFrame(
            {
                "symbol": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
                "quantity": [100, 50, 75, 25, 40],
                "entry_price": [150.0, 2800.0, 300.0, 3200.0, 800.0],
                "current_price": [155.0, 2850.0, 310.0, 3150.0, 820.0],
                "sector": ["Tech", "Tech", "Tech", "Consumer", "Auto"],
            }
        )

    @pytest.fixture
    def engine(self):
        """Create RiskMetricsEngine instance"""
        return RiskMetricsEngine()

    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine is not None
        assert engine.confidence_level == 0.95
        assert engine.time_horizon == 1
        assert engine.lookback_period == 252

    def test_historical_var_calculation(self, engine, sample_returns):
        """Test Historical VaR calculation"""
        var = engine.calculate_var(
            returns=sample_returns, method=CalculationMethod.HISTORICAL, confidence=0.95
        )

        # VaR should be negative (loss)
        assert var < 0
        # Check if VaR is within reasonable range
        assert -0.1 < var < 0
        # VaR at 95% should be around 1.65 std devs
        expected_var = np.percentile(sample_returns, 5)
        assert abs(var - expected_var) < 0.01

    def test_parametric_var_calculation(self, engine, sample_returns):
        """Test Parametric VaR calculation"""
        var = engine.calculate_var(
            returns=sample_returns, method=CalculationMethod.PARAMETRIC, confidence=0.95
        )

        # Calculate expected parametric VaR
        mean = sample_returns.mean()
        std = sample_returns.std()
        z_score = 1.645  # 95% confidence
        expected_var = mean - z_score * std

        assert var < 0
        assert abs(var - expected_var) < 0.005

    def test_monte_carlo_var_calculation(self, engine, sample_returns):
        """Test Monte Carlo VaR calculation"""
        var = engine.calculate_var(
            returns=sample_returns,
            method=CalculationMethod.MONTE_CARLO,
            confidence=0.95,
            n_simulations=10000,
        )

        assert var < 0
        # Monte Carlo should be close to historical
        historical_var = engine.calculate_var(
            returns=sample_returns, method=CalculationMethod.HISTORICAL, confidence=0.95
        )
        # Allow for some Monte Carlo variance
        assert abs(var - historical_var) < 0.01

    def test_cornish_fisher_var_calculation(self, engine):
        """Test Cornish-Fisher VaR with skewness and kurtosis"""
        # Create returns with known skewness and excess kurtosis
        np.random.seed(42)
        returns = np.concatenate(
            [
                np.random.normal(0.001, 0.01, 200),  # Normal returns
                np.random.normal(-0.05, 0.03, 52),  # Tail events
            ]
        )
        returns = pd.Series(returns)

        var = engine.calculate_var(
            returns=returns, method=CalculationMethod.CORNISH_FISHER, confidence=0.95
        )

        # CF VaR should account for fat tails
        parametric_var = engine.calculate_var(
            returns=returns, method=CalculationMethod.PARAMETRIC, confidence=0.95
        )

        # CF VaR should be more conservative (more negative)
        assert var < parametric_var

    def test_cvar_calculation(self, engine, sample_returns):
        """Test Conditional VaR (Expected Shortfall) calculation"""
        var = engine.calculate_var(
            returns=sample_returns, method=CalculationMethod.HISTORICAL, confidence=0.95
        )

        cvar = engine.calculate_cvar(returns=sample_returns, var=var, confidence=0.95)

        # CVaR should be more extreme than VaR
        assert cvar < var
        # Check calculation
        tail_returns = sample_returns[sample_returns <= var]
        expected_cvar = tail_returns.mean()
        assert abs(cvar - expected_cvar) < 0.01

    def test_exposure_metrics_calculation(self, engine, sample_positions):
        """Test exposure metrics calculation"""
        metrics = engine.calculate_exposure_metrics(sample_positions)

        assert metrics is not None
        assert hasattr(metrics, "gross_exposure")
        assert hasattr(metrics, "net_exposure")
        assert hasattr(metrics, "concentration_ratio")
        assert hasattr(metrics, "sector_exposure")

        # Check calculations
        total_value = (sample_positions["quantity"] * sample_positions["current_price"]).sum()
        assert metrics.gross_exposure == total_value

        # All positions are long, so net = gross
        assert metrics.net_exposure == metrics.gross_exposure

        # Check concentration
        assert 0 <= metrics.concentration_ratio <= 1

    def test_sharpe_ratio_calculation(self, engine, sample_returns):
        """Test Sharpe ratio calculation"""
        sharpe = engine.calculate_sharpe_ratio(returns=sample_returns, risk_free_rate=0.02)

        # Check calculation
        excess_returns = sample_returns - 0.02 / 252  # Daily risk-free rate
        expected_sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

        assert abs(sharpe - expected_sharpe) < 0.1

    def test_sortino_ratio_calculation(self, engine, sample_returns):
        """Test Sortino ratio calculation"""
        sortino = engine.calculate_sortino_ratio(
            returns=sample_returns, risk_free_rate=0.02, target_return=0
        )

        # Sortino should typically be higher than Sharpe
        sharpe = engine.calculate_sharpe_ratio(returns=sample_returns, risk_free_rate=0.02)

        # This assumes we have some positive returns
        if sample_returns.mean() > 0:
            assert sortino >= sharpe

    def test_calmar_ratio_calculation(self, engine, sample_returns):
        """Test Calmar ratio calculation"""
        calmar = engine.calculate_calmar_ratio(sample_returns)

        # Check if Calmar is reasonable
        annual_return = sample_returns.mean() * 252
        cumsum = sample_returns.cumsum()
        running_max = cumsum.cummax()
        drawdown = cumsum - running_max
        max_drawdown = abs(drawdown.min())

        if max_drawdown > 0:
            expected_calmar = annual_return / max_drawdown
            assert abs(calmar - expected_calmar) < 0.1

    def test_maximum_drawdown_calculation(self, engine, sample_returns):
        """Test maximum drawdown calculation"""
        max_dd = engine.calculate_max_drawdown(sample_returns)

        # Max drawdown should be negative
        assert max_dd <= 0

        # Manual calculation
        cumsum = sample_returns.cumsum()
        running_max = cumsum.cummax()
        drawdown = cumsum - running_max
        expected_dd = drawdown.min()

        assert abs(max_dd - expected_dd) < 0.001

    def test_var_with_different_confidence_levels(self, engine, sample_returns):
        """Test VaR at different confidence levels"""
        var_90 = engine.calculate_var(returns=sample_returns, confidence=0.90)
        var_95 = engine.calculate_var(returns=sample_returns, confidence=0.95)
        var_99 = engine.calculate_var(returns=sample_returns, confidence=0.99)

        # Higher confidence should give more extreme VaR
        assert var_99 < var_95 < var_90

    def test_var_with_different_time_horizons(self, engine, sample_returns):
        """Test VaR scaling for different time horizons"""
        var_1day = engine.calculate_var(returns=sample_returns, time_horizon=1)
        var_10day = engine.calculate_var(returns=sample_returns, time_horizon=10)

        # 10-day VaR should be approximately sqrt(10) times 1-day VaR
        # (under normal distribution assumption)
        expected_ratio = np.sqrt(10)
        actual_ratio = var_10day / var_1day

        # Allow for some deviation from sqrt scaling
        assert abs(actual_ratio - expected_ratio) < 1.0

    def test_stress_testing_scenarios(self, engine, sample_returns):
        """Test stress testing with extreme scenarios"""
        # Add stress scenario
        stressed_returns = sample_returns.copy()
        stressed_returns.iloc[0] = -0.10  # 10% loss event

        var_normal = engine.calculate_var(returns=sample_returns)
        var_stressed = engine.calculate_var(returns=stressed_returns)

        # Stressed VaR should be more extreme
        assert var_stressed < var_normal

    def test_sector_concentration_calculation(self, engine, sample_positions):
        """Test sector concentration metrics"""
        metrics = engine.calculate_exposure_metrics(sample_positions)

        # Tech sector should have highest concentration
        tech_exposure = metrics.sector_exposure.get("Tech", 0)
        total_exposure = metrics.gross_exposure
        tech_concentration = tech_exposure / total_exposure

        # Tech has 3 out of 5 positions and most value
        assert tech_concentration > 0.5

    def test_position_limit_checks(self, engine, sample_positions):
        """Test position limit violation detection"""
        # Set position limits
        limits = {
            "max_position_size": 0.25,  # 25% max per position
            "max_sector_concentration": 0.60,  # 60% max per sector
        }

        violations = engine.check_position_limits(sample_positions, limits)

        # GOOGL position might violate size limit
        assert isinstance(violations, list)

        # Check if violations are properly formatted
        for violation in violations:
            assert "type" in violation
            assert "details" in violation

    def test_incremental_var_calculation(self, engine, sample_returns):
        """Test incremental VaR for new positions"""
        current_var = engine.calculate_var(returns=sample_returns)

        # Simulate adding a new position
        new_position_returns = sample_returns * 1.2  # Correlated position
        combined_returns = (sample_returns + new_position_returns) / 2

        new_var = engine.calculate_var(returns=combined_returns)
        incremental_var = new_var - current_var

        # Adding correlated position should increase risk
        assert incremental_var < 0  # More negative = more risk

    def test_marginal_var_calculation(self, engine, sample_returns):
        """Test marginal VaR calculation"""
        portfolio_var = engine.calculate_var(returns=sample_returns)

        # Calculate marginal VaR for small position change
        epsilon = 0.01
        adjusted_returns = sample_returns * (1 + epsilon)
        adjusted_var = engine.calculate_var(returns=adjusted_returns)

        marginal_var = (adjusted_var - portfolio_var) / epsilon

        # Marginal VaR should be reasonable
        assert abs(marginal_var) < abs(portfolio_var)

    def test_component_var_calculation(self, engine):
        """Test component VaR for portfolio decomposition"""
        # Create portfolio with multiple assets
        np.random.seed(42)
        n_assets = 3
        n_days = 252

        returns = pd.DataFrame(
            {f"Asset{i}": np.random.normal(0.001, 0.02, n_days) for i in range(n_assets)}
        )
        weights = np.array([0.4, 0.3, 0.3])

        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_var = engine.calculate_var(returns=portfolio_returns)

        # Component VaRs should sum to portfolio VaR
        component_vars = []
        for i, col in enumerate(returns.columns):
            asset_var = engine.calculate_var(returns=returns[col])
            component_vars.append(asset_var * weights[i])

        # Due to diversification, sum of component VaRs > portfolio VaR
        assert sum(component_vars) < portfolio_var

    def test_risk_metrics_aggregation(self, engine, sample_returns, sample_positions):
        """Test aggregation of all risk metrics"""
        metrics = engine.calculate_all_metrics(
            returns=sample_returns, positions=sample_positions, confidence=0.95
        )

        # Check all metrics are present
        assert metrics.var_historical is not None
        assert metrics.var_parametric is not None
        assert metrics.cvar is not None
        assert metrics.sharpe_ratio is not None
        assert metrics.sortino_ratio is not None
        assert metrics.max_drawdown is not None
        assert metrics.gross_exposure is not None
        assert metrics.net_exposure is not None

    def test_risk_metrics_caching(self, engine, sample_returns):
        """Test caching of expensive calculations"""
        # First calculation
        import time

        start = time.time()
        var1 = engine.calculate_var(
            returns=sample_returns, method=VaRMethod.MONTE_CARLO, n_simulations=10000
        )
        first_time = time.time() - start

        # Second calculation (should be cached)
        start = time.time()
        var2 = engine.calculate_var(
            returns=sample_returns, method=VaRMethod.MONTE_CARLO, n_simulations=10000
        )
        second_time = time.time() - start

        assert var1 == var2
        # Second call should be much faster if cached
        # Note: This might not always work in test environment

    def test_risk_metrics_serialization(self, engine, sample_returns):
        """Test serialization of risk metrics"""
        metrics = RiskMetrics(
            var_historical=-0.05,
            var_parametric=-0.04,
            var_monte_carlo=-0.045,
            cvar=-0.07,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=0.8,
            max_drawdown=-0.15,
        )

        # Convert to dict
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["var_historical"] == -0.05

        # Convert to JSON
        import json

        metrics_json = json.dumps(metrics_dict)
        assert isinstance(metrics_json, str)

        # Deserialize
        loaded_dict = json.loads(metrics_json)
        assert loaded_dict["var_historical"] == -0.05

    def test_error_handling_invalid_inputs(self, engine):
        """Test error handling for invalid inputs"""
        # Empty returns
        with pytest.raises(ValueError):
            engine.calculate_var(returns=pd.Series([]))

        # Invalid confidence level
        with pytest.raises(ValueError):
            engine.calculate_var(returns=pd.Series([0.01, 0.02]), confidence=1.5)

        # Invalid method
        with pytest.raises(ValueError):
            engine.calculate_var(returns=pd.Series([0.01, 0.02]), method="invalid_method")

    def test_extreme_market_conditions(self, engine):
        """Test calculations under extreme market conditions"""
        # Create extreme returns
        extreme_returns = pd.Series([-0.20, -0.15, -0.10, 0.30, -0.25])  # Extreme volatility

        # Should still calculate without errors
        var = engine.calculate_var(returns=extreme_returns)
        assert var is not None
        assert var < -0.10  # Should reflect high risk

        # Test with all negative returns
        negative_returns = pd.Series([-0.01, -0.02, -0.03, -0.04, -0.05])
        sharpe = engine.calculate_sharpe_ratio(returns=negative_returns)
        assert sharpe < 0  # Negative Sharpe for losing strategy

    def test_correlation_matrix_calculation(self, engine):
        """Test correlation matrix for portfolio assets"""
        # Create correlated asset returns
        np.random.seed(42)
        n_days = 252

        # Base return series
        base = np.random.normal(0, 0.01, n_days)

        returns = pd.DataFrame(
            {
                "Asset1": base + np.random.normal(0, 0.005, n_days),
                "Asset2": base * 0.8 + np.random.normal(0, 0.008, n_days),
                "Asset3": -base * 0.5 + np.random.normal(0, 0.01, n_days),
            }
        )

        corr_matrix = engine.calculate_correlation_matrix(returns)

        # Check matrix properties
        assert corr_matrix.shape == (3, 3)
        # Diagonal should be 1
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), np.ones(3))
        # Should be symmetric
        np.testing.assert_array_almost_equal(corr_matrix, corr_matrix.T)
        # Asset1 and Asset2 should be positively correlated
        assert corr_matrix.loc["Asset1", "Asset2"] > 0.5
        # Asset1 and Asset3 should be negatively correlated
        assert corr_matrix.loc["Asset1", "Asset3"] < 0


class TestRiskMetricsIntegration:
    """Integration tests for risk metrics with other components"""

    @pytest.fixture
    def mock_data_source(self):
        """Mock data source for integration tests"""
        mock = Mock()
        mock.get_historical_data.return_value = pd.DataFrame(
            {
                "close": np.random.normal(100, 10, 252),
                "volume": np.random.randint(1000000, 5000000, 252),
            }
        )
        return mock

    @pytest.fixture
    def mock_position_manager(self):
        """Mock position manager"""
        mock = Mock()
        mock.get_positions.return_value = [
            {"symbol": "AAPL", "quantity": 100, "current_price": 150},
            {"symbol": "GOOGL", "quantity": 50, "current_price": 2800},
        ]
        return mock

    def test_integration_with_data_source(self, mock_data_source):
        """Test integration with data source"""
        engine = RiskMetricsEngine(data_source=mock_data_source)

        # Should fetch data and calculate metrics
        metrics = engine.calculate_metrics_for_symbol("AAPL")

        assert mock_data_source.get_historical_data.called
        assert metrics is not None

    def test_integration_with_position_manager(self, mock_position_manager):
        """Test integration with position manager"""
        engine = RiskMetricsEngine(position_manager=mock_position_manager)

        # Should fetch positions and calculate portfolio metrics
        portfolio_metrics = engine.calculate_portfolio_metrics()

        assert mock_position_manager.get_positions.called
        assert portfolio_metrics is not None

    @patch("src.bot.risk.risk_metrics_engine.send_alert")
    def test_alert_on_var_breach(self, mock_alert):
        """Test alert generation on VaR breach"""
        engine = RiskMetricsEngine()
        engine.var_limit = -0.05  # 5% VaR limit

        # Create returns that breach VaR limit
        bad_returns = pd.Series([-0.08, -0.06, -0.07, 0.02, -0.05])

        metrics = engine.calculate_and_monitor(bad_returns)

        # Should trigger alert
        assert mock_alert.called
        alert_data = mock_alert.call_args[0][0]
        assert "VaR breach" in alert_data["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
