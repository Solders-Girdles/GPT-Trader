"""
Comprehensive tests for main analyze module orchestration.

Tests cover:
- analyze_symbol function (complete analysis)
- analyze_portfolio function (multi-symbol analysis)
- compare_strategies function (strategy comparison)
- Helper functions (calculate_indicators, determine_regime, generate_recommendation)
- Integration with data providers
"""

import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime

from bot_v2.features.analyze.analyze import (
    analyze_symbol,
    analyze_portfolio,
    compare_strategies,
    calculate_indicators,
    determine_regime,
    generate_recommendation,
    identify_levels,
    calculate_correlations,
    calculate_portfolio_risk,
    generate_rebalance_suggestions,
)
from bot_v2.features.analyze.types import (
    TechnicalIndicators,
    MarketRegime,
    AnalysisResult,
    StrategySignals,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range("2025-01-01", periods=250, freq="D")
    np.random.seed(42)

    # Generate realistic price data
    close_prices = 100 * np.exp(np.cumsum(np.random.randn(250) * 0.02))
    high_prices = close_prices * (1 + np.abs(np.random.randn(250)) * 0.01)
    low_prices = close_prices * (1 - np.abs(np.random.randn(250)) * 0.01)
    open_prices = close_prices + np.random.randn(250) * 0.5
    volumes = np.random.randint(1000000, 5000000, 250)

    return pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        },
        index=dates,
    )


@pytest.fixture
def mock_data_provider(sample_market_data):
    """Create mock data provider."""
    provider = Mock()
    provider.get_historical_data.return_value = sample_market_data.copy()
    return provider


@pytest.fixture
def sample_indicators():
    """Create sample technical indicators."""
    return TechnicalIndicators(
        sma_20=100.0,
        sma_50=98.0,
        sma_200=95.0,
        ema_12=101.0,
        ema_26=99.0,
        rsi=55.0,
        macd=0.5,
        macd_signal=0.3,
        macd_histogram=0.2,
        bollinger_upper=105.0,
        bollinger_middle=100.0,
        bollinger_lower=95.0,
        atr=2.5,
        volume_sma=2000000.0,
        obv=50000000.0,
        stochastic_k=60.0,
        stochastic_d=58.0,
    )


@pytest.fixture
def sample_strategy_signals():
    """Create sample strategy signals."""
    return [
        StrategySignals(strategy_name="MA", signal=1, confidence=0.7, reason="Golden cross"),
        StrategySignals(
            strategy_name="Momentum", signal=1, confidence=0.6, reason="Positive momentum"
        ),
        StrategySignals(strategy_name="Mean Reversion", signal=0, confidence=0.4, reason="Neutral"),
    ]


# ============================================================================
# Test: analyze_symbol
# ============================================================================


class TestAnalyzeSymbol:
    """Test main analyze_symbol function."""

    @patch("bot_v2.features.analyze.analyze.get_data_provider")
    def test_analyzes_symbol_successfully(
        self, mock_get_provider, mock_data_provider, sample_market_data
    ):
        """Test successful symbol analysis."""
        mock_get_provider.return_value = mock_data_provider

        result = analyze_symbol("AAPL", lookback_days=90)

        assert isinstance(result, AnalysisResult)
        assert result.symbol == "AAPL"
        assert isinstance(result.timestamp, datetime)
        assert result.current_price > 0
        assert isinstance(result.indicators, TechnicalIndicators)
        assert isinstance(result.regime, MarketRegime)
        assert isinstance(result.patterns, list)
        assert isinstance(result.strategy_signals, list)
        assert result.recommendation in ["strong_buy", "buy", "hold", "sell", "strong_sell"]
        assert 0 <= result.confidence <= 1

    @patch("bot_v2.features.analyze.analyze.get_data_provider")
    def test_raises_error_on_no_data(self, mock_get_provider):
        """Test error handling when no data available."""
        provider = Mock()
        # Return DataFrame with correct columns structure but no rows
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        provider.get_historical_data.return_value = empty_df
        mock_get_provider.return_value = provider

        with pytest.raises(ValueError, match="No data available"):
            analyze_symbol("INVALID")

    @patch("bot_v2.features.analyze.analyze.get_data_provider")
    def test_include_patterns_false(self, mock_get_provider, mock_data_provider):
        """Test analysis without pattern detection."""
        mock_get_provider.return_value = mock_data_provider

        result = analyze_symbol("AAPL", include_patterns=False)

        assert result.patterns == []

    @patch("bot_v2.features.analyze.analyze.get_data_provider")
    def test_include_strategies_false(self, mock_get_provider, mock_data_provider):
        """Test analysis without strategy signals."""
        mock_get_provider.return_value = mock_data_provider

        result = analyze_symbol("AAPL", include_strategies=False)

        assert result.strategy_signals == []


# ============================================================================
# Test: analyze_portfolio
# ============================================================================


class TestAnalyzePortfolio:
    """Test analyze_portfolio function."""

    @patch("bot_v2.features.analyze.analyze.get_data_provider")
    @patch("bot_v2.features.analyze.analyze.analyze_symbol")
    def test_analyzes_portfolio(self, mock_analyze_symbol, mock_get_provider, mock_data_provider):
        """Test portfolio analysis."""
        mock_get_provider.return_value = mock_data_provider

        # Mock analyze_symbol to return simple results
        mock_result = Mock(spec=AnalysisResult)
        mock_result.recommendation = "buy"
        mock_result.confidence = 0.7
        mock_analyze_symbol.return_value = mock_result

        symbols = ["AAPL", "GOOGL", "MSFT"]
        result = analyze_portfolio(symbols, lookback_days=90)

        assert result.total_value > 0
        assert len(result.symbol_analyses) <= len(symbols)  # May be fewer due to errors
        assert isinstance(result.correlation_matrix, pd.DataFrame)
        assert isinstance(result.sector_allocation, dict)
        assert isinstance(result.risk_metrics, dict)
        assert isinstance(result.rebalance_suggestions, list)

    @patch("bot_v2.features.analyze.analyze.get_data_provider")
    @patch("bot_v2.features.analyze.analyze.analyze_symbol")
    def test_uses_equal_weights_when_none_provided(
        self, mock_analyze_symbol, mock_get_provider, mock_data_provider
    ):
        """Test equal weights when weights not specified."""
        mock_get_provider.return_value = mock_data_provider

        # Create proper mock with required attributes
        mock_result = Mock(spec=AnalysisResult)
        mock_result.recommendation = "hold"
        mock_result.confidence = 0.5
        mock_analyze_symbol.return_value = mock_result

        symbols = ["AAPL", "GOOGL"]
        result = analyze_portfolio(symbols)

        # Should use equal weights and complete successfully
        assert len(result.symbol_analyses) == 2

    @patch("bot_v2.features.analyze.analyze.get_data_provider")
    @patch("bot_v2.features.analyze.analyze.analyze_symbol")
    def test_handles_symbol_analysis_errors(
        self, mock_analyze_symbol, mock_get_provider, mock_data_provider, caplog
    ):
        """Test handling of errors for individual symbols."""
        mock_get_provider.return_value = mock_data_provider

        # Create proper mocks with required attributes
        mock_result1 = Mock(spec=AnalysisResult)
        mock_result1.recommendation = "buy"
        mock_result1.confidence = 0.7

        mock_result2 = Mock(spec=AnalysisResult)
        mock_result2.recommendation = "sell"
        mock_result2.confidence = 0.6

        mock_analyze_symbol.side_effect = [
            mock_result1,
            Exception("API error"),
            mock_result2,
        ]

        symbols = ["AAPL", "INVALID", "GOOGL"]
        result = analyze_portfolio(symbols)

        # Should have 2 successful analyses
        assert len(result.symbol_analyses) == 2
        assert "Could not analyze" in caplog.text


# ============================================================================
# Test: compare_strategies
# ============================================================================


class TestCompareStrategies:
    """Test compare_strategies function."""

    @patch("bot_v2.features.analyze.analyze.get_data_provider")
    def test_compares_strategies(self, mock_get_provider, mock_data_provider):
        """Test strategy comparison."""
        mock_get_provider.return_value = mock_data_provider

        result = compare_strategies("AAPL", lookback_days=90)

        assert isinstance(result.strategies, list)
        assert len(result.strategies) == 5  # Default strategies
        assert isinstance(result.metrics, dict)
        assert isinstance(result.rankings, dict)
        assert result.best_strategy in result.strategies
        assert isinstance(result.recommendation, str)

    @patch("bot_v2.features.analyze.analyze.get_data_provider")
    def test_uses_custom_strategies(self, mock_get_provider, mock_data_provider):
        """Test with custom strategy list."""
        mock_get_provider.return_value = mock_data_provider

        custom_strategies = ["SimpleMA", "Momentum"]
        result = compare_strategies("AAPL", strategies=custom_strategies)

        assert result.strategies == custom_strategies
        assert len(result.metrics) == len(custom_strategies)


# ============================================================================
# Test: calculate_indicators
# ============================================================================


class TestCalculateIndicators:
    """Test calculate_indicators function."""

    def test_calculates_all_indicators(self, sample_market_data):
        """Test that all indicators are calculated."""
        indicators = calculate_indicators(sample_market_data)

        assert isinstance(indicators, TechnicalIndicators)
        assert indicators.sma_20 > 0
        assert indicators.sma_50 > 0
        assert indicators.ema_12 > 0
        assert 0 <= indicators.rsi <= 100
        assert indicators.atr >= 0
        assert indicators.volume_sma > 0

    def test_handles_short_data_gracefully(self):
        """Test with minimal data."""
        data = pd.DataFrame(
            {
                "open": [100] * 30,
                "high": [102] * 30,
                "low": [98] * 30,
                "close": [100, 101, 100, 99, 100] * 6,
                "volume": [1000000] * 30,
            }
        )

        indicators = calculate_indicators(data)

        # Should complete without errors
        assert isinstance(indicators, TechnicalIndicators)

    def test_uses_fallback_for_missing_data(self):
        """Test fallback values for insufficient data."""
        data = pd.DataFrame(
            {
                "open": [100] * 25,
                "high": [102] * 25,
                "low": [98] * 25,
                "close": [100] * 25,
                "volume": [1000000] * 25,
            }
        )

        indicators = calculate_indicators(data)

        # SMA_50 should use SMA_20 as fallback (not enough data for 50-period)
        assert indicators.sma_50 == indicators.sma_20


# ============================================================================
# Test: determine_regime
# ============================================================================


class TestDetermineRegime:
    """Test determine_regime function."""

    def test_determines_regime(self, sample_market_data, sample_indicators):
        """Test regime determination."""
        regime = determine_regime(sample_market_data, sample_indicators)

        assert isinstance(regime, MarketRegime)
        assert regime.trend in ["bullish", "bearish", "neutral"]
        assert regime.volatility in ["low", "medium", "high"]
        assert regime.momentum in ["strong_up", "weak_up", "neutral", "weak_down", "strong_down"]
        assert regime.volume_profile in ["accumulation", "distribution", "neutral"]
        assert 0 <= regime.strength <= 100

    def test_bullish_regime_with_high_rsi(self, sample_market_data):
        """Test bullish regime detection."""
        indicators = TechnicalIndicators(
            sma_20=100,
            sma_50=95,
            sma_200=90,
            ema_12=101,
            ema_26=99,
            rsi=75.0,  # Overbought
            macd=1.0,
            macd_signal=0.5,
            macd_histogram=0.5,
            bollinger_upper=105,
            bollinger_middle=100,
            bollinger_lower=95,
            atr=2.0,
            volume_sma=2000000,
            obv=50000000,
            stochastic_k=70,
            stochastic_d=68,
        )

        regime = determine_regime(sample_market_data, indicators)

        assert regime.momentum == "strong_up"

    def test_bearish_regime_with_low_rsi(self, sample_market_data):
        """Test bearish regime detection."""
        indicators = TechnicalIndicators(
            sma_20=100,
            sma_50=105,
            sma_200=110,
            ema_12=99,
            ema_26=101,
            rsi=25.0,  # Oversold
            macd=-1.0,
            macd_signal=-0.5,
            macd_histogram=-0.5,
            bollinger_upper=105,
            bollinger_middle=100,
            bollinger_lower=95,
            atr=3.0,
            volume_sma=2000000,
            obv=40000000,
            stochastic_k=25,
            stochastic_d=23,
        )

        regime = determine_regime(sample_market_data, indicators)

        assert regime.momentum == "strong_down"


# ============================================================================
# Test: generate_recommendation
# ============================================================================


class TestGenerateRecommendation:
    """Test generate_recommendation function."""

    def test_strong_buy_recommendation(self, sample_indicators, sample_strategy_signals):
        """Test strong buy signal generation."""
        # Bullish indicators
        indicators = TechnicalIndicators(
            sma_20=100,
            sma_50=95,
            sma_200=90,
            ema_12=102,
            ema_26=98,
            rsi=25.0,  # Oversold - buy signal
            macd=1.0,
            macd_signal=0.5,
            macd_histogram=0.5,  # Bullish
            bollinger_upper=105,
            bollinger_middle=100,
            bollinger_lower=95,
            atr=2.0,
            volume_sma=2000000,
            obv=50000000,
            stochastic_k=70,
            stochastic_d=68,
        )

        regime = MarketRegime(
            trend="bullish",
            volatility="low",
            momentum="strong_up",
            volume_profile="accumulation",
            strength=80.0,
        )

        # All bullish signals
        signals = [
            StrategySignals("MA", 1, 0.8, "Golden cross"),
            StrategySignals("Momentum", 1, 0.7, "Strong momentum"),
            StrategySignals("Breakout", 1, 0.75, "Breakout"),
        ]

        recommendation, confidence = generate_recommendation(indicators, regime, [], signals)

        assert recommendation in ["strong_buy", "buy"]
        assert confidence > 0.6

    def test_strong_sell_recommendation(self, sample_indicators, sample_strategy_signals):
        """Test strong sell signal generation."""
        # Bearish indicators
        indicators = TechnicalIndicators(
            sma_20=100,
            sma_50=105,
            sma_200=110,
            ema_12=98,
            ema_26=102,
            rsi=75.0,  # Overbought - sell signal
            macd=-1.0,
            macd_signal=-0.5,
            macd_histogram=-0.5,  # Bearish
            bollinger_upper=105,
            bollinger_middle=100,
            bollinger_lower=95,
            atr=3.0,
            volume_sma=2000000,
            obv=40000000,
            stochastic_k=25,
            stochastic_d=23,
        )

        regime = MarketRegime(
            trend="bearish",
            volatility="high",
            momentum="strong_down",
            volume_profile="distribution",
            strength=20.0,
        )

        # All bearish signals
        signals = [
            StrategySignals("MA", -1, 0.8, "Death cross"),
            StrategySignals("Momentum", -1, 0.7, "Negative momentum"),
            StrategySignals("Breakout", -1, 0.75, "Breakdown"),
        ]

        recommendation, confidence = generate_recommendation(indicators, regime, [], signals)

        assert recommendation in ["strong_sell", "sell"]
        assert confidence > 0.6

    def test_hold_recommendation_mixed_signals(self, sample_indicators):
        """Test hold recommendation with mixed signals."""
        regime = MarketRegime(
            trend="neutral",
            volatility="medium",
            momentum="neutral",
            volume_profile="neutral",
            strength=50.0,
        )

        # All neutral signals (score should be close to 0)
        signals = [
            StrategySignals("MA", 0, 0.4, "Neutral"),
            StrategySignals("Momentum", 0, 0.3, "Neutral"),
            StrategySignals("Mean Reversion", 0, 0.4, "Neutral"),
        ]

        recommendation, confidence = generate_recommendation(sample_indicators, regime, [], signals)

        # With neutral signals, should get hold or mild buy/sell
        assert recommendation in ["hold", "buy", "sell"]
        assert confidence <= 0.7  # Should not be highly confident


# ============================================================================
# Test: Helper Functions
# ============================================================================


class TestHelperFunctions:
    """Test helper functions."""

    def test_identify_levels(self, sample_market_data):
        """Test support/resistance identification."""
        levels = identify_levels(sample_market_data)

        assert levels.immediate_support > 0
        assert levels.strong_support > 0
        assert levels.immediate_resistance > 0
        assert levels.strong_resistance > 0
        assert levels.pivot_point > 0

    @patch("bot_v2.features.analyze.analyze.get_data_provider")
    def test_calculate_correlations(self, mock_get_provider, sample_market_data):
        """Test correlation calculation."""
        provider = Mock()
        # Return data with 'Close' column (capital C)
        data_with_capital = sample_market_data.copy()
        data_with_capital["Close"] = data_with_capital["close"]
        provider.get_historical_data.return_value = data_with_capital
        mock_get_provider.return_value = provider

        symbols = ["AAPL", "GOOGL"]
        corr_matrix = calculate_correlations(symbols, 90)

        assert isinstance(corr_matrix, pd.DataFrame)

    def test_calculate_portfolio_risk(self):
        """Test portfolio risk calculation."""
        symbols = ["AAPL", "GOOGL"]
        weights = {"AAPL": 0.6, "GOOGL": 0.4}
        corr_matrix = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]],
            index=symbols,
            columns=symbols,
        )

        risk_metrics = calculate_portfolio_risk(symbols, weights, corr_matrix)

        assert "beta" in risk_metrics
        assert "correlation_risk" in risk_metrics
        assert "concentration_risk" in risk_metrics
        assert risk_metrics["concentration_risk"] == 0.6  # Max weight

    def test_generate_rebalance_suggestions(self):
        """Test rebalance suggestion generation."""
        # Create mock analyses
        analysis1 = Mock(spec=AnalysisResult)
        analysis1.recommendation = "strong_buy"
        analysis1.confidence = 0.8

        analysis2 = Mock(spec=AnalysisResult)
        analysis2.recommendation = "strong_sell"
        analysis2.confidence = 0.75

        analyses = {"AAPL": analysis1, "GOOGL": analysis2}
        weights = {"AAPL": 0.1, "GOOGL": 0.3}

        suggestions = generate_rebalance_suggestions(analyses, weights)

        assert isinstance(suggestions, list)
        # Should suggest increasing AAPL (strong buy, low weight)
        # Should suggest decreasing GOOGL (strong sell, high weight)
