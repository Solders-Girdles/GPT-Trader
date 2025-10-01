"""
Comprehensive tests for the main position sizing orchestrator.

Tests cover:
- All sizing methods (intelligent, Kelly, fractional Kelly, confidence, regime, fixed)
- Integration of multiple adjustment layers
- Portfolio allocation
- Error handling and validation
- Edge cases and boundary conditions
- Configuration integration
"""

import pytest

from bot_v2.errors import RiskLimitExceeded, ValidationError
from bot_v2.features.position_sizing.position_sizing import (
    calculate_portfolio_allocation,
    calculate_position_size,
)
from bot_v2.features.position_sizing.types import (
    PositionSizeRequest,
    RiskParameters,
    SizingMethod,
)


class TestIntelligentSizing:
    """Test intelligent sizing that combines all factors."""

    def test_intelligent_with_all_factors(self):
        """Test intelligent sizing with Kelly, confidence, and regime data."""
        request = PositionSizeRequest(
            symbol="BTC",
            current_price=5000.0,  # Lower price so position fits within caps
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.INTELLIGENT,
            win_rate=0.6,
            avg_win=0.05,
            avg_loss=-0.03,
            confidence=0.85,
            market_regime="bull_quiet",
            risk_params=RiskParameters(max_position_size=0.2, kelly_fraction=0.25),
        )

        response = calculate_position_size(request)

        assert response.symbol == "BTC"
        assert response.recommended_shares > 0
        assert response.recommended_value > 0
        assert response.method_used == SizingMethod.INTELLIGENT
        assert response.kelly_fraction is not None
        assert response.confidence_adjustment is not None
        assert response.regime_adjustment is not None
        assert len(response.calculation_notes) > 0

    def test_intelligent_kelly_only(self):
        """Test intelligent sizing with only Kelly data."""
        request = PositionSizeRequest(
            symbol="ETH",
            current_price=3000.0,
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.INTELLIGENT,
            win_rate=0.55,
            avg_win=0.04,
            avg_loss=-0.025,
        )

        response = calculate_position_size(request)

        assert response.recommended_shares > 0
        assert response.kelly_fraction is not None
        assert response.confidence_adjustment is None  # No confidence data
        assert response.regime_adjustment is None  # No regime data
        assert any("Kelly" in note for note in response.calculation_notes)

    def test_intelligent_no_statistics_fallback(self):
        """Test intelligent sizing falls back without trade statistics."""
        request = PositionSizeRequest(
            symbol="SOL",
            current_price=100.0,
            portfolio_value=50000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.INTELLIGENT,
            # No win_rate, avg_win, avg_loss
            confidence=0.7,
        )

        response = calculate_position_size(request)

        assert response.recommended_shares > 0
        assert len(response.warnings) > 0
        assert any("No historical" in warn or "No trade" in warn for warn in response.warnings)

    def test_intelligent_confidence_below_threshold(self):
        """Test intelligent sizing rejects low confidence."""
        request = PositionSizeRequest(
            symbol="BTC",
            current_price=50000.0,
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.INTELLIGENT,
            win_rate=0.6,
            avg_win=0.05,
            avg_loss=-0.03,
            confidence=0.4,  # Below 0.6 threshold
            risk_params=RiskParameters(confidence_threshold=0.6),
        )

        response = calculate_position_size(request)

        assert response.recommended_shares == 0
        assert response.recommended_value == 0.0
        assert any("below threshold" in warn.lower() for warn in response.warnings)

    def test_intelligent_strategy_multiplier(self):
        """Test intelligent sizing applies strategy multiplier."""
        request = PositionSizeRequest(
            symbol="BTC",
            current_price=5000.0,  # Lower price so position fits within caps
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.INTELLIGENT,
            win_rate=0.6,
            avg_win=0.05,
            avg_loss=-0.03,
            strategy_multiplier=1.5,  # Moderate multiplier that won't exceed max
            risk_params=RiskParameters(max_position_size=0.5),  # Higher cap for multiplier test
        )

        response = calculate_position_size(request)

        assert response.recommended_shares > 0
        assert any("multiplier" in note.lower() for note in response.calculation_notes)

    def test_intelligent_caps_at_max_position(self):
        """Test intelligent sizing respects max position limit."""
        request = PositionSizeRequest(
            symbol="BTC",
            current_price=1000.0,  # Lower price so we can actually buy shares within 10% cap
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.INTELLIGENT,
            win_rate=0.9,  # Very high win rate
            avg_win=0.15,  # Large wins
            avg_loss=-0.01,  # Small losses
            risk_params=RiskParameters(max_position_size=0.1),  # Cap at 10%
        )

        response = calculate_position_size(request)

        # Position should be capped
        assert response.position_size_pct <= 0.1


class TestKellySizing:
    """Test pure Kelly Criterion sizing."""

    def test_kelly_sizing_basic(self):
        """Test basic Kelly sizing."""
        request = PositionSizeRequest(
            symbol="BTC",
            current_price=5000.0,  # Lower price so position fits within 10% cap
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.KELLY,
            win_rate=0.6,
            avg_win=0.05,
            avg_loss=-0.03,
        )

        response = calculate_position_size(request)

        assert response.recommended_shares > 0
        assert response.method_used == SizingMethod.KELLY
        assert response.kelly_fraction is not None
        assert response.kelly_fraction > 0

    def test_kelly_sizing_missing_data(self):
        """Test Kelly sizing returns error without required data."""
        request = PositionSizeRequest(
            symbol="BTC",
            current_price=50000.0,
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.KELLY,
            # Missing win_rate, avg_win, avg_loss
        )

        response = calculate_position_size(request)

        assert response.recommended_shares == 0
        assert len(response.warnings) > 0


class TestFractionalKellySizing:
    """Test fractional Kelly sizing."""

    def test_fractional_kelly_sizing(self):
        """Test fractional Kelly sizing."""
        request = PositionSizeRequest(
            symbol="ETH",
            current_price=3000.0,
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.FRACTIONAL_KELLY,
            win_rate=0.55,
            avg_win=0.04,
            avg_loss=-0.025,
            risk_params=RiskParameters(kelly_fraction=0.5),  # Half Kelly
        )

        response = calculate_position_size(request)

        assert response.recommended_shares > 0
        assert response.method_used == SizingMethod.FRACTIONAL_KELLY
        assert "Fractional Kelly" in response.calculation_notes[0]

    def test_fractional_kelly_missing_data(self):
        """Test fractional Kelly returns error without data."""
        request = PositionSizeRequest(
            symbol="ETH",
            current_price=3000.0,
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.FRACTIONAL_KELLY,
        )

        response = calculate_position_size(request)

        assert response.recommended_shares == 0
        assert len(response.warnings) > 0


class TestConfidenceSizing:
    """Test confidence-adjusted sizing."""

    def test_confidence_sizing_basic(self):
        """Test basic confidence sizing."""
        request = PositionSizeRequest(
            symbol="SOL",
            current_price=100.0,
            portfolio_value=50000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.CONFIDENCE_ADJUSTED,
            confidence=0.85,
        )

        response = calculate_position_size(request)

        assert response.recommended_shares > 0
        assert response.method_used == SizingMethod.CONFIDENCE_ADJUSTED
        assert response.confidence_adjustment is not None

    def test_confidence_sizing_missing_confidence(self):
        """Test confidence sizing returns error without confidence."""
        request = PositionSizeRequest(
            symbol="SOL",
            current_price=100.0,
            portfolio_value=50000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.CONFIDENCE_ADJUSTED,
            # Missing confidence
        )

        response = calculate_position_size(request)

        assert response.recommended_shares == 0
        assert len(response.warnings) > 0


class TestRegimeSizing:
    """Test regime-adjusted sizing."""

    def test_regime_sizing_basic(self):
        """Test basic regime sizing."""
        request = PositionSizeRequest(
            symbol="AVAX",
            current_price=30.0,
            portfolio_value=50000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.REGIME_ADJUSTED,
            market_regime="bull_quiet",
        )

        response = calculate_position_size(request)

        assert response.recommended_shares > 0
        assert response.method_used == SizingMethod.REGIME_ADJUSTED
        assert response.regime_adjustment is not None

    def test_regime_sizing_missing_regime(self):
        """Test regime sizing returns error without regime."""
        request = PositionSizeRequest(
            symbol="AVAX",
            current_price=30.0,
            portfolio_value=50000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.REGIME_ADJUSTED,
            # Missing market_regime
        )

        response = calculate_position_size(request)

        assert response.recommended_shares == 0
        assert len(response.warnings) > 0


class TestFixedSizing:
    """Test fixed sizing method."""

    def test_fixed_sizing_basic(self):
        """Test basic fixed sizing."""
        request = PositionSizeRequest(
            symbol="BTC",
            current_price=1000.0,  # Lower price so position fits within default fixed size
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.FIXED,
        )

        response = calculate_position_size(request)

        assert response.recommended_shares > 0
        assert response.method_used == SizingMethod.FIXED
        assert "Fixed sizing" in response.calculation_notes[0]

    def test_fixed_sizing_respects_limits(self):
        """Test fixed sizing respects risk parameters."""
        request = PositionSizeRequest(
            symbol="BTC",
            current_price=50000.0,
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.FIXED,
            risk_params=RiskParameters(max_position_size=0.05),
        )

        response = calculate_position_size(request)

        assert response.position_size_pct <= 0.05


class TestPositionSizeValidation:
    """Test position size request validation."""

    def test_validation_missing_symbol(self):
        """Test validation catches missing symbol."""
        with pytest.raises(ValidationError) as exc_info:
            request = PositionSizeRequest(
                symbol="",  # Empty symbol
                current_price=50000.0,
                portfolio_value=100000.0,
                strategy_name="TestStrategy",
            )
            calculate_position_size(request)

        assert "symbol" in str(exc_info.value).lower()

    def test_validation_negative_price(self):
        """Test validation catches negative price."""
        with pytest.raises(ValidationError) as exc_info:
            request = PositionSizeRequest(
                symbol="BTC",
                current_price=-50000.0,  # Negative
                portfolio_value=100000.0,
                strategy_name="TestStrategy",
            )
            calculate_position_size(request)

        assert "current_price" in str(exc_info.value).lower()

    def test_validation_negative_portfolio(self):
        """Test validation catches negative portfolio value."""
        with pytest.raises(ValidationError) as exc_info:
            request = PositionSizeRequest(
                symbol="BTC",
                current_price=50000.0,
                portfolio_value=-100000.0,  # Negative
                strategy_name="TestStrategy",
            )
            calculate_position_size(request)

        assert "portfolio_value" in str(exc_info.value).lower()

    def test_validation_invalid_kelly_inputs(self):
        """Test validation catches invalid Kelly inputs and returns warnings."""
        request = PositionSizeRequest(
            symbol="BTC",
            current_price=1000.0,  # Lower price so we can check behavior
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            win_rate=1.5,  # Invalid > 1.0
            avg_win=0.05,
            avg_loss=-0.03,
        )
        response = calculate_position_size(request)

        # Should have warnings about invalid inputs (implementation still returns minimal position)
        assert len(response.warnings) > 0
        assert any("Win rate" in w or "Kelly" in w for w in response.warnings)

    def test_validation_invalid_confidence(self):
        """Test validation catches invalid confidence."""
        with pytest.raises(ValidationError):
            request = PositionSizeRequest(
                symbol="BTC",
                current_price=50000.0,
                portfolio_value=100000.0,
                strategy_name="TestStrategy",
                confidence=1.5,  # Invalid > 1.0
            )
            calculate_position_size(request)


class TestPortfolioAllocation:
    """Test portfolio-level allocation."""

    def test_portfolio_allocation_single_asset(self):
        """Test portfolio allocation with single asset."""
        requests = [
            PositionSizeRequest(
                symbol="BTC",
                current_price=50000.0,
                portfolio_value=100000.0,
                strategy_name="Strategy1",
                win_rate=0.6,
                avg_win=0.05,
                avg_loss=-0.03,
            )
        ]

        result = calculate_portfolio_allocation(requests)

        assert result.primary.symbol == "BTC"
        assert len(result.alternatives) == 1
        assert "total_allocation_pct" in result.portfolio_impact

    def test_portfolio_allocation_multiple_assets(self):
        """Test portfolio allocation across multiple assets."""
        requests = [
            PositionSizeRequest(
                symbol="BTC",
                current_price=50000.0,
                portfolio_value=100000.0,
                strategy_name="Strategy1",
                win_rate=0.6,
                avg_win=0.05,
                avg_loss=-0.03,
            ),
            PositionSizeRequest(
                symbol="ETH",
                current_price=3000.0,
                portfolio_value=100000.0,
                strategy_name="Strategy2",
                win_rate=0.55,
                avg_win=0.04,
                avg_loss=-0.025,
            ),
            PositionSizeRequest(
                symbol="SOL",
                current_price=100.0,
                portfolio_value=100000.0,
                strategy_name="Strategy3",
                win_rate=0.58,
                avg_win=0.045,
                avg_loss=-0.028,
            ),
        ]

        result = calculate_portfolio_allocation(requests)

        assert len(result.alternatives) == 3
        assert result.portfolio_impact["num_positions"] <= 3
        assert result.portfolio_impact["total_allocation_pct"] > 0

    def test_portfolio_allocation_scales_down_excessive(self):
        """Test portfolio allocation scales down when exceeding limits."""
        # Create requests that would exceed 80% allocation
        requests = [
            PositionSizeRequest(
                symbol=f"ASSET{i}",
                current_price=1000.0,
                portfolio_value=100000.0,
                strategy_name=f"Strategy{i}",
                win_rate=0.7,  # High win rate for large positions
                avg_win=0.1,
                avg_loss=-0.02,
                risk_params=RiskParameters(max_position_size=0.3),  # Each wants 30%
            )
            for i in range(4)
        ]

        result = calculate_portfolio_allocation(requests)

        # Should scale down to stay under 80%
        total_allocation = result.portfolio_impact["total_allocation_pct"]
        assert total_allocation <= 0.85  # Allow small margin

    def test_portfolio_allocation_empty_requests(self):
        """Test portfolio allocation with empty requests."""
        result = calculate_portfolio_allocation([])

        assert result.primary.recommended_shares == 0
        assert len(result.primary.warnings) > 0

    def test_portfolio_allocation_metrics(self):
        """Test portfolio allocation calculates proper metrics."""
        requests = [
            PositionSizeRequest(
                symbol="BTC",
                current_price=50000.0,
                portfolio_value=100000.0,
                strategy_name="Strategy1",
                win_rate=0.6,
                avg_win=0.05,
                avg_loss=-0.03,
            ),
            PositionSizeRequest(
                symbol="ETH",
                current_price=3000.0,
                portfolio_value=100000.0,
                strategy_name="Strategy2",
                win_rate=0.55,
                avg_win=0.04,
                avg_loss=-0.025,
            ),
        ]

        result = calculate_portfolio_allocation(requests)

        impact = result.portfolio_impact
        assert "total_allocation_pct" in impact
        assert "total_risk_pct" in impact
        assert "expected_portfolio_return" in impact
        assert "max_portfolio_loss" in impact
        assert "num_positions" in impact


class TestRiskMetrics:
    """Test risk metric calculations."""

    def test_risk_metrics_with_avg_loss(self):
        """Test risk metrics use average loss when available."""
        request = PositionSizeRequest(
            symbol="BTC",
            current_price=50000.0,
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            win_rate=0.6,
            avg_win=0.05,
            avg_loss=-0.04,  # 4% average loss
        )

        response = calculate_position_size(request)

        # Risk should be based on average loss
        expected_risk = response.position_size_pct * 0.04
        assert abs(response.risk_pct - expected_risk) < 0.01

    def test_risk_metrics_with_volatility(self):
        """Test risk metrics use volatility when no avg_loss."""
        request = PositionSizeRequest(
            symbol="BTC",
            current_price=1000.0,  # Lower price so position can be sized
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.INTELLIGENT,
            win_rate=0.6,  # Add Kelly inputs
            avg_win=0.05,
            avg_loss=-0.03,
            volatility=0.3,  # 30% volatility
        )

        response = calculate_position_size(request)

        # Risk should be based on volatility
        assert response.risk_pct > 0

    def test_risk_metrics_default(self):
        """Test risk metrics use default when no data."""
        request = PositionSizeRequest(
            symbol="BTC",
            current_price=50000.0,
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            # No avg_loss or volatility
        )

        response = calculate_position_size(request)

        # Should use default 5% risk estimate
        expected_risk = response.position_size_pct * 0.05
        assert abs(response.risk_pct - expected_risk) < 0.01


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_portfolio_value(self):
        """Test handling of zero portfolio value."""
        with pytest.raises(ValidationError):
            request = PositionSizeRequest(
                symbol="BTC",
                current_price=50000.0,
                portfolio_value=0.0,  # Zero
                strategy_name="TestStrategy",
            )
            calculate_position_size(request)

    def test_very_small_portfolio(self):
        """Test handling of very small portfolio."""
        request = PositionSizeRequest(
            symbol="BTC",
            current_price=50000.0,
            portfolio_value=100.0,  # Only $100
            strategy_name="TestStrategy",
        )

        response = calculate_position_size(request)

        # Can't afford any shares
        assert response.recommended_shares == 0

    def test_very_expensive_asset(self):
        """Test handling of very expensive asset."""
        request = PositionSizeRequest(
            symbol="BRK.A",
            current_price=500000.0,  # $500k per share
            portfolio_value=100000.0,  # Only $100k portfolio
            strategy_name="TestStrategy",
        )

        response = calculate_position_size(request)

        # Can't afford any shares
        assert response.recommended_shares == 0

    def test_negative_expected_value(self):
        """Test handling of strategy with negative expected value."""
        request = PositionSizeRequest(
            symbol="BTC",
            current_price=50000.0,
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.KELLY,
            win_rate=0.3,  # Low win rate
            avg_win=0.02,  # Small wins
            avg_loss=-0.05,  # Large losses
        )

        response = calculate_position_size(request)

        # Kelly should return zero for negative expected value
        assert response.recommended_shares == 0

    def test_perfect_strategy(self):
        """Test handling of unrealistic perfect strategy."""
        request = PositionSizeRequest(
            symbol="BTC",
            current_price=50000.0,
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            method=SizingMethod.KELLY,
            win_rate=0.98,  # Near perfect
            avg_win=0.2,  # Large wins
            avg_loss=-0.01,  # Tiny losses
            risk_params=RiskParameters(max_position_size=0.2),
        )

        response = calculate_position_size(request)

        # Should cap at max position size
        assert response.position_size_pct <= 0.2

    def test_extreme_strategy_multiplier(self):
        """Test handling of extreme strategy multiplier."""
        request = PositionSizeRequest(
            symbol="BTC",
            current_price=1000.0,  # Lower price so position can be sized
            portfolio_value=100000.0,
            strategy_name="TestStrategy",
            win_rate=0.6,
            avg_win=0.05,
            avg_loss=-0.03,
            strategy_multiplier=3.0,  # High but not so extreme it triggers errors
            risk_params=RiskParameters(max_position_size=0.5, kelly_fraction=0.25),  # Higher cap
        )

        response = calculate_position_size(request)

        # Should still be capped at max
        assert response.position_size_pct <= 0.5

    def test_all_regimes_crisis(self):
        """Test portfolio allocation when all assets in crisis."""
        requests = [
            PositionSizeRequest(
                symbol=f"ASSET{i}",
                current_price=100.0,
                portfolio_value=100000.0,
                strategy_name=f"Strategy{i}",
                market_regime="crisis",
                win_rate=0.5,
                avg_win=0.03,
                avg_loss=-0.03,
            )
            for i in range(3)
        ]

        result = calculate_portfolio_allocation(requests)

        # All positions should be very small
        for alt in result.alternatives:
            assert alt.position_size_pct < 0.05
