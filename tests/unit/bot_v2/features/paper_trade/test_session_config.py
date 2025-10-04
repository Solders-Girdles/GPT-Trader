"""
Tests for paper trading session configuration.

Tests cover:
- PaperSessionConfig creation with defaults
- Custom parameter values
- Validation rules for all parameters
- SessionConfigBuilder.from_kwargs pattern
- Strategy parameter passthrough
"""

import pytest

from bot_v2.features.paper_trade.session_config import (
    PaperSessionConfig,
    SessionConfigBuilder,
)


# ============================================================================
# Test: PaperSessionConfig Creation
# ============================================================================


class TestPaperSessionConfigCreation:
    """Test PaperSessionConfig dataclass creation."""

    def test_config_with_defaults(self):
        """Test config creation with default values."""
        config = PaperSessionConfig(
            strategy_name="SimpleMAStrategy",
            symbols=["AAPL", "MSFT"],
        )

        assert config.strategy_name == "SimpleMAStrategy"
        assert config.symbols == ["AAPL", "MSFT"]
        assert config.initial_capital == 100000.0
        assert config.commission == 0.001
        assert config.slippage == 0.0005
        assert config.max_positions == 10
        assert config.position_size == 0.95
        assert config.update_interval == 60
        assert config.strategy_params == {}

    def test_config_with_custom_values(self):
        """Test config creation with custom values."""
        config = PaperSessionConfig(
            strategy_name="MomentumStrategy",
            symbols=["BTC-USD"],
            initial_capital=50000.0,
            commission=0.002,
            slippage=0.001,
            max_positions=5,
            position_size=0.8,
            update_interval=30,
            strategy_params={"lookback": 15, "threshold": 0.03},
        )

        assert config.initial_capital == 50000.0
        assert config.commission == 0.002
        assert config.slippage == 0.001
        assert config.max_positions == 5
        assert config.position_size == 0.8
        assert config.update_interval == 30
        assert config.strategy_params == {"lookback": 15, "threshold": 0.03}

    def test_config_with_empty_symbols(self):
        """Test config with empty symbols list (valid but edge case)."""
        config = PaperSessionConfig(
            strategy_name="SimpleMAStrategy",
            symbols=[],
        )

        assert config.symbols == []

    def test_config_with_zero_capital(self):
        """Test config with zero initial capital (valid)."""
        config = PaperSessionConfig(
            strategy_name="SimpleMAStrategy",
            symbols=["AAPL"],
            initial_capital=0.0,
        )

        assert config.initial_capital == 0.0

    def test_config_with_zero_update_interval(self):
        """Test config with zero update interval (valid for testing)."""
        config = PaperSessionConfig(
            strategy_name="SimpleMAStrategy",
            symbols=["AAPL"],
            update_interval=0,
        )

        assert config.update_interval == 0


# ============================================================================
# Test: PaperSessionConfig Validation
# ============================================================================
#
# NOTE: Validation tests intentionally omitted for Phase 1 to preserve
# backward compatibility. The original PaperTradingSession accepted all
# parameter values without validation. Validation can be added in a future
# phase if desired.
#
# Potential future validation tests:
# - Negative initial_capital
# - Negative/excessive commission
# - Negative/excessive slippage
# - Invalid max_positions
# - Invalid position_size
# - Negative update_interval
# - Non-list symbols
# ============================================================================


# ============================================================================
# Test: SessionConfigBuilder
# ============================================================================


class TestSessionConfigBuilder:
    """Test SessionConfigBuilder.from_kwargs pattern."""

    def test_from_kwargs_default_params(self):
        """Test builder with default parameters."""
        config = SessionConfigBuilder.from_kwargs(
            strategy="SimpleMAStrategy",
            symbols=["AAPL", "MSFT"],
        )

        assert config.strategy_name == "SimpleMAStrategy"
        assert config.symbols == ["AAPL", "MSFT"]
        assert config.initial_capital == 100000.0
        assert config.commission == 0.001
        assert config.slippage == 0.0005
        assert config.max_positions == 10
        assert config.position_size == 0.95
        assert config.update_interval == 60
        assert config.strategy_params == {}

    def test_from_kwargs_custom_session_params(self):
        """Test builder with custom session parameters."""
        config = SessionConfigBuilder.from_kwargs(
            strategy="MomentumStrategy",
            symbols=["BTC-USD"],
            initial_capital=50000,
            commission=0.002,
            slippage=0.001,
            max_positions=5,
            position_size=0.8,
            update_interval=30,
        )

        assert config.initial_capital == 50000
        assert config.commission == 0.002
        assert config.slippage == 0.001
        assert config.max_positions == 5
        assert config.position_size == 0.8
        assert config.update_interval == 30
        assert config.strategy_params == {}

    def test_from_kwargs_strategy_params_passthrough(self):
        """Test that non-session kwargs are passed to strategy_params."""
        config = SessionConfigBuilder.from_kwargs(
            strategy="SimpleMAStrategy",
            symbols=["AAPL"],
            fast_period=5,
            slow_period=20,
        )

        assert config.strategy_params == {"fast_period": 5, "slow_period": 20}

    def test_from_kwargs_mixed_params(self):
        """Test builder with both session and strategy parameters."""
        config = SessionConfigBuilder.from_kwargs(
            strategy="MomentumStrategy",
            symbols=["BTC-USD"],
            initial_capital=50000,
            commission=0.002,
            lookback=15,
            threshold=0.03,
        )

        assert config.initial_capital == 50000
        assert config.commission == 0.002
        assert config.strategy_params == {"lookback": 15, "threshold": 0.03}

    def test_from_kwargs_accepts_edge_case_values(self):
        """Test that builder accepts edge case values (no validation in Phase 1)."""
        # Should accept negative capital (preserves old behavior)
        config = SessionConfigBuilder.from_kwargs(
            strategy="SimpleMAStrategy",
            symbols=["AAPL"],
            initial_capital=-1000,
        )
        assert config.initial_capital == -1000

    def test_from_kwargs_preserves_all_strategy_params(self):
        """Test that all non-session kwargs are preserved."""
        config = SessionConfigBuilder.from_kwargs(
            strategy="CustomStrategy",
            symbols=["AAPL"],
            param1="value1",
            param2=42,
            param3=True,
            param4=[1, 2, 3],
        )

        assert config.strategy_params == {
            "param1": "value1",
            "param2": 42,
            "param3": True,
            "param4": [1, 2, 3],
        }


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestSessionConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_position_size_exactly_1(self):
        """Test that position_size of exactly 1.0 is valid."""
        config = PaperSessionConfig(
            strategy_name="SimpleMAStrategy",
            symbols=["AAPL"],
            position_size=1.0,
        )

        assert config.position_size == 1.0

    def test_commission_boundary_valid(self):
        """Test commission just under 100% is valid."""
        config = PaperSessionConfig(
            strategy_name="SimpleMAStrategy",
            symbols=["AAPL"],
            commission=0.999,
        )

        assert config.commission == 0.999

    def test_slippage_boundary_valid(self):
        """Test slippage just under 100% is valid."""
        config = PaperSessionConfig(
            strategy_name="SimpleMAStrategy",
            symbols=["AAPL"],
            slippage=0.999,
        )

        assert config.slippage == 0.999

    def test_very_large_capital(self):
        """Test with very large initial capital."""
        config = PaperSessionConfig(
            strategy_name="SimpleMAStrategy",
            symbols=["AAPL"],
            initial_capital=1_000_000_000.0,
        )

        assert config.initial_capital == 1_000_000_000.0

    def test_very_small_position_size(self):
        """Test with very small position size."""
        config = PaperSessionConfig(
            strategy_name="SimpleMAStrategy",
            symbols=["AAPL"],
            position_size=0.01,
        )

        assert config.position_size == 0.01

    def test_max_positions_one(self):
        """Test max_positions boundary of 1."""
        config = PaperSessionConfig(
            strategy_name="SimpleMAStrategy",
            symbols=["AAPL"],
            max_positions=1,
        )

        assert config.max_positions == 1

    def test_very_large_max_positions(self):
        """Test with very large max_positions."""
        config = PaperSessionConfig(
            strategy_name="SimpleMAStrategy",
            symbols=["AAPL"],
            max_positions=1000,
        )

        assert config.max_positions == 1000

    def test_many_symbols(self):
        """Test with many symbols."""
        symbols = [f"SYM{i}" for i in range(100)]
        config = PaperSessionConfig(
            strategy_name="SimpleMAStrategy",
            symbols=symbols,
        )

        assert len(config.symbols) == 100
