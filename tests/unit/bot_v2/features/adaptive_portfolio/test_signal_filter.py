"""Tests for SignalFilter."""

from unittest.mock import Mock

import pytest

from bot_v2.features.adaptive_portfolio.signal_filter import SignalFilter
from bot_v2.features.adaptive_portfolio.types import (
    PortfolioSnapshot,
    PortfolioTier,
    PositionConstraints,
    PositionInfo,
    RiskProfile,
    TierConfig,
    TradingRules,
    TradingSignal,
)


@pytest.fixture
def portfolio_snapshot() -> PortfolioSnapshot:
    """Create sample portfolio snapshot with one existing position."""
    return PortfolioSnapshot(
        total_value=3000.0,
        cash=2000.0,
        positions=[PositionInfo("AAPL", 10, 100.0, 100.0, 1000.0, 0.0, 0.0, 1)],
        daily_pnl=0.0,
        daily_pnl_pct=0.0,
        quarterly_pnl_pct=0.0,
        current_tier=PortfolioTier.MICRO,
        positions_count=1,
        largest_position_pct=33.33,
        sector_exposures={},
    )


@pytest.fixture
def market_constraints() -> Mock:
    """Create mock market constraints."""
    constraints = Mock()
    constraints.excluded_symbols = ["TSLA", "GME"]
    return constraints


class TestSignalFilter:
    """Test SignalFilter."""

    def test_filters_existing_positions(
        self, portfolio_snapshot: PortfolioSnapshot, market_constraints: Mock
    ) -> None:
        """Filters out signals for symbols already in portfolio."""
        tier_config = TierConfig(
            name="micro",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        signals = [
            TradingSignal("AAPL", "BUY", 0.8, 500.0, 5.0, "momentum", "existing"),
            TradingSignal("MSFT", "BUY", 0.8, 500.0, 5.0, "momentum", "new"),
        ]

        filter_obj = SignalFilter()
        filtered = filter_obj.filter_signals(
            signals, tier_config, portfolio_snapshot, market_constraints
        )

        # AAPL should be filtered out (already owned), MSFT should pass
        assert len(filtered) == 1
        assert filtered[0].symbol == "MSFT"

    def test_micro_tier_requires_high_confidence(
        self, portfolio_snapshot: PortfolioSnapshot, market_constraints: Mock
    ) -> None:
        """Micro tier requires 0.7+ confidence."""
        tier_config = TierConfig(
            name="Micro Portfolio",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        # Empty portfolio to avoid existing position filter
        empty_snapshot = PortfolioSnapshot(
            total_value=3000.0,
            cash=3000.0,
            positions=[],
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=0,
            largest_position_pct=0.0,
            sector_exposures={},
        )

        signals = [
            TradingSignal("SYM1", "BUY", 0.6, 500.0, 5.0, "momentum", "too low"),
            TradingSignal("SYM2", "BUY", 0.75, 500.0, 5.0, "momentum", "passes"),
        ]

        filter_obj = SignalFilter()
        filtered = filter_obj.filter_signals(
            signals, tier_config, empty_snapshot, market_constraints
        )

        # Only 0.75 confidence signal should pass (micro requires 0.7+)
        assert len(filtered) == 1
        assert filtered[0].symbol == "SYM2"

    def test_small_tier_requires_moderate_confidence(self, market_constraints: Mock) -> None:
        """Small tier requires 0.6+ confidence."""
        tier_config = TierConfig(
            name="Small Portfolio",
            range=(5000, 25000),
            positions=PositionConstraints(3, 8, 5, 500.0),
            min_position_size=500.0,
            strategies=["momentum"],
            risk=RiskProfile(1.5, 8.0, 4.0, 40.0),
            trading=TradingRules(10, "cash", 1, True),
        )

        empty_snapshot = PortfolioSnapshot(
            total_value=10000.0,
            cash=10000.0,
            positions=[],
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.SMALL,
            positions_count=0,
            largest_position_pct=0.0,
            sector_exposures={},
        )

        signals = [
            TradingSignal("SYM1", "BUY", 0.55, 600.0, 4.0, "momentum", "too low"),
            TradingSignal("SYM2", "BUY", 0.65, 600.0, 4.0, "momentum", "passes"),
        ]

        filter_obj = SignalFilter()
        filtered = filter_obj.filter_signals(
            signals, tier_config, empty_snapshot, market_constraints
        )

        # Only 0.65 confidence signal should pass (small requires 0.6+)
        assert len(filtered) == 1
        assert filtered[0].symbol == "SYM2"

    def test_medium_tier_requires_standard_confidence(self, market_constraints: Mock) -> None:
        """Medium tier requires 0.5+ confidence."""
        tier_config = TierConfig(
            name="Medium Portfolio",
            range=(25000, 100000),
            positions=PositionConstraints(5, 15, 10, 1000.0),
            min_position_size=1000.0,
            strategies=["momentum"],
            risk=RiskProfile(1.0, 6.0, 3.0, 30.0),
            trading=TradingRules(15, "margin", 0, False),
        )

        empty_snapshot = PortfolioSnapshot(
            total_value=50000.0,
            cash=50000.0,
            positions=[],
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.MEDIUM,
            positions_count=0,
            largest_position_pct=0.0,
            sector_exposures={},
        )

        signals = [
            TradingSignal("SYM1", "BUY", 0.45, 1200.0, 3.0, "momentum", "too low"),
            TradingSignal("SYM2", "BUY", 0.55, 1200.0, 3.0, "momentum", "passes"),
        ]

        filter_obj = SignalFilter()
        filtered = filter_obj.filter_signals(
            signals, tier_config, empty_snapshot, market_constraints
        )

        # Only 0.55 confidence signal should pass (medium requires 0.5+)
        assert len(filtered) == 1
        assert filtered[0].symbol == "SYM2"

    def test_large_tier_allows_lower_confidence(self, market_constraints: Mock) -> None:
        """Large tier requires 0.4+ confidence."""
        tier_config = TierConfig(
            name="Large Portfolio",
            range=(100000, float("inf")),
            positions=PositionConstraints(10, 30, 20, 1000.0),
            min_position_size=1000.0,
            strategies=["momentum"],
            risk=RiskProfile(1.0, 5.0, 3.0, 25.0),
            trading=TradingRules(20, "margin", 0, False),
        )

        empty_snapshot = PortfolioSnapshot(
            total_value=200000.0,
            cash=200000.0,
            positions=[],
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.LARGE,
            positions_count=0,
            largest_position_pct=0.0,
            sector_exposures={},
        )

        signals = [
            TradingSignal("SYM1", "BUY", 0.35, 2000.0, 2.5, "momentum", "too low"),
            TradingSignal("SYM2", "BUY", 0.45, 2000.0, 2.5, "momentum", "passes"),
        ]

        filter_obj = SignalFilter()
        filtered = filter_obj.filter_signals(
            signals, tier_config, empty_snapshot, market_constraints
        )

        # Only 0.45 confidence signal should pass (large requires 0.4+)
        assert len(filtered) == 1
        assert filtered[0].symbol == "SYM2"

    def test_filters_positions_below_minimum_size(self, market_constraints: Mock) -> None:
        """Filters signals with position size below tier minimum."""
        tier_config = TierConfig(
            name="micro",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=500.0,  # Minimum $500
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        empty_snapshot = PortfolioSnapshot(
            total_value=3000.0,
            cash=3000.0,
            positions=[],
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=0,
            largest_position_pct=0.0,
            sector_exposures={},
        )

        signals = [
            TradingSignal("SYM1", "BUY", 0.75, 400.0, 5.0, "momentum", "too small"),
            TradingSignal("SYM2", "BUY", 0.75, 600.0, 5.0, "momentum", "passes"),
        ]

        filter_obj = SignalFilter()
        filtered = filter_obj.filter_signals(
            signals, tier_config, empty_snapshot, market_constraints
        )

        # Only $600 position should pass ($500 minimum)
        assert len(filtered) == 1
        assert filtered[0].symbol == "SYM2"

    def test_filters_excluded_symbols(self, market_constraints: Mock) -> None:
        """Filters symbols in market constraints excluded list."""
        tier_config = TierConfig(
            name="micro",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        empty_snapshot = PortfolioSnapshot(
            total_value=3000.0,
            cash=3000.0,
            positions=[],
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=0,
            largest_position_pct=0.0,
            sector_exposures={},
        )

        # market_constraints.excluded_symbols = ["TSLA", "GME"]
        signals = [
            TradingSignal("TSLA", "BUY", 0.75, 500.0, 5.0, "momentum", "excluded"),
            TradingSignal("GME", "BUY", 0.75, 500.0, 5.0, "momentum", "excluded"),
            TradingSignal("MSFT", "BUY", 0.75, 500.0, 5.0, "momentum", "passes"),
        ]

        filter_obj = SignalFilter()
        filtered = filter_obj.filter_signals(
            signals, tier_config, empty_snapshot, market_constraints
        )

        # Only MSFT should pass (TSLA and GME are excluded)
        assert len(filtered) == 1
        assert filtered[0].symbol == "MSFT"

    def test_applies_all_filters_together(
        self, portfolio_snapshot: PortfolioSnapshot, market_constraints: Mock
    ) -> None:
        """Applies all filters (position, confidence, size, constraints) together."""
        tier_config = TierConfig(
            name="Micro Portfolio",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=400.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        # portfolio_snapshot has AAPL position
        # market_constraints excludes TSLA, GME
        signals = [
            TradingSignal("AAPL", "BUY", 0.75, 500.0, 5.0, "momentum", "existing"),
            TradingSignal("MSFT", "BUY", 0.65, 500.0, 5.0, "momentum", "low confidence"),
            TradingSignal("GOOGL", "BUY", 0.75, 300.0, 5.0, "momentum", "too small"),
            TradingSignal("TSLA", "BUY", 0.75, 500.0, 5.0, "momentum", "excluded"),
            TradingSignal("NVDA", "BUY", 0.75, 500.0, 5.0, "momentum", "PASSES"),
        ]

        filter_obj = SignalFilter()
        filtered = filter_obj.filter_signals(
            signals, tier_config, portfolio_snapshot, market_constraints
        )

        # Only NVDA passes all filters
        assert len(filtered) == 1
        assert filtered[0].symbol == "NVDA"

    def test_handles_empty_signal_list(
        self, portfolio_snapshot: PortfolioSnapshot, market_constraints: Mock
    ) -> None:
        """Handles empty signal list gracefully."""
        tier_config = TierConfig(
            name="micro",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        signals = []

        filter_obj = SignalFilter()
        filtered = filter_obj.filter_signals(
            signals, tier_config, portfolio_snapshot, market_constraints
        )

        assert filtered == []

    def test_returns_empty_when_all_filtered_out(
        self, portfolio_snapshot: PortfolioSnapshot, market_constraints: Mock
    ) -> None:
        """Returns empty list when all signals are filtered out."""
        tier_config = TierConfig(
            name="Micro Portfolio",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        # All signals fail confidence threshold (micro requires 0.7+)
        empty_snapshot = PortfolioSnapshot(
            total_value=3000.0,
            cash=3000.0,
            positions=[],
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            quarterly_pnl_pct=0.0,
            current_tier=PortfolioTier.MICRO,
            positions_count=0,
            largest_position_pct=0.0,
            sector_exposures={},
        )

        signals = [
            TradingSignal("SYM1", "BUY", 0.5, 500.0, 5.0, "momentum", "low"),
            TradingSignal("SYM2", "BUY", 0.6, 500.0, 5.0, "momentum", "low"),
        ]

        filter_obj = SignalFilter()
        filtered = filter_obj.filter_signals(
            signals, tier_config, empty_snapshot, market_constraints
        )

        assert filtered == []
