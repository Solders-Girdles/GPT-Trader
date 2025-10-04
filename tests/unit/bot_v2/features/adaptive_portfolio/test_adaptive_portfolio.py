"""Tests for AdaptivePortfolioManager - tier-based portfolio management.

This module tests the AdaptivePortfolioManager's ability to automatically
adjust trading strategies, risk parameters, and position sizing based on
portfolio capital and tier configuration.

Critical behaviors tested:
- Initialization and configuration loading
- Tier determination based on capital
- Tier transition detection and handling
- Portfolio snapshot creation
- Signal generation appropriate to tier
- Risk metric calculation
- Recommendation and warning generation
- Backtest execution with tier transitions
- Position management and updates
- Trade execution simulation

Business Context:
    The AdaptivePortfolioManager is responsible for ensuring that trading
    behavior scales appropriately with account size. Failures here can result in:

    - Small accounts using aggressive strategies meant for large accounts
    - Large accounts constrained by overly conservative micro-account rules
    - Incorrect position sizing leading to excessive risk or missed opportunities
    - PDT violations for accounts under $25K
    - Over-diversification in small accounts or under-diversification in large ones

    This is the intelligence layer that prevents "one size fits all" disasters.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import Mock, patch

import pytest

from bot_v2.features.adaptive_portfolio.adaptive_portfolio import (
    AdaptivePortfolioManager,
    run_adaptive_backtest,
    run_adaptive_strategy,
)
from bot_v2.features.adaptive_portfolio.types import (
    AdaptiveResult,
    BacktestMetrics,
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
def mock_data_provider() -> Mock:
    """Create a mock data provider."""
    provider = Mock()
    provider.get_historical_data = Mock(return_value=Mock())
    return provider


@pytest.fixture
def sample_tier_config() -> TierConfig:
    """Create a sample tier configuration using proper dataclasses."""
    return TierConfig(
        name="micro",
        range=(0, 5000),
        positions=PositionConstraints(
            min_positions=1,
            max_positions=3,
            target_positions=2,
            min_position_size=100,
        ),
        min_position_size=100,
        strategies=["momentum"],
        risk=RiskProfile(
            daily_limit_pct=2.0,
            quarterly_limit_pct=10.0,
            position_stop_loss_pct=5.0,
            max_sector_exposure_pct=50.0,
        ),
        trading=TradingRules(
            max_trades_per_week=5,
            account_type="cash",
            settlement_days=2,
            pdt_compliant=True,
        ),
    )


@pytest.fixture
def sample_positions() -> list[PositionInfo]:
    """Create sample positions for testing."""
    return [
        PositionInfo(
            symbol="AAPL",
            shares=10,
            entry_price=150.0,
            current_price=155.0,
            position_value=1550.0,
            unrealized_pnl=50.0,
            unrealized_pnl_pct=3.33,
            days_held=5,
            stop_loss_price=142.5,
        ),
        PositionInfo(
            symbol="MSFT",
            shares=5,
            entry_price=300.0,
            current_price=310.0,
            position_value=1550.0,
            unrealized_pnl=50.0,
            unrealized_pnl_pct=3.33,
            days_held=3,
            stop_loss_price=285.0,
        ),
    ]


class TestAdaptivePortfolioManagerInitialization:
    """Test AdaptivePortfolioManager initialization."""

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.get_data_provider")
    def test_initializes_with_defaults(
        self, mock_get_provider: Mock, mock_load_config: Mock
    ) -> None:
        """Initializes with default configuration and mock data provider.

        Default behavior allows quick testing without external dependencies.
        """
        mock_config = Mock()
        mock_config.tiers = {}
        mock_load_config.return_value = mock_config
        mock_get_provider.return_value = Mock()

        manager = AdaptivePortfolioManager(prefer_real_data=False)

        assert manager.config is not None
        assert manager.data_provider is not None
        assert manager.data_provider_type == "mock"

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.get_data_provider")
    def test_initializes_with_real_data_provider(
        self, mock_get_provider: Mock, mock_load_config: Mock
    ) -> None:
        """Initializes with real data provider when requested.

        Production mode uses real market data.
        """
        mock_config = Mock()
        mock_config.tiers = {}
        mock_load_config.return_value = mock_config
        mock_get_provider.return_value = Mock()

        manager = AdaptivePortfolioManager(prefer_real_data=True)

        mock_get_provider.assert_called_with("yfinance")
        assert manager.data_provider_type == "yfinance"

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    def test_initializes_with_custom_data_provider(
        self, mock_load_config: Mock, mock_data_provider: Mock
    ) -> None:
        """Accepts custom data provider for testing/flexibility.

        Allows injection of custom data sources.
        """
        mock_config = Mock()
        mock_config.tiers = {}
        mock_load_config.return_value = mock_config

        manager = AdaptivePortfolioManager(data_provider=mock_data_provider)

        assert manager.data_provider is mock_data_provider
        assert manager.data_provider_type == "custom"

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.get_data_provider")
    def test_initializes_submodules(self, mock_get_provider: Mock, mock_load_config: Mock) -> None:
        """Initializes tier manager, risk manager, and strategy selector.

        All subsystems must be created and wired together.
        """
        mock_config = Mock()
        mock_config.tiers = {}
        mock_load_config.return_value = mock_config
        mock_get_provider.return_value = Mock()

        manager = AdaptivePortfolioManager(prefer_real_data=False)

        assert manager.tier_manager is not None
        assert manager.risk_manager is not None
        assert manager.strategy_selector is not None

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.get_data_provider")
    def test_loads_custom_config_path(
        self, mock_get_provider: Mock, mock_load_config: Mock
    ) -> None:
        """Loads configuration from custom path when provided.

        Allows per-strategy or per-environment configurations.
        """
        mock_config = Mock()
        mock_config.tiers = {}
        mock_load_config.return_value = mock_config
        mock_get_provider.return_value = Mock()

        custom_path = "/custom/config.json"
        manager = AdaptivePortfolioManager(config_path=custom_path, prefer_real_data=False)

        mock_load_config.assert_called_with(custom_path)
        assert manager.config_path == custom_path


class TestPortfolioAnalysis:
    """Test analyze_portfolio method."""

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.get_data_provider")
    def test_analyzes_portfolio_and_returns_result(
        self,
        mock_get_provider: Mock,
        mock_load_config: Mock,
        sample_tier_config: dict[str, Any],
    ) -> None:
        """Analyzes portfolio and returns comprehensive result.

        Core functionality - must return actionable analysis.
        """
        mock_config = Mock()
        mock_config.tiers = {"micro": sample_tier_config}
        mock_load_config.return_value = mock_config
        mock_get_provider.return_value = Mock()

        manager = AdaptivePortfolioManager(prefer_real_data=False)
        manager.tier_manager.detect_tier = Mock(
            return_value=(PortfolioTier.MICRO, sample_tier_config)
        )
        manager.tier_manager.should_transition = Mock(return_value=(False, None))
        manager.risk_manager.calculate_risk_metrics = Mock(return_value={})
        manager.strategy_selector.generate_signals = Mock(return_value=[])

        result = manager.analyze_portfolio(current_capital=2000.0)

        assert isinstance(result, AdaptiveResult)
        assert result.current_tier == PortfolioTier.MICRO
        assert result.portfolio_snapshot.total_value == 2000.0

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.get_data_provider")
    def test_creates_portfolio_snapshot(
        self,
        mock_get_provider: Mock,
        mock_load_config: Mock,
        sample_tier_config: dict[str, Any],
        sample_positions: list[PositionInfo],
    ) -> None:
        """Creates accurate portfolio snapshot with positions.

        Snapshot must reflect current portfolio state accurately.
        """
        mock_config = Mock()
        mock_config.tiers = {"micro": sample_tier_config}
        mock_load_config.return_value = mock_config
        mock_get_provider.return_value = Mock()

        manager = AdaptivePortfolioManager(prefer_real_data=False)
        manager.tier_manager.detect_tier = Mock(
            return_value=(PortfolioTier.MICRO, sample_tier_config)
        )
        manager.tier_manager.should_transition = Mock(return_value=(False, None))
        manager.risk_manager.calculate_risk_metrics = Mock(return_value={})
        manager.strategy_selector.generate_signals = Mock(return_value=[])

        result = manager.analyze_portfolio(current_capital=5000.0, positions=sample_positions)

        snapshot = result.portfolio_snapshot
        assert snapshot.total_value == 5000.0
        assert snapshot.positions_count == 2
        assert len(snapshot.positions) == 2
        assert snapshot.daily_pnl == 100.0  # 50 + 50

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.get_data_provider")
    def test_detects_tier_transition_needed(
        self,
        mock_get_provider: Mock,
        mock_load_config: Mock,
        sample_tier_config: dict[str, Any],
    ) -> None:
        """Detects when portfolio needs to transition to different tier.

        Critical: Must identify when account size crosses tier boundaries.
        """
        mock_config = Mock()
        mock_config.tiers = {"micro": sample_tier_config}
        mock_load_config.return_value = mock_config
        mock_get_provider.return_value = Mock()

        manager = AdaptivePortfolioManager(prefer_real_data=False)
        manager.tier_manager.detect_tier = Mock(
            return_value=(PortfolioTier.MICRO, sample_tier_config)
        )
        manager.tier_manager.should_transition = Mock(return_value=(True, PortfolioTier.SMALL))
        manager.risk_manager.calculate_risk_metrics = Mock(return_value={})
        manager.strategy_selector.generate_signals = Mock(return_value=[])

        result = manager.analyze_portfolio(current_capital=6000.0)  # Above micro tier

        assert result.tier_transition_needed is True
        assert result.tier_transition_target == PortfolioTier.SMALL

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.get_data_provider")
    def test_generates_tier_appropriate_signals(
        self,
        mock_get_provider: Mock,
        mock_load_config: Mock,
        sample_tier_config: dict[str, Any],
    ) -> None:
        """Generates trading signals appropriate for current tier.

        Micro accounts should get different signals than large accounts.
        """
        mock_config = Mock()
        mock_config.tiers = {"micro": sample_tier_config}
        mock_load_config.return_value = mock_config
        mock_get_provider.return_value = Mock()

        test_signal = TradingSignal(
            symbol="AAPL",
            action="BUY",
            confidence=0.8,
            target_position_size=500.0,
            stop_loss_pct=5.0,
            strategy_source="momentum",
            reasoning="Strong momentum",
        )

        manager = AdaptivePortfolioManager(prefer_real_data=False)
        manager.tier_manager.detect_tier = Mock(
            return_value=(PortfolioTier.MICRO, sample_tier_config)
        )
        manager.tier_manager.should_transition = Mock(return_value=(False, None))
        manager.risk_manager.calculate_risk_metrics = Mock(return_value={})
        manager.strategy_selector.generate_signals = Mock(return_value=[test_signal])

        result = manager.analyze_portfolio(current_capital=2000.0)

        assert len(result.signals) == 1
        assert result.signals[0].symbol == "AAPL"


class TestRecommendations:
    """Test recommendation generation."""

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.get_data_provider")
    def test_recommends_adding_positions_when_under_target(
        self,
        mock_get_provider: Mock,
        mock_load_config: Mock,
        sample_tier_config: dict[str, Any],
    ) -> None:
        """Recommends adding positions when below target count.

        Helps achieve proper diversification for tier.
        """
        mock_config = Mock()
        tier_cfg = sample_tier_config
        tier_cfg.positions.target_positions = 3
        mock_config.tiers = {"micro": tier_cfg}
        mock_load_config.return_value = mock_config
        mock_get_provider.return_value = Mock()

        manager = AdaptivePortfolioManager(prefer_real_data=False)
        manager.tier_manager.detect_tier = Mock(
            return_value=(PortfolioTier.MICRO, sample_tier_config)
        )
        manager.tier_manager.should_transition = Mock(return_value=(False, None))
        manager.risk_manager.calculate_risk_metrics = Mock(return_value={})
        manager.strategy_selector.generate_signals = Mock(return_value=[])

        # Portfolio with only 1 position, target is 3
        positions = [
            PositionInfo(
                symbol="AAPL",
                shares=10,
                entry_price=150.0,
                current_price=155.0,
                position_value=1550.0,
                unrealized_pnl=50.0,
                unrealized_pnl_pct=3.33,
                days_held=5,
            )
        ]

        result = manager.analyze_portfolio(current_capital=3000.0, positions=positions)

        assert any("adding" in rec.lower() for rec in result.recommended_actions)

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.get_data_provider")
    def test_recommends_reducing_positions_when_over_max(
        self,
        mock_get_provider: Mock,
        mock_load_config: Mock,
        sample_tier_config: dict[str, Any],
    ) -> None:
        """Recommends reducing positions when exceeding maximum.

        Prevents over-diversification in small accounts.
        """
        mock_config = Mock()
        tier_cfg = sample_tier_config
        tier_cfg.positions.max_positions = 2
        mock_config.tiers = {"micro": tier_cfg}
        mock_load_config.return_value = mock_config
        mock_get_provider.return_value = Mock()

        manager = AdaptivePortfolioManager(prefer_real_data=False)
        manager.tier_manager.detect_tier = Mock(
            return_value=(PortfolioTier.MICRO, sample_tier_config)
        )
        manager.tier_manager.should_transition = Mock(return_value=(False, None))
        manager.risk_manager.calculate_risk_metrics = Mock(return_value={})
        manager.strategy_selector.generate_signals = Mock(return_value=[])

        # Portfolio with 3 positions, max is 2
        positions = [
            PositionInfo("AAPL", 10, 150.0, 155.0, 1550.0, 50.0, 3.33, 5),
            PositionInfo("MSFT", 5, 300.0, 310.0, 1550.0, 50.0, 3.33, 3),
            PositionInfo("GOOGL", 3, 140.0, 145.0, 435.0, 15.0, 3.57, 1),
        ]

        result = manager.analyze_portfolio(current_capital=4000.0, positions=positions)

        assert any("reducing" in rec.lower() for rec in result.recommended_actions)


class TestWarnings:
    """Test warning generation."""

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.get_data_provider")
    def test_warns_about_concentrated_position(
        self,
        mock_get_provider: Mock,
        mock_load_config: Mock,
        sample_tier_config: dict[str, Any],
    ) -> None:
        """Warns when single position exceeds 25% of portfolio.

        Critical: Concentration risk must be flagged.
        """
        mock_config = Mock()
        mock_config.tiers = {"micro": sample_tier_config}
        mock_load_config.return_value = mock_config
        mock_get_provider.return_value = Mock()

        manager = AdaptivePortfolioManager(prefer_real_data=False)
        manager.tier_manager.detect_tier = Mock(
            return_value=(PortfolioTier.MICRO, sample_tier_config)
        )
        manager.tier_manager.should_transition = Mock(return_value=(False, None))
        manager.risk_manager.calculate_risk_metrics = Mock(return_value={})
        manager.strategy_selector.generate_signals = Mock(return_value=[])

        # One large position (90% of portfolio)
        positions = [
            PositionInfo("AAPL", 100, 150.0, 180.0, 18000.0, 3000.0, 20.0, 10),  # $18K position
        ]

        result = manager.analyze_portfolio(current_capital=20000.0, positions=positions)

        assert any("largest position" in warn.lower() for warn in result.warnings)

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.get_data_provider")
    def test_warns_about_pdt_risk_for_small_accounts(
        self,
        mock_get_provider: Mock,
        mock_load_config: Mock,
        sample_tier_config: dict[str, Any],
    ) -> None:
        """Warns about PDT violations for accounts under $25K.

        Critical: PDT violations can freeze account for 90 days.
        """
        mock_config = Mock()
        tier_cfg = sample_tier_config
        tier_cfg.trading.pdt_compliant = True
        mock_config.tiers = {"micro": tier_cfg}
        mock_load_config.return_value = mock_config
        mock_get_provider.return_value = Mock()

        manager = AdaptivePortfolioManager(prefer_real_data=False)
        manager.tier_manager.detect_tier = Mock(
            return_value=(PortfolioTier.MICRO, sample_tier_config)
        )
        manager.tier_manager.should_transition = Mock(return_value=(False, None))
        manager.risk_manager.calculate_risk_metrics = Mock(return_value={})
        manager.strategy_selector.generate_signals = Mock(return_value=[])

        result = manager.analyze_portfolio(current_capital=15000.0)  # Under $25K

        assert any("pdt" in warn.lower() or "day trade" in warn.lower() for warn in result.warnings)


class TestConvenienceFunctions:
    """Test convenience functions for external use."""

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.AdaptivePortfolioManager")
    def test_run_adaptive_strategy_convenience_function(self, mock_manager_class: Mock) -> None:
        """run_adaptive_strategy provides simple entry point.

        Convenience function for quick analysis without manager instantiation.
        """
        mock_manager = Mock()
        mock_result = Mock(spec=AdaptiveResult)
        mock_manager.analyze_portfolio.return_value = mock_result
        mock_manager_class.return_value = mock_manager

        result = run_adaptive_strategy(current_capital=5000.0)

        mock_manager_class.assert_called_once()
        mock_manager.analyze_portfolio.assert_called_once()
        assert result is mock_result

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.AdaptivePortfolioManager")
    def test_run_adaptive_backtest_convenience_function(self, mock_manager_class: Mock) -> None:
        """run_adaptive_backtest provides simple backtest entry point.

        Convenience function for quick backtesting.
        """
        mock_manager = Mock()
        mock_metrics = Mock(spec=BacktestMetrics)
        mock_manager.run_adaptive_backtest.return_value = mock_metrics
        mock_manager_class.return_value = mock_manager

        result = run_adaptive_backtest(
            symbols=["AAPL", "MSFT"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            initial_capital=10000.0,
        )

        mock_manager_class.assert_called_once()
        mock_manager.run_adaptive_backtest.assert_called_once()
        assert result is mock_metrics


class TestPortfolioSnapshot:
    """Test portfolio snapshot creation."""

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.get_data_provider")
    def test_calculates_cash_from_positions(
        self, mock_get_provider: Mock, mock_load_config: Mock, sample_positions: list[PositionInfo]
    ) -> None:
        """Calculates cash as total value minus position values.

        Cash = Total Portfolio Value - Sum(Position Values)
        """
        mock_config = Mock()
        mock_config.tiers = {}
        mock_load_config.return_value = mock_config
        mock_get_provider.return_value = Mock()

        manager = AdaptivePortfolioManager(prefer_real_data=False)

        snapshot = manager._create_portfolio_snapshot(
            total_value=5000.0, positions=sample_positions, current_tier=PortfolioTier.MICRO
        )

        # Total value: 5000, Position values: 1550 + 1550 = 3100
        assert snapshot.cash == 1900.0

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.get_data_provider")
    def test_calculates_largest_position_percentage(
        self, mock_get_provider: Mock, mock_load_config: Mock
    ) -> None:
        """Calculates largest position as percentage of total portfolio.

        Used for concentration risk monitoring.
        """
        mock_config = Mock()
        mock_config.tiers = {}
        mock_load_config.return_value = mock_config
        mock_get_provider.return_value = Mock()

        manager = AdaptivePortfolioManager(prefer_real_data=False)

        positions = [
            PositionInfo("AAPL", 10, 150.0, 200.0, 2000.0, 500.0, 33.33, 5),  # 20%
            PositionInfo("MSFT", 5, 300.0, 320.0, 1600.0, 100.0, 6.67, 3),  # 16%
        ]

        snapshot = manager._create_portfolio_snapshot(
            total_value=10000.0, positions=positions, current_tier=PortfolioTier.SMALL
        )

        assert snapshot.largest_position_pct == 20.0


class TestErrorHandling:
    """Test error handling and edge cases."""

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.get_data_provider")
    def test_handles_zero_capital_gracefully(
        self,
        mock_get_provider: Mock,
        mock_load_config: Mock,
        sample_tier_config: dict[str, Any],
    ) -> None:
        """Handles zero capital without division by zero errors.

        Edge case: Empty account should not crash analysis.
        """
        mock_config = Mock()
        mock_config.tiers = {"micro": sample_tier_config}
        mock_load_config.return_value = mock_config
        mock_get_provider.return_value = Mock()

        manager = AdaptivePortfolioManager(prefer_real_data=False)
        manager.tier_manager.detect_tier = Mock(
            return_value=(PortfolioTier.MICRO, sample_tier_config)
        )
        manager.tier_manager.should_transition = Mock(return_value=(False, None))
        manager.risk_manager.calculate_risk_metrics = Mock(return_value={})
        manager.strategy_selector.generate_signals = Mock(return_value=[])

        result = manager.analyze_portfolio(current_capital=0.0)

        assert result.portfolio_snapshot.total_value == 0.0
        assert result.portfolio_snapshot.daily_pnl_pct == 0.0

    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.load_portfolio_config")
    @patch("bot_v2.features.adaptive_portfolio.adaptive_portfolio.get_data_provider")
    def test_handles_empty_positions_list(
        self,
        mock_get_provider: Mock,
        mock_load_config: Mock,
        sample_tier_config: dict[str, Any],
    ) -> None:
        """Handles empty positions list (all cash portfolio).

        Portfolio with no positions should analyze correctly.
        """
        mock_config = Mock()
        mock_config.tiers = {"micro": sample_tier_config}
        mock_load_config.return_value = mock_config
        mock_get_provider.return_value = Mock()

        manager = AdaptivePortfolioManager(prefer_real_data=False)
        manager.tier_manager.detect_tier = Mock(
            return_value=(PortfolioTier.MICRO, sample_tier_config)
        )
        manager.tier_manager.should_transition = Mock(return_value=(False, None))
        manager.risk_manager.calculate_risk_metrics = Mock(return_value={})
        manager.strategy_selector.generate_signals = Mock(return_value=[])

        result = manager.analyze_portfolio(current_capital=5000.0, positions=[])

        assert result.portfolio_snapshot.positions_count == 0
        assert result.portfolio_snapshot.cash == 5000.0
        assert result.portfolio_snapshot.largest_position_pct == 0.0
