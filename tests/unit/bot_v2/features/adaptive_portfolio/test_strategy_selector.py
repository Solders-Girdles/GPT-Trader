"""StrategySelector integration tests focused on registry coordination."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from bot_v2.features.adaptive_portfolio.position_size_calculator import PositionSizeCalculator
from bot_v2.features.adaptive_portfolio.signal_filter import SignalFilter
from bot_v2.features.adaptive_portfolio.strategy_selector import StrategySelector
from bot_v2.features.adaptive_portfolio.symbol_universe_builder import SymbolUniverseBuilder
from bot_v2.features.adaptive_portfolio.types import (
    PortfolioConfig,
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
    """Return a simple data provider mock."""

    provider = Mock()
    provider.get_historical_data = Mock()
    return provider


@pytest.fixture
def portfolio_config() -> PortfolioConfig:
    """Construct a minimal portfolio configuration for tests."""

    tier_config = TierConfig(
        name="micro",
        range=(0, 5000),
        positions=PositionConstraints(1, 3, 2, 100.0),
        min_position_size=100.0,
        strategies=["momentum"],
        risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
        trading=TradingRules(5, "cash", 2, True),
    )

    market_constraints = Mock()
    market_constraints.excluded_symbols = []

    return PortfolioConfig(
        version="1.0",
        last_updated="2024-01-01",
        description="Test config",
        tiers={"micro": tier_config},
        costs=Mock(),
        market_constraints=market_constraints,
        validation={},
        rebalancing={},
    )


@pytest.fixture
def portfolio_snapshot() -> PortfolioSnapshot:
    """Build a representative snapshot for micro-tier accounts."""

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
def mock_universe_builder() -> Mock:
    """Provide a universe builder returning a predictable symbol set."""

    builder = Mock(spec=SymbolUniverseBuilder)
    builder.build_universe.return_value = ["AAPL", "ETH", "TSLA"]
    return builder


@pytest.fixture
def mock_signal_filter() -> Mock:
    """Return a signal filter mock that passes signals through by default."""

    signal_filter = Mock(spec=SignalFilter)
    signal_filter.filter_signals.side_effect = lambda signals, *_, **__: signals
    return signal_filter


@pytest.fixture
def strategy_registry() -> dict[str, Mock]:
    """Build a registry mapping strategy names to handler mocks."""

    momentum = Mock(spec=["generate_signals"])
    momentum.generate_signals.return_value = []

    mean_reversion = Mock(spec=["generate_signals"])
    mean_reversion.generate_signals.return_value = []

    return {"momentum": momentum, "mean_reversion": mean_reversion}


@pytest.fixture
def strategy_selector(
    portfolio_config: PortfolioConfig,
    mock_data_provider: Mock,
    mock_universe_builder: Mock,
    mock_signal_filter: Mock,
    strategy_registry: dict[str, Mock],
) -> StrategySelector:
    """Instantiate StrategySelector with injected collaborators."""

    return StrategySelector(
        portfolio_config,
        mock_data_provider,
        universe_builder=mock_universe_builder,
        signal_filter=mock_signal_filter,
        strategy_registry=strategy_registry,
    )


class TestStrategySelectorInitialization:
    """Confirm StrategySelector wiring during initialization."""

    def test_initializes_with_config_and_provider(
        self,
        portfolio_config: PortfolioConfig,
        mock_data_provider: Mock,
        mock_universe_builder: Mock,
        mock_signal_filter: Mock,
        strategy_registry: dict[str, Mock],
    ) -> None:
        selector = StrategySelector(
            portfolio_config,
            mock_data_provider,
            universe_builder=mock_universe_builder,
            signal_filter=mock_signal_filter,
            strategy_registry=strategy_registry,
        )

        assert selector.config is portfolio_config
        assert selector.data_provider is mock_data_provider
        assert selector._explicit_registry == strategy_registry


class TestSignalGeneration:
    """Validate signal generation orchestrates registry handlers."""

    def test_delegates_to_all_registered_handlers(
        self,
        strategy_selector: StrategySelector,
        strategy_registry: dict[str, Mock],
        mock_universe_builder: Mock,
        mock_signal_filter: Mock,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        tier_config = TierConfig(
            name="micro",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum", "mean_reversion"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        momentum_signal = TradingSignal(
            symbol="AAPL",
            action="BUY",
            confidence=0.6,
            target_position_size=150.0,
            stop_loss_pct=5.0,
            strategy_source="momentum",
            reasoning="trend up",
        )
        mean_reversion_signal = TradingSignal(
            symbol="ETH",
            action="BUY",
            confidence=0.9,
            target_position_size=200.0,
            stop_loss_pct=5.0,
            strategy_source="mean_reversion",
            reasoning="oversold",
        )

        strategy_registry["momentum"].generate_signals.return_value = [momentum_signal]
        strategy_registry["mean_reversion"].generate_signals.return_value = [mean_reversion_signal]

        signals = strategy_selector.generate_signals(tier_config, portfolio_snapshot)

        mock_universe_builder.build_universe.assert_called_once_with(
            tier_config, portfolio_snapshot
        )
        strategy_registry["momentum"].generate_signals.assert_called_once_with(
            mock_universe_builder.build_universe.return_value,
            tier_config,
            portfolio_snapshot,
        )
        strategy_registry["mean_reversion"].generate_signals.assert_called_once_with(
            mock_universe_builder.build_universe.return_value,
            tier_config,
            portfolio_snapshot,
        )
        mock_signal_filter.filter_signals.assert_called_once()
        assert signals == [mean_reversion_signal]

    def test_unknown_strategy_is_skipped(
        self,
        strategy_selector: StrategySelector,
        portfolio_snapshot: PortfolioSnapshot,
        caplog: Any,
    ) -> None:
        tier_config = TierConfig(
            name="micro",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["unknown"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        with caplog.at_level("WARNING"):
            signals = strategy_selector.generate_signals(tier_config, portfolio_snapshot)

        assert signals == []
        assert "Unknown strategy" in caplog.text

    def test_applies_filter_and_ranking(
        self,
        strategy_selector: StrategySelector,
        strategy_registry: dict[str, Mock],
        mock_signal_filter: Mock,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        tier_config = TierConfig(
            name="balanced",
            range=(10000, 50000),
            positions=PositionConstraints(2, 5, 4, 100.0),
            min_position_size=200.0,
            strategies=["momentum"],
            risk=RiskProfile(1.5, 8.0, 4.0, 35.0),
            trading=TradingRules(10, "margin", 1, True),
        )

        strategy_registry["momentum"].generate_signals.return_value = [
            TradingSignal("AAA", "BUY", 0.2, 150.0, 5.0, "momentum", "low"),
            TradingSignal("BBB", "BUY", 0.8, 200.0, 5.0, "momentum", "high"),
            TradingSignal("CCC", "BUY", 0.5, 180.0, 5.0, "momentum", "mid"),
        ]

        def filter_override(
            signals: list[TradingSignal], *_: Any, **__: Any
        ) -> list[TradingSignal]:
            return [signals[0], signals[2], signals[1]]  # Intentional reordering

        mock_signal_filter.filter_signals.side_effect = filter_override

        signals = strategy_selector.generate_signals(tier_config, portfolio_snapshot)

        confidences = [signal.confidence for signal in signals]
        assert confidences == sorted(confidences, reverse=True)


class TestSymbolUniverse:
    """Ensure symbol universe builder behaves for different tiers."""

    def test_micro_tier_gets_limited_universe(self, portfolio_snapshot: PortfolioSnapshot) -> None:
        tier_config = TierConfig(
            name="Micro Portfolio",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        universe = SymbolUniverseBuilder().build_universe(tier_config, portfolio_snapshot)

        assert len(universe) <= 8

    def test_large_tier_gets_full_universe(self, portfolio_snapshot: PortfolioSnapshot) -> None:
        tier_config = TierConfig(
            name="Large Portfolio",
            range=(100000, float("inf")),
            positions=PositionConstraints(10, 30, 20, 1000.0),
            min_position_size=1000.0,
            strategies=["momentum", "mean_reversion", "trend_following", "ml_enhanced"],
            risk=RiskProfile(1.0, 5.0, 3.0, 25.0),
            trading=TradingRules(20, "margin", 0, False),
        )

        universe = SymbolUniverseBuilder().build_universe(tier_config, portfolio_snapshot)

        assert len(universe) > 15


class TestPositionSizing:
    """Validate position sizing helper remains monotonic and bounded."""

    def test_calculates_position_size_based_on_confidence(
        self,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        tier_config = TierConfig(
            name="micro",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        calculator = PositionSizeCalculator()

        size_low = calculator.calculate(0.3, tier_config, portfolio_snapshot)
        size_high = calculator.calculate(0.9, tier_config, portfolio_snapshot)

        assert size_high > size_low

    def test_respects_minimum_position_size(
        self,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        tier_config = TierConfig(
            name="micro",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        calculator = PositionSizeCalculator()

        size = calculator.calculate(0.1, tier_config, portfolio_snapshot)

        assert size >= tier_config.min_position_size


class TestSignalFiltering:
    """Cover internal helpers that cap signal counts."""

    def test_limits_signals_to_tier_capacity(
        self,
        strategy_selector: StrategySelector,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        tier_config = TierConfig(
            name="micro",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        max_signals = strategy_selector._calculate_max_signals(tier_config, portfolio_snapshot)

        assert max_signals <= tier_config.positions.max_positions


class TestConfigDrivenRegistry:
    """Test config-driven strategy registry building."""

    def test_builds_registry_from_tier_config_when_not_injected(
        self,
        portfolio_config: PortfolioConfig,
        mock_data_provider: Mock,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Builds strategy registry from tier config when no explicit registry provided."""
        # Create selector without explicit registry
        selector = StrategySelector(portfolio_config, mock_data_provider)

        tier_config = TierConfig(
            name="test_tier",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum", "mean_reversion"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        # Should build registry on-demand
        # Note: generate_signals will fail without real data, but registry should be built
        try:
            selector.generate_signals(tier_config, portfolio_snapshot)
        except Exception:
            pass  # Expected to fail due to mock data provider

        # Verify registry was cached
        assert "test_tier" in selector._tier_registries
        registry = selector._tier_registries["test_tier"]
        assert "momentum" in registry
        assert "mean_reversion" in registry
        assert len(registry) == 2

    def test_caches_registry_per_tier(
        self,
        portfolio_config: PortfolioConfig,
        mock_data_provider: Mock,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Caches built registries per tier to avoid rebuilding."""
        selector = StrategySelector(portfolio_config, mock_data_provider)

        tier_config_1 = TierConfig(
            name="tier_1",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        tier_config_2 = TierConfig(
            name="tier_2",
            range=(5000, 50000),
            positions=PositionConstraints(2, 8, 5, 250.0),
            min_position_size=250.0,
            strategies=["momentum", "mean_reversion", "trend_following"],
            risk=RiskProfile(1.5, 8.0, 4.0, 35.0),
            trading=TradingRules(10, "margin", 1, True),
        )

        # Build registries
        registry_1 = selector._get_strategy_registry(tier_config_1)
        registry_2 = selector._get_strategy_registry(tier_config_2)

        # Should be cached
        assert selector._get_strategy_registry(tier_config_1) is registry_1
        assert selector._get_strategy_registry(tier_config_2) is registry_2

        # Should be different registries
        assert len(registry_1) == 1
        assert len(registry_2) == 3

    def test_uses_explicit_registry_when_provided(
        self,
        portfolio_config: PortfolioConfig,
        mock_data_provider: Mock,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Uses explicit registry when provided (test injection)."""
        mock_handler = Mock()
        mock_handler.generate_signals.return_value = []
        explicit_registry = {"momentum": mock_handler}

        selector = StrategySelector(
            portfolio_config, mock_data_provider, strategy_registry=explicit_registry
        )

        tier_config = TierConfig(
            name="test",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        # Should use explicit registry, not build from config
        registry = selector._get_strategy_registry(tier_config)
        assert registry is explicit_registry
        assert "test" not in selector._tier_registries  # Should not cache
