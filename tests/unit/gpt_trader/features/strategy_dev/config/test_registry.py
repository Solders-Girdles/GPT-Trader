"""Tests for strategy registry module."""

import pytest

from gpt_trader.features.strategy_dev.config.registry import StrategyRegistry
from gpt_trader.features.strategy_dev.config.strategy_profile import (
    SignalConfig,
    StrategyProfile,
)


class TestStrategyRegistry:
    """Tests for StrategyRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a registry without persistence."""
        return StrategyRegistry()

    @pytest.fixture
    def sample_profile(self):
        """Create a sample profile."""
        return StrategyProfile(
            name="test_strategy",
            description="Test strategy",
            signals=[SignalConfig(name="momentum", weight=1.0)],
            tags=["test"],
        )

    def test_register(self, registry, sample_profile):
        """Test registering a strategy."""
        entry = registry.register(sample_profile)

        assert entry.profile.name == "test_strategy"
        assert "test_strategy" in registry

    def test_get(self, registry, sample_profile):
        """Test getting a strategy."""
        registry.register(sample_profile)

        profile = registry.get("test_strategy")

        assert profile is not None
        assert profile.name == "test_strategy"

    def test_get_nonexistent(self, registry):
        """Test getting a nonexistent strategy."""
        profile = registry.get("nonexistent")
        assert profile is None

    def test_list_strategies(self, registry):
        """Test listing strategies."""
        profiles = [
            StrategyProfile(name="strategy1", tags=["production"]),
            StrategyProfile(name="strategy2", tags=["test"]),
            StrategyProfile(name="strategy3", tags=["production"]),
        ]

        for p in profiles:
            registry.register(p)

        # All strategies
        all_strategies = registry.list_strategies()
        assert len(all_strategies) == 3

        # Filter by tags
        prod = registry.list_strategies(tags=["production"])
        assert len(prod) == 2

    def test_activate(self, registry, sample_profile):
        """Test activating a strategy."""
        registry.register(sample_profile)

        active = registry.activate("test_strategy")

        assert active.name == "test_strategy"
        assert registry.get_active() is not None
        assert registry.get_active().name == "test_strategy"

    def test_activate_replaces_previous(self, registry):
        """Test activating replaces previous active."""
        p1 = StrategyProfile(name="strategy1")
        p2 = StrategyProfile(name="strategy2")

        registry.register(p1)
        registry.register(p2)

        registry.activate("strategy1")
        registry.activate("strategy2")

        assert registry.get_active().name == "strategy2"

        # Check entry is_active flags
        assert registry.get_entry("strategy1").is_active is False
        assert registry.get_entry("strategy2").is_active is True

    def test_remove(self, registry, sample_profile):
        """Test removing a strategy."""
        registry.register(sample_profile)

        removed = registry.remove("test_strategy")

        assert removed is True
        assert "test_strategy" not in registry

    def test_remove_active(self, registry, sample_profile):
        """Test removing active strategy."""
        registry.register(sample_profile)
        registry.activate("test_strategy")

        registry.remove("test_strategy")

        assert registry.get_active() is None

    def test_compare(self, registry):
        """Test comparing strategies."""
        p1 = StrategyProfile(
            name="strategy1",
            signals=[SignalConfig(name="a", weight=0.5)],
        )
        p2 = StrategyProfile(
            name="strategy2",
            signals=[SignalConfig(name="b", weight=0.6)],
        )

        registry.register(p1)
        registry.register(p2)

        comparison = registry.compare("strategy1", "strategy2")

        assert comparison["strategies"] == ["strategy1", "strategy2"]
        assert len(comparison["differences"]) > 0

    def test_search(self, registry):
        """Test searching strategies."""
        profiles = [
            StrategyProfile(name="momentum_fast", description="Fast momentum"),
            StrategyProfile(name="momentum_slow", description="Slow momentum"),
            StrategyProfile(name="trend_following", description="Trend strategy"),
        ]

        for p in profiles:
            registry.register(p)

        # Search by name
        results = registry.search("momentum")
        assert len(results) == 2

        # Search by description
        results = registry.search("trend")
        assert len(results) == 1

    def test_summary(self, registry):
        """Test registry summary."""
        profiles = [
            StrategyProfile(name="s1", environment="paper", tags=["test"]),
            StrategyProfile(name="s2", environment="paper", tags=["test", "dev"]),
            StrategyProfile(name="s3", environment="live", tags=["production"]),
        ]

        for p in profiles:
            registry.register(p)

        registry.activate("s1")

        summary = registry.summary()

        assert summary["total_strategies"] == 3
        assert summary["active_strategy"] == "s1"
        assert summary["by_environment"]["paper"] == 2
        assert summary["by_environment"]["live"] == 1

    def test_len(self, registry, sample_profile):
        """Test len() on registry."""
        assert len(registry) == 0

        registry.register(sample_profile)
        assert len(registry) == 1

    def test_contains(self, registry, sample_profile):
        """Test 'in' operator."""
        assert "test_strategy" not in registry

        registry.register(sample_profile)
        assert "test_strategy" in registry

    def test_iter(self, registry):
        """Test iterating over registry."""
        profiles = [
            StrategyProfile(name="s1"),
            StrategyProfile(name="s2"),
            StrategyProfile(name="s3"),
        ]

        for p in profiles:
            registry.register(p)

        names = [p.name for p in registry]
        assert set(names) == {"s1", "s2", "s3"}

    def test_callback(self, registry, sample_profile):
        """Test registry callbacks."""
        events = []

        def callback(entry):
            events.append(entry)

        registry.on("registered", callback)
        registry.register(sample_profile)

        assert len(events) == 1
        assert events[0].profile.name == "test_strategy"

    def test_validation_on_register(self, registry):
        """Test validation on register."""
        # Profile with validation error
        invalid = StrategyProfile(
            name="",  # Empty name
        )

        with pytest.raises(ValueError):
            registry.register(invalid)
