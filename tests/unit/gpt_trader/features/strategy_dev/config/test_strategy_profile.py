"""Tests for strategy profile module."""

from typing import Any

from gpt_trader.features.strategy_dev.config.diff import compute_profile_diff
from gpt_trader.features.strategy_dev.config.strategy_profile import (
    ExecutionConfig,
    RegimeConfig,
    RiskConfig,
    SignalConfig,
    StrategyProfile,
)


class TestSignalConfig:
    """Tests for SignalConfig."""

    def test_create_signal(self):
        """Test creating a signal config."""
        signal = SignalConfig(
            name="momentum",
            weight=0.5,
            parameters={"period": 20},
        )

        assert signal.name == "momentum"
        assert signal.weight == 0.5
        assert signal.parameters["period"] == 20

    def test_to_from_dict(self):
        """Test serialization."""
        signal = SignalConfig(
            name="rsi",
            weight=0.3,
            parameters={"period": 14, "oversold": 30},
        )

        data = signal.to_dict()
        restored = SignalConfig.from_dict(data)

        assert restored.name == signal.name
        assert restored.weight == signal.weight
        assert restored.parameters == signal.parameters


class TestRiskConfig:
    """Tests for RiskConfig."""

    def test_defaults(self):
        """Test default values."""
        config = RiskConfig()

        assert config.max_position_size == 0.10
        assert config.max_portfolio_heat == 0.06
        assert config.stop_loss_percent == 0.02

    def test_custom_values(self):
        """Test custom values."""
        config = RiskConfig(
            max_position_size=0.05,
            stop_loss_percent=0.03,
            take_profit_percent=0.06,
        )

        assert config.max_position_size == 0.05
        assert config.take_profit_percent == 0.06


class TestRegimeConfig:
    """Tests for RegimeConfig."""

    def test_defaults(self):
        """Test default scale factors."""
        config = RegimeConfig()

        assert config.scale_factors["CRISIS"] == 0.2
        assert config.scale_factors["BULL_QUIET"] == 1.2
        assert config.pause_in_crisis is True

    def test_custom_factors(self):
        """Test custom scale factors."""
        config = RegimeConfig(
            scale_factors={"BULL_QUIET": 1.5, "CRISIS": 0.0},
        )

        assert config.scale_factors["BULL_QUIET"] == 1.5


class TestStrategyProfile:
    """Tests for StrategyProfile."""

    def test_create_profile(self):
        """Test creating a profile."""
        profile = StrategyProfile(
            name="test_strategy",
            description="A test strategy",
            signals=[
                SignalConfig(name="momentum", weight=0.5),
                SignalConfig(name="trend", weight=0.5),
            ],
        )

        assert profile.name == "test_strategy"
        assert len(profile.signals) == 2

    def test_config_hash(self):
        """Test configuration hash."""
        profile1 = StrategyProfile(
            name="strategy1",
            signals=[SignalConfig(name="a", weight=0.5)],
        )

        StrategyProfile(
            name="strategy2",  # Different name
            signals=[SignalConfig(name="a", weight=0.5)],  # Same signals
        )

        profile3 = StrategyProfile(
            name="strategy1",
            signals=[SignalConfig(name="b", weight=0.5)],  # Different signal
        )

        # Different names but same config should have different hash
        assert profile1.config_hash != profile3.config_hash

    def test_get_signal_weights(self):
        """Test getting signal weights."""
        profile = StrategyProfile(
            name="test",
            signals=[
                SignalConfig(name="a", weight=0.3, enabled=True),
                SignalConfig(name="b", weight=0.5, enabled=True),
                SignalConfig(name="c", weight=0.2, enabled=False),
            ],
        )

        weights = profile.get_signal_weights()

        assert weights == {"a": 0.3, "b": 0.5}
        assert "c" not in weights

    def test_get_regime_scale(self):
        """Test getting regime scale factors."""
        profile = StrategyProfile(
            name="test",
            regime=RegimeConfig(
                scale_factors={"BULL_QUIET": 1.2, "CRISIS": 0.2},
            ),
        )

        assert profile.get_regime_scale("BULL_QUIET") == 1.2
        assert profile.get_regime_scale("CRISIS") == 0.2
        assert profile.get_regime_scale("UNKNOWN") == 0.5  # Default

    def test_should_trade_in_regime(self):
        """Test trading permission by regime."""
        profile = StrategyProfile(
            name="test",
            regime=RegimeConfig(
                enabled=True,
                pause_in_crisis=True,
                min_confidence=0.6,
            ),
        )

        # Should trade in normal regime with confidence
        assert profile.should_trade_in_regime("BULL_QUIET", 0.8) is True

        # Should not trade in crisis
        assert profile.should_trade_in_regime("CRISIS", 0.9) is False

        # Should not trade with low confidence
        assert profile.should_trade_in_regime("BULL_QUIET", 0.5) is False

    def test_clone(self):
        """Test cloning a profile."""
        profile = StrategyProfile(
            name="original",
            signals=[SignalConfig(name="a", weight=0.5)],
            tags=["test"],
        )

        cloned = profile.clone("cloned")

        assert cloned.name == "cloned"
        assert len(cloned.signals) == len(profile.signals)
        assert cloned.tags == profile.tags

    def test_validate(self):
        """Test profile validation."""
        # Valid profile
        valid = StrategyProfile(
            name="valid",
            signals=[SignalConfig(name="a", weight=1.0)],
        )
        assert valid.validate() == []

        # Invalid - no name
        invalid1 = StrategyProfile(name="", signals=[])
        errors1 = invalid1.validate()
        assert any("name" in e.lower() for e in errors1)

        # Invalid - bad risk config
        invalid2 = StrategyProfile(
            name="test",
            risk=RiskConfig(max_position_size=2.0),
        )
        errors2 = invalid2.validate()
        assert any("max_position_size" in e for e in errors2)

    def test_to_from_dict(self):
        """Test serialization round-trip."""
        profile = StrategyProfile(
            name="test_profile",
            description="Test description",
            version="2.0.0",
            symbols=["BTC-USD", "ETH-USD"],
            signals=[
                SignalConfig(name="momentum", weight=0.4),
                SignalConfig(name="trend", weight=0.6),
            ],
            risk=RiskConfig(max_position_size=0.05),
            regime=RegimeConfig(pause_in_crisis=False),
            execution=ExecutionConfig(order_type="market"),
            tags=["test", "backtest"],
        )

        data = profile.to_dict()
        restored = StrategyProfile.from_dict(data)

        assert restored.name == profile.name
        assert restored.version == profile.version
        assert len(restored.signals) == 2
        assert restored.risk.max_position_size == 0.05
        assert restored.tags == ["test", "backtest"]


class TestStrategyProfileDiff:
    """Tests for profile diff helpers."""

    def _index_by_path(self, entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {entry["path"]: entry for entry in entries}

    def test_diff_detects_changes_and_missing(self) -> None:
        baseline = {
            "name": "strategy",
            "risk": {"max_position_size": 0.1},
            "signals": ["a"],
        }
        runtime = {
            "name": "strategy",
            "risk": {"max_position_size": 0.2},
            "signals": ["b"],
            "extra": True,
        }

        entries = compute_profile_diff(baseline, runtime)
        indexed = self._index_by_path(entries)

        assert indexed["name"]["status"] == "unchanged"
        assert indexed["risk.max_position_size"]["status"] == "changed"
        assert indexed["signals"]["status"] == "changed"
        assert indexed["extra"]["status"] == "changed"
        assert indexed["extra"]["baseline_value"] is None

    def test_diff_handles_missing_runtime_key(self) -> None:
        baseline = {"name": "strategy", "version": "1.0"}
        runtime = {"name": "strategy"}

        entries = compute_profile_diff(baseline, runtime)
        indexed = self._index_by_path(entries)

        assert indexed["version"]["status"] == "missing"
        assert indexed["version"]["runtime_value"] is None

    def test_diff_includes_nested_entries(self) -> None:
        baseline = {"risk": {"max": 1, "min": 0}}
        runtime = {"risk": {"max": 1, "min": 0}}

        entries = compute_profile_diff(baseline, runtime)
        indexed = self._index_by_path(entries)

        assert indexed["risk.max"]["status"] == "unchanged"
        assert indexed["risk.min"]["status"] == "unchanged"

    def test_diff_skips_ignored_fields(self) -> None:
        baseline = {"created_at": "2025-02-01T00:00:00Z", "name": "strategy"}
        runtime = {"created_at": "2025-02-02T00:00:00Z", "name": "strategy"}

        entries = compute_profile_diff(baseline, runtime)
        indexed = self._index_by_path(entries)

        assert "created_at" not in indexed
        assert indexed["name"]["status"] == "unchanged"
