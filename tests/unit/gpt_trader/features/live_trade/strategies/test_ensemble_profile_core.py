"""Tests for EnsembleProfile objects and helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from gpt_trader.features.live_trade.strategies.ensemble_profile import (
    EnsembleProfile,
    SignalProfileConfig,
    get_aggressive_profile,
    get_conservative_profile,
    get_default_profile,
    get_microstructure_profile,
)


class TestEnsembleProfile:
    """Tests for EnsembleProfile."""

    def test_default_profile_is_valid(self):
        """Default profile should pass validation."""
        profile = get_default_profile()
        errors = profile.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_microstructure_profile_is_valid(self):
        """Microstructure profile should pass validation."""
        profile = get_microstructure_profile()
        errors = profile.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_conservative_profile_is_valid(self):
        """Conservative profile should pass validation."""
        profile = get_conservative_profile()
        errors = profile.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_aggressive_profile_is_valid(self):
        """Aggressive profile should pass validation."""
        profile = get_aggressive_profile()
        errors = profile.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_build_signals_creates_instances(self):
        """build_signals should create SignalGenerator instances."""
        profile = get_default_profile()
        signals = profile.build_signals()
        assert len(signals) > 0
        # Check they're valid signal generators
        for signal in signals:
            assert hasattr(signal, "generate")

    def test_build_signals_respects_enabled_flag(self):
        """Disabled signals should not be built."""
        profile = EnsembleProfile(
            name="test",
            signals=[
                SignalProfileConfig(name="trend", enabled=True),
                SignalProfileConfig(name="momentum", enabled=False),
            ],
        )
        signals = profile.build_signals()
        assert len(signals) == 1

    def test_build_combiner(self):
        """build_combiner should create a RegimeAwareCombiner."""
        profile = get_default_profile()
        combiner = profile.build_combiner()
        assert combiner is not None
        assert hasattr(combiner, "combine")

    def test_yaml_roundtrip(self):
        """Profile should survive YAML serialization."""
        profile = get_microstructure_profile()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = Path(f.name)

        try:
            profile.to_yaml(path)
            restored = EnsembleProfile.from_yaml(path)

            assert restored.name == profile.name
            assert restored.version == profile.version
            assert len(restored.signals) == len(profile.signals)
            assert restored.decision.buy_threshold == profile.decision.buy_threshold
        finally:
            path.unlink()

    def test_validation_requires_name(self):
        """Profile without name should fail validation."""
        profile = EnsembleProfile(name="", signals=[SignalProfileConfig(name="trend")])
        errors = profile.validate()
        assert any("name" in e.lower() for e in errors)

    def test_validation_requires_signals(self):
        """Profile without signals should fail validation."""
        profile = EnsembleProfile(name="test", signals=[])
        errors = profile.validate()
        assert any("signal" in e.lower() for e in errors)

    def test_validation_requires_enabled_signals(self):
        """Profile with all disabled signals should fail validation."""
        profile = EnsembleProfile(
            name="test",
            signals=[SignalProfileConfig(name="trend", enabled=False)],
        )
        errors = profile.validate()
        assert any("enabled" in e.lower() for e in errors)
