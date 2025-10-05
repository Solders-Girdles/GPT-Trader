"""
Characterization Tests for PerpsBot Builder Pattern

Tests documenting builder-centric construction pattern.
"""

import pytest
import warnings

from bot_v2.orchestration.builders import PerpsBotBuilder
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.service_registry import empty_registry


@pytest.mark.integration
@pytest.mark.characterization
class TestPerpsBotBuilderPattern:
    """Characterize builder-centric construction now that it is the default."""

    def test_constructor_uses_builder_pipeline(self, monkeypatch, tmp_path, minimal_config):
        """Direct construction routes through PerpsBotBuilder.build_into."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        build_calls: list[None] = []
        original_build_into = PerpsBotBuilder.build_into

        def tracking_build_into(self, bot):  # type: ignore[override]
            build_calls.append(None)
            return original_build_into(self, bot)

        monkeypatch.setattr(PerpsBotBuilder, "build_into", tracking_build_into)

        bot = PerpsBot(minimal_config)

        assert build_calls, "Expected PerpsBotBuilder.build_into to be invoked"
        assert bot.bot_id == "perps_bot"
        assert bot.config == minimal_config
        assert hasattr(bot, "strategy_orchestrator")
        assert hasattr(bot, "execution_coordinator")

    def test_constructor_ignores_legacy_env_flag(self, monkeypatch, tmp_path, minimal_config):
        """Historical USE_PERPS_BOT_BUILDER flag no longer flips code paths."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("USE_PERPS_BOT_BUILDER", "false")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        build_calls: list[None] = []
        original_build_into = PerpsBotBuilder.build_into

        def tracking_build_into(self, bot):  # type: ignore[override]
            build_calls.append(None)
            return original_build_into(self, bot)

        monkeypatch.setattr(PerpsBotBuilder, "build_into", tracking_build_into)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bot = PerpsBot(minimal_config)

        legacy_warnings = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning) and "legacy" in str(w.message).lower()
        ]

        assert build_calls, "Builder path should run even when flag is false"
        assert not legacy_warnings, "Legacy builder warnings should be gone"
        assert bot.config == minimal_config

    def test_constructor_and_builder_produce_identical_state(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Direct construction and explicit builder flow return equivalent bots."""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        direct_bot = PerpsBot(minimal_config)
        builder_bot = PerpsBot.from_builder(PerpsBotBuilder(minimal_config))

        core_attrs = [
            "bot_id",
            "running",
            "symbols",
            "_derivatives_enabled",
        ]

        for attr in core_attrs:
            assert getattr(direct_bot, attr) == getattr(builder_bot, attr), attr

        service_attrs = [
            "strategy_orchestrator",
            "execution_coordinator",
            "system_monitor",
            "runtime_coordinator",
            "account_manager",
            "account_telemetry",
            "_market_monitor",
            "event_store",
            "orders_store",
            "config_controller",
            "registry",
        ]

        for attr in service_attrs:
            assert getattr(direct_bot, attr) is not None, f"direct missing {attr}"
            assert getattr(builder_bot, attr) is not None, f"builder missing {attr}"

    def test_builder_from_classmethod_works(self, monkeypatch, tmp_path, minimal_config):
        """PerpsBot.from_builder() creates valid instance"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))

        builder = PerpsBotBuilder(minimal_config)
        bot = PerpsBot.from_builder(builder)

        assert isinstance(bot, PerpsBot)
        assert bot.bot_id == "perps_bot"
        assert bot.config == minimal_config
        assert hasattr(bot, "strategy_orchestrator")

    def test_builder_respects_custom_registry(self, monkeypatch, tmp_path, minimal_config):
        """Builder uses provided custom registry"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        custom_registry = empty_registry(minimal_config)
        bot = PerpsBot(minimal_config, registry=custom_registry)

        # Verify registry was used (config should match)
        assert bot.registry.config == minimal_config
