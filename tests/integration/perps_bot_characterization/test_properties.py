"""
Characterization Tests for PerpsBot Properties

Tests documenting property descriptor behavior and error conditions.
"""

import pytest
from bot_v2.orchestration.perps_bot import PerpsBot


@pytest.mark.integration
@pytest.mark.characterization
class TestPerpsBotProperties:
    """Characterize property descriptor behavior"""

    def test_broker_property_raises_when_none(self, monkeypatch, tmp_path, minimal_config):
        """Document: broker property must raise RuntimeError if None"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        # Registry is frozen dataclass, must use with_updates
        bot.registry = bot.registry.with_updates(broker=None)

        with pytest.raises(RuntimeError) as exc_info:
            _ = bot.broker

        assert "Broker is not configured" in str(exc_info.value)

    def test_risk_manager_property_raises_when_none(self, monkeypatch, tmp_path, minimal_config):
        """Document: risk_manager property must raise RuntimeError if None"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        # Registry is frozen dataclass, must use with_updates
        bot.registry = bot.registry.with_updates(risk_manager=None)

        with pytest.raises(RuntimeError) as exc_info:
            _ = bot.risk_manager

        assert "Risk manager is not configured" in str(exc_info.value)

    def test_exec_engine_property_raises_when_none(self, monkeypatch, tmp_path, minimal_config):
        """Document: exec_engine property must raise RuntimeError if None"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        bot._exec_engine = None

        with pytest.raises(RuntimeError) as exc_info:
            _ = bot.exec_engine

        assert "Execution engine not initialized" in str(exc_info.value)

    def test_property_setters_update_registry(self, monkeypatch, tmp_path, minimal_config):
        """Document: Property setters must update registry using with_updates()"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        original_registry = bot.registry

        # Test broker setter
        from unittest.mock import Mock

        new_broker = Mock()
        bot.broker = new_broker

        # Verify registry was updated (new instance created)
        assert bot.registry is not original_registry
        assert bot.registry.broker is new_broker

        # Test risk_manager setter
        new_risk = Mock()
        bot.risk_manager = new_risk

        assert bot.registry.risk_manager is new_risk

    def test_properties_after_builder_construction(self, monkeypatch, tmp_path, minimal_config):
        """Document: Properties must be accessible after builder construction"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # Verify all properties are accessible
        assert bot.broker is not None
        assert bot.risk_manager is not None

        # Verify property setters work
        from unittest.mock import Mock

        new_broker = Mock()
        bot.broker = new_broker
        assert bot.broker is new_broker
