import pytest

from bot_v2.orchestration.perps_bot import (
    BotConfig,
    ConfigManager,
    ConfigValidationError,
    Profile,
)


def test_config_manager_build_and_snapshot(monkeypatch):
    monkeypatch.delenv('ORDER_PREVIEW_ENABLED', raising=False)
    manager = ConfigManager(profile=Profile.DEV, auto_build=False)
    config = manager.build()

    assert isinstance(config, BotConfig)
    assert config.metadata['profile'] == Profile.DEV.value
    assert 'config_snapshot' in config.metadata
    assert not manager.has_changes()

    monkeypatch.setenv('ORDER_PREVIEW_ENABLED', '1')
    assert manager.has_changes()

    updated = manager.refresh_if_changed()
    assert updated is not None
    assert updated.enable_order_preview is True
    assert not manager.has_changes()


def test_config_manager_from_config_reuses_snapshot(monkeypatch):
    base_manager = ConfigManager(profile=Profile.DEV)
    config = base_manager.get_config()
    assert config is not None

    followup = ConfigManager.from_config(config)
    assert not followup.has_changes()

    monkeypatch.setenv('ORDER_PREVIEW_ENABLED', 'true')
    assert followup.has_changes()


def test_config_manager_validation_failure():
    with pytest.raises(ConfigValidationError):
        ConfigManager(profile=Profile.DEV, overrides={'update_interval': 0})


def test_canary_configuration_guards(monkeypatch):
    monkeypatch.delenv('ORDER_PREVIEW_ENABLED', raising=False)
    manager = ConfigManager(
        profile=Profile.CANARY,
        overrides={'max_leverage': 5, 'reduce_only_mode': False},
        auto_build=False,
    )
    config = manager.build()

    assert config.max_leverage == 1
    assert config.reduce_only_mode is True
    assert config.time_in_force == 'IOC'
    assert config.symbols
