from __future__ import annotations

import threading

import bot_v2.orchestration.runtime_settings as runtime_settings_module
from bot_v2.security.secrets_manager import SecretsManager


def test_file_backed_secrets_roundtrip(temp_home, runtime_settings_factory) -> None:
    settings = runtime_settings_factory()

    manager = SecretsManager(vault_enabled=False, settings=settings)
    secret_path = "brokers/coinbase"
    payload = {"key": "value", "_key": "old"}

    assert manager.store_secret(secret_path, payload)
    manager._lock = threading.RLock()
    cached = manager.get_secret(secret_path)
    assert cached == payload

    manager.clear_cache()
    loaded = manager.get_secret(secret_path)
    assert loaded == payload

    assert manager.rotate_key(secret_path)
    rotated = manager.get_secret(secret_path)
    assert rotated is not None and rotated["_key"] != "old"

    listed = manager.list_secrets()
    assert secret_path in listed

    assert manager.delete_secret(secret_path)
    assert manager.get_secret(secret_path) is None


def test_vault_backed_secrets_use_stub(
    hvac_stub,
    runtime_settings_factory,
    monkeypatch,
) -> None:
    settings = runtime_settings_factory(
        env_overrides={
            "VAULT_TOKEN": "token",
            "VAULT_ADDR": "http://vault.local",
        }
    )

    real_loader = runtime_settings_module.load_runtime_settings
    monkeypatch.setattr(
        runtime_settings_module,
        "load_runtime_settings",
        lambda env=None: settings if env is None else real_loader(env),
    )
    monkeypatch.setattr(
        "bot_v2.security.secrets_manager.load_runtime_settings",
        lambda env=None: settings if env is None else real_loader(env),
    )

    manager = SecretsManager(vault_enabled=True)
    secret_path = "brokers/binance"
    payload = {"key": "secret"}

    assert manager.store_secret(secret_path, payload)
    assert secret_path in hvac_stub._storage

    manager.clear_cache()
    retrieved = manager.get_secret(secret_path)
    assert retrieved == payload

    listed = manager.list_secrets()
    assert secret_path in listed

    assert manager.delete_secret(secret_path)
    assert secret_path not in hvac_stub._storage
