from __future__ import annotations

import json
from pathlib import Path

from gpt_trader.features.brokerages.coinbase.credentials import resolve_coinbase_credentials


def _clear_coinbase_credential_env(monkeypatch) -> None:
    for var in (
        "COINBASE_CREDENTIALS_FILE",
        "COINBASE_PROD_CDP_API_KEY",
        "COINBASE_PROD_CDP_PRIVATE_KEY",
        "COINBASE_CDP_API_KEY",
        "COINBASE_CDP_PRIVATE_KEY",
        "COINBASE_API_KEY_NAME",
        "COINBASE_PRIVATE_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_resolve_prefers_prod_cdp_env_vars(monkeypatch) -> None:
    _clear_coinbase_credential_env(monkeypatch)
    monkeypatch.setenv("COINBASE_PROD_CDP_API_KEY", "organizations/prod/apiKeys/abc")
    monkeypatch.setenv("COINBASE_PROD_CDP_PRIVATE_KEY", "-----BEGIN EC PRIVATE KEY-----\nprod\n")

    creds = resolve_coinbase_credentials()

    assert creds is not None
    assert creds.key_name == "organizations/prod/apiKeys/abc"
    assert "COINBASE_PROD_CDP_API_KEY" in creds.source


def test_resolve_falls_back_to_cdp_env_vars(monkeypatch) -> None:
    _clear_coinbase_credential_env(monkeypatch)
    monkeypatch.setenv("COINBASE_CDP_API_KEY", "organizations/dev/apiKeys/def")
    monkeypatch.setenv("COINBASE_CDP_PRIVATE_KEY", "-----BEGIN EC PRIVATE KEY-----\ndev\n")

    creds = resolve_coinbase_credentials()

    assert creds is not None
    assert creds.key_name == "organizations/dev/apiKeys/def"
    assert "COINBASE_CDP_API_KEY" in creds.source


def test_resolve_supports_legacy_env_vars(monkeypatch) -> None:
    _clear_coinbase_credential_env(monkeypatch)
    monkeypatch.setenv("COINBASE_API_KEY_NAME", "organizations/legacy/apiKeys/xyz")
    monkeypatch.setenv("COINBASE_PRIVATE_KEY", "-----BEGIN EC PRIVATE KEY-----\nlegacy\n")

    creds = resolve_coinbase_credentials()

    assert creds is not None
    assert creds.key_name == "organizations/legacy/apiKeys/xyz"
    assert creds.source.endswith("COINBASE_API_KEY_NAME+COINBASE_PRIVATE_KEY")


def test_resolve_warns_on_mixed_env_vars(monkeypatch) -> None:
    _clear_coinbase_credential_env(monkeypatch)
    monkeypatch.setenv("COINBASE_API_KEY_NAME", "organizations/old/apiKeys/old")
    monkeypatch.setenv("COINBASE_PRIVATE_KEY", "-----BEGIN EC PRIVATE KEY-----\nold\n")
    monkeypatch.setenv("COINBASE_CDP_API_KEY", "organizations/new/apiKeys/new")
    monkeypatch.setenv("COINBASE_CDP_PRIVATE_KEY", "-----BEGIN EC PRIVATE KEY-----\nnew\n")

    creds = resolve_coinbase_credentials()

    assert creds is not None
    assert creds.key_name == "organizations/new/apiKeys/new"
    assert any("COINBASE_API_KEY_NAME" in w for w in creds.warnings)
    assert any("COINBASE_PRIVATE_KEY" in w for w in creds.warnings)


def test_resolve_uses_credentials_file(tmp_path, monkeypatch) -> None:
    _clear_coinbase_credential_env(monkeypatch)
    key_path = Path(tmp_path) / "coinbase.json"
    key_path.write_text(
        json.dumps(
            {
                "name": "organizations/file/apiKeys/file1",
                "privateKey": "-----BEGIN EC PRIVATE KEY-----\nfile\n",
            }
        )
    )
    monkeypatch.setenv("COINBASE_CREDENTIALS_FILE", str(key_path))

    # Also set env vars to ensure file takes precedence.
    monkeypatch.setenv("COINBASE_CDP_API_KEY", "organizations/env/apiKeys/env1")
    monkeypatch.setenv("COINBASE_CDP_PRIVATE_KEY", "-----BEGIN EC PRIVATE KEY-----\nenv\n")

    creds = resolve_coinbase_credentials()

    assert creds is not None
    assert creds.key_name == "organizations/file/apiKeys/file1"
    assert creds.source.startswith("file:")


def test_resolve_invalid_credentials_file_falls_back_to_env(tmp_path, monkeypatch) -> None:
    _clear_coinbase_credential_env(monkeypatch)
    key_path = Path(tmp_path) / "coinbase.json"
    key_path.write_text(json.dumps({"name": "organizations/file/apiKeys/file1"}))
    monkeypatch.setenv("COINBASE_CREDENTIALS_FILE", str(key_path))

    monkeypatch.setenv("COINBASE_CDP_API_KEY", "organizations/env/apiKeys/env1")
    monkeypatch.setenv("COINBASE_CDP_PRIVATE_KEY", "-----BEGIN EC PRIVATE KEY-----\nenv\n")

    creds = resolve_coinbase_credentials()

    assert creds is not None
    assert creds.key_name == "organizations/env/apiKeys/env1"
    assert any("COINBASE_CREDENTIALS_FILE" in w for w in creds.warnings)
