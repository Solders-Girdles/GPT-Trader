"""Tests for PreflightContext CDP credential resolution."""

from __future__ import annotations

import os

import pytest

from gpt_trader.preflight.context import PreflightContext


def _set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str], *, clear: bool = True) -> None:
    if clear:
        for key in list(os.environ):
            monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


class TestPreflightContextCredentials:
    """Test CDP credential resolution."""

    def test_resolve_cdp_credentials_from_prod_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should prefer PROD env vars."""
        with monkeypatch.context() as mp:
            _set_env(
                mp,
                {
                    "COINBASE_PROD_CDP_API_KEY": "prod_key",
                    "COINBASE_PROD_CDP_PRIVATE_KEY": "prod_private",
                    "COINBASE_CDP_API_KEY": "fallback_key",
                    "COINBASE_CDP_PRIVATE_KEY": "fallback_private",
                },
                clear=False,
            )
            ctx = PreflightContext()
            api_key, private_key = ctx.resolve_cdp_credentials()
            resolved = ctx.resolve_cdp_credentials_info()

            assert api_key == "prod_key"
            assert private_key == "prod_private"
            assert resolved is not None
            assert resolved.key_name == "prod_key"
            assert resolved.private_key == "prod_private"

    def test_resolve_cdp_credentials_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should fall back to non-PROD env vars."""
        with monkeypatch.context() as mp:
            _set_env(
                mp,
                {
                    "COINBASE_CDP_API_KEY": "fallback_key",
                    "COINBASE_CDP_PRIVATE_KEY": "fallback_private",
                },
                clear=True,
            )
            ctx = PreflightContext()
            api_key, private_key = ctx.resolve_cdp_credentials()
            resolved = ctx.resolve_cdp_credentials_info()

            assert api_key == "fallback_key"
            assert private_key == "fallback_private"
            assert resolved is not None
            assert resolved.key_name == "fallback_key"
            assert resolved.private_key == "fallback_private"

    def test_resolve_cdp_credentials_returns_none_when_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return None when credentials not set."""
        with monkeypatch.context() as mp:
            _set_env(mp, {}, clear=True)
            ctx = PreflightContext()
            api_key, private_key = ctx.resolve_cdp_credentials()
            resolved = ctx.resolve_cdp_credentials_info()

            assert api_key is None
            assert private_key is None
            assert resolved is None

    def test_has_real_cdp_credentials_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return True for valid CDP credentials."""
        with monkeypatch.context() as mp:
            _set_env(
                mp,
                {
                    "COINBASE_CDP_API_KEY": "organizations/abc/apiKeys/xyz",
                    "COINBASE_CDP_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
                },
                clear=True,
            )
            ctx = PreflightContext()
            assert ctx.has_real_cdp_credentials() is True

    def test_has_real_cdp_credentials_invalid_key_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return False for invalid key format."""
        with monkeypatch.context() as mp:
            _set_env(
                mp,
                {
                    "COINBASE_CDP_API_KEY": "invalid_format",
                    "COINBASE_CDP_PRIVATE_KEY": "-----BEGIN EC PRIVATE KEY-----\ntest\n-----END EC PRIVATE KEY-----",
                },
                clear=True,
            )
            ctx = PreflightContext()
            assert ctx.has_real_cdp_credentials() is False

    def test_has_real_cdp_credentials_invalid_private_key_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return False for invalid private key format."""
        with monkeypatch.context() as mp:
            _set_env(
                mp,
                {
                    "COINBASE_CDP_API_KEY": "organizations/abc/apiKeys/xyz",
                    "COINBASE_CDP_PRIVATE_KEY": "not-a-valid-key",
                },
                clear=True,
            )
            ctx = PreflightContext()
            assert ctx.has_real_cdp_credentials() is False

    def test_has_real_cdp_credentials_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return False when credentials missing."""
        with monkeypatch.context() as mp:
            _set_env(mp, {}, clear=True)
            ctx = PreflightContext()
            assert ctx.has_real_cdp_credentials() is False
