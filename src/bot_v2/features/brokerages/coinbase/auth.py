"""Authentication helpers for Coinbase Advanced Trade and derivatives APIs."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings

logger = logging.getLogger(__name__)


class AuthStrategy(Protocol):
    """Protocol for auth objects capable of signing REST requests."""

    def sign(
        self, method: str, path: str, body: dict[str, Any] | None
    ) -> dict[str, str]:  # pragma: no cover - typing helper
        ...


@dataclass
class CoinbaseAuth:
    """HMAC authentication for Coinbase REST APIs (Advanced Trade and Exchange)."""

    api_key: str
    api_secret: str
    passphrase: str | None = None
    key_version: str | None = None
    api_mode: str | None = None  # "advanced" or "exchange"

    def sign(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, str]:
        import base64 as _base64

        mode = (self.api_mode or "").lower()
        is_advanced = mode == "advanced"
        is_exchange = mode == "exchange"
        if not (is_advanced or is_exchange):
            is_exchange = bool(self.passphrase)
            is_advanced = not is_exchange

        timestamp = str(time.time()) if is_exchange else str(int(time.time()))
        body_str = json.dumps(body, separators=(",", ":")) if body else ""
        prehash = (timestamp + method.upper() + path + body_str).encode()

        try:
            key = _base64.b64decode(self.api_secret, validate=True)
            if not key:
                key = self.api_secret.encode()
        except Exception:
            key = self.api_secret.encode()

        digest = hmac.new(key, prehash, hashlib.sha256).digest()
        signature = _base64.b64encode(digest).decode()

        headers = {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }
        if is_exchange and self.passphrase:
            headers["CB-ACCESS-PASSPHRASE"] = self.passphrase
        if self.key_version:
            headers["CB-ACCESS-KEY-VERSION"] = self.key_version
        return headers


@dataclass
class CDPJWTAuth:
    """JWT authentication using EC private keys for Coinbase Developer Platform."""

    api_key_name: str
    private_key_pem: str
    base_host: str = "api.coinbase.com"  # Host without scheme
    issuer: str = "cdp"
    audience: tuple[str, ...] | None = None
    include_host_in_uri: bool = True
    add_nonce_header: bool = True

    def generate_jwt(self, method: str | None = None, path: str | None = None) -> str:
        method = (method or "GET").upper()
        path_only = path or "/users/self"
        if not path_only.startswith("/"):
            path_only = f"/{path_only}"
        uri = (
            f"{method} {self.base_host}{path_only}"
            if self.include_host_in_uri
            else f"{method} {path_only}"
        )

        try:
            import jwt
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import serialization
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "CDP authentication requires 'pyjwt' and 'cryptography' packages. "
                "Install with: pip install pyjwt cryptography"
            ) from exc

        try:
            serialization.load_pem_private_key(
                self.private_key_pem.encode(),
                password=None,
                backend=default_backend(),
            )
        except Exception as exc:
            logger.error("Failed to load CDP private key", exc_info=True)
            raise ValueError("Invalid private key") from exc

        current_time = int(time.time())
        claims: dict[str, Any] = {
            "sub": self.api_key_name,
            "iss": self.issuer,
            "nbf": current_time,
            "exp": current_time + 120,
            "uri": uri,
        }
        if self.audience:
            claims["aud"] = list(self.audience)

        headers = {"kid": self.api_key_name}
        if self.add_nonce_header:
            headers["nonce"] = secrets.token_hex()

        encoded_token = jwt.encode(claims, self.private_key_pem, algorithm="ES256", headers=headers)
        if isinstance(encoded_token, bytes):
            token = encoded_token.decode("utf-8")
        else:
            token = encoded_token

        fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]
        logger.debug("Generated CDP JWT uri=%s fingerprint=%s", uri, fingerprint)
        return token

    def sign(
        self, method: str, path: str, body: dict[str, Any] | None
    ) -> dict[str, str]:  # noqa: ARG002 - body kept for signature parity
        token = self.generate_jwt(method, path)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }


def _normalize_private_key(private_key_pem: str | bytes) -> str:
    key = private_key_pem.decode() if isinstance(private_key_pem, bytes) else private_key_pem
    key = key.strip().replace("\r\n", "\n").replace("\r", "\n")
    if not key.startswith("-----BEGIN"):
        key = key.replace("\\n", "\n")
    return key


def _host_from_base_url(base_url: str | None, default: str = "api.coinbase.com") -> str:
    if not base_url:
        return default
    try:
        parsed = urlparse(base_url)
        if parsed.netloc:
            return parsed.netloc
        cleaned = base_url.replace("https://", "").replace("http://", "").strip("/")
        return cleaned or default
    except Exception:  # pragma: no cover - defensive parsing
        return default


def create_cdp_jwt_auth(
    *,
    api_key_name: str,
    private_key_pem: str | bytes,
    base_url: str | None = None,
    issuer: str = "cdp",
    audience: Iterable[str] | None = None,
    include_host_in_uri: bool = True,
    add_nonce_header: bool = True,
) -> CDPJWTAuth:
    host = _host_from_base_url(base_url)
    key = _normalize_private_key(private_key_pem)
    audience_tuple = tuple(audience) if audience else None
    return CDPJWTAuth(
        api_key_name=api_key_name,
        private_key_pem=key,
        base_host=host,
        issuer=issuer,
        audience=audience_tuple,
        include_host_in_uri=include_host_in_uri,
        add_nonce_header=add_nonce_header,
    )


def create_cdp_auth(
    api_key_name: str, private_key_pem: str | bytes, base_url: str | None = None
) -> CDPJWTAuth:
    """Legacy helper mirroring the behaviour of the original CDPAuth."""
    return create_cdp_jwt_auth(
        api_key_name=api_key_name,
        private_key_pem=private_key_pem,
        base_url=base_url,
        issuer="coinbase-cloud",
        audience=("retail_rest_api_proxy",),
        include_host_in_uri=False,
    )


def build_rest_auth(config: APIConfig) -> AuthStrategy:
    """Create the appropriate REST authentication object for Coinbase APIs."""

    if config.cdp_api_key and config.cdp_private_key:
        logger.info("Using CDP JWT authentication (SDK-compatible)")
        api_key = config.cdp_api_key
        private_key = config.cdp_private_key
        return create_cdp_jwt_auth(
            api_key_name=api_key,
            private_key_pem=private_key,
            base_url=config.base_url,
        )

    logger.info("Using HMAC authentication with %s mode", config.api_mode)
    return CoinbaseAuth(
        api_key=config.api_key,
        api_secret=config.api_secret,
        passphrase=config.passphrase,
        api_mode=config.api_mode,
    )


def build_ws_auth_provider(
    config: APIConfig,
    client_auth: Any,
    *,
    settings: RuntimeSettings | None = None,
) -> Callable[[], dict[str, str] | None] | None:
    """Return a callable that produces WebSocket auth payloads when required."""

    if config is None:
        return None

    runtime_settings = settings or load_runtime_settings()

    ws_user_auth_raw = runtime_settings.raw_env.get("COINBASE_WS_USER_AUTH")
    if ws_user_auth_raw and getattr(client_auth, "generate_jwt", None):

        def provider() -> dict[str, str] | None:
            try:
                generator = client_auth.generate_jwt
                token = generator("GET", "/users/self") if callable(generator) else None
                if isinstance(token, str):
                    return {"jwt": token}
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Failed to generate WS auth via client auth object: %s", exc)
            return None

        return provider

    if (
        config.enable_derivatives
        and getattr(config, "auth_type", None) == "JWT"
        and config.cdp_api_key
        and config.cdp_private_key
    ):

        def provider() -> dict[str, str] | None:
            try:
                api_key = config.cdp_api_key
                private_key = config.cdp_private_key
                if api_key is None or private_key is None:
                    return None
                auth = create_cdp_auth(
                    api_key_name=api_key,
                    private_key_pem=private_key,
                    base_url=config.base_url,
                )
                token = auth.generate_jwt("GET", "/users/self")
                return {"jwt": token}
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Failed to generate WS auth token: %s", exc)
                return None

        return provider

    return None


__all__ = [
    "AuthStrategy",
    "CoinbaseAuth",
    "CDPJWTAuth",
    "create_cdp_auth",
    "create_cdp_jwt_auth",
    "build_rest_auth",
    "build_ws_auth_provider",
]
