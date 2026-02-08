from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ResolvedCoinbaseCredentials:
    """Resolved Coinbase CDP JWT credentials.

    Values are sourced from (in priority order):
    - COINBASE_CREDENTIALS_FILE (JSON key file)
    - COINBASE_PROD_CDP_* / COINBASE_CDP_* env vars
    - COINBASE_API_KEY_NAME / COINBASE_PRIVATE_KEY legacy env vars
    """

    key_name: str
    private_key: str = field(repr=False)
    source: str
    warnings: tuple[str, ...] = ()

    @property
    def masked_key_name(self) -> str:
        return mask_key_name(self.key_name)


def mask_key_name(key_name: str) -> str:
    """Mask a CDP key name for safe display/logging."""
    key_name = (key_name or "").strip()
    if not key_name:
        return ""
    if len(key_name) <= 20:
        return f"{key_name[:6]}…"
    return f"{key_name[:20]}…{key_name[-8:]}"


def _read_credentials_file(path: Path) -> tuple[str | None, str | None]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        return None, None
    key_name = data.get("name") or data.get("key_name") or data.get("api_key_name")
    private_key = data.get("privateKey") or data.get("private_key") or data.get("privateKeyPem")
    if key_name is not None:
        key_name = str(key_name)
    if private_key is not None:
        private_key = str(private_key)
    return key_name, private_key


def resolve_coinbase_credentials() -> ResolvedCoinbaseCredentials | None:
    """Resolve Coinbase credentials used for JWT auth without exposing secrets."""
    warnings: list[str] = []
    allow_ambiguous = os.getenv("COINBASE_ALLOW_AMBIGUOUS_CREDENTIALS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    env_key_candidates = (
        ("COINBASE_PROD_CDP_API_KEY", os.getenv("COINBASE_PROD_CDP_API_KEY")),
        ("COINBASE_CDP_API_KEY", os.getenv("COINBASE_CDP_API_KEY")),
        ("COINBASE_API_KEY_NAME", os.getenv("COINBASE_API_KEY_NAME")),
    )
    env_priv_candidates = (
        ("COINBASE_PROD_CDP_PRIVATE_KEY", os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY")),
        ("COINBASE_CDP_PRIVATE_KEY", os.getenv("COINBASE_CDP_PRIVATE_KEY")),
        ("COINBASE_PRIVATE_KEY", os.getenv("COINBASE_PRIVATE_KEY")),
    )

    # Conflict detection: only flag same-tier vars (CDP vs legacy).
    # Cross-tier differences (PROD overriding CDP) are the normal deployment pattern.
    cdp_key = os.getenv("COINBASE_CDP_API_KEY")
    legacy_key = os.getenv("COINBASE_API_KEY_NAME")
    cdp_priv = os.getenv("COINBASE_CDP_PRIVATE_KEY")
    legacy_priv = os.getenv("COINBASE_PRIVATE_KEY")

    key_conflict = bool(cdp_key and legacy_key and cdp_key != legacy_key)
    private_conflict = bool(cdp_priv and legacy_priv and cdp_priv != legacy_priv)
    if key_conflict:
        warnings.append(
            "COINBASE_CDP_API_KEY and COINBASE_API_KEY_NAME are set with different values"
        )
    if private_conflict:
        warnings.append(
            "COINBASE_CDP_PRIVATE_KEY and COINBASE_PRIVATE_KEY are set with different values"
        )

    key_name_var = ""
    key_name = ""
    for var, val in env_key_candidates:
        if val:
            key_name_var = var
            key_name = val
            break

    private_key_var = ""
    private_key = ""
    for var, val in env_priv_candidates:
        if val:
            private_key_var = var
            private_key = val
            break

    creds_file = os.getenv("COINBASE_CREDENTIALS_FILE")
    if creds_file:
        path = Path(creds_file)
        if path.exists():
            file_key_name, file_private_key = _read_credentials_file(path)
            if file_key_name and file_private_key:
                if key_name and key_name != file_key_name:
                    warnings.append(
                        "COINBASE_CREDENTIALS_FILE key name differs from env key vars; "
                        "refusing to resolve ambiguous credentials"
                    )
                if private_key and private_key != file_private_key:
                    warnings.append(
                        "COINBASE_CREDENTIALS_FILE private key differs from env private key vars; "
                        "refusing to resolve ambiguous credentials"
                    )
                has_file_env_mismatch = any(
                    "refusing to resolve ambiguous credentials" in w for w in warnings
                )
                has_conflict = key_conflict or private_conflict or has_file_env_mismatch
                if has_conflict and not allow_ambiguous:
                    return None
                return ResolvedCoinbaseCredentials(
                    key_name=file_key_name,
                    private_key=file_private_key,
                    source=f"file:{path}",
                    warnings=tuple(warnings),
                )

            warnings.append(
                "COINBASE_CREDENTIALS_FILE is set but missing required fields ('name', 'privateKey')"
            )

    if (key_conflict or private_conflict) and not allow_ambiguous:
        return None

    if not key_name or not private_key:
        return None

    return ResolvedCoinbaseCredentials(
        key_name=key_name,
        private_key=private_key,
        source=f"env:{key_name_var}+{private_key_var}",
        warnings=tuple(warnings),
    )


__all__ = [
    "ResolvedCoinbaseCredentials",
    "mask_key_name",
    "resolve_coinbase_credentials",
]
