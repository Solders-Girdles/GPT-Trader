from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from gpt_trader.app.config.profile_loader import (
    get_env_defaults_for_profile,
    is_dev_profile,
)
from gpt_trader.features.brokerages.coinbase.credentials import (
    ResolvedCoinbaseCredentials,
    resolve_coinbase_credentials,
)
from gpt_trader.preflight.validation_result import (
    PreflightResultPayload,
    normalize_preflight_result,
)


class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


@dataclass(slots=True)
class PreflightContext:
    """Shared state and logging utilities for preflight checks."""

    verbose: bool = False
    profile: str = "canary"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    successes: list[str] = field(default_factory=list)
    results: list[PreflightResultPayload] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    # ----- Logging helpers -------------------------------------------------
    def _record_result(
        self,
        *,
        status: str,
        message: str,
        details: Mapping[str, Any] | str | None = None,
    ) -> None:
        self.results.append(
            normalize_preflight_result(status=status, message=message, details=details)
        )

    def log_success(self, message: str, details: Mapping[str, Any] | str | None = None) -> None:
        self.successes.append(message)
        self._record_result(status="pass", message=message, details=details)
        print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")

    def log_warning(self, message: str, details: Mapping[str, Any] | str | None = None) -> None:
        self.warnings.append(message)
        self._record_result(status="warn", message=message, details=details)
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")

    def log_error(self, message: str, details: Mapping[str, Any] | str | None = None) -> None:
        self.errors.append(message)
        self._record_result(status="fail", message=message, details=details)
        print(f"{Colors.RED}❌ {message}{Colors.RESET}")

    def log_info(self, message: str) -> None:
        if self.verbose:
            print(f"{Colors.CYAN}ℹ️  {message}{Colors.RESET}")

    def section_header(self, title: str) -> None:
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")

    # ----- Environment helpers ---------------------------------------------
    def resolve_cdp_credentials(self) -> tuple[str | None, str | None]:
        creds = resolve_coinbase_credentials()
        if not creds:
            return None, None
        return creds.key_name, creds.private_key

    def resolve_cdp_credentials_info(self) -> ResolvedCoinbaseCredentials | None:
        return resolve_coinbase_credentials()

    def has_real_cdp_credentials(self) -> bool:
        api_key, private_key = self.resolve_cdp_credentials()
        if not api_key or not private_key:
            return False
        if not (api_key.startswith("organizations/") and "/apiKeys/" in api_key):
            return False
        if "BEGIN EC PRIVATE KEY" not in private_key:
            return False
        return True

    def should_skip_remote_checks(self) -> bool:
        if os.getenv("COINBASE_PREFLIGHT_FORCE_REMOTE") == "1":
            return False
        if os.getenv("COINBASE_PREFLIGHT_SKIP_REMOTE") == "1":
            return True
        if is_dev_profile(self.profile) and not self.has_real_cdp_credentials():
            return True
        return False

    def expected_env_defaults(self) -> Mapping[str, tuple[str, bool]]:
        return get_env_defaults_for_profile(self.profile)

    def _env_bool(self, key: str, default: bool = False) -> bool:
        raw = os.getenv(key)
        if raw is None or raw == "":
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    def trading_modes(self) -> list[str]:
        raw = os.getenv("TRADING_MODES", "")
        if not raw:
            return ["spot"]
        modes = [mode.strip().lower() for mode in raw.split(",") if mode.strip()]
        return modes or ["spot"]

    def cfm_enabled(self) -> bool:
        return self._env_bool("CFM_ENABLED", False)

    def intx_perps_enabled(self) -> bool:
        return self._env_bool("COINBASE_ENABLE_INTX_PERPS", False)

    def intends_real_orders(self) -> bool:
        if self._env_bool("DRY_RUN", False):
            return False
        if self._env_bool("PAPER_MODE", False):
            return False
        if self._env_bool("PERPS_PAPER", False):
            return False
        if os.getenv("COINBASE_SANDBOX", "0") == "1":
            return False
        return True

    def requires_trade_permission(self) -> bool:
        if not self.intends_real_orders():
            return False
        modes = set(self.trading_modes())
        if modes.intersection({"spot", "cfm"}):
            return True
        return self.intx_perps_enabled()

    # ----- Coinbase connectivity helpers -----------------------------------
    def build_cdp_client(self) -> tuple[Any, Any] | None:
        try:
            from gpt_trader.features.brokerages.coinbase.client import (
                CoinbaseClient,
                create_cdp_jwt_auth,
            )
        except Exception as exc:  # pragma: no cover - defensive import guard
            self.log_error(f"Coinbase client import failed: {exc}")
            return None

        api_key, private_key = self.resolve_cdp_credentials()

        if not api_key or not private_key:
            if self.should_skip_remote_checks():
                self.log_info("CDP credentials not configured; skipping remote connectivity checks")
            else:
                self.log_error("CDP credentials missing (export API key and private key)")
            return None

        try:
            auth = create_cdp_jwt_auth(
                api_key=api_key,
                private_key=private_key,
            )
        except Exception as exc:
            self.log_error(f"Failed to initialize CDP JWT auth: {exc}")
            return None

        client = CoinbaseClient(
            base_url="https://api.coinbase.com",
            auth=auth,
            api_mode="advanced",
        )
        return client, auth
