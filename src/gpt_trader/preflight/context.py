from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


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
    config: dict[str, Any] = field(default_factory=dict)

    # ----- Logging helpers -------------------------------------------------
    def log_success(self, message: str) -> None:
        self.successes.append(message)
        print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")

    def log_warning(self, message: str) -> None:
        self.warnings.append(message)
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")

    def log_error(self, message: str) -> None:
        self.errors.append(message)
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
        api_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
        private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv(
            "COINBASE_CDP_PRIVATE_KEY"
        )
        return api_key, private_key

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
        if self.profile == "dev" and not self.has_real_cdp_credentials():
            return True
        return False

    def expected_env_defaults(self) -> dict[str, tuple[str, bool]]:
        if self.profile == "dev":
            return {
                "BROKER": ("coinbase", True),
                "COINBASE_SANDBOX": ("1", False),
                "COINBASE_API_MODE": ("advanced", False),
                "COINBASE_ENABLE_DERIVATIVES": ("0", False),
            }
        return {
            "BROKER": ("coinbase", True),
            "COINBASE_SANDBOX": ("0", True),
            "COINBASE_API_MODE": ("advanced", True),
            "COINBASE_ENABLE_DERIVATIVES": ("1", True),
        }

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
                api_key_name=api_key,
                private_key_pem=private_key,
                base_url="https://api.coinbase.com",
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
