"""Core validation service for API credentials.

This service validates Coinbase CDP API credentials against the requirements
of the selected trading mode, providing detailed findings about format,
connectivity, and permission status.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError

from gpt_trader.features.brokerages.coinbase.credentials import resolve_coinbase_credentials
from gpt_trader.preflight.validation_result import (
    CredentialValidationResult,
    PermissionDetails,
    ValidationCategory,
)

if TYPE_CHECKING:
    from textual.app import App

logger = logging.getLogger(__name__)


# Mode requirements - what each mode needs from the API key
MODE_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "demo": {
        "requires_credentials": False,
        "requires_view": False,
        "requires_trade": False,
        "description": "Demo mode - mock data, no API credentials needed",
    },
    "read_only": {
        "requires_credentials": True,
        "requires_view": True,
        "requires_trade": False,
        "description": "Observation mode - view-only access to real account",
    },
    "paper": {
        "requires_credentials": True,
        "requires_view": True,
        "requires_trade": False,
        "description": "Paper trading - real data, simulated execution",
    },
    "live": {
        "requires_credentials": True,
        "requires_view": True,
        "requires_trade": True,
        "description": "Live trading - real money at risk",
    },
}


class CredentialValidator:
    """Validates API credentials against trading mode requirements.

    This class performs comprehensive validation of Coinbase CDP credentials:
    1. Key format validation (without network calls)
    2. Connectivity testing (JWT generation, API calls)
    3. Permission retrieval and parsing
    4. Mode compatibility evaluation

    Example:
        validator = CredentialValidator(app)
        result = await validator.validate_for_mode("paper")
        if result.valid_for_mode:
            print("Credentials are valid for paper trading")
    """

    KEY_FORMAT_PREFIX = "organizations/"
    KEY_FORMAT_CONTAINS = "/apiKeys/"
    PRIVATE_KEY_MARKER = "BEGIN EC PRIVATE KEY"

    # Private key format markers for algorithm detection
    EC_KEY_MARKER = "BEGIN EC PRIVATE KEY"
    ED25519_KEY_MARKER = "BEGIN PRIVATE KEY"
    OPENSSH_KEY_MARKER = "BEGIN OPENSSH PRIVATE KEY"
    RSA_KEY_MARKER = "BEGIN RSA PRIVATE KEY"

    def __init__(self, app: App | None = None) -> None:
        """Initialize the validator.

        Args:
            app: Optional Textual app for UI integration.
        """
        self.app = app
        self._client: Any = None
        self._auth: Any = None

    def compute_credential_fingerprint(self) -> str | None:
        """Generate fingerprint from API key for cache comparison.

        Creates a safe fingerprint using first 8 and last 8 chars of the API key.
        This allows detecting credential changes without storing the full key.

        Returns:
            Fingerprint string, or None if no API key is configured.
        """
        creds = resolve_coinbase_credentials()
        api_key = creds.key_name if creds else ""
        if not api_key or len(api_key) < 16:
            return None
        # Use first 8 + last 8 chars as fingerprint (safe, doesn't expose full key)
        return f"{api_key[:8]}...{api_key[-8:]}"

    async def validate_for_mode(
        self,
        mode: str,
        force_refresh: bool = False,
    ) -> CredentialValidationResult:
        """Main validation entry point.

        Validates credentials against the requirements of the specified mode.

        Args:
            mode: Trading mode to validate for (demo, paper, read_only, live).
            force_refresh: Whether to force re-validation even if cached.

        Returns:
            CredentialValidationResult with all findings and overall status.
        """
        result = CredentialValidationResult(mode=mode)
        requirements = MODE_REQUIREMENTS.get(mode, MODE_REQUIREMENTS["demo"])

        logger.debug("Validating credentials for mode: %s", mode)

        # Step 1: Check if mode requires credentials at all
        if not requirements["requires_credentials"]:
            result.valid_for_mode = True
            result.add_success(
                ValidationCategory.MODE_COMPATIBILITY,
                f"{mode.upper()} mode does not require API credentials",
                requirements["description"],
            )
            logger.debug("Mode %s skips credential validation", mode)
            return result

        # Step 2: Validate key format (no network)
        self._validate_key_format(result)
        if not result.credentials_configured:
            result.add_error(
                ValidationCategory.MODE_COMPATIBILITY,
                f"{mode.upper()} mode requires API credentials",
                suggestion="Export COINBASE_CDP_API_KEY and COINBASE_CDP_PRIVATE_KEY",
            )
            return result

        # Step 3: Test connectivity (async)
        await self._validate_connectivity(result)
        if not result.connectivity_ok:
            # Can't proceed without connectivity
            result.add_error(
                ValidationCategory.MODE_COMPATIBILITY,
                "Cannot validate permissions without API connectivity",
            )
            return result

        # Step 4: Retrieve and parse permissions
        await self._validate_permissions(result)

        # Step 5: Evaluate mode compatibility
        self._evaluate_mode_compatibility(result, requirements)

        return result

    def _detect_key_type(self, api_key: str, private_key: str) -> dict[str, Any]:
        """Detect the type of API key and provide diagnostic info.

        Analyzes the format of both API key and private key to determine:
        - Whether it's a CDP or legacy key format
        - What signature algorithm the private key uses
        - Any compatibility issues

        Args:
            api_key: The API key string to analyze.
            private_key: The private key string to analyze.

        Returns:
            Dictionary with detection results including key_type, algorithm,
            and any issues/suggestions.
        """
        result: dict[str, Any] = {
            "key_type": "unknown",
            "algorithm": "unknown",
            "is_cdp_format": False,
            "is_legacy": False,
            "is_ed25519": False,
            "issues": [],
            "suggestions": [],
        }

        # Check API key format
        if api_key.startswith(self.KEY_FORMAT_PREFIX) and self.KEY_FORMAT_CONTAINS in api_key:
            result["is_cdp_format"] = True
            result["key_type"] = "cdp"
        elif len(api_key) == 36 and api_key.count("-") == 4:
            # UUID format - likely legacy Coinbase API key
            result["is_legacy"] = True
            result["key_type"] = "legacy_uuid"
            result["issues"].append("Legacy UUID-format API key detected")
            result["suggestions"].append(
                "Generate a new CDP API key at https://portal.cdp.coinbase.com/"
            )
        elif api_key and len(api_key) < 50 and not api_key.startswith(self.KEY_FORMAT_PREFIX):
            # Short key - likely legacy hex format or other non-CDP key
            result["is_legacy"] = True
            result["key_type"] = "legacy_short"
            result["issues"].append("Legacy short-format API key detected")
            result["suggestions"].append(
                "Generate a new CDP API key at https://portal.cdp.coinbase.com/"
            )

        # Check private key format for algorithm detection
        if self.EC_KEY_MARKER in private_key:
            result["algorithm"] = "ES256"
            # This is the correct format for our implementation
        elif self.ED25519_KEY_MARKER in private_key and self.EC_KEY_MARKER not in private_key:
            # "BEGIN PRIVATE KEY" without "EC" indicates Ed25519 or other PKCS#8 format
            result["algorithm"] = "Ed25519"
            result["is_ed25519"] = True
            result["issues"].append("Ed25519 key detected - not yet supported")
            result["suggestions"].append(
                "Create a new CDP key with ES256/ECDSA algorithm (not Ed25519)"
            )
        elif self.OPENSSH_KEY_MARKER in private_key:
            result["algorithm"] = "Ed25519_OpenSSH"
            result["is_ed25519"] = True
            result["issues"].append("OpenSSH Ed25519 key detected - not supported")
            result["suggestions"].append("Create a new CDP key with ES256/ECDSA algorithm")
        elif self.RSA_KEY_MARKER in private_key:
            result["algorithm"] = "RSA"
            result["issues"].append("RSA key detected - wrong algorithm")
            result["suggestions"].append("CDP requires EC (ES256) keys, not RSA")
        elif private_key and not private_key.strip().startswith("-----BEGIN"):
            # Not a PEM format at all - might be legacy API secret
            result["algorithm"] = "none_pem"
            result["issues"].append("Private key is not in PEM format")
            result["suggestions"].append(
                "Ensure you're using the complete private key including BEGIN/END markers"
            )

        return result

    def _validate_key_format(self, result: CredentialValidationResult) -> None:
        """Validate key format with enhanced diagnostics.

        Checks environment variables and validates the format of the API key
        and private key, providing specific guidance based on detected key type.
        """
        resolved = resolve_coinbase_credentials()
        if resolved:
            api_key = resolved.key_name
            private_key = resolved.private_key
        else:
            api_key = (
                os.getenv("COINBASE_PROD_CDP_API_KEY")
                or os.getenv("COINBASE_CDP_API_KEY")
                or os.getenv("COINBASE_API_KEY_NAME")
                or ""
            )
            private_key = (
                os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY")
                or os.getenv("COINBASE_CDP_PRIVATE_KEY")
                or os.getenv("COINBASE_PRIVATE_KEY")
                or ""
            )

        # Check if any credentials are configured
        if not api_key and not private_key:
            result.credentials_configured = False
            result.add_error(
                ValidationCategory.KEY_FORMAT,
                "No API credentials configured",
                suggestion="Set COINBASE_CDP_API_KEY and COINBASE_CDP_PRIVATE_KEY, or COINBASE_CREDENTIALS_FILE",
            )
            return

        result.credentials_configured = True
        if resolved:
            result.add_info(
                ValidationCategory.KEY_FORMAT,
                "Credential source detected",
                details=resolved.source,
            )
            for warning in resolved.warnings:
                result.add_warning(
                    ValidationCategory.KEY_FORMAT,
                    "Credential configuration warning",
                    details=warning,
                )

        # Detect key type for better diagnostics
        key_info = self._detect_key_type(api_key or "", private_key or "")

        # Validate API key format
        if api_key:
            if key_info["is_cdp_format"]:
                result.key_format_valid = True
                # Mask the key for display
                masked = f"{api_key[:20]}...{api_key[-8:]}" if len(api_key) > 30 else api_key
                result.add_success(
                    ValidationCategory.KEY_FORMAT,
                    "API key format valid",
                    masked,
                )
            elif key_info["is_legacy"]:
                result.add_error(
                    ValidationCategory.KEY_FORMAT,
                    f"Legacy API key format detected ({key_info['key_type']})",
                    details="This system requires CDP API keys, not legacy keys",
                    suggestion="Create a new CDP API key at https://portal.cdp.coinbase.com/",
                )
            else:
                result.add_error(
                    ValidationCategory.KEY_FORMAT,
                    "Invalid API key format",
                    details="Expected: organizations/<org_id>/apiKeys/<key_id>",
                    suggestion="Ensure you're using a CDP API key from the Developer Portal",
                )
        else:
            result.add_error(
                ValidationCategory.KEY_FORMAT,
                "API key not configured",
                suggestion="Set COINBASE_CDP_API_KEY or COINBASE_PROD_CDP_API_KEY",
            )

        # Validate private key format with algorithm check
        if private_key:
            if key_info["algorithm"] == "ES256":
                result.add_success(
                    ValidationCategory.KEY_FORMAT,
                    "Private key format valid (ES256/ECDSA)",
                )
            elif key_info["is_ed25519"]:
                result.add_error(
                    ValidationCategory.KEY_FORMAT,
                    "Ed25519 key detected - not yet supported",
                    details=f"Algorithm: {key_info['algorithm']}",
                    suggestion="Create a new CDP key with ES256/ECDSA algorithm (not Ed25519)",
                )
            elif key_info["algorithm"] == "RSA":
                result.add_error(
                    ValidationCategory.KEY_FORMAT,
                    "RSA key detected - incompatible algorithm",
                    details="CDP API requires EC (ES256) or Ed25519 keys",
                    suggestion="Create a new key with the correct algorithm",
                )
            elif key_info["algorithm"] == "none_pem":
                result.add_error(
                    ValidationCategory.KEY_FORMAT,
                    "Private key not in PEM format",
                    details="Expected: -----BEGIN EC PRIVATE KEY-----",
                    suggestion="Ensure you copied the complete private key including BEGIN/END markers",
                )
            else:
                result.add_error(
                    ValidationCategory.KEY_FORMAT,
                    "Invalid private key format",
                    details="Unable to determine key algorithm",
                    suggestion="Verify the private key is complete and correctly formatted",
                )
        else:
            result.add_error(
                ValidationCategory.KEY_FORMAT,
                "Private key not configured",
                suggestion="Set COINBASE_CDP_PRIVATE_KEY or COINBASE_PROD_CDP_PRIVATE_KEY",
            )

    async def _validate_connectivity(self, result: CredentialValidationResult) -> None:
        """Test connectivity with retry logic.

        Attempts to generate a JWT token and make a test API call,
        retrying on transient network errors.
        """
        creds = resolve_coinbase_credentials()
        api_key = creds.key_name if creds else ""
        private_key = creds.private_key if creds else ""

        if not api_key or not private_key:
            return

        try:
            from gpt_trader.features.brokerages.coinbase.client import (
                CoinbaseClient,
                create_cdp_jwt_auth,
            )
        except ImportError as exc:
            result.add_error(
                ValidationCategory.CONNECTIVITY,
                "Failed to import Coinbase client",
                details=str(exc),
            )
            return

        # Try to create auth and test JWT generation
        try:
            self._auth = create_cdp_jwt_auth(
                api_key=api_key,
                private_key=private_key,
            )
            # Test JWT generation
            self._auth.generate_jwt("GET", "/api/v3/brokerage/accounts")
            result.jwt_generation_ok = True
            result.add_success(
                ValidationCategory.CONNECTIVITY,
                "JWT token generation successful",
            )
        except Exception as exc:
            result.add_error(
                ValidationCategory.CONNECTIVITY,
                "JWT token generation failed",
                details=str(exc),
                suggestion="Check that your private key is valid and matches the API key",
            )
            return

        # Create client and test API connectivity
        self._client = CoinbaseClient(
            base_url="https://api.coinbase.com",
            auth=self._auth,
            api_mode="advanced",
        )

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                start = time.perf_counter()
                self._client.get_time()
                result.api_latency_ms = (time.perf_counter() - start) * 1000
                result.connectivity_ok = True
                result.add_success(
                    ValidationCategory.CONNECTIVITY,
                    f"API connectivity OK ({result.api_latency_ms:.0f}ms)",
                )
                return
            except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
                if attempt == max_attempts:
                    result.add_error(
                        ValidationCategory.CONNECTIVITY,
                        "API connectivity failed after retries",
                        details=str(exc),
                        suggestion="Check network connection and firewall settings",
                    )
                else:
                    delay = min(5.0, 1.5**attempt)
                    logger.warning(
                        f"Transient connectivity error ({exc}); retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
            except Exception as exc:
                result.add_error(
                    ValidationCategory.CONNECTIVITY,
                    "API connectivity failed",
                    details=str(exc),
                    suggestion="Verify API key is active and not revoked",
                )
                return

    async def _validate_permissions(self, result: CredentialValidationResult) -> None:
        """Retrieve and parse permissions from the API.

        Calls the key_permissions endpoint and parses the response
        into structured permission details.
        """
        if not self._client:
            return

        max_attempts = 3
        permissions: dict[str, Any] | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                permissions = self._client.get_key_permissions() or {}
                break
            except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
                if attempt == max_attempts:
                    result.add_error(
                        ValidationCategory.PERMISSIONS,
                        "Failed to retrieve key permissions after retries",
                        details=str(exc),
                    )
                    return
                delay = min(5.0, 1.5**attempt)
                await asyncio.sleep(delay)
            except Exception as exc:
                result.add_error(
                    ValidationCategory.PERMISSIONS,
                    "Failed to retrieve key permissions",
                    details=str(exc),
                    suggestion="API key may be invalid or revoked",
                )
                return

        if not permissions:
            result.add_error(
                ValidationCategory.PERMISSIONS,
                "Empty permissions response from API",
            )
            return

        # Parse permissions
        result.permissions = PermissionDetails(
            can_trade=bool(permissions.get("can_trade")),
            can_view=bool(permissions.get("can_view")),
            portfolio_type=str(permissions.get("portfolio_type") or ""),
            portfolio_uuid=str(permissions.get("portfolio_uuid") or ""),
        )

        # Report view permission
        if result.permissions.can_view:
            result.add_success(
                ValidationCategory.PERMISSIONS,
                "View permission granted",
            )
        else:
            result.add_warning(
                ValidationCategory.PERMISSIONS,
                "View permission not granted",
                suggestion="Enable 'View' permission in Coinbase Developer Portal",
            )

        # Report trade permission
        if result.permissions.can_trade:
            result.add_success(
                ValidationCategory.PERMISSIONS,
                "Trade permission granted",
            )
        else:
            result.add_info(
                ValidationCategory.PERMISSIONS,
                "Trade permission not granted",
                details="This key cannot place orders (view-only)",
                suggestion="Enable 'Trade' permission for live trading",
            )

        # Report portfolio info
        if result.permissions.portfolio_uuid:
            result.add_info(
                ValidationCategory.ACCOUNT_STATUS,
                f"Portfolio: {result.permissions.portfolio_uuid[:8]}...",
            )
        else:
            result.add_warning(
                ValidationCategory.ACCOUNT_STATUS,
                "Portfolio UUID not returned",
                suggestion="Verify API key has portfolio access",
            )

        # Report portfolio type
        if result.permissions.portfolio_type:
            portfolio_type = result.permissions.portfolio_type.upper()
            if portfolio_type == "INTX":
                result.add_success(
                    ValidationCategory.ACCOUNT_STATUS,
                    "INTX portfolio (perpetuals enabled)",
                )
            else:
                result.add_info(
                    ValidationCategory.ACCOUNT_STATUS,
                    f"Portfolio type: {portfolio_type}",
                    details="INTX portfolio required for perpetuals trading",
                )

    def _evaluate_mode_compatibility(
        self,
        result: CredentialValidationResult,
        requirements: dict[str, Any],
    ) -> None:
        """Check if permissions satisfy mode requirements.

        Evaluates the retrieved permissions against what the mode requires
        and sets the final valid_for_mode status.
        """
        mode = result.mode.upper()
        all_satisfied = True

        # Check view permission requirement
        if requirements["requires_view"] and not result.permissions.can_view:
            result.add_error(
                ValidationCategory.MODE_COMPATIBILITY,
                f"{mode} mode requires view permission",
                suggestion="Create API key with 'View' permission enabled",
            )
            all_satisfied = False

        # Check trade permission requirement
        if requirements["requires_trade"] and not result.permissions.can_trade:
            result.add_error(
                ValidationCategory.MODE_COMPATIBILITY,
                f"{mode} mode requires trade permission",
                details="Live trading requires the ability to place orders",
                suggestion="Create API key with 'Trade' permission enabled",
            )
            all_satisfied = False

        # Set final status
        if all_satisfied and result.connectivity_ok:
            result.valid_for_mode = True
            result.add_success(
                ValidationCategory.MODE_COMPATIBILITY,
                f"Credentials valid for {mode} mode",
                requirements["description"],
            )
        elif not all_satisfied:
            # Errors were already added above
            pass

    @staticmethod
    def get_mode_requirements(mode: str) -> dict[str, Any]:
        """Get the requirements for a specific mode.

        Args:
            mode: Trading mode name.

        Returns:
            Dictionary of requirements for the mode.
        """
        return MODE_REQUIREMENTS.get(mode, MODE_REQUIREMENTS["demo"])
