from __future__ import annotations

import time
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError

if TYPE_CHECKING:
    from gpt_trader.preflight.core import PreflightCheck


def check_api_connectivity(checker: PreflightCheck) -> bool:
    """Test connectivity to Coinbase API endpoints."""
    checker.section_header("4. API CONNECTIVITY TEST")

    ctx = checker.context
    if ctx.should_skip_remote_checks():
        checker.log_success("API connectivity checks bypassed (offline/stub credentials)")
        return True

    built = checker._build_cdp_client()
    if not built:
        return False
    client, auth = built

    try:
        auth.generate_jwt("GET", "/api/v3/brokerage/accounts")
        checker.log_success("JWT token generated successfully")
    except Exception as exc:
        checker.log_error(f"JWT generation failed: {exc}")
        return False

    tests = [
        ("Server time", lambda: client.get_time()),
        ("Accounts", lambda: client.get_accounts()),
        ("Products", lambda: client.list_products()),
    ]

    all_good = True
    for test_name, test_func in tests:
        try:
            start = time.perf_counter()
            result = test_func()
            latency = (time.perf_counter() - start) * 1000

            if result:
                checker.log_success(f"{test_name}: OK ({latency:.0f}ms)")
            else:
                checker.log_warning(f"{test_name}: Empty response")
        except Exception as exc:
            checker.log_error(f"{test_name}: {str(exc)[:100]}")
            all_good = False

    return all_good


def check_key_permissions(checker: PreflightCheck) -> bool:
    """Validate Coinbase key permissions."""
    checker.section_header("5. KEY PERMISSIONS")

    ctx = checker.context
    if ctx.should_skip_remote_checks():
        checker.log_success("Key permission checks bypassed (offline/stub credentials)")
        return True

    built = checker._build_cdp_client()
    if not built:
        return False
    client, _auth = built

    max_attempts = 3
    permissions: dict[str, object] | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            permissions = client.get_key_permissions()
            break
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            if attempt == max_attempts:
                checker.log_error(f"Key permissions check failed after retries: {exc}")
                return False
            delay = min(5.0, 1.5**attempt)
            checker.log_warning(
                f"Transient error fetching key permissions ({exc}); retrying in {delay:.1f}s"
            )
            time.sleep(delay)
        except Exception as exc:
            checker.log_error(f"Failed to fetch key permissions: {exc}")
            return False

    if not permissions:
        checker.log_error("Key permissions response empty; cannot validate entitlements")
        return False

    can_trade = bool(permissions.get("can_trade"))
    can_view = bool(permissions.get("can_view"))
    portfolio_uuid = permissions.get("portfolio_uuid")
    requires_trade_permission = ctx.requires_trade_permission()

    all_good = True

    # View permission is always required
    if can_view:
        checker.log_success("API key has view permission")
    else:
        checker.log_error("API key missing portfolio view permission (can_view=False)")
        all_good = False

    # Trade permission is required when live orders are intended (spot/cfm/perps)
    if can_trade:
        checker.log_success("API key has trade permission")
    else:
        if requires_trade_permission:
            checker.log_error(
                "API key missing trade permission (can_trade=False) - required for live trading"
            )
            all_good = False
        else:
            checker.log_info(
                "API key is view-only (can_trade=False) - "
                "sufficient for observation and paper trading"
            )

    if portfolio_uuid:
        checker.log_info(f"Portfolio UUID detected: {portfolio_uuid}")
    else:
        checker.log_warning("Portfolio UUID not returned; verify CDP key portfolio access")

    return all_good
