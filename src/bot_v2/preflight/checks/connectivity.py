from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError

if TYPE_CHECKING:
    from bot_v2.preflight.core import PreflightCheck


def check_api_connectivity(checker: "PreflightCheck") -> bool:
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

    try:
        products = client.list_products()
        perps = [p for p in products if p.get("product_id", "").endswith("-PERP")]
        if perps:
            checker.log_success(f"Found {len(perps)} perpetual products")
            if checker.verbose:
                for product in perps[:3]:
                    checker.log_info(f"  - {product.get('product_id')}")
        else:
            checker.log_error("No perpetual products found")
            all_good = False
    except Exception as exc:
        checker.log_error(f"Failed to list products: {exc}")
        all_good = False

    return all_good


def check_key_permissions(checker: "PreflightCheck") -> bool:
    """Validate Coinbase key permissions and INTX portfolio readiness."""
    checker.section_header("5. KEY PERMISSIONS & INTX READINESS")

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
            permissions = client.get_key_permissions() or {}
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

    if permissions is None:
        checker.log_error("Key permissions response empty; cannot validate entitlements")
        return False

    can_trade = bool(permissions.get("can_trade"))
    can_view = bool(permissions.get("can_view"))
    portfolio_type = (permissions.get("portfolio_type") or "").upper()
    portfolio_uuid = permissions.get("portfolio_uuid")

    all_good = True

    if can_trade and can_view:
        checker.log_success("API key has trade + view permissions")
    else:
        if not can_trade:
            checker.log_error("API key missing trade permission (can_trade=False)")
            all_good = False
        if not can_view:
            checker.log_error("API key missing portfolio view permission (can_view=False)")
            all_good = False

    if portfolio_uuid:
        checker.log_info(f"Portfolio UUID detected: {portfolio_uuid}")
    else:
        checker.log_warning("Portfolio UUID not returned; verify CDP key portfolio access")

    derivatives_enabled = os.getenv("COINBASE_ENABLE_DERIVATIVES") == "1"
    if derivatives_enabled:
        if portfolio_type == "INTX":
            checker.log_success("INTX portfolio detected (derivatives can execute)")
        else:
            display_type = portfolio_type or "UNKNOWN"
            checker.log_error(
                f"INTX gating check failed: portfolio_type={display_type} "
                "(expected INTX when derivatives are enabled)"
            )
            all_good = False
    else:
        if portfolio_type == "INTX":
            checker.log_info("INTX portfolio available; derivatives flag is currently disabled")

    return all_good
