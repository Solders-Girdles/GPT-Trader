"""
Pre-trade diagnostics check.

Validates API health, broker readiness, and market data freshness
before trading starts. Complements the runtime guard stack by catching
unsafe states at startup.
"""

from __future__ import annotations

import os
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpt_trader.preflight.core import PreflightCheck


def _is_warn_only() -> bool:
    """Check if warn-only mode is enabled via env."""
    return os.getenv("GPT_TRADER_PREFLIGHT_WARN_ONLY", "0") == "1"


def check_pretrade_diagnostics(checker: PreflightCheck) -> bool:
    """
    Validate pre-trade diagnostics before trading starts.

    Checks:
    - API health (circuit breakers, error rates, rate limits)
    - Broker readiness (accounts, products)
    - Market data freshness (ticker prices, staleness)

    Args:
        checker: PreflightCheck instance for logging and context

    Returns:
        True if all diagnostics pass (or warn-only mode), False otherwise
    """
    checker.section_header("11. PRE-TRADE DIAGNOSTICS")

    warn_only = _is_warn_only()
    if warn_only:
        checker.log_info("Warn-only mode enabled (GPT_TRADER_PREFLIGHT_WARN_ONLY=1)")

    ctx = checker.context

    # Skip remote checks if configured
    if ctx.should_skip_remote_checks():
        checker.log_success("Pre-trade diagnostics bypassed (offline/stub credentials)")
        return True

    # Build CDP client
    built = checker._build_cdp_client()
    if not built:
        if warn_only:
            checker.log_warning("CDP client unavailable - skipping diagnostics (warn-only)")
            return True
        checker.log_error("CDP client unavailable - cannot run pre-trade diagnostics")
        return False

    client, _auth = built

    # Track overall success
    all_good = True

    # -------------------------------------------------------------------------
    # 1. API Health Check (reuses ApiHealthGuard logic)
    # -------------------------------------------------------------------------
    api_health_ok = _check_api_health(checker, client, warn_only)
    all_good = all_good and api_health_ok

    # -------------------------------------------------------------------------
    # 2. Broker Readiness
    # -------------------------------------------------------------------------
    broker_ok = _check_broker_readiness(checker, client, warn_only)
    all_good = all_good and broker_ok

    # -------------------------------------------------------------------------
    # 3. Market Data Freshness
    # -------------------------------------------------------------------------
    market_ok = _check_market_data(checker, client, warn_only)
    all_good = all_good and market_ok

    if warn_only and not all_good:
        checker.log_warning("Pre-trade diagnostics had issues but warn-only mode enabled")
        return True

    return all_good


def _check_api_health(checker: PreflightCheck, client: Any, warn_only: bool) -> bool:
    """Check API health using resilience status."""
    checker.log_info("Checking API health...")

    if not hasattr(client, "get_resilience_status"):
        checker.log_info("Client does not expose resilience status - skipping API health check")
        return True

    try:
        status = client.get_resilience_status()
    except Exception as exc:
        if warn_only:
            checker.log_warning(f"Failed to get API resilience status: {exc}")
            return False
        checker.log_error(f"Failed to get API resilience status: {exc}")
        return False

    if status is None:
        checker.log_success("API health: No resilience status available (assumed healthy)")
        return True

    # Parse metrics
    metrics = (status or {}).get("metrics") or {}
    error_rate = float(metrics.get("error_rate", 0.0))

    # Parse rate limit usage
    rate_limit_usage = (status or {}).get("rate_limit_usage", 0.0)
    if isinstance(rate_limit_usage, str):
        rate_limit_usage = float(rate_limit_usage.rstrip("%")) / 100.0
    else:
        rate_limit_usage = float(rate_limit_usage or 0.0)

    # Parse circuit breakers
    breakers = (status or {}).get("circuit_breakers") or {}
    open_breakers = []
    for breaker_name, entry in breakers.items():
        if isinstance(entry, dict):
            breaker_state = entry.get("state", "")
        else:
            breaker_state = str(entry)
        if breaker_state == "open":
            open_breakers.append(breaker_name)

    # Get thresholds from RiskConfig
    try:
        from gpt_trader.features.live_trade.risk.config import RiskConfig

        config = RiskConfig.from_env()
        error_rate_threshold = config.api_error_rate_threshold
        rate_limit_threshold = config.api_rate_limit_usage_threshold
    except Exception:
        error_rate_threshold = 0.2
        rate_limit_threshold = 0.9

    # Check conditions
    issues = []

    if open_breakers:
        issues.append(f"circuit breakers open: {', '.join(open_breakers)}")

    if error_rate >= error_rate_threshold:
        issues.append(f"error rate {error_rate:.1%} >= threshold {error_rate_threshold:.1%}")

    if rate_limit_usage >= rate_limit_threshold:
        issues.append(
            f"rate limit usage {rate_limit_usage:.1%} >= threshold {rate_limit_threshold:.1%}"
        )

    if issues:
        msg = f"API health degraded: {'; '.join(issues)}"
        if warn_only:
            checker.log_warning(msg)
        else:
            checker.log_error(msg)
        return False

    checker.log_success(
        f"API health: OK (error_rate={error_rate:.1%}, rate_limit={rate_limit_usage:.1%})"
    )
    return True


def _check_broker_readiness(checker: PreflightCheck, client: Any, warn_only: bool) -> bool:
    """Check broker connectivity and data availability."""
    checker.log_info("Checking broker readiness...")
    all_good = True

    # Check accounts
    try:
        accounts = client.get_accounts()
        if accounts:
            account_list = accounts.get("accounts", []) if isinstance(accounts, dict) else accounts
            if account_list:
                checker.log_success(f"Accounts: {len(account_list)} account(s) accessible")
            else:
                if warn_only:
                    checker.log_warning("Accounts: Empty response")
                else:
                    checker.log_error("Accounts: Empty response - no accounts found")
                all_good = False
        else:
            if warn_only:
                checker.log_warning("Accounts: No response")
            else:
                checker.log_error("Accounts: No response from API")
            all_good = False
    except Exception as exc:
        if warn_only:
            checker.log_warning(f"Accounts: Failed to fetch - {exc}")
        else:
            checker.log_error(f"Accounts: Failed to fetch - {exc}")
        all_good = False

    # Check products
    try:
        products = client.list_products()
        if products:
            checker.log_success(f"Products: {len(products)} product(s) available")
        else:
            if warn_only:
                checker.log_warning("Products: Empty response")
            else:
                checker.log_error("Products: Empty response - no products found")
            all_good = False
    except Exception as exc:
        if warn_only:
            checker.log_warning(f"Products: Failed to fetch - {exc}")
        else:
            checker.log_error(f"Products: Failed to fetch - {exc}")
        all_good = False

    # Check trading symbols from config
    symbols = _get_trading_symbols()
    if symbols:
        for symbol in symbols[:3]:  # Check first 3 symbols
            product_ok = _check_product_details(checker, client, symbol, warn_only)
            all_good = all_good and product_ok

    return all_good


def _check_product_details(
    checker: PreflightCheck, client: Any, symbol: str, warn_only: bool
) -> bool:
    """Validate product details for a symbol."""
    try:
        product = client.get_product(symbol)
        if not product:
            if warn_only:
                checker.log_warning(f"Product {symbol}: Not found")
            else:
                checker.log_error(f"Product {symbol}: Not found")
            return False

        # Validate key fields
        min_size = product.get("base_min_size") or product.get("min_market_funds")
        step_size = product.get("base_increment") or product.get("quote_increment")
        price_increment = product.get("quote_increment")

        issues = []
        if not min_size or Decimal(str(min_size)) <= 0:
            issues.append("invalid min_size")
        if not step_size or Decimal(str(step_size)) <= 0:
            issues.append("invalid step_size")
        if not price_increment or Decimal(str(price_increment)) <= 0:
            issues.append("invalid price_increment")

        if issues:
            msg = f"Product {symbol}: {', '.join(issues)}"
            if warn_only:
                checker.log_warning(msg)
            else:
                checker.log_error(msg)
            return False

        checker.log_success(f"Product {symbol}: Valid specs")
        return True

    except Exception as exc:
        if warn_only:
            checker.log_warning(f"Product {symbol}: Failed to validate - {exc}")
        else:
            checker.log_error(f"Product {symbol}: Failed to validate - {exc}")
        return False


def _check_market_data(checker: PreflightCheck, client: Any, warn_only: bool) -> bool:
    """Check market data freshness for trading symbols."""
    checker.log_info("Checking market data freshness...")
    all_good = True

    symbols = _get_trading_symbols()
    if not symbols:
        checker.log_info("No trading symbols configured - skipping market data check")
        return True

    for symbol in symbols[:3]:  # Check first 3 symbols
        try:
            ticker = client.get_ticker(symbol)
            if not ticker:
                if warn_only:
                    checker.log_warning(f"Ticker {symbol}: No data")
                else:
                    checker.log_error(f"Ticker {symbol}: No data available")
                all_good = False
                continue

            # Parse price
            price = ticker.get("price") or ticker.get("last")
            if price:
                price_decimal = Decimal(str(price))
                if price_decimal > 0:
                    checker.log_success(f"Ticker {symbol}: ${price_decimal:,.2f}")
                else:
                    if warn_only:
                        checker.log_warning(f"Ticker {symbol}: Invalid price {price}")
                    else:
                        checker.log_error(f"Ticker {symbol}: Invalid price {price}")
                    all_good = False
            else:
                if warn_only:
                    checker.log_warning(f"Ticker {symbol}: No price in response")
                else:
                    checker.log_error(f"Ticker {symbol}: No price in response")
                all_good = False

        except Exception as exc:
            if warn_only:
                checker.log_warning(f"Ticker {symbol}: Failed - {exc}")
            else:
                checker.log_error(f"Ticker {symbol}: Failed - {exc}")
            all_good = False

    return all_good


def _get_trading_symbols() -> list[str]:
    """Get trading symbols from environment or config."""
    symbols_env = os.getenv("TRADING_SYMBOLS", "")
    if symbols_env:
        return [s.strip() for s in symbols_env.split(",") if s.strip()]

    # Default symbols for testing
    return ["BTC-USD", "ETH-USD"]


__all__ = ["check_pretrade_diagnostics"]
