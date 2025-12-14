"""
Coinbase API constants.

Centralizes API URLs, prefixes, and endpoint definitions following the
pattern used by the official Coinbase Python SDK.

Reference: https://github.com/coinbase/coinbase-advanced-py
"""

from typing import Final

# =============================================================================
# REST API Constants
# =============================================================================

BASE_URL: Final[str] = "https://api.coinbase.com"
"""Base URL for all REST API requests."""

API_PREFIX: Final[str] = "/api/v3/brokerage"
"""Versioned path prefix for Advanced Trade API endpoints."""

DEFAULT_API_VERSION: Final[str] = "2024-10-24"
"""Default CB-VERSION header value."""

# =============================================================================
# WebSocket Constants
# =============================================================================

WS_BASE_URL: Final[str] = "wss://advanced-trade-ws.coinbase.com"
"""Public WebSocket endpoint for market data."""

WS_USER_BASE_URL: Final[str] = "wss://advanced-trade-ws-user.coinbase.com"
"""Authenticated WebSocket endpoint for user-specific data."""

# WebSocket channel names
WS_CHANNEL_HEARTBEATS: Final[str] = "heartbeats"
WS_CHANNEL_CANDLES: Final[str] = "candles"
WS_CHANNEL_MARKET_TRADES: Final[str] = "market_trades"
WS_CHANNEL_STATUS: Final[str] = "status"
WS_CHANNEL_TICKER: Final[str] = "ticker"
WS_CHANNEL_TICKER_BATCH: Final[str] = "ticker_batch"
WS_CHANNEL_LEVEL2: Final[str] = "level2"
WS_CHANNEL_USER: Final[str] = "user"
WS_CHANNEL_FUTURES_BALANCE_SUMMARY: Final[str] = "futures_balance_summary"

# Channels requiring authentication
WS_AUTH_CHANNELS: Final[frozenset[str]] = frozenset({
    WS_CHANNEL_USER,
    WS_CHANNEL_FUTURES_BALANCE_SUMMARY,
})

# =============================================================================
# Endpoint Map
# =============================================================================

# Using f-strings with API_PREFIX for consistency with official SDK pattern.
# Each endpoint can include {placeholder} values for runtime substitution.

ADVANCED_ENDPOINTS: Final[dict[str, str]] = {
    # Products & Market Data
    "products": f"{API_PREFIX}/products",
    "product": f"{API_PREFIX}/products/{{product_id}}",
    "ticker": f"{API_PREFIX}/products/{{product_id}}/ticker",
    "candles": f"{API_PREFIX}/products/{{product_id}}/candles",
    "order_book": f"{API_PREFIX}/product_book",
    "best_bid_ask": f"{API_PREFIX}/best_bid_ask",
    # Public (unauthenticated) market endpoints
    "public_products": f"{API_PREFIX}/market/products",
    "public_product": f"{API_PREFIX}/market/products/{{product_id}}",
    "public_ticker": f"{API_PREFIX}/market/products/{{product_id}}/ticker",
    "public_candles": f"{API_PREFIX}/market/products/{{product_id}}/candles",
    # Accounts
    "accounts": f"{API_PREFIX}/accounts",
    "account": f"{API_PREFIX}/accounts/{{account_uuid}}",
    # Orders
    "orders": f"{API_PREFIX}/orders",
    "order": f"{API_PREFIX}/orders/historical/{{order_id}}",
    "orders_historical": f"{API_PREFIX}/orders/historical",
    "orders_batch_cancel": f"{API_PREFIX}/orders/batch_cancel",
    "order_preview": f"{API_PREFIX}/orders/preview",
    "order_edit": f"{API_PREFIX}/orders/edit",
    "close_position": f"{API_PREFIX}/orders/close_position",
    "fills": f"{API_PREFIX}/orders/historical/fills",
    # System
    "time": f"{API_PREFIX}/time",
    "key_permissions": f"{API_PREFIX}/key_permissions",
    "fees": f"{API_PREFIX}/fees",
    "limits": f"{API_PREFIX}/limits",
    # Portfolios
    "portfolios": f"{API_PREFIX}/portfolios",
    "portfolio": f"{API_PREFIX}/portfolios/{{portfolio_uuid}}",
    "portfolio_breakdown": f"{API_PREFIX}/portfolios/{{portfolio_uuid}}/breakdown",
    "move_funds": f"{API_PREFIX}/portfolios/move_funds",
    # Convert
    "convert_quote": f"{API_PREFIX}/convert/quote",
    "convert_trade": f"{API_PREFIX}/convert/trade/{{trade_id}}",
    # Payment Methods
    "payment_methods": f"{API_PREFIX}/payment_methods",
    "payment_method": f"{API_PREFIX}/payment_methods/{{payment_method_id}}",
    # INTX (International Perpetuals)
    "intx_portfolio": f"{API_PREFIX}/intx/portfolio/{{portfolio_uuid}}",
    "intx_allocate": f"{API_PREFIX}/intx/allocate",
    "intx_balances": f"{API_PREFIX}/intx/balances/{{portfolio_uuid}}",
    "intx_positions": f"{API_PREFIX}/intx/positions/{{portfolio_uuid}}",
    "intx_position": f"{API_PREFIX}/intx/positions/{{portfolio_uuid}}/{{symbol}}",
    "intx_multi_asset_collateral": f"{API_PREFIX}/intx/multi_asset_collateral",
    # CFM (Coinbase Financial Markets - US Futures)
    "cfm_balance_summary": f"{API_PREFIX}/cfm/balance_summary",
    "cfm_positions": f"{API_PREFIX}/cfm/positions",
    "cfm_position": f"{API_PREFIX}/cfm/positions/{{product_id}}",
    "cfm_sweeps": f"{API_PREFIX}/cfm/sweeps",
    "cfm_schedule_sweep": f"{API_PREFIX}/cfm/sweeps/schedule",
    "cfm_intraday_margin_setting": f"{API_PREFIX}/cfm/intraday/margin_setting",
    "cfm_intraday_current_margin_window": f"{API_PREFIX}/cfm/intraday/current_margin_window",
}

EXCHANGE_ENDPOINTS: Final[dict[str, str]] = {
    "products": "/products",
    "product": "/products/{product_id}",
    "accounts": "/accounts",
    "order_book": "/products/{product_id}/book",
}

ENDPOINT_MAP: Final[dict[str, dict[str, str]]] = {
    "advanced": ADVANCED_ENDPOINTS,
    "exchange": EXCHANGE_ENDPOINTS,
}


__all__ = [
    # REST
    "BASE_URL",
    "API_PREFIX",
    "DEFAULT_API_VERSION",
    # WebSocket
    "WS_BASE_URL",
    "WS_USER_BASE_URL",
    "WS_CHANNEL_HEARTBEATS",
    "WS_CHANNEL_CANDLES",
    "WS_CHANNEL_MARKET_TRADES",
    "WS_CHANNEL_STATUS",
    "WS_CHANNEL_TICKER",
    "WS_CHANNEL_TICKER_BATCH",
    "WS_CHANNEL_LEVEL2",
    "WS_CHANNEL_USER",
    "WS_CHANNEL_FUTURES_BALANCE_SUMMARY",
    "WS_AUTH_CHANNELS",
    # Endpoints
    "ADVANCED_ENDPOINTS",
    "EXCHANGE_ENDPOINTS",
    "ENDPOINT_MAP",
]
