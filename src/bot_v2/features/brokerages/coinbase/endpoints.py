"""
Endpoint registry for Coinbase Advanced Trade (scaffold, verify against docs).

This acts as a single source of truth for routes we intend to support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Method = Literal["GET", "POST", "DELETE"]


@dataclass(frozen=True)
class Endpoint:
    name: str
    method: Method
    path: str  # path relative to base, e.g., "/api/v3/brokerage/products"
    auth: bool
    notes: str = ""


# Core set based on Coinbase Advanced Trade public docs (to verify)
ENDPOINTS: list[Endpoint] = [
    # Products and Market Data
    Endpoint("list_products", "GET", "/api/v3/brokerage/products", False, "List tradable products"),
    Endpoint(
        "get_product",
        "GET",
        "/api/v3/brokerage/products/{product_id}",
        False,
        "Get product details",
    ),
    Endpoint(
        "get_product_ticker",
        "GET",
        "/api/v3/brokerage/products/{product_id}/ticker",
        False,
        "Best bid/ask and last",
    ),
    Endpoint(
        "get_product_candles",
        "GET",
        "/api/v3/brokerage/products/{product_id}/candles",
        False,
        "OHLCV candles",
    ),
    Endpoint(
        "get_product_book",
        "GET",
        "/api/v3/brokerage/product_book",
        False,
        "Order book (levels) - aggregated",
    ),
    # Public market namespace variants
    Endpoint(
        "list_public_products",
        "GET",
        "/api/v3/brokerage/market/products",
        False,
        "List products (public market namespace)",
    ),
    Endpoint(
        "get_public_product",
        "GET",
        "/api/v3/brokerage/market/products/{product_id}",
        False,
        "Product details (public market)",
    ),
    Endpoint(
        "get_public_product_ticker",
        "GET",
        "/api/v3/brokerage/market/products/{product_id}/ticker",
        False,
        "Ticker (public market)",
    ),
    Endpoint(
        "get_public_product_candles",
        "GET",
        "/api/v3/brokerage/market/products/{product_id}/candles",
        False,
        "Candles (public market)",
    ),
    Endpoint(
        "get_public_product_book",
        "GET",
        "/api/v3/brokerage/market/product_book",
        False,
        "Order book (public market)",
    ),
    Endpoint(
        "get_best_bid_ask",
        "GET",
        "/api/v3/brokerage/best_bid_ask",
        False,
        "Best bid/ask for products",
    ),
    # Accounts
    Endpoint(
        "list_accounts", "GET", "/api/v3/brokerage/accounts", True, "List trading accounts/balances"
    ),
    Endpoint(
        "get_account", "GET", "/api/v3/brokerage/accounts/{account_uuid}", True, "Account detail"
    ),
    Endpoint("get_time", "GET", "/api/v3/brokerage/time", False, "Server time"),
    Endpoint(
        "get_key_permissions",
        "GET",
        "/api/v3/brokerage/key_permissions",
        True,
        "API key permissions",
    ),
    # Orders
    Endpoint("place_order", "POST", "/api/v3/brokerage/orders", True, "Create order"),
    Endpoint("preview_order", "POST", "/api/v3/brokerage/orders/preview", True, "Preview order"),
    Endpoint(
        "edit_order_preview",
        "POST",
        "/api/v3/brokerage/orders/edit_preview",
        True,
        "Preview order edit",
    ),
    Endpoint("edit_order", "POST", "/api/v3/brokerage/orders/edit", True, "Edit order"),
    Endpoint(
        "close_position",
        "POST",
        "/api/v3/brokerage/orders/close_position",
        True,
        "Close derivative position",
    ),
    Endpoint(
        "cancel_orders",
        "POST",
        "/api/v3/brokerage/orders/batch_cancel",
        True,
        "Batch cancel by IDs",
    ),
    Endpoint(
        "get_order_historical",
        "GET",
        "/api/v3/brokerage/orders/historical/{order_id}",
        True,
        "Order status",
    ),
    Endpoint(
        "list_orders_historical",
        "GET",
        "/api/v3/brokerage/orders/historical",
        True,
        "Historical orders",
    ),
    Endpoint(
        "list_orders_historical_batch",
        "GET",
        "/api/v3/brokerage/orders/historical/batch",
        True,
        "Batch order lookup",
    ),
    Endpoint("list_fills", "GET", "/api/v3/brokerage/orders/historical/fills", True, "Trade fills"),
    # Fees and limits
    Endpoint("get_fees", "GET", "/api/v3/brokerage/fees", True, "Trading and funding fees"),
    Endpoint("get_limits", "GET", "/api/v3/brokerage/limits", True, "Trading limits"),
    Endpoint(
        "get_transaction_summary",
        "GET",
        "/api/v3/brokerage/transaction_summary",
        True,
        "Transaction summary",
    ),
    # Conversions (if available)
    Endpoint(
        "convert_quote", "POST", "/api/v3/brokerage/convert/quote", True, "Create convert quote"
    ),
    Endpoint(
        "get_convert_trade",
        "GET",
        "/api/v3/brokerage/convert/trade/{trade_id}",
        True,
        "Get convert trade status",
    ),
    # Payment methods
    Endpoint(
        "list_payment_methods",
        "GET",
        "/api/v3/brokerage/payment_methods",
        True,
        "List payment methods",
    ),
    Endpoint(
        "get_payment_method",
        "GET",
        "/api/v3/brokerage/payment_methods/{payment_method_id}",
        True,
        "Payment method detail",
    ),
    # Portfolios (institutional/advanced)
    Endpoint("list_portfolios", "GET", "/api/v3/brokerage/portfolios", True, "List portfolios"),
    Endpoint(
        "get_portfolio",
        "GET",
        "/api/v3/brokerage/portfolios/{portfolio_uuid}",
        True,
        "Get portfolio",
    ),
    Endpoint(
        "move_funds",
        "POST",
        "/api/v3/brokerage/portfolios/move_funds",
        True,
        "Move funds between portfolios",
    ),
    # INTX endpoints (institutional)
    Endpoint("intx_allocate", "POST", "/api/v3/brokerage/intx/allocate", True, "INTX allocation"),
    Endpoint(
        "intx_balances",
        "GET",
        "/api/v3/brokerage/intx/balances/{portfolio_uuid}",
        True,
        "INTX balances",
    ),
    Endpoint(
        "intx_portfolio",
        "GET",
        "/api/v3/brokerage/intx/portfolio/{portfolio_uuid}",
        True,
        "INTX portfolio",
    ),
    Endpoint(
        "intx_positions",
        "GET",
        "/api/v3/brokerage/intx/positions/{portfolio_uuid}",
        True,
        "INTX positions",
    ),
    Endpoint(
        "intx_position",
        "GET",
        "/api/v3/brokerage/intx/positions/{portfolio_uuid}/{symbol}",
        True,
        "INTX position detail",
    ),
    Endpoint(
        "intx_multi_asset_collateral",
        "GET",
        "/api/v3/brokerage/intx/multi_asset_collateral",
        True,
        "INTX MAC",
    ),
    # CFM endpoints (cross-futures margin / derivatives)
    Endpoint(
        "cfm_balance_summary",
        "GET",
        "/api/v3/brokerage/cfm/balance_summary",
        True,
        "CFM balance summary",
    ),
    Endpoint("cfm_positions", "GET", "/api/v3/brokerage/cfm/positions", True, "CFM positions"),
    Endpoint(
        "cfm_position",
        "GET",
        "/api/v3/brokerage/cfm/positions/{product_id}",
        True,
        "CFM position detail",
    ),
    Endpoint("cfm_sweeps", "GET", "/api/v3/brokerage/cfm/sweeps", True, "CFM sweeps"),
    Endpoint(
        "cfm_sweeps_schedule",
        "GET",
        "/api/v3/brokerage/cfm/sweeps/schedule",
        True,
        "CFM sweep schedule",
    ),
    Endpoint(
        "cfm_intraday_current_margin_window",
        "GET",
        "/api/v3/brokerage/cfm/intraday/current_margin_window",
        True,
        "CFM current margin window",
    ),
    Endpoint(
        "cfm_intraday_margin_setting",
        "POST",
        "/api/v3/brokerage/cfm/intraday/margin_setting",
        True,
        "CFM set intraday margin setting (verify method)",
    ),
]


class CoinbaseEndpoints:
    """Mode-aware endpoint builder for Coinbase APIs."""

    def __init__(
        self,
        mode: Literal["advanced", "exchange"] = "advanced",
        sandbox: bool = False,
        enable_derivatives: bool = True,
    ) -> None:
        self.mode = mode
        self.sandbox = sandbox
        self.enable_derivatives = enable_derivatives

        # Resolve REST and WS base URLs based on mode + sandbox
        if self.mode == "exchange":
            # Legacy Exchange (Pro) API — supports sandbox
            if sandbox:
                self.base_url = "https://api-public.sandbox.exchange.coinbase.com"
                self.ws_url = "wss://ws-feed-public.sandbox.exchange.coinbase.com"
            else:
                self.base_url = "https://api.exchange.coinbase.com"
                self.ws_url = "wss://ws-feed.exchange.coinbase.com"
        else:
            # Advanced Trade v3 — no public sandbox; always production endpoints
            if sandbox:
                # Keep using production base but callers should be aware this is not a true sandbox
                # This mirrors broker_factory's warning behavior.
                pass
            self.base_url = "https://api.coinbase.com"
            self.ws_url = "wss://advanced-trade-ws.coinbase.com"

    def list_products(self) -> str:
        """List tradable products endpoint."""
        if self.mode == "advanced":
            return f"{self.base_url}/api/v3/brokerage/products"
        else:
            return f"{self.base_url}/products"

    def get_product(self, product_id: str) -> str:
        """Get product details endpoint."""
        if self.mode == "advanced":
            return f"{self.base_url}/api/v3/brokerage/products/{product_id}"
        else:
            return f"{self.base_url}/products/{product_id}"

    def place_order(self) -> str:
        """Place order endpoint."""
        if self.mode == "advanced":
            return f"{self.base_url}/api/v3/brokerage/orders"
        else:
            return f"{self.base_url}/orders"

    def list_orders(self) -> str:
        """List orders endpoint."""
        if self.mode == "advanced":
            return f"{self.base_url}/api/v3/brokerage/orders/historical"
        else:
            return f"{self.base_url}/orders"

    def get_order(self, order_id: str) -> str:
        """Get single order endpoint."""
        if self.mode == "advanced":
            return f"{self.base_url}/api/v3/brokerage/orders/historical/{order_id}"
        else:
            return f"{self.base_url}/orders/{order_id}"

    def cancel_orders(self) -> str:
        """Cancel orders endpoint."""
        if self.mode == "advanced":
            return f"{self.base_url}/api/v3/brokerage/orders/batch_cancel"
        else:
            return f"{self.base_url}/orders/cancel"

    def list_accounts(self) -> str:
        """List accounts endpoint."""
        if self.mode == "advanced":
            return f"{self.base_url}/api/v3/brokerage/accounts"
        else:
            return f"{self.base_url}/accounts"

    def list_positions(self) -> str:
        """List positions endpoint (CFM derivatives)."""
        if self.mode == "advanced" and self.enable_derivatives:
            return f"{self.base_url}/api/v3/brokerage/cfm/positions"
        else:
            # Fallback for non-derivatives mode
            return f"{self.base_url}/api/v3/brokerage/accounts"

    def get_position(self, product_id: str) -> str:
        """Get single position endpoint for derivatives."""
        if self.mode == "advanced" and self.enable_derivatives:
            return f"{self.base_url}/api/v3/brokerage/cfm/positions/{product_id}"
        raise NotImplementedError("Position lookup not supported for this mode")

    def close_position(self) -> str:
        """Close position endpoint."""
        if self.mode == "advanced":
            return f"{self.base_url}/api/v3/brokerage/orders/close_position"
        else:
            return f"{self.base_url}/orders"

    def get_fills(self) -> str:
        """Get fills endpoint."""
        if self.mode == "advanced":
            return f"{self.base_url}/api/v3/brokerage/orders/historical/fills"
        else:
            return f"{self.base_url}/fills"

    def websocket_url(self) -> str:
        """WebSocket URL for market data."""
        return self.ws_url

    def supports_derivatives(self) -> bool:
        """Return True if derivatives (CFM/perpetuals) endpoints are available.

        Advanced Trade mode with derivatives enabled supports CFM endpoints. The
        legacy Exchange mode does not expose these.
        """
        return self.mode == "advanced" and bool(self.enable_derivatives)


def get_perps_symbols() -> set:
    """Get expected perpetuals symbols for validation."""
    return {"BTC-PERP", "ETH-PERP", "SOL-PERP", "XRP-PERP"}
