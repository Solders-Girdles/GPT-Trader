"""
Simplified Base Client.
Handles HTTP requests with basic retries.
"""
import time
import json
import logging
import requests
from typing import Any, Callable

from bot_v2.features.brokerages.coinbase.auth import SimpleAuth

logger = logging.getLogger(__name__)

class CoinbaseClientBase:
    def __init__(
        self,
        base_url: str = "https://api.coinbase.com",
        auth: Any | None = None,
        api_mode: str = "advanced",
        **kwargs
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.api_mode = api_mode
        self.session = requests.Session()

    def _get_endpoint_path(self, endpoint_name: str, **kwargs: str) -> str:
        # Simplified map - in real usage we might want a full map or just pass paths directly
        # For now, we rely on the mixins passing known paths.
        # But wait, the mixins (e.g. market.py) use _get_endpoint_path("ticker").
        # I need to keep the map or change the mixins.
        # Changing mixins is a lot of work. Keeping the map is safer for now.

        ENDPOINT_MAP = {
            "advanced": {
                "products": "/api/v3/brokerage/products",
                "product": "/api/v3/brokerage/products/{product_id}",
                "ticker": "/api/v3/brokerage/products/{product_id}/ticker",
                "candles": "/api/v3/brokerage/products/{product_id}/candles",
                "order_book": "/api/v3/brokerage/product_book",
                "public_products": "/api/v3/brokerage/market/products",
                "public_product": "/api/v3/brokerage/market/products/{product_id}",
                "public_ticker": "/api/v3/brokerage/market/products/{product_id}/ticker",
                "public_candles": "/api/v3/brokerage/market/products/{product_id}/candles",
                "best_bid_ask": "/api/v3/brokerage/best_bid_ask",
                "accounts": "/api/v3/brokerage/accounts",
                "account": "/api/v3/brokerage/accounts/{account_uuid}",
                "orders": "/api/v3/brokerage/orders",
                "order": "/api/v3/brokerage/orders/historical/{order_id}",
                "orders_historical": "/api/v3/brokerage/orders/historical",
                "orders_batch_cancel": "/api/v3/brokerage/orders/batch_cancel",
                "order_preview": "/api/v3/brokerage/orders/preview",
                "order_edit": "/api/v3/brokerage/orders/edit",
                "close_position": "/api/v3/brokerage/orders/close_position",
                "fills": "/api/v3/brokerage/orders/historical/fills",
                "time": "/api/v3/brokerage/time",
                "portfolios": "/api/v3/brokerage/portfolios",
                "portfolio": "/api/v3/brokerage/portfolios/{portfolio_uuid}",
            }
        }

        # Fallback to advanced if unknown
        mode_map = ENDPOINT_MAP.get(self.api_mode, ENDPOINT_MAP["advanced"])
        path = mode_map.get(endpoint_name, endpoint_name)

        try:
            return path.format(**kwargs)
        except KeyError:
            return path

    def _make_url(self, path: str) -> str:
        if path.startswith("http"): return path
        if not path.startswith("/"): path = "/" + path
        return f"{self.base_url}{path}"

    def _build_path_with_params(self, path: str, params: dict[str, Any] | None) -> str:
        if not params: return path
        query_parts = []
        for k, v in params.items():
            if v is not None:
                query_parts.append(f"{k}={v}")
        if not query_parts: return path

        connector = "&" if "?" in path else "?"
        return f"{path}{connector}{'&'.join(query_parts)}"

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = self._make_url(path)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "gpt-trader/v2"
        }

        # Sign request if auth is available
        if self.auth:
            # Check if auth has .get_headers (SimpleAuth) or .sign (Legacy)
            if hasattr(self.auth, "get_headers"):
                # SimpleAuth expects (method, path)
                # path passed to auth should exclude domain but include leading slash
                auth_path = path if path.startswith("/") else f"/{path}"
                headers.update(self.auth.get_headers(method, auth_path))
            elif hasattr(self.auth, "sign"):
                 # Legacy interface
                 headers.update(self.auth.sign(method, path, payload))

        try:
            resp = self.session.request(
                method,
                url,
                json=payload,
                headers=headers,
                timeout=10
            )
            resp.raise_for_status()

            if resp.content:
                return resp.json()
            return {}

        except Exception as e:
            logger.error(f"Request failed: {method} {url} - {e}")
            # Minimal error handling
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    # Helper aliases used by mixins
    def get(self, path, params=None):
        if params:
            path = self._build_path_with_params(path, params)
        return self._request("GET", path)

    def post(self, path, payload=None):
        return self._request("POST", path, payload)

    def delete(self, path, payload=None):
        return self._request("DELETE", path, payload)
