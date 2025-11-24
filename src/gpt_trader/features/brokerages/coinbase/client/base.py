"""
Simplified Base Client.
Handles HTTP requests with basic retries.
"""
import time
import json
import logging
import requests
from typing import Any, Callable

from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.errors import InvalidRequestError

logger = logging.getLogger(__name__)

class CoinbaseClientBase:
    def __init__(
        self,
        base_url: str = "https://api.coinbase.com",
        auth: Any | None = None,
        api_mode: str = "advanced",
        timeout: int = 30,
        api_version: str = "2024-10-24",
        rate_limit_per_minute: int = 100,
        enable_throttle: bool = True,
        enable_keep_alive: bool = True,
        **kwargs
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.api_mode = api_mode
        self.timeout = timeout
        self.api_version = api_version
        self.rate_limit_per_minute = rate_limit_per_minute
        self.enable_throttle = enable_throttle
        self.enable_keep_alive = enable_keep_alive
        self._is_cdp = hasattr(auth, "key_name") and auth.key_name.startswith("organizations/") if auth else False
        
        self.session = requests.Session()
        self._transport = None # For testing
        self._request_times = []

    def set_transport_for_testing(self, transport: Any) -> None:
        self._transport = transport

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
            },
            "exchange": {
                "products": "/products",
                "product": "/products/{product_id}",
                "accounts": "/accounts",
            }
        }

        if self.api_mode not in ENDPOINT_MAP:
             raise InvalidRequestError(f"Unknown API mode: {self.api_mode}")

        mode_map = ENDPOINT_MAP.get(self.api_mode, {})
        path = mode_map.get(endpoint_name)
        
        if path is None:
             # Check if it exists in advanced mode but not current mode
             if endpoint_name in ENDPOINT_MAP["advanced"] and self.api_mode != "advanced":
                 raise InvalidRequestError(f"Endpoint {endpoint_name} not available in {self.api_mode} mode")
             raise InvalidRequestError(f"Unknown endpoint: {endpoint_name}")

        try:
            return path.format(**kwargs)
        except KeyError:
            return path

    def _make_url(self, path: str) -> str:
        if path.startswith("http"): return path
        if not path.startswith("/"): path = "/" + path
        return f"{self.base_url}{path}"
    
    def _normalize_path(self, path: str) -> str:
        if path.startswith(self.base_url):
            path = path[len(self.base_url):]
        if path.startswith("/"):
            return path
        return path

    def _build_path_with_params(self, path: str, params: dict[str, Any] | None) -> str:
        if not params: return path
        query_parts = []
        for k, v in params.items():
            if v is not None:
                query_parts.append(f"{k}={v}")
        if not query_parts: return path

        connector = "&" if "?" in path else "?"
        return f"{path}{connector}{'&'.join(query_parts)}"

    def _check_rate_limit(self) -> None:
        if not self.enable_throttle:
            return

        now = time.time()
        # Remove requests older than 1 minute
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self.rate_limit_per_minute:
            logger.info(f"Rate limit reached ({len(self._request_times)}/{self.rate_limit_per_minute}). throttling...")
            sleep_time = 60 - (now - self._request_times[0]) + 1
            if sleep_time > 0:
                time.sleep(sleep_time)
            # After sleep, we can proceed (or check again, but simple sleep is ok)
            # Clean up again
            now = time.time()
            self._request_times = [t for t in self._request_times if now - t < 60]
        elif len(self._request_times) >= self.rate_limit_per_minute * 0.8:
             logger.warning("Approaching rate limit: %d/%d requests in last minute", len(self._request_times), self.rate_limit_per_minute)

        self._request_times.append(now)

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        self._check_rate_limit()
        url = self._make_url(path)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "gpt-trader/v2",
            "CB-VERSION": self.api_version
        }

        # Sign request if auth is available
        if self.auth:
            # Check if auth has .get_headers (SimpleAuth) or .sign (Legacy)
            if hasattr(self.auth, "get_headers"):
                # SimpleAuth expects (method, path)
                # path passed to auth should exclude domain but include leading slash
                auth_path = path if path.startswith("/") else f"/{path}"
                # normalize path for signing if it's a full url (though _request takes path usually)
                if path.startswith("http"):
                     auth_path = "/" + path.split("/", 3)[-1]
                
                headers.update(self.auth.get_headers(method, auth_path))
            elif hasattr(self.auth, "sign"):
                 # Legacy interface
                 headers.update(self.auth.sign(method, path, payload))
        
        # Add correlation ID if available
        # (Simplified: assuming we might add it later or it's handled elsewhere, 
        # but tests expect it if patched)
        # For now, rely on tests patching get_correlation_id if needed or add it here if imported.
        from gpt_trader.utilities.logging_patterns import get_correlation_id
        corr_id = get_correlation_id()
        if corr_id:
            headers["X-Correlation-Id"] = corr_id

        if self._transport:
             # Use mock transport
             status, resp_headers, text = self._transport(method, url, headers, json.dumps(payload) if payload else None, self.timeout)
             # Mimic requests response
             class MockResponse:
                 def __init__(self, status_code, text, headers):
                     self.status_code = status_code
                     self.text = text
                     self.content = text.encode()
                     self.headers = headers
                 def json(self):
                     return json.loads(self.text)
                 def raise_for_status(self):
                     if 400 <= self.status_code < 600:
                         # Simplified error raising
                         if self.status_code == 429:
                             raise Exception("Rate limit exceeded") # Should be requests.HTTPError
                         msg = "Error"
                         try:
                             data = self.json()
                             msg = data.get("message", msg)
                         except: pass
                         raise InvalidRequestError(msg)

             resp = MockResponse(status, text, resp_headers)
             resp.raise_for_status()
             if resp.content:
                return resp.json()
             return {}

        try:
            # Retry logic (simplified)
            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    resp = self.session.request(
                        method,
                        url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout
                    )
                    
                    if resp.status_code == 429:
                        retry_after = int(resp.headers.get("retry-after", 1))
                        time.sleep(retry_after)
                        continue
                        
                    if 500 <= resp.status_code < 600:
                        if attempt < max_retries:
                            time.sleep(0.5 * (2 ** attempt))
                            continue
                    
                    if 400 <= resp.status_code < 500:
                         # Map 400s to InvalidRequestError
                         try:
                             data = resp.json()
                             msg = data.get("message", "Bad request")
                         except:
                             msg = resp.text
                         raise InvalidRequestError(msg)

                    resp.raise_for_status()

                    if resp.content:
                        try:
                            return resp.json()
                        except json.JSONDecodeError:
                            return {"raw": resp.text}
                    return {}
                except (requests.ConnectionError, requests.Timeout):
                    if attempt < max_retries:
                        time.sleep(0.5 * (2 ** attempt))
                        continue
                    raise

        except Exception as e:
            logger.error(f"Request failed: {method} {url} - {e}")
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