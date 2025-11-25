"""
Simplified Base Client.
Handles HTTP requests with basic retries.
"""

import json
import time
from typing import Any

import requests

from gpt_trader.config.constants import (
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_RATE_LIMIT_PER_MINUTE,
    MAX_HTTP_RETRIES,
    RATE_LIMIT_WARNING_THRESHOLD,
    RETRY_BACKOFF_MULTIPLIER,
    RETRY_BASE_DELAY,
)
from gpt_trader.features.brokerages.coinbase.errors import InvalidRequestError, map_http_error
from gpt_trader.utilities.logging_patterns import get_correlation_id, get_logger

logger = get_logger(__name__, component="coinbase_client")


class CoinbaseClientBase:
    def __init__(
        self,
        base_url: str = "https://api.coinbase.com",
        auth: Any | None = None,
        api_mode: str = "advanced",
        timeout: int | None = None,
        api_version: str = "2024-10-24",
        rate_limit_per_minute: int | None = None,
        enable_throttle: bool = True,
        enable_keep_alive: bool = True,
        **kwargs,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.api_mode = api_mode
        self.timeout = timeout if timeout is not None else DEFAULT_HTTP_TIMEOUT
        self.api_version = api_version
        self.rate_limit_per_minute = (
            rate_limit_per_minute
            if rate_limit_per_minute is not None
            else DEFAULT_RATE_LIMIT_PER_MINUTE
        )
        self.enable_throttle = enable_throttle
        self.enable_keep_alive = enable_keep_alive
        self._is_cdp = (
            hasattr(auth, "key_name") and auth.key_name.startswith("organizations/")
            if auth
            else False
        )

        self.session = requests.Session()
        self._transport = None  # For testing
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
                "key_permissions": "/api/v3/brokerage/key_permissions",
                # INTX Endpoints
                "intx_portfolio": "/api/v3/brokerage/intx/portfolio/{portfolio_uuid}",
                "intx_allocate": "/api/v3/brokerage/intx/allocate",
                "intx_balances": "/api/v3/brokerage/intx/balances/{portfolio_uuid}",
                "intx_positions": "/api/v3/brokerage/intx/positions/{portfolio_uuid}",
                "intx_position": "/api/v3/brokerage/intx/positions/{portfolio_uuid}/{symbol}",
                "intx_multi_asset_collateral": "/api/v3/brokerage/intx/multi_asset_collateral",
                "portfolios": "/api/v3/brokerage/portfolios",
                "portfolio": "/api/v3/brokerage/portfolios/{portfolio_uuid}",
                "fees": "/api/v3/brokerage/fees",
                "limits": "/api/v3/brokerage/limits",
                "payment_methods": "/api/v3/brokerage/payment_methods",
                "payment_method": "/api/v3/brokerage/payment_methods/{payment_method_id}",
                "convert_trade": "/api/v3/brokerage/convert/trade/{trade_id}",
                "convert_quote": "/api/v3/brokerage/convert/quote",
                # CFM Endpoints
                "cfm_balance_summary": "/api/v3/brokerage/cfm/balance_summary",
                "cfm_positions": "/api/v3/brokerage/cfm/positions",
                "cfm_position": "/api/v3/brokerage/cfm/positions/{product_id}",
                "cfm_intraday_current_margin_window": "/api/v3/brokerage/cfm/intraday/current_margin_window",  # Corrected
                "cfm_intraday_margin_setting": "/api/v3/brokerage/cfm/intraday/margin_setting",  # Corrected
                "cfm_sweeps": "/api/v3/brokerage/cfm/sweeps",
                "cfm_schedule_sweep": "/api/v3/brokerage/cfm/sweeps/schedule",  # Added
                # Portfolio endpoints
                "portfolio_breakdown": "/api/v3/brokerage/portfolios/{portfolio_uuid}/breakdown",
                "move_funds": "/api/v3/brokerage/portfolios/move_funds",  # Corrected
            },
            "exchange": {
                "products": "/products",
                "product": "/products/{product_id}",
                "accounts": "/accounts",
                "order_book": "/products/{product_id}/book",
            },
        }

        if self.api_mode not in ENDPOINT_MAP:
            raise InvalidRequestError(f"Unknown API mode: {self.api_mode}")

        mode_map = ENDPOINT_MAP.get(self.api_mode, {})
        path = mode_map.get(endpoint_name)

        if path is None:
            # Check if it exists in advanced mode but not current mode
            if endpoint_name in ENDPOINT_MAP["advanced"] and self.api_mode != "advanced":
                raise InvalidRequestError(
                    f"Endpoint {endpoint_name} not available in {self.api_mode} mode"
                )
            raise InvalidRequestError(f"Unknown endpoint: {endpoint_name}")

        try:
            return path.format(**kwargs)
        except KeyError:
            return path

    def _make_url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _normalize_path(self, path: str) -> str:
        if path.startswith(self.base_url):
            path = path[len(self.base_url) :]
        if path.startswith("/"):
            return path
        return path

    def _build_path_with_params(self, path: str, params: dict[str, Any] | None) -> str:
        if not params:
            return path
        query_parts = []
        for k, v in params.items():
            if v is not None:
                query_parts.append(f"{k}={v}")
        if not query_parts:
            return path

        connector = "&" if "?" in path else "?"
        return f"{path}{connector}{'&'.join(query_parts)}"

    def paginate(
        self,
        path: str,
        params: dict = None,
        pagination_key: str = None,
        cursor_param: str = "cursor",
        cursor_field: str = None,
    ) -> Any:
        """
        Generator that yields items from paginated endpoints.
        """
        params = params or {}
        cursor = params.get(cursor_param)
        has_more = True

        while has_more:
            current_params = params.copy()
            if cursor:
                current_params[cursor_param] = cursor

            response = self.get(path, current_params)

            # Extract data list
            data = response
            if pagination_key and isinstance(response, dict):
                data = response.get(pagination_key, [])

            if isinstance(data, list):
                yield from data
            elif data:
                # Yield single item if not a list (rare for pagination but possible)
                yield data

            # Update cursor
            if isinstance(response, dict):
                # Coinbase APIs vary: 'cursor', 'next_cursor', 'pagination' dict
                next_cursor = None

                if cursor_field:
                    next_cursor = response.get(cursor_field)
                else:
                    next_cursor = response.get("cursor") or response.get("next_cursor")

                # Some APIs return a 'pagination' object
                if not next_cursor and "pagination" in response:
                    # If cursor_field is inside pagination, we assume standard structure
                    # or user must handle complex extraction.
                    # For now, fallback to standard pagination.next_cursor
                    next_cursor = response["pagination"].get("next_cursor")

                if next_cursor and next_cursor != cursor:
                    cursor = next_cursor
                else:
                    has_more = False
            else:
                has_more = False

    def _check_rate_limit(self) -> None:
        if not self.enable_throttle:
            return

        now = time.time()
        # Remove requests older than 1 minute
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self.rate_limit_per_minute:
            logger.info(
                f"Rate limit reached ({len(self._request_times)}/{self.rate_limit_per_minute}). throttling..."
            )
            sleep_time = 60 - (now - self._request_times[0]) + 1
            if sleep_time > 0:
                time.sleep(sleep_time)
            # After sleep, we can proceed (or check again, but simple sleep is ok)
            # Clean up again
            now = time.time()
            self._request_times = [t for t in self._request_times if now - t < 60]
        elif len(self._request_times) >= self.rate_limit_per_minute * RATE_LIMIT_WARNING_THRESHOLD:
            logger.warning(
                "Approaching rate limit: %d/%d requests in last minute",
                len(self._request_times),
                self.rate_limit_per_minute,
            )

        self._request_times.append(now)

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        self._check_rate_limit()
        url = self._make_url(path)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "gpt-trader/v2",
            "CB-VERSION": self.api_version,
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

                if payload:
                    headers.update(self.auth.get_headers(method, auth_path, payload))
                else:
                    headers.update(self.auth.get_headers(method, auth_path))
            elif hasattr(self.auth, "sign"):
                # Legacy interface
                headers.update(self.auth.sign(method, path, payload))

        # Add correlation ID if available
        corr_id = get_correlation_id()
        if corr_id:
            headers["X-Correlation-Id"] = corr_id

        def perform_request():
            if self._transport:
                # Use mock transport
                status, resp_headers, text = self._transport(
                    method, url, headers, json.dumps(payload) if payload else None, self.timeout
                )
                resp = requests.Response()
                resp.status_code = status
                resp._content = text.encode() if text else b""
                resp.headers = resp_headers
                return resp
            else:
                return self.session.request(
                    method, url, json=payload, headers=headers, timeout=self.timeout
                )

        try:
            # Retry logic with configurable parameters
            max_retries = MAX_HTTP_RETRIES
            for attempt in range(max_retries + 1):
                try:
                    resp = perform_request()

                    if resp.status_code == 429:
                        try:
                            retry_after = float(resp.headers.get("retry-after", 1))
                        except ValueError:
                            retry_after = 1.0
                        time.sleep(retry_after)
                        continue

                    if 500 <= resp.status_code < 600:
                        if attempt < max_retries:
                            time.sleep(RETRY_BASE_DELAY * (RETRY_BACKOFF_MULTIPLIER**attempt))
                            continue

                    if 400 <= resp.status_code < 500:
                        # Map 400s to specific errors
                        try:
                            data = resp.json()
                            msg = data.get("message", "Bad request")
                            code = data.get("error")
                        except (json.JSONDecodeError, ValueError, KeyError):
                            msg = resp.text
                            code = None

                        if resp.status_code == 400:
                            raise InvalidRequestError(msg)
                        raise map_http_error(resp.status_code, code, msg)

                    resp.raise_for_status()

                    if resp.content:
                        try:
                            return resp.json()
                        except ValueError:
                            return {"raw": resp.text}
                    return {}

                except requests.exceptions.HTTPError as e:
                    # Catch HTTPError from raise_for_status (e.g. 500s)
                    try:
                        data = e.response.json()
                        msg = data.get("message", str(e))
                        code = data.get("error")
                    except (json.JSONDecodeError, ValueError, KeyError, AttributeError):
                        msg = str(e)
                        code = None
                    raise map_http_error(e.response.status_code, code, msg)

                except (requests.ConnectionError, requests.Timeout, ConnectionError):
                    if attempt < max_retries:
                        time.sleep(RETRY_BASE_DELAY * (RETRY_BACKOFF_MULTIPLIER**attempt))
                        continue
                    raise

            # If loop finishes without return (e.g. all 429s handled but retries exhausted)
            if "resp" in locals() and resp is not None and resp.status_code == 429:
                raise map_http_error(429, "rate_limited", "Rate limit exceeded (rate_limited)")

            return {}

        except Exception as e:
            # Catch other errors during request performance (like map_http_error raises)
            raise e

        except Exception as e:
            logger.error(f"Request failed: {method} {url} - {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    # Helper aliases used by mixins
    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if params:
            path = self._build_path_with_params(path, params)
        return self._request("GET", path)

    def post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, payload)

    def delete(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        return self._request("DELETE", path, payload)
