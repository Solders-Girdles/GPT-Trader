"""
REST client for Coinbase Advanced Trade and derivatives.

Implements signing, retries/backoff, error mapping, and a pluggable transport
to enable unit testing without network access.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from bot_v2.features.monitor import get_logger

from ....config import get_config
from ..core.interfaces import BrokerageError
from .cdp_auth import CDPAuth
from .cdp_auth_v2 import CDPAuthV2
from .errors import InvalidRequestError, NotFoundError, map_http_error

_ul: Any
_ue: Any
try:
    # Prefer stdlib; avoid adding external deps
    import urllib.error as _ue
    import urllib.request as _ul
except Exception:  # pragma: no cover
    _ul = None
    _ue = None

logger = logging.getLogger(__name__)


@dataclass
class CoinbaseAuth:
    api_key: str
    api_secret: str
    passphrase: str | None = None
    key_version: str | None = None  # reserved for future versioned keys
    # api_mode: 'advanced' for Advanced Trade v3 (api.coinbase.com)
    #           'exchange' for legacy Exchange/Pro (exchange.coinbase.com or sandbox)
    #           None (auto) will infer from presence of passphrase (legacy behavior)
    api_mode: str | None = None

    def sign(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, str]:
        """Return headers per Coinbase HMAC spec (supports both Advanced Trade and Exchange API).

        Prehash: <timestamp><method><path><body_json_or_empty>
        Signature: base64(HMAC_SHA256(base64_decode(api_secret), prehash))
        """
        import base64

        # Determine API mode
        mode = (self.api_mode or "").lower()
        is_advanced = mode == "advanced"
        is_exchange = mode == "exchange"
        if not (is_advanced or is_exchange):
            # Auto-detect: presence of passphrase historically implies Exchange API
            is_exchange = bool(self.passphrase)
            is_advanced = not is_exchange

        # Use fractional timestamp for Exchange API, integer seconds for Advanced Trade
        timestamp = str(time.time()) if is_exchange else str(int(time.time()))

        body_str = json.dumps(body, separators=(",", ":")) if body else ""
        prehash = (timestamp + method.upper() + path + body_str).encode()

        # Both APIs use base64-encoded secrets
        try:
            key = base64.b64decode(self.api_secret, validate=True)
            if not key:
                key = self.api_secret.encode()
        except Exception:
            key = self.api_secret.encode()

        digest = hmac.new(key, prehash, hashlib.sha256).digest()
        signature = base64.b64encode(digest).decode()

        headers = {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }
        # Only include passphrase for Exchange API
        if is_exchange and self.passphrase:
            headers["CB-ACCESS-PASSPHRASE"] = self.passphrase
        if self.key_version:
            headers["CB-ACCESS-KEY-VERSION"] = self.key_version
        return headers


class CoinbaseClient:
    def __init__(
        self,
        base_url: str,
        auth: CoinbaseAuth | CDPAuth | CDPAuthV2 | None = None,
        timeout: int = 30,
        api_version: str = "2024-10-24",
        rate_limit_per_minute: int = 100,
        enable_throttle: bool = True,
        api_mode: str = "advanced",
        enable_keep_alive: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.timeout = timeout
        self.api_version = api_version  # CB-VERSION header value
        self.api_mode = api_mode  # "advanced" or "exchange"
        # Pluggable transport: (method, url, headers, body_bytes, timeout) -> (status, headers, text)
        self._transport: Callable[[str, str, dict[str, str], bytes | None, int], tuple[int, dict[str, str], str]] = self._urllib_transport
        # Enhanced rate limit tracking with throttling
        self._request_count = 0
        self._request_window_start = time.time()
        self._request_times: list[float] = []  # Track request timestamps for sliding window
        self.rate_limit_per_minute = rate_limit_per_minute
        self.enable_throttle = enable_throttle
        # Detect auth type
        self._is_cdp = isinstance(auth, CDPAuth | CDPAuthV2)
        # Connection reuse: shared opener for keep-alive
        self.enable_keep_alive = enable_keep_alive
        self._opener: Any | None = None
        if enable_keep_alive:
            self._setup_opener()

    def set_transport_for_testing(
        self,
        transport: Callable[[str, str, dict[str, str], bytes | None, int], tuple[int, dict[str, str], str]],
    ) -> None:
        self._transport = transport

    def _setup_opener(self) -> None:
        """Setup a shared opener with keep-alive for connection reuse."""
        if _ul is None:  # pragma: no cover
            return

        # Create an opener with HTTP handler that supports keep-alive
        http_handler = _ul.HTTPHandler()
        https_handler = _ul.HTTPSHandler()
        self._opener = _ul.build_opener(http_handler, https_handler)

    def _get_endpoint_path(self, endpoint_name: str, **kwargs: str) -> str:
        """Get the correct endpoint path based on API mode."""
        from .errors import InvalidRequestError

        # Define endpoint mappings for each mode
        ENDPOINT_MAP = {
            'advanced': {
                # Market data
                'products': '/api/v3/brokerage/market/products',
                'product': '/api/v3/brokerage/market/products/{product_id}',
                'ticker': '/api/v3/brokerage/market/products/{product_id}/ticker',
                'candles': '/api/v3/brokerage/market/products/{product_id}/candles',
                'order_book': '/api/v3/brokerage/market/product_book',
                'best_bid_ask': '/api/v3/brokerage/best_bid_ask',
                # Account
                'accounts': '/api/v3/brokerage/accounts',
                'account': '/api/v3/brokerage/accounts/{account_uuid}',
                # Orders
                'orders': '/api/v3/brokerage/orders',
                'order': '/api/v3/brokerage/orders/historical/{order_id}',
                'orders_historical': '/api/v3/brokerage/orders/historical',
                'orders_batch_cancel': '/api/v3/brokerage/orders/batch_cancel',
                'order_preview': '/api/v3/brokerage/orders/preview',
                'order_edit': '/api/v3/brokerage/orders/edit',
                'close_position': '/api/v3/brokerage/orders/close_position',
                # Fills
                'fills': '/api/v3/brokerage/orders/historical/fills',
                # System
                'time': '/api/v3/brokerage/time',
                'fees': '/api/v3/brokerage/fees',
                'limits': '/api/v3/brokerage/limits',
                'key_permissions': '/api/v3/brokerage/key_permissions',
                'transaction_summary': '/api/v3/brokerage/transaction_summary',
                # Portfolios
                'portfolios': '/api/v3/brokerage/portfolios',
                'portfolio': '/api/v3/brokerage/portfolios/{portfolio_uuid}',
                'portfolio_breakdown': '/api/v3/brokerage/portfolios/{portfolio_uuid}/breakdown',
                # Convert
                'convert_quote': '/api/v3/brokerage/convert/quote',
                'convert_trade': '/api/v3/brokerage/convert/trade/{trade_id}',
                # Payment
                'payment_methods': '/api/v3/brokerage/payment_methods',
                'payment_method': '/api/v3/brokerage/payment_methods/{payment_method_id}',
                # Order management (advanced features)
                'orders_batch': '/api/v3/brokerage/orders/historical/batch',
                'order_edit_preview': '/api/v3/brokerage/orders/edit_preview',
                'move_funds': '/api/v3/brokerage/portfolios/move_funds',
                # INTX (derivatives)
                'intx_allocate': '/api/v3/brokerage/intx/allocate',
                'intx_balances': '/api/v3/brokerage/intx/balances/{portfolio_uuid}',
                'intx_portfolio': '/api/v3/brokerage/intx/portfolio/{portfolio_uuid}',
                'intx_positions': '/api/v3/brokerage/intx/positions/{portfolio_uuid}',
                'intx_position': '/api/v3/brokerage/intx/positions/{portfolio_uuid}/{symbol}',
                'intx_multi_asset_collateral': '/api/v3/brokerage/intx/multi_asset_collateral',
                # CFM
                'cfm_balance_summary': '/api/v3/brokerage/cfm/balance_summary',
                'cfm_positions': '/api/v3/brokerage/cfm/positions',
                'cfm_position': '/api/v3/brokerage/cfm/positions/{product_id}',
                'cfm_sweeps': '/api/v3/brokerage/cfm/sweeps',
                'cfm_schedule_sweep': '/api/v3/brokerage/cfm/sweeps/schedule',
                'cfm_intraday_margin_window': '/api/v3/brokerage/cfm/intraday/current_margin_window',
                'cfm_intraday_margin_setting': '/api/v3/brokerage/cfm/intraday/margin_setting',
            },
            'exchange': {
                # Market data
                'products': '/products',
                'product': '/products/{product_id}',
                'ticker': '/products/{product_id}/ticker',
                'candles': '/products/{product_id}/candles',
                'order_book': '/products/{product_id}/book',
                'trades': '/products/{product_id}/trades',
                # Account
                'accounts': '/accounts',
                'account': '/accounts/{account_id}',
                # Orders
                'orders': '/orders',
                'order': '/orders/{order_id}',
                'cancel_order': '/orders/{order_id}',
                # Fills
                'fills': '/fills',
                # System
                'time': '/time',
                'fees': '/fees',
                # Note: Many advanced features not available in exchange mode
            }
        }

        if self.api_mode not in ENDPOINT_MAP:
            raise InvalidRequestError(f"Unknown API mode: {self.api_mode}")

        mode_endpoints = ENDPOINT_MAP[self.api_mode]

        if endpoint_name not in mode_endpoints:
            available_modes = [mode for mode in ENDPOINT_MAP if endpoint_name in ENDPOINT_MAP[mode]]
            if available_modes:
                raise InvalidRequestError(
                    f"Endpoint '{endpoint_name}' not available in {self.api_mode} mode. "
                    f"Available in: {', '.join(available_modes)}. "
                    f"Set COINBASE_API_MODE={available_modes[0]} to use this endpoint."
                )
            else:
                raise InvalidRequestError(f"Unknown endpoint: {endpoint_name}")

        path = mode_endpoints[endpoint_name]

        # Format path with provided kwargs
        if kwargs:
            path = path.format(**kwargs)

        return path

    def _make_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _build_path_with_params(self, path: str, params: dict[str, Any] | None) -> str:
        if not params:
            return path
        # Simple querystring builder (no URL-encoding for simplicity in tests)
        qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        sep = "?" if "?" not in path else "&"
        return f"{path}{sep}{qs}" if qs else path

    def _normalize_path(self, url_or_path: str) -> str:
        """Convert an absolute broker URL into a relative path for signing."""
        if url_or_path.startswith(self.base_url):
            path = url_or_path[len(self.base_url):]
        else:
            path = url_or_path
        if not path.startswith("/"):
            path = "/" + path
        return path

    def get(self, url_or_path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Backwards-compatible GET helper that mirrors the old adapter API."""
        path = self._normalize_path(url_or_path)
        path = self._build_path_with_params(path, params)
        return self._request("GET", path)

    def post(self, url_or_path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        path = self._normalize_path(url_or_path)
        return self._request("POST", path, payload)

    def delete(self, url_or_path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        path = self._normalize_path(url_or_path)
        return self._request("DELETE", path, payload)

    def _urllib_transport(self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: int) -> tuple[int, dict[str, str], str]:
        if _ul is None:  # pragma: no cover
            raise RuntimeError("urllib not available")

        # Normalize and augment headers
        headers = headers.copy()
        # Add keep-alive header for connection reuse if enabled
        if self.enable_keep_alive:
            headers["Connection"] = "keep-alive"
        # Some Coinbase endpoints/CDNs block default urllib user-agent; set explicit UA and Accept
        headers.setdefault("User-Agent", "GPT-Trader/0.1 (+https://github.com)")
        headers.setdefault("Accept", "application/json")

        req = _ul.Request(url=url, data=body, method=method)
        for k, v in headers.items():
            req.add_header(k, v)

        try:
            # Use shared opener if available for connection reuse
            if self._opener and self.enable_keep_alive:
                resp = self._opener.open(req, timeout=timeout)
            else:
                resp = _ul.urlopen(req, timeout=timeout)

            with resp:
                status = resp.getcode()
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
                text = resp.read().decode("utf-8")
                return status, hdrs, text
        except _ue.HTTPError as e:  # type: ignore[attr-defined]
            status = e.code
            hdrs = {k.lower(): v for k, v in (e.headers or {}).items()}
            text = e.read().decode("utf-8") if hasattr(e, 'read') else (e.reason or "")
            return status, hdrs, text
        except _ue.URLError as exc:  # type: ignore[attr-defined]
            mapped = map_http_error(0, None, f"Network error: {getattr(exc, 'reason', exc)}")
            raise mapped from exc

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limits with throttling."""
        if not self.enable_throttle:
            return

        current_time = time.time()

        # Clean up old request times (older than 1 minute)
        self._request_times = [t for t in self._request_times if current_time - t < 60]

        # Check if we're approaching the limit
        if len(self._request_times) >= self.rate_limit_per_minute * 0.8:  # Warn at 80%
            logger.warning(f"Approaching rate limit: {len(self._request_times)}/{self.rate_limit_per_minute} requests in last minute")

        # If at limit, sleep until oldest request expires
        if len(self._request_times) >= self.rate_limit_per_minute:
            oldest_request = self._request_times[0]
            sleep_time = 60 - (current_time - oldest_request) + 0.1  # Add 100ms buffer
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
                # Clean up again after sleep
                current_time = time.time()
                self._request_times = [t for t in self._request_times if current_time - t < 60]

        # Record this request
        self._request_times.append(current_time)

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        # Enhanced rate limit tracking with throttling
        self._check_rate_limit()

        # Legacy tracking for backward compatibility
        self._request_count += 1
        elapsed = time.time() - self._request_window_start
        if elapsed > 60:  # Reset every minute
            if self._request_count > self.rate_limit_per_minute:
                logger.warning(f"High request rate: {self._request_count} requests in {elapsed:.1f}s")
            self._request_count = 0
            self._request_window_start = time.time()

        url = self._make_url(path)

        # Normalize body: ensure JSON for non-GET methods to avoid auth mismatches
        # Coinbase APIs expect JSON; signing must match the transmitted payload.
        if method.upper() != "GET" and body is None:
            body = {}
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "CB-VERSION": self.api_version  # Add required API version header
        }
        if self.auth:
            # Sign with path only (not full URL) per API specs
            path_only = path if path.startswith("/") else "/" + path
            # If using HMAC auth and api_mode wasn't set on auth, sync it from client
            try:
                if isinstance(self.auth, CoinbaseAuth) and not getattr(self.auth, "api_mode", None):
                    self.auth.api_mode = self.api_mode  # keep signing rules consistent
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Failed to sync auth api_mode", exc_info=exc)
            headers.update(self.auth.sign(method, path_only, body))

        data_bytes = json.dumps(body, separators=(",", ":")).encode("utf-8") if body is not None else None

        # Retries/backoff with jitter
        sys_cfg = get_config("system")
        max_retries = int(sys_cfg.get("max_retries", 3))
        base_delay = float(sys_cfg.get("retry_delay", 1.0))
        jitter_factor = float(sys_cfg.get("jitter_factor", 0.1))  # 10% jitter by default

        attempt = 0
        while True:
            attempt += 1
            _t0 = time.perf_counter()
            status, resp_headers, text = self._transport(method, url, headers, data_bytes, self.timeout)
            _dt_ms = (time.perf_counter() - _t0) * 1000.0
            try:
                get_logger().log_rest_response(endpoint=path, method=method, status_code=status, duration_ms=_dt_ms)
            except Exception as exc:  # pragma: no cover - telemetry resilience
                logger.debug("log_rest_response failed", exc_info=exc)
            # Parse JSON if possible
            try:
                payload = json.loads(text) if text else {}
            except json.JSONDecodeError:
                payload = {"raw": text or ""}

            if 200 <= status < 300:
                return payload if isinstance(payload, dict) else {"data": payload}

            # Error handling and retry logic
            code = None
            message = None
            if isinstance(payload, dict):
                code = payload.get("error") or payload.get("code")
                message = payload.get("message") or payload.get("error_message")

            # Retry on 429/5xx
            if status == 429 or (500 <= status < 600):
                if attempt <= max_retries:
                    retry_after = resp_headers.get("retry-after")
                    delay = base_delay * (2 ** (attempt - 1))
                    try:
                        if retry_after:
                            delay = float(retry_after)
                    except ValueError:
                        pass

                    # Add deterministic jitter for testing
                    if jitter_factor > 0:
                        # Use attempt number as seed for deterministic jitter in tests
                        jitter = delay * jitter_factor * ((attempt % 10) / 10.0)
                        delay = delay + jitter

                    logger.debug(f"Retrying after {delay:.2f}s (attempt {attempt}/{max_retries})")

                    # Sleep with jitter; tests can monkeypatch time.sleep
                    try:
                        time.sleep(delay)
                    except Exception as exc:  # pragma: no cover - defensive sleep fallback
                        logger.debug("sleep interrupted", exc_info=exc)
                    continue
            # Map and raise
            raise map_http_error(status, code, message)

    # Pagination helper for GET endpoints that return cursor-based pages
    def paginate(
        self,
        path: str,
        params: dict[str, Any],
        items_key: str,
        cursor_param: str = "cursor",
        cursor_field: str = "cursor",
    ) -> Iterator[dict[str, Any]]:
        next_cursor: str | None = None
        while True:
            p = dict(params or {})
            if next_cursor:
                p[cursor_param] = next_cursor
            final_path = self._build_path_with_params(path, p)
            page = self._request("GET", final_path)
            items = page.get(items_key) or []
            yield from items
            # Extract next cursor
            next_cursor = page.get(cursor_field) or page.get("next_cursor")
            if not next_cursor:
                break

    # Example endpoint stubs
    def get_products(self) -> dict[str, Any]:
        return self._request("GET", self._get_endpoint_path('products'))

    def get_accounts(self) -> dict[str, Any]:
        return self._request("GET", self._get_endpoint_path('accounts'))

    def get_ticker(self, product_id: str) -> dict[str, Any]:
        return self._request("GET", self._get_endpoint_path('ticker', product_id=product_id))

    def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", self._get_endpoint_path('orders'), payload)

    def get_candles(
        self,
        product_id: str,
        granularity: str,
        limit: int = 200,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:

        path = self._get_endpoint_path('candles', product_id=product_id)
        query = [f"granularity={granularity}"]
        if limit:
            query.append(f"limit={int(limit)}")
        if start:
            if start.tzinfo is None:
                start = start.replace(tzinfo=UTC)
            start_ts = start.astimezone(UTC).isoformat().replace('+00:00', 'Z')
            query.append(f"start={start_ts}")
        if end:
            if end.tzinfo is None:
                end = end.replace(tzinfo=UTC)
            end_ts = end.astimezone(UTC).isoformat().replace('+00:00', 'Z')
            query.append(f"end={end_ts}")
        path = f"{path}?{'&'.join(query)}"
        return self._request("GET", path)

    # Additional endpoints from registry (stubs)
    def get_product(self, product_id: str) -> dict[str, Any]:
        return self._request("GET", self._get_endpoint_path('product', product_id=product_id))

    def get_product_book(self, product_id: str, level: int = 2) -> dict[str, Any]:
        path = self._get_endpoint_path('order_book', product_id=product_id)
        if self.api_mode == 'exchange':
            path = f"{path}?level={level}"
        else:
            path = f"{path}?product_id={product_id}&level={level}"
        return self._request("GET", path)

    # Market namespace variants
    def get_market_products(self) -> dict[str, Any]:
        return self._request("GET", self._get_endpoint_path('products'))

    def get_market_product(self, product_id: str) -> dict[str, Any]:
        return self._request("GET", self._get_endpoint_path('product', product_id=product_id))

    def get_market_product_ticker(self, product_id: str) -> dict[str, Any]:
        return self._request("GET", self._get_endpoint_path('ticker', product_id=product_id))

    def get_market_product_candles(self, product_id: str, granularity: str, limit: int = 200) -> dict[str, Any]:
        path = self._get_endpoint_path('candles', product_id=product_id)
        path = f"{path}?granularity={granularity}&limit={limit}"
        return self._request("GET", path)

    def get_market_product_book(self, product_id: str, level: int = 2) -> dict[str, Any]:
        # Use order_book endpoint which exists in both modes
        path = self._get_endpoint_path('order_book', product_id=product_id)

        # Add query params based on mode
        if self.api_mode == 'exchange':
            # Exchange API uses level parameter directly
            path = f"{path}?level={level}"
        else:
            # Advanced API includes product_id in query
            path = f"{path}?product_id={product_id}&level={level}"

        return self._request("GET", path)

    def get_best_bid_ask(self, product_ids: list[str]) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "get_best_bid_ask not available in exchange mode. "
                "Use get_ticker for individual products instead. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )

        path = self._get_endpoint_path('best_bid_ask')
        q = ",".join(product_ids)
        return self._request("GET", f"{path}?product_ids={q}")

    def get_account(self, account_uuid: str) -> dict[str, Any]:
        # Use both kwargs to support both placeholder names
        # ENDPOINT_MAP will format whichever placeholder exists
        path = self._get_endpoint_path('account', account_uuid=account_uuid, account_id=account_uuid)
        return self._request("GET", path)

    def get_time(self) -> dict[str, Any]:
        return self._request("GET", self._get_endpoint_path('time'))

    def get_key_permissions(self) -> dict[str, Any]:
        path = self._get_endpoint_path('key_permissions')
        return self._request("GET", path)

    def cancel_orders(self, order_ids: list[str]) -> dict[str, Any]:
        path = self._get_endpoint_path('orders_batch_cancel')
        return self._request("POST", path, {"order_ids": order_ids})

    def preview_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "preview_order not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to call order previews."
            )
        path = self._get_endpoint_path('order_preview')
        return self._request("POST", path, payload)

    def edit_order_preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "edit_order_preview not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to edit orders."
            )
        path = self._get_endpoint_path('order_edit_preview')
        return self._request("POST", path, payload)

    def edit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "edit_order not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to edit orders."
            )
        path = self._get_endpoint_path('order_edit')
        return self._request("POST", path, payload)

    def close_position(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "close_position not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced with derivatives enabled."
            )
        path = self._get_endpoint_path('close_position')
        return self._request("POST", path, payload)

    def get_order_historical(self, order_id: str) -> dict[str, Any]:
        path = self._get_endpoint_path('order', order_id=order_id)
        return self._request("GET", path)

    def list_orders(self, **params: Any) -> dict[str, Any]:
        # params may include product_id, order_status, start_date, end_date, etc.
        # Prefer historical listing in Advanced mode; gracefully fall back to open orders endpoint
        q = "&".join(f"{k}={v}" for k, v in params.items())
        suffix = f"?{q}" if q else ""
        # Exchange mode: only '/orders' is available
        if self.api_mode == 'exchange':
            path_open = self._get_endpoint_path('orders')
            return self._request("GET", f"{path_open}{suffix}")

        # Advanced mode: try historical first, then fallback to open orders on 404
        path_hist = self._get_endpoint_path('orders_historical')
        try:
            return self._request("GET", f"{path_hist}{suffix}")
        except NotFoundError:
            # Some environments may not expose historical listing; return open orders instead
            path_open = self._get_endpoint_path('orders')
            try:
                return self._request("GET", f"{path_open}{suffix}")
            except BrokerageError as e:
                # Some orgs return 501 Method Not Allowed on open-orders listing
                # Treat as empty listing to avoid hard-failing startup
                if 'method not allowed' in str(e).lower():
                    return {"orders": []}
                raise

    def list_positions(self) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "list_positions not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced with derivatives enabled."
            )
        path = self._get_endpoint_path('cfm_positions')
        return self._request("GET", path)

    def get_position(self, product_id: str) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "get_position not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced with derivatives enabled."
            )
        path = self._get_endpoint_path('cfm_position', product_id=product_id)
        return self._request("GET", path)

    def list_orders_batch(self, order_ids: list[str]) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "list_orders_batch not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('orders_batch')
        # Placeholder: actual API may use query params
        return self._request("GET", path)

    def list_fills(self, **params: Any) -> dict[str, Any]:
        path = self._get_endpoint_path('fills')
        q = "&".join(f"{k}={v}" for k, v in params.items())
        suffix = f"?{q}" if q else ""
        return self._request("GET", f"{path}{suffix}")

    def get_fees(self) -> dict[str, Any]:
        return self._request("GET", self._get_endpoint_path('fees'))

    def get_limits(self) -> dict[str, Any]:
        path = self._get_endpoint_path('limits')
        return self._request("GET", path)

    def get_transaction_summary(self) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            # Not available in legacy Exchange API
            raise InvalidRequestError(
                "get_transaction_summary not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('transaction_summary')
        return self._request("GET", path)

    def convert_quote(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._get_endpoint_path('convert_quote')
        return self._request("POST", path, payload)

    def get_convert_trade(self, trade_id: str) -> dict[str, Any]:
        path = self._get_endpoint_path('convert_trade', trade_id=trade_id)
        return self._request("GET", path)

    def commit_convert_trade(self, trade_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "commit_convert_trade not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('convert_trade', trade_id=trade_id)
        return self._request("POST", path, payload or {})

    # Payment methods
    def list_payment_methods(self) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "list_payment_methods not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('payment_methods')
        return self._request("GET", path)

    def get_payment_method(self, payment_method_id: str) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "get_payment_method not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('payment_method', payment_method_id=payment_method_id)
        return self._request("GET", path)

    # Portfolios
    def list_portfolios(self) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "list_portfolios not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('portfolios')
        return self._request("GET", path)

    def get_portfolio(self, portfolio_uuid: str) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "get_portfolio not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('portfolio', portfolio_uuid=portfolio_uuid)
        return self._request("GET", path)

    def get_portfolio_breakdown(self, portfolio_uuid: str) -> dict[str, Any]:
        """Get detailed portfolio breakdown including cash and crypto balances."""
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "get_portfolio_breakdown not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('portfolio_breakdown', portfolio_uuid=portfolio_uuid)
        return self._request("GET", path)

    def move_funds(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "move_funds not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('move_funds')
        return self._request("POST", path, payload)

    # INTX
    def intx_allocate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "intx_allocate not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('intx_allocate')
        return self._request("POST", path, payload)

    def intx_balances(self, portfolio_uuid: str) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "intx_balances not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('intx_balances', portfolio_uuid=portfolio_uuid)
        return self._request("GET", path)

    def intx_portfolio(self, portfolio_uuid: str) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "intx_portfolio not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('intx_portfolio', portfolio_uuid=portfolio_uuid)
        return self._request("GET", path)

    def intx_positions(self, portfolio_uuid: str) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "intx_positions not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('intx_positions', portfolio_uuid=portfolio_uuid)
        return self._request("GET", path)

    def intx_position(self, portfolio_uuid: str, symbol: str) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "intx_position not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('intx_position', portfolio_uuid=portfolio_uuid, symbol=symbol)
        return self._request("GET", path)

    def intx_multi_asset_collateral(self) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "intx_multi_asset_collateral not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('intx_multi_asset_collateral')
        return self._request("GET", path)

    # CFM
    def cfm_balance_summary(self) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "cfm_balance_summary not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('cfm_balance_summary')
        return self._request("GET", path)

    def cfm_positions(self) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "cfm_positions not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('cfm_positions')
        return self._request("GET", path)

    def cfm_position(self, product_id: str) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "cfm_position not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('cfm_position', product_id=product_id)
        return self._request("GET", path)

    def cfm_sweeps(self) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "cfm_sweeps not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('cfm_sweeps')
        return self._request("GET", path)

    def cfm_sweeps_schedule(self) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "cfm_sweeps_schedule not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('cfm_schedule_sweep')
        return self._request("GET", path)

    def cfm_intraday_current_margin_window(self) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "cfm_intraday_current_margin_window not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('cfm_intraday_margin_window')
        return self._request("GET", path)

    def cfm_intraday_margin_setting(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.api_mode == 'exchange':
            raise InvalidRequestError(
                "cfm_intraday_margin_setting not available in exchange mode. "
                "Set COINBASE_API_MODE=advanced to use this feature."
            )
        path = self._get_endpoint_path('cfm_intraday_margin_setting')
        return self._request("POST", path, payload)
