"""Core HTTP machinery for the Coinbase REST client."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, cast

from bot_v2.features.brokerages.coinbase.auth import AuthStrategy, CDPJWTAuth, CoinbaseAuth
from bot_v2.features.brokerages.coinbase.errors import InvalidRequestError, map_http_error
from bot_v2.monitoring.system import get_correlation_id
from bot_v2.monitoring.system import get_logger as get_production_logger
from bot_v2.utilities.logging_patterns import get_logger as get_structured_logger

if TYPE_CHECKING:
    from bot_v2.orchestration.runtime_settings import RuntimeSettings
else:  # pragma: no cover - runtime type alias
    RuntimeSettings = Any  # type: ignore[misc]

_ul: Any
_ue: Any
_requests_exceptions: Any
try:
    import urllib.error as _ue  # type: ignore
    import urllib.request as _ul  # type: ignore
except Exception:  # pragma: no cover - urllib unavailable in some runtimes
    _ul = None
    _ue = None

logger = get_structured_logger(__name__, component="coinbase_client")

_TRUTHY = {"1", "true", "yes", "on"}


def _load_system_config() -> dict[str, Any]:
    """Resolve configuration loader, allowing package-level monkeypatching in tests."""
    pkg = sys.modules.get("bot_v2.features.brokerages.coinbase.client")
    if pkg and hasattr(pkg, "get_config"):
        return cast(dict[str, Any], pkg.get_config("system"))  # type: ignore[attr-defined]
    from bot_v2.config import get_config as fallback_get_config

    return cast(dict[str, Any], fallback_get_config("system"))


def _load_runtime_settings_snapshot() -> RuntimeSettings:
    from bot_v2.orchestration.runtime_settings import load_runtime_settings as _loader

    return _loader()


class CoinbaseClientBase:
    """Provides transport, retry, and endpoint plumbing for Coinbase APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        auth: AuthStrategy | None,
        timeout: int = 30,
        api_version: str = "2024-10-24",
        rate_limit_per_minute: int = 100,
        enable_throttle: bool = True,
        api_mode: str = "advanced",
        enable_keep_alive: bool = True,
        settings: RuntimeSettings | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.timeout = timeout
        self.api_version = api_version
        self.api_mode = api_mode
        self._transport: Callable[
            [str, str, dict[str, str], bytes | None, int], tuple[int, dict[str, str], str]
        ] = self._urllib_transport
        self._request_count = 0
        self._request_window_start = time.time()
        self._request_times: list[float] = []
        self.rate_limit_per_minute = rate_limit_per_minute
        self.enable_throttle = enable_throttle
        self._is_cdp = isinstance(auth, CDPJWTAuth)
        self.enable_keep_alive = enable_keep_alive
        self._opener: Any | None = None
        if enable_keep_alive:
            self._setup_opener()

        self._static_settings = settings is not None
        self._settings = settings or _load_runtime_settings_snapshot()
        self._production_logger: Any | None = None
        self._configure_logger()

    def set_transport_for_testing(
        self,
        transport: Callable[
            [str, str, dict[str, str], bytes | None, int], tuple[int, dict[str, str], str]
        ],
    ) -> None:
        self._transport = transport

    def _configure_logger(self) -> None:
        try:
            self._production_logger = get_production_logger(settings=self._settings)
        except Exception:  # pragma: no cover - telemetry optional
            try:
                self._production_logger = get_production_logger()
            except Exception:  # pragma: no cover - telemetry optional
                self._production_logger = None

    def _ensure_runtime_settings(self) -> None:
        if self._static_settings:
            return
        self._settings = _load_runtime_settings_snapshot()
        self._configure_logger()

    def _log_rest_response(
        self,
        *,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        logger_instance = self._production_logger
        if logger_instance is None:
            try:
                logger_instance = get_production_logger()
            except Exception:  # pragma: no cover - telemetry optional
                logger_instance = None
        if logger_instance is None:
            return
        try:
            if error is not None:
                logger_instance.log_rest_response(
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code,
                    duration_ms=duration_ms,
                    error=error,
                )
            else:
                logger_instance.log_rest_response(
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code,
                    duration_ms=duration_ms,
                )
        except Exception as exc:  # pragma: no cover - telemetry resilience
            logger.debug("log_rest_response failed", exc_info=exc)

    # ------------------------------------------------------------------
    # Endpoint handling
    # ------------------------------------------------------------------
    def _setup_opener(self) -> None:
        if _ul is None:  # pragma: no cover
            return
        http_handler = _ul.HTTPHandler()
        https_handler = _ul.HTTPSHandler()
        self._opener = _ul.build_opener(http_handler, https_handler)

    def _get_endpoint_path(self, endpoint_name: str, **kwargs: str) -> str:
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
                "fees": "/api/v3/brokerage/fees",
                "limits": "/api/v3/brokerage/limits",
                "key_permissions": "/api/v3/brokerage/key_permissions",
                "transaction_summary": "/api/v3/brokerage/transaction_summary",
                "portfolios": "/api/v3/brokerage/portfolios",
                "portfolio": "/api/v3/brokerage/portfolios/{portfolio_uuid}",
                "portfolio_breakdown": "/api/v3/brokerage/portfolios/{portfolio_uuid}/breakdown",
                "convert_quote": "/api/v3/brokerage/convert/quote",
                "convert_trade": "/api/v3/brokerage/convert/trade/{trade_id}",
                "payment_methods": "/api/v3/brokerage/payment_methods",
                "payment_method": "/api/v3/brokerage/payment_methods/{payment_method_id}",
                "orders_batch": "/api/v3/brokerage/orders/historical/batch",
                "order_edit_preview": "/api/v3/brokerage/orders/edit_preview",
                "move_funds": "/api/v3/brokerage/portfolios/move_funds",
                "intx_allocate": "/api/v3/brokerage/intx/allocate",
                "intx_balances": "/api/v3/brokerage/intx/balances/{portfolio_uuid}",
                "intx_portfolio": "/api/v3/brokerage/intx/portfolio/{portfolio_uuid}",
                "intx_positions": "/api/v3/brokerage/intx/positions/{portfolio_uuid}",
                "intx_position": "/api/v3/brokerage/intx/positions/{portfolio_uuid}/{symbol}",
                "intx_multi_asset_collateral": "/api/v3/brokerage/intx/multi_asset_collateral",
                "cfm_balance_summary": "/api/v3/brokerage/cfm/balance_summary",
                "cfm_positions": "/api/v3/brokerage/cfm/positions",
                "cfm_position": "/api/v3/brokerage/cfm/positions/{product_id}",
                "cfm_sweeps": "/api/v3/brokerage/cfm/sweeps",
                "cfm_schedule_sweep": "/api/v3/brokerage/cfm/sweeps/schedule",
                "cfm_intraday_margin_window": "/api/v3/brokerage/cfm/intraday/current_margin_window",
                "cfm_intraday_margin_setting": "/api/v3/brokerage/cfm/intraday/margin_setting",
            },
            "exchange": {
                "products": "/products",
                "product": "/products/{product_id}",
                "ticker": "/products/{product_id}/ticker",
                "candles": "/products/{product_id}/candles",
                "order_book": "/products/{product_id}/book",
                "trades": "/products/{product_id}/trades",
                "accounts": "/accounts",
                "account": "/accounts/{account_id}",
                "orders": "/orders",
                "order": "/orders/{order_id}",
                "cancel_order": "/orders/{order_id}",
                "fills": "/fills",
                "time": "/time",
                "fees": "/fees",
            },
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
            raise InvalidRequestError(f"Unknown endpoint: {endpoint_name}")

        path = mode_endpoints[endpoint_name]
        try:
            return path.format(**kwargs)
        except KeyError:
            return path

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------
    def _make_url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}{path if path.startswith('/') else '/' + path}"

    def _build_path_with_params(self, path: str, params: dict[str, Any] | None) -> str:
        if not params:
            return path
        query = "&".join(f"{key}={value}" for key, value in params.items())
        joiner = "?" if "?" not in path else "&"
        return f"{path}{joiner}{query}"

    def _normalize_path(self, url_or_path: str) -> str:
        if url_or_path.startswith(self.base_url):
            return url_or_path[len(self.base_url) :]
        return url_or_path

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------
    def get(self, url_or_path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        path = self._normalize_path(url_or_path)
        if params:
            path = self._build_path_with_params(path, params)
        return self._request("GET", path)

    def post(self, url_or_path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        path = self._normalize_path(url_or_path)
        return self._request("POST", path, payload)

    def delete(self, url_or_path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        path = self._normalize_path(url_or_path)
        return self._request("DELETE", path, payload)

    # ------------------------------------------------------------------
    # Transport + retry handling
    # ------------------------------------------------------------------
    def _urllib_transport(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: int,
    ) -> tuple[int, dict[str, str], str]:
        if _ul is None or _ue is None:  # pragma: no cover
            raise RuntimeError("urllib not available in this environment")

        if not url.startswith(("http://", "https://")):
            raise InvalidRequestError(f"Unsupported URL scheme: {url!r}")

        req = _ul.Request(url, data=body, method=method.upper())
        for key, value in headers.items():
            req.add_header(key, value)
        if self.enable_keep_alive and "Connection" not in req.headers:
            req.add_header("Connection", "keep-alive")

        try:
            if self._opener and self.enable_keep_alive:
                resp = self._opener.open(req, timeout=timeout)  # nosec B310
            else:
                resp = _ul.urlopen(req, timeout=timeout)  # nosec B310

            with resp:
                status = resp.getcode()
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
                text = resp.read().decode("utf-8")
                return status, hdrs, text
        except _ue.HTTPError as exc:  # type: ignore[attr-defined]
            status = exc.code
            hdrs = {k.lower(): v for k, v in (exc.headers or {}).items()}
            text = exc.read().decode("utf-8") if hasattr(exc, "read") else (exc.reason or "")
            return status, hdrs, text
        except _ue.URLError as exc:  # type: ignore[attr-defined]
            mapped = map_http_error(0, None, f"Network error: {getattr(exc, 'reason', exc)}")
            raise mapped from exc

    def _check_rate_limit(self) -> None:
        if not self.enable_throttle:
            return

        current_time = time.time()
        self._request_times = [t for t in self._request_times if current_time - t < 60]

        pending_count = len(self._request_times) + 1

        if pending_count >= self.rate_limit_per_minute * 0.8:
            logger.warning(
                "Approaching rate limit: %d/%d requests in last minute",
                pending_count,
                self.rate_limit_per_minute,
            )

        if pending_count > self.rate_limit_per_minute:
            oldest_request = self._request_times[0]
            sleep_time = 60 - (current_time - oldest_request) + 0.1
            if sleep_time > 0:
                logger.info("Rate limit reached, sleeping for %.1fs", sleep_time)
                time.sleep(sleep_time)
                current_time = time.time()
                self._request_times = [t for t in self._request_times if current_time - t < 60]

        self._request_times.append(current_time)

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self._ensure_runtime_settings()
        self._check_rate_limit()

        self._request_count += 1
        elapsed = time.time() - self._request_window_start
        if elapsed > 60:
            if self._request_count > self.rate_limit_per_minute:
                logger.warning(
                    "High request rate: %d requests in %.1fs",
                    self._request_count,
                    elapsed,
                )
            self._request_count = 0
            self._request_window_start = time.time()

        url = self._make_url(path)

        if method.upper() != "GET" and body is None:
            body = {}
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "CB-VERSION": self.api_version,
        }
        if self.auth:
            path_only = path if path.startswith("/") else "/" + path
            try:
                if isinstance(self.auth, CoinbaseAuth) and not getattr(self.auth, "api_mode", None):
                    self.auth.api_mode = self.api_mode
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Failed to sync auth api_mode", exc_info=exc)
            headers.update(self.auth.sign(method, path_only, body))
        try:
            correlation_id = get_correlation_id()
        except Exception:  # pragma: no cover - defensive guard
            correlation_id = None
        if correlation_id:
            headers.setdefault("X-Correlation-Id", correlation_id)

        data_bytes = (
            json.dumps(body, separators=(",", ":")).encode("utf-8") if body is not None else None
        )

        sys_config = dict(_load_system_config())
        fast_retry_flag = (self._settings.raw_env.get("COINBASE_FAST_RETRY") or "").strip().lower()
        if fast_retry_flag in _TRUTHY:
            sys_config.setdefault("max_retries", sys_config.get("max_retries", 3))
            sys_config["retry_delay"] = 0.0
            sys_config["jitter_factor"] = 0.0
        max_retries = int(sys_config.get("max_retries", 3))
        base_delay = float(sys_config.get("retry_delay", 1.0))
        jitter_factor = float(sys_config.get("jitter_factor", 0.1))

        attempt = 0

        try:
            from requests import exceptions as _requests_exceptions  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            _requests_exceptions = cast(Any, None)

        retryable_exceptions: tuple[type[Exception], ...]
        if _requests_exceptions is not None:
            retryable_exceptions = (
                ConnectionError,
                TimeoutError,
                OSError,
                _requests_exceptions.ConnectionError,
                _requests_exceptions.Timeout,
            )
        else:
            retryable_exceptions = (ConnectionError, TimeoutError, OSError)

        def _should_retry_retryable_error(attempt_idx: int, reason: str | None = None) -> bool:
            if attempt_idx > max_retries:
                return False
            delay = base_delay * (2 ** (attempt_idx - 1))
            if jitter_factor > 0:
                jitter = delay * jitter_factor * ((attempt_idx % 10) / 10.0)
                delay += jitter
            if reason:
                logger.debug(
                    "Retrying after %.2fs (attempt %d/%d) due to %s",
                    delay,
                    attempt_idx,
                    max_retries,
                    reason,
                )
            else:
                logger.debug(
                    "Retrying after %.2fs (attempt %d/%d)",
                    delay,
                    attempt_idx,
                    max_retries,
                )
            try:
                time.sleep(delay)
            except Exception as exc:  # pragma: no cover - defensive sleep fallback
                logger.debug("sleep interrupted", exc_info=exc)
            return True

        def _handle_retryable_status(attempt_idx: int, retry_after: str | None) -> bool:
            if attempt_idx > max_retries:
                return False
            delay = base_delay * (2 ** (attempt_idx - 1))
            used_retry_after = False
            try:
                if retry_after:
                    delay = float(retry_after)
                    used_retry_after = True
            except ValueError:
                pass
            if jitter_factor > 0 and not used_retry_after:
                jitter = delay * jitter_factor * ((attempt_idx % 10) / 10.0)
                delay += jitter
            logger.debug(
                "Retrying after %.2fs (attempt %d/%d)",
                delay,
                attempt_idx,
                max_retries,
            )
            try:
                time.sleep(delay)
            except Exception as exc:  # pragma: no cover - defensive sleep fallback
                logger.debug("sleep interrupted", exc_info=exc)
            return True

        while True:
            attempt += 1
            start = time.perf_counter()
            try:
                status, resp_headers, text = self._transport(
                    method, url, headers, data_bytes, self.timeout
                )
            except retryable_exceptions as exc:
                duration_ms = (time.perf_counter() - start) * 1000.0
                self._log_rest_response(
                    endpoint=path,
                    method=method,
                    status_code=0,
                    duration_ms=duration_ms,
                    error=str(exc),
                )

                reason = f"network error: {exc}" if str(exc) else exc.__class__.__name__
                if _should_retry_retryable_error(attempt, reason):
                    continue
                raise map_http_error(0, None, str(exc)) from exc
            duration_ms = (time.perf_counter() - start) * 1000.0
            self._log_rest_response(
                endpoint=path,
                method=method,
                status_code=status,
                duration_ms=duration_ms,
            )

            try:
                payload = json.loads(text) if text else {}
            except json.JSONDecodeError:
                payload = {"raw": text or ""}

            if 200 <= status < 300:
                return payload if isinstance(payload, dict) else {"data": payload}

            code = None
            message = None
            if isinstance(payload, dict):
                code = payload.get("error") or payload.get("code")
                message = payload.get("message") or payload.get("error_message")

            if status == 429 or (500 <= status < 600):
                retry_after = resp_headers.get("retry-after")
                if _handle_retryable_status(attempt, retry_after):
                    continue
            raise map_http_error(status, code, message)

    # ------------------------------------------------------------------
    # Pagination helper
    # ------------------------------------------------------------------
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
            query = dict(params or {})
            if next_cursor:
                query[cursor_param] = next_cursor
            final_path = self._build_path_with_params(path, query)
            page = self._request("GET", final_path)
            items = page.get(items_key) or []
            yield from items
            next_cursor = page.get(cursor_field) or page.get("next_cursor")
            if not next_cursor:
                break


__all__ = ["CoinbaseClientBase"]
