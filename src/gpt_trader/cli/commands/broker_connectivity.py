"""Broker connectivity CLI commands."""

from __future__ import annotations

import json
from argparse import Namespace
from typing import Any

import requests

from gpt_trader.cli import options
from gpt_trader.cli.response import CliErrorCode, CliResponse
from gpt_trader.config.constants import DEFAULT_HTTP_TIMEOUT
from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.credentials import resolve_coinbase_credentials
from gpt_trader.features.brokerages.coinbase.errors import (
    AuthError,
    InvalidRequestError,
    PermissionDeniedError,
    RateLimitError,
    TransientBrokerError,
)

COMMAND_NAME = "broker-check coinbase"
DEFAULT_ENDPOINT = "time"


def register(subparsers: Any) -> None:
    """Register broker connectivity commands."""
    parser = subparsers.add_parser("broker-check", help="Broker connectivity checks")
    broker_subparsers = parser.add_subparsers(dest="broker_command", required=True)

    coinbase = broker_subparsers.add_parser(
        "coinbase",
        help="Test Coinbase API connectivity",
        description="Run a read-only connectivity check against the Coinbase REST API",
    )
    options.add_output_options(coinbase, include_quiet=False)
    coinbase.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_HTTP_TIMEOUT,
        help=f"HTTP timeout in seconds (default: {DEFAULT_HTTP_TIMEOUT})",
    )
    coinbase.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
        help="Endpoint name or full path/URL to request (default: time)",
    )
    coinbase.set_defaults(handler=_handle_coinbase_check, subcommand="coinbase")


def _handle_coinbase_check(args: Namespace) -> CliResponse | int:
    output_format = getattr(args, "output_format", "text")
    endpoint = getattr(args, "endpoint", DEFAULT_ENDPOINT)
    timeout = getattr(args, "timeout", DEFAULT_HTTP_TIMEOUT)

    client: CoinbaseClient | None = None
    resolved_endpoint = ""
    try:
        client = _build_coinbase_client(timeout=timeout)
        resolved_endpoint = _resolve_endpoint(client, endpoint)
        response = client.get(resolved_endpoint)
        success, summary, raw_response = _evaluate_response(response, resolved_endpoint)
        return _emit_result(
            output_format,
            success,
            summary,
            raw_response,
            endpoint,
            resolved_endpoint,
            timeout,
        )
    except InvalidRequestError as exc:
        summary = f"Invalid endpoint '{endpoint}': {exc}"
        return _emit_error(
            output_format,
            CliErrorCode.INVALID_ARGUMENT,
            summary,
            endpoint,
            resolved_endpoint,
            timeout,
            None,
        )
    except (AuthError, PermissionDeniedError) as exc:
        summary = f"Authentication failed: {exc}"
        return _emit_error(
            output_format,
            CliErrorCode.AUTHENTICATION_FAILED,
            summary,
            endpoint,
            resolved_endpoint,
            timeout,
            None,
        )
    except RateLimitError as exc:
        summary = f"Rate limited: {exc}"
        return _emit_error(
            output_format,
            CliErrorCode.RATE_LIMITED,
            summary,
            endpoint,
            resolved_endpoint,
            timeout,
            None,
        )
    except TransientBrokerError as exc:
        summary = f"Service unavailable: {exc}"
        return _emit_error(
            output_format,
            CliErrorCode.API_ERROR,
            summary,
            endpoint,
            resolved_endpoint,
            timeout,
            None,
        )
    except requests.exceptions.SSLError as exc:
        summary = f"SSL error: {exc}"
        return _emit_error(
            output_format,
            CliErrorCode.NETWORK_ERROR,
            summary,
            endpoint,
            resolved_endpoint,
            timeout,
            None,
        )
    except (requests.Timeout, requests.ConnectionError, ConnectionError) as exc:
        summary = f"Network timeout/error: {exc}"
        return _emit_error(
            output_format,
            CliErrorCode.NETWORK_ERROR,
            summary,
            endpoint,
            resolved_endpoint,
            timeout,
            None,
        )
    except Exception as exc:  # noqa: BLE001
        summary = f"{type(exc).__name__}: {exc}"
        return _emit_error(
            output_format,
            CliErrorCode.OPERATION_FAILED,
            summary,
            endpoint,
            resolved_endpoint,
            timeout,
            None,
        )
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass


def _build_coinbase_client(*, timeout: int | None) -> CoinbaseClient:
    creds = resolve_coinbase_credentials()
    if not creds:
        raise RuntimeError(
            "Coinbase credentials not found. Set COINBASE_CREDENTIALS_FILE or "
            "COINBASE_CDP_API_KEY + COINBASE_CDP_PRIVATE_KEY."
        )
    auth = SimpleAuth(key_name=creds.key_name, private_key=creds.private_key)
    return CoinbaseClient(
        base_url="https://api.coinbase.com",
        auth=auth,
        api_mode="advanced",
        timeout=timeout,
    )


def _resolve_endpoint(client: CoinbaseClient, endpoint: str) -> str:
    endpoint = endpoint.strip()
    if endpoint.startswith("http"):
        return endpoint
    if endpoint.startswith("/"):
        return endpoint
    return client._get_endpoint_path(endpoint)


def _evaluate_response(response: Any, resolved_endpoint: str) -> tuple[bool, str, Any | None]:
    # None means we didn't get a usable response at all.
    if response is None:
        summary = f"No response from Coinbase endpoint '{resolved_endpoint}'"
        return False, summary, response

    # Empty collections can be valid for many read-only endpoints (e.g., no open orders).
    # Treat these as reachability success to avoid false negatives.
    if response == {} or response == []:
        summary = f"Coinbase endpoint '{resolved_endpoint}' reachable (empty response)"
        return True, summary, response

    if isinstance(response, dict) and "raw" in response and len(response) == 1:
        summary = f"Malformed JSON response from Coinbase endpoint '{resolved_endpoint}'"
        return False, summary, response.get("raw")

    server_time = _extract_server_time(response)
    if server_time:
        summary = f"Coinbase endpoint '{resolved_endpoint}' reachable (server time: {server_time})"
    else:
        summary = f"Coinbase endpoint '{resolved_endpoint}' reachable"
    return True, summary, response


def _extract_server_time(response: Any) -> str:
    if not isinstance(response, dict):
        return ""
    for key in ("iso", "time", "epoch"):
        value = response.get(key)
        if value:
            return str(value)
    return ""


def _emit_result(
    output_format: str,
    success: bool,
    summary: str,
    raw_response: Any | None,
    endpoint: str,
    resolved_endpoint: str,
    timeout: int | None,
) -> CliResponse | int:
    if output_format == "json":
        data = {
            "broker": "coinbase",
            "endpoint": endpoint,
            "resolved_endpoint": resolved_endpoint,
            "timeout": timeout,
            "summary": summary,
            "raw_response": raw_response,
        }
        if success:
            return CliResponse.success_response(command=COMMAND_NAME, data=data)
        return CliResponse.error_response(
            command=COMMAND_NAME,
            code=CliErrorCode.OPERATION_FAILED,
            message=summary,
            details=data,
        )

    status = "OK" if success else "FAILED"
    print(f"Broker check {status}: {summary}")
    print(f"Raw response: {_format_raw_response(raw_response)}")
    return 0 if success else 1


def _emit_error(
    output_format: str,
    code: CliErrorCode,
    summary: str,
    endpoint: str,
    resolved_endpoint: str,
    timeout: int | None,
    raw_response: Any | None,
) -> CliResponse | int:
    if output_format == "json":
        return CliResponse.error_response(
            command=COMMAND_NAME,
            code=code,
            message=summary,
            details={
                "broker": "coinbase",
                "endpoint": endpoint,
                "resolved_endpoint": resolved_endpoint,
                "timeout": timeout,
                "summary": summary,
                "raw_response": raw_response,
            },
        )

    print(f"Broker check FAILED: {summary}")
    print(f"Raw response: {_format_raw_response(raw_response)}")
    return 1


def _format_raw_response(raw_response: Any | None) -> str:
    if raw_response is None:
        return "(none)"
    if isinstance(raw_response, str):
        return raw_response
    try:
        return json.dumps(raw_response, default=str)
    except TypeError:
        return str(raw_response)


__all__ = ["register"]
