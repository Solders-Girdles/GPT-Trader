"""Coinbase CLI commands."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from gpt_trader.cli import options
from gpt_trader.cli.response import CliErrorCode, CliResponse
from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.credentials import resolve_coinbase_credentials

COMMAND_NAME = "coinbase connectivity"


def register(subparsers: Any) -> None:
    """Register Coinbase-related commands."""
    parser = subparsers.add_parser("coinbase", help="Coinbase diagnostics")
    coinbase_subparsers = parser.add_subparsers(dest="coinbase_command", required=True)

    connectivity = coinbase_subparsers.add_parser(
        "connectivity",
        help="Test Coinbase API connectivity",
        description="Run a read-only connectivity test against the Coinbase API",
    )
    options.add_output_options(connectivity, include_quiet=False)
    connectivity.set_defaults(handler=_handle_connectivity, subcommand="connectivity")


def _handle_connectivity(args: Namespace) -> CliResponse | int:
    output_format = getattr(args, "output_format", "text")

    client = None
    try:
        client = _build_coinbase_client()
    except Exception as exc:  # noqa: BLE001
        message = f"Failed to initialize Coinbase client: {exc}"
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.CONFIG_INVALID,
                message=message,
            )
        print(f"Coinbase connectivity FAILED: {message}")
        return 1

    try:
        response = client.get_time()
        if not response:
            message = "Empty response from Coinbase time endpoint"
            if output_format == "json":
                return CliResponse.error_response(
                    command=COMMAND_NAME,
                    code=CliErrorCode.NETWORK_ERROR,
                    message=message,
                )
            print(f"Coinbase connectivity FAILED: {message}")
            return 1

        server_time = _extract_server_time(response)
        data = {
            "endpoint": "time",
            "server_time": server_time,
        }

        if output_format == "json":
            return CliResponse.success_response(command=COMMAND_NAME, data=data)

        if server_time:
            print(f"Coinbase connectivity OK (server time: {server_time})")
        else:
            print("Coinbase connectivity OK")
        return 0
    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {exc}"
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.OPERATION_FAILED,
                message=message,
            )
        print(f"Coinbase connectivity FAILED: {message}")
        return 1
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


def _build_coinbase_client() -> CoinbaseClient:
    creds = resolve_coinbase_credentials()
    if not creds:
        raise RuntimeError(
            "Coinbase credentials not found. Set COINBASE_CREDENTIALS_FILE or "
            "COINBASE_CDP_API_KEY + COINBASE_CDP_PRIVATE_KEY."
        )
    auth = SimpleAuth(key_name=creds.key_name, private_key=creds.private_key)
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=auth, api_mode="advanced")


def _extract_server_time(response: dict[str, Any]) -> str:
    for key in ("iso", "time", "epoch"):
        value = response.get(key)
        if value:
            return str(value)
    return ""


__all__ = ["register"]
