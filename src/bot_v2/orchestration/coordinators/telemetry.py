"""Telemetry coordinator public interface and compatibility exports."""

from __future__ import annotations

from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.monitoring.system import get_logger as _get_plog
from bot_v2.orchestration.account_telemetry import AccountTelemetryService
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.intx_portfolio_service import IntxPortfolioService
from bot_v2.orchestration.market_monitor import MarketActivityMonitor
from bot_v2.utilities import emit_metric
from bot_v2.utilities.logging_patterns import get_logger

from .telemetry_coordinator import TelemetryCoordinator

logger = get_logger(__name__, component="telemetry_coordinator")

__all__ = [
    "TelemetryCoordinator",
    "AccountTelemetryService",
    "CoinbaseAccountManager",
    "CoinbaseBrokerage",
    "IntxPortfolioService",
    "MarketActivityMonitor",
    "Profile",
    "emit_metric",
    "_get_plog",
    "logger",
]
