"""Logging utilities for GPT-Trader."""

from __future__ import annotations

from bot_v2.logging.correlation import (
    add_domain_field,
    correlation_context,
    generate_correlation_id,
    get_correlation_id,
    get_domain_context,
    get_log_context,
    order_context,
    set_correlation_id,
    set_domain_context,
    symbol_context,
    update_domain_context,
)
from bot_v2.logging.json_formatter import (
    StructuredJSONFormatter,
    StructuredJSONFormatterWithTimestamp,
)
from bot_v2.logging.orchestration_helpers import (
    get_orchestration_logger,
    log_execution_error,
    log_market_data_update,
    log_order_event,
    log_risk_event,
    log_strategy_decision,
    log_trading_operation,
    with_order_context,
    with_symbol_context,
    with_trading_context,
)
from bot_v2.logging.setup import configure_logging

__all__ = [
    "configure_logging",
    "StructuredJSONFormatter",
    "StructuredJSONFormatterWithTimestamp",
    "get_correlation_id",
    "set_correlation_id",
    "generate_correlation_id",
    "get_domain_context",
    "set_domain_context",
    "update_domain_context",
    "add_domain_field",
    "get_log_context",
    "correlation_context",
    "symbol_context",
    "order_context",
    "get_orchestration_logger",
    "log_trading_operation",
    "log_order_event",
    "log_strategy_decision",
    "log_execution_error",
    "log_risk_event",
    "log_market_data_update",
    "with_trading_context",
    "with_symbol_context",
    "with_order_context",
]
