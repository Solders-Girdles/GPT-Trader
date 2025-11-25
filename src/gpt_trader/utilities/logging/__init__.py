"""
Unified logging facade for GPT-Trader.

This module provides a single entry point for all logging functionality,
consolidating the interface (logging_patterns) with the backend infrastructure
(gpt_trader.logging).

Usage:
    from gpt_trader.utilities.logging import get_logger, configure_logging

    # Get a structured logger
    logger = get_logger(__name__, component="my_component")
    logger.info("Message", extra_field="value")

    # Configure logging (typically done at application startup)
    configure_logging(settings=my_settings)
"""

from __future__ import annotations

# Re-export backend infrastructure from gpt_trader.logging (System A)
from gpt_trader.logging import (
    StructuredJSONFormatter,
    StructuredJSONFormatterWithTimestamp,
    add_domain_field,
    configure_logging,
    correlation_context,
    generate_correlation_id,
    get_domain_context,
    get_log_context,
    get_orchestration_logger,
    log_execution_error,
    log_order_event,
    log_risk_event,
    log_strategy_decision,
    log_trading_operation,
    order_context,
    set_correlation_id,
    set_domain_context,
    symbol_context,
    update_domain_context,
    with_order_context,
    with_symbol_context,
    with_trading_context,
)

# Re-export interface from logging_patterns (System B) - widely used
from gpt_trader.utilities.logging_patterns import (
    LOG_FIELDS,
    StructuredLogger,
    UnifiedLogger,
    get_correlation_id,
    get_logger,
    log_configuration_change,
    log_error_with_context,
    log_execution,
    log_market_data_update,
    log_operation,
    log_position_update,
    log_system_health,
    log_trade_event,
)

# Note: gpt_trader.logging also exports get_correlation_id, but we prefer
# the one from logging_patterns for consistency with existing code.
# The logging_patterns version returns None (simplified), while the
# gpt_trader.logging version uses contextvars for actual correlation tracking.
# For production use with correlation IDs, import directly from gpt_trader.logging.

__all__ = [
    # Primary interface (from logging_patterns)
    "get_logger",
    "StructuredLogger",
    "UnifiedLogger",
    "get_correlation_id",
    "log_operation",
    "log_trade_event",
    "log_position_update",
    "log_system_health",
    "log_error_with_context",
    "log_configuration_change",
    "log_market_data_update",
    "log_execution",
    "LOG_FIELDS",
    # Backend infrastructure (from gpt_trader.logging)
    "configure_logging",
    "StructuredJSONFormatter",
    "StructuredJSONFormatterWithTimestamp",
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
    "with_trading_context",
    "with_symbol_context",
    "with_order_context",
]
