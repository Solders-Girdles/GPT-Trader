"""General-purpose utility helpers for bot_v2."""

from .console_logging import (
    ConsoleLogger,
    console_analysis,
    console_cache,
    console_data,
    console_error,
    console_info,
    console_key_value,
    console_ml,
    console_network,
    console_order,
    console_position,
    console_section,
    console_storage,
    console_success,
    console_table,
    console_trading,
    console_warning,
    get_console_logger,
)
from .datetime_helpers import (
    normalize_to_utc,
    to_iso_utc,
    utc_now,
    utc_now_iso,
    utc_timestamp,
)
from .import_utils import optional_import
from .iterators import empty_stream
from .logging_patterns import get_logger, log_operation
from .quantities import quantity_from
from .quantization import (
    quantize_price,
    quantize_price_side_aware,
    quantize_size,
)
from .telemetry import emit_metric
from .trading_operations import (
    PositionManager,
    TradingOperations,
    create_position_manager,
    create_trading_operations,
)

__all__ = [
    # Datetime utilities
    "utc_now",
    "utc_now_iso",
    "utc_timestamp",
    "to_iso_utc",
    "normalize_to_utc",
    # Iterator utilities
    "empty_stream",
    # Telemetry utilities
    "emit_metric",
    # Quantity utilities
    "quantity_from",
    "quantize_size",
    "quantize_price",
    "quantize_price_side_aware",
    # Trading operations
    "TradingOperations",
    "PositionManager",
    "create_trading_operations",
    "create_position_manager",
    # Console logging
    "ConsoleLogger",
    "get_console_logger",
    "console_success",
    "console_error",
    "console_warning",
    "console_info",
    "console_data",
    "console_trading",
    "console_order",
    "console_position",
    "console_cache",
    "console_storage",
    "console_network",
    "console_analysis",
    "console_ml",
    "console_section",
    "console_table",
    "console_key_value",
    # Import utilities
    "optional_import",
    # Logging utilities
    "log_operation",
    "get_logger",
]
