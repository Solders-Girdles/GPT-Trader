"""Log formatters for TUI display.

Contains:
- Helper functions: _abbreviate_logger, _format_number, _extract_key_values
- Formatter classes: ImprovedExceptionFormatter, CompactTuiFormatter, StructuredTuiFormatter
"""

from __future__ import annotations

import logging
import time

from gpt_trader.tui.log_constants import (
    KEYVALUE_PATTERN,
    LEVEL_ICONS,
    LOGGER_ABBREVIATIONS,
    ORDER_PATTERN,
    PRICE_PATTERN,
    STRATEGY_DEBUG_PATTERN,
    STRATEGY_DECISION_PATTERN,
)


def _abbreviate_logger(name: str, max_len: int = 10) -> str:
    """Abbreviate logger name to fit in fixed width.

    Args:
        name: Full logger name (e.g., 'gpt_trader.features.strategy')
        max_len: Maximum length for abbreviated name

    Returns:
        Abbreviated name, right-padded to max_len
    """
    short = name.rsplit(".", 1)[-1]

    # Apply known abbreviations
    abbrev = LOGGER_ABBREVIATIONS.get(short, short)

    # Truncate if still too long
    if len(abbrev) > max_len:
        abbrev = abbrev[: max_len - 1] + "…"

    return abbrev.ljust(max_len)


def _format_number(value: str, decimals: int = 2) -> str:
    """Format a numeric string to specified decimal places.

    Args:
        value: String representation of number
        decimals: Number of decimal places

    Returns:
        Formatted number string
    """
    try:
        num = float(value)
        if num >= 1000:
            return f"{num:,.{decimals}f}"
        return f"{num:.{decimals}f}"
    except (ValueError, TypeError):
        return value


def _extract_key_values(message: str) -> dict[str, str]:
    """Extract key=value pairs from log message.

    Args:
        message: Log message string

    Returns:
        Dictionary of extracted key-value pairs
    """
    return dict(KEYVALUE_PATTERN.findall(message))


class ImprovedExceptionFormatter(logging.Formatter):
    """Custom formatter that improves exception/traceback display."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with improved exception formatting."""
        # Use parent formatter for base message
        base_message = super().format(record)

        # If there's exception info, format it better
        if record.exc_info:
            # Get the formatted exception from the record
            if record.exc_text:
                exc_text = record.exc_text
            else:
                exc_text = self.formatException(record.exc_info)

            # Add indentation to exception lines for better readability
            indented_exc = "\n".join(
                f"  │ {line}" if line.strip() else "  │" for line in exc_text.split("\n")
            )

            # Combine base message with formatted exception
            return f"{base_message}\n  ╰─ Exception:\n{indented_exc}"

        return base_message


class CompactTuiFormatter(logging.Formatter):
    """Compact formatter for TUI with icons and short logger names.

    Format: [short_logger] icon message

    Example outputs:
    - [bot_lifecycle] ✓ Mode switch completed
    - [factory] ⚠ Failed to create strategy
    - [tui] ✗ Widget initialization failed
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record in compact style with icon."""
        icon = LEVEL_ICONS.get(record.levelno, "·")
        short_name = record.name.rsplit(".", 1)[-1]
        message = record.getMessage()

        # Handle exceptions compactly
        if record.exc_info:
            exc_type = record.exc_info[0]
            exc_value = record.exc_info[1]
            if exc_type and exc_value:
                message = f"{message} ({exc_type.__name__}: {exc_value})"

        return f"[{short_name}] {icon} {message}"

    def format_with_timestamp(self, record: logging.LogRecord) -> str:
        """Format log record with timestamp prefix."""
        timestamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        compact = self.format(record)
        return f"{timestamp} {compact}"


class StructuredTuiFormatter(logging.Formatter):
    """Structured formatter with fixed columns for easy scanning.

    Layout: TIME ICON LOGGER     | MESSAGE
            ──── ─── ────────── | ─────────────────────────────

    Features:
    - Fixed-width columns for visual alignment
    - Abbreviated logger names (max 10 chars)
    - Domain-aware message formatting (strategy, orders, positions)
    - Rounded decimals for readability
    - Key fields extracted and highlighted

    Example outputs:
        09:31 ✓ strat      BTC-USD NEUTRAL  MA 115.25/113.40
        09:31 ⚠ portfolio  Position $0.00 < share $150.00
        09:31 ✓ exec       BUY 0.05 BTC-USD @ $98,450.00
    """

    # Column widths
    TIME_WIDTH = 8  # HH:MM:SS
    ICON_WIDTH = 2  # Icon + space
    LOGGER_WIDTH = 10  # Abbreviated logger name

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured columns."""
        # Time column
        timestamp = time.strftime("%H:%M:%S", time.localtime(record.created))

        # Icon column
        icon = LEVEL_ICONS.get(record.levelno, "·")

        # Message - apply domain-specific formatting (check if system log first)
        message = self._format_message(record)
        logger_name = record.name.lower()

        # System/startup logs - use more descriptive logger names
        system_keywords = [
            "startup",
            "initializ",
            "mount",
            "ready",
            "connect",
            "disconnect",
            "error",
            "failed",
            "critical",
            "warning",
            "status",
            "state",
            "config",
            "mode",
            "credential",
            "validation",
            "bot",
            "engine",
            "app",
            "tui",
            "lifecycle",
            "coordinator",
            "service",
        ]
        is_system_log = any(
            keyword in logger_name or keyword in message.lower() for keyword in system_keywords
        )

        if is_system_log:
            # For system logs, use more descriptive logger name (last 2 components)
            # e.g., "gpt_trader.tui.app" -> "tui.app" instead of just "app"
            parts = record.name.rsplit(".", 2)
            if len(parts) >= 2:
                logger_abbrev = parts[-2] + "." + parts[-1]
                # Truncate if still too long, but preserve more context
                if len(logger_abbrev) > self.LOGGER_WIDTH:
                    logger_abbrev = logger_abbrev[-(self.LOGGER_WIDTH - 1) :]
            else:
                logger_abbrev = _abbreviate_logger(record.name, self.LOGGER_WIDTH)
        else:
            # Logger column (abbreviated, fixed width) for non-system logs
            logger_abbrev = _abbreviate_logger(record.name, self.LOGGER_WIDTH)

        # Handle exceptions
        if record.exc_info:
            exc_type = record.exc_info[0]
            exc_value = record.exc_info[1]
            if exc_type and exc_value:
                exc_name = exc_type.__name__
                # Keep exception message short
                exc_msg = str(exc_value)[:50]
                if len(str(exc_value)) > 50:
                    exc_msg += "…"
                message = f"{message} [{exc_name}: {exc_msg}]"

        return f"{timestamp} {icon} {logger_abbrev:<{self.LOGGER_WIDTH}} {message}"

    def _format_message(self, record: logging.LogRecord) -> str:
        """Apply domain-specific formatting to message.

        Args:
            record: Log record to format

        Returns:
            Formatted message string
        """
        message = record.getMessage()
        logger_name = record.name.lower()

        # System/startup logs - preserve full context for better debugging
        # These logs are critical for understanding system state
        system_keywords = [
            "startup",
            "initializ",
            "mount",
            "ready",
            "connect",
            "disconnect",
            "error",
            "failed",
            "critical",
            "warning",
            "status",
            "state",
            "config",
            "mode",
            "credential",
            "validation",
            "bot",
            "engine",
        ]
        is_system_log = any(
            keyword in logger_name or keyword in message.lower() for keyword in system_keywords
        )

        # For system logs, preserve more context - don't condense as aggressively
        if is_system_log:
            # Still apply basic formatting but preserve important details
            # Remove only truly redundant prefixes, keep the message informative
            prefixes_to_remove = [
                "Strategy decision debug: ",
                "Strategy decision: ",
            ]
            for prefix in prefixes_to_remove:
                if message.startswith(prefix):
                    message = message[len(prefix) :]
                    break
            # Return full message for system logs to preserve context
            return message

        # Strategy decision logs - condense the verbose output
        if "strategy" in logger_name:
            # Actual decision format: "Strategy Decision for BTC-USD: BUY (momentum crossover)"
            decision_match = STRATEGY_DECISION_PATTERN.search(message)
            if decision_match:
                symbol, action, reason = decision_match.groups()
                # Truncate long reasons
                if len(reason) > 25:
                    reason = reason[:22] + "..."
                return f"{symbol:<8} {action.upper():<6} {reason}"

            # Debug format with MA values
            debug_match = STRATEGY_DEBUG_PATTERN.search(message)
            if debug_match:
                symbol, short_ma, long_ma, label = debug_match.groups()
                short_ma_fmt = _format_number(short_ma)
                long_ma_fmt = _format_number(long_ma)
                return f"{symbol:<8} {label.upper():<8} MA {short_ma_fmt}/{long_ma_fmt}"

        # Order logs - extract key info
        if "order" in logger_name or "execution" in logger_name:
            match = ORDER_PATTERN.search(message)
            if match:
                _, side, quantity, symbol = match.groups()
                return f"{side.upper()} {quantity} {symbol}"

        # Position logs
        if "position" in logger_name or "portfolio" in logger_name:
            # Extract and format dollar amounts
            message = PRICE_PATTERN.sub(
                lambda m: f"${_format_number(m.group(1).replace(',', ''))}",
                message,
            )
            # Shorten common prefixes
            message = message.replace("Position value ", "Pos ")
            message = message.replace("Kelly position ", "Kelly ")
            return message

        # Generic key=value formatting - round long decimals
        # Optimization: only run if '=' is present
        if "=" in message:
            kv_pairs = _extract_key_values(message)
            for key, value in kv_pairs.items():
                # Round long decimal values
                if "." in value and len(value.split(".")[-1]) > 4:
                    formatted = _format_number(value)
                    message = message.replace(f"{key}={value}", f"{key}={formatted}")

        # Remove redundant prefixes
        prefixes_to_remove = [
            "Strategy decision debug: ",
            "Strategy decision: ",
        ]
        for prefix in prefixes_to_remove:
            if message.startswith(prefix):
                message = message[len(prefix) :]
                break

        return message
