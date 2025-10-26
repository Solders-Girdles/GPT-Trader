"""Unified logger implementation with structured context."""

from __future__ import annotations

import logging
import sys
from typing import Any, TextIO

LOG_FIELDS = {
    "operation": "operation",
    "component": "component",
    "symbol": "symbol",
    "side": "side",
    "quantity": "quantity",
    "price": "price",
    "order_id": "order_id",
    "position_size": "position_size",
    "pnl": "pnl",
    "equity": "equity",
    "leverage": "leverage",
    "duration_ms": "duration_ms",
    "error_type": "error_type",
    "status": "status",
}

RESERVED_LOG_RECORD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
}


class UnifiedLogger:
    """Unified logger with structured context and optional console mirroring."""

    LEVEL_PREFIXES: dict[int, str] = {
        logging.INFO: "ℹ️",
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "🚨",
    }

    CHANNEL_PREFIXES: dict[str, tuple[int, str]] = {
        "success": (logging.INFO, "✅"),
        "data": (logging.INFO, "📊"),
        "trading": (logging.INFO, "💰"),
        "order": (logging.INFO, "📝"),
        "position": (logging.INFO, "📈"),
        "cache": (logging.DEBUG, "📦"),
        "storage": (logging.INFO, "💾"),
        "network": (logging.INFO, "🌐"),
        "analysis": (logging.INFO, "🔍"),
        "ml": (logging.INFO, "🧠"),
    }

    def __init__(
        self,
        name: str,
        *,
        component: str | None = None,
        enable_console: bool = False,
        output_stream: TextIO | None = None,
    ) -> None:
        self.logger = logging.getLogger(name)
        self.component = component
        self.enable_console = enable_console
        self.output_stream = output_stream or sys.stdout

    @property
    def name(self) -> str:
        return self.logger.name

    def _emit_console(self, message: str, prefix: str | None = None) -> None:
        text = f"{prefix} {message}" if prefix else message
        try:
            print(text, file=self.output_stream)
        except Exception:
            print(text)

    def log(
        self,
        level: int,
        message: str,
        *args: Any,
        console_prefix: str | None = None,
        console_message: str | None = None,
        **kwargs: Any,
    ) -> None:
        exc_info = kwargs.pop("exc_info", None)
        stack_info = kwargs.pop("stack_info", None)
        stacklevel = kwargs.pop("stacklevel", None)
        extra_param = kwargs.pop("extra", None)
        console_message_override = kwargs.pop("console_message", None)
        console_prefix_override = kwargs.pop("console_prefix", None)

        reserved_kwargs = {"exc_info", "stack_info", "stacklevel", "extra", "raw_message"}
        if console_message_override is not None:
            console_message = console_message_override
        if console_prefix_override is not None:
            console_prefix = console_prefix_override

        raw_message = kwargs.pop("raw_message", False)

        context_fields: dict[str, Any] = {}
        for key in list(kwargs.keys()):
            if key in reserved_kwargs:
                continue
            context_fields[key] = kwargs.pop(key)

        if args:
            try:
                rendered_message = message % args
            except Exception:
                rendered_message = message
        else:
            rendered_message = message

        log_kwargs: dict[str, Any] = {}
        if exc_info is not None:
            log_kwargs["exc_info"] = exc_info
        if stack_info is not None:
            log_kwargs["stack_info"] = stack_info
        if stacklevel is not None:
            log_kwargs["stacklevel"] = stacklevel

        extra: dict[str, Any] = extra_param or {}
        extra_context = context_fields.pop("extra_context", None)
        if extra_context:
            extra.update(extra_context)

        structured_fields = {
            field: context_fields.pop(field)
            for field in list(context_fields.keys())
            if field in LOG_FIELDS
        }

        if not raw_message and structured_fields:
            structured = " ".join(
                f"{key}={structured_fields[key]}" for key in sorted(structured_fields)
            )
            rendered_message = f"{rendered_message} | {structured}"

        if self.component and "component" not in context_fields:
            context_fields["component"] = self.component

        for key, value in context_fields.items():
            if key not in RESERVED_LOG_RECORD_ATTRS:
                extra[key] = value

        if extra:
            log_kwargs["extra"] = extra

        self.logger.log(level, rendered_message, **log_kwargs)

        if self.enable_console or console_prefix or console_message:
            prefix = console_prefix
            if prefix is None:
                prefix = self.LEVEL_PREFIXES.get(level)

            message_to_emit = console_message or rendered_message
            if console_message:
                message_to_emit = console_message
            elif raw_message:
                message_to_emit = rendered_message
            else:
                context_text = self._format_context(dict(extra))
                if context_text:
                    message_to_emit = f"{message_to_emit} | {context_text}"

            self._emit_console(message_to_emit, prefix=prefix)

    def _format_context(self, context: dict[str, Any]) -> str:
        if not context:
            return ""
        formatted = " ".join(f"{key}={value}" for key, value in sorted(context.items()))
        return formatted

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.log(logging.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.log(logging.ERROR, message, *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.log(logging.DEBUG, message, *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.log(logging.CRITICAL, message, *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("exc_info", True)
        self.error(message, *args, **kwargs)

    def is_enabled_for(self, level: int) -> bool:
        return self.logger.isEnabledFor(level)

    def isEnabledFor(self, level: int) -> bool:  # pragma: no cover - legacy alias
        return self.is_enabled_for(level)

    def success(self, message: str, **kwargs: Any) -> None:
        level, prefix = self.CHANNEL_PREFIXES["success"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def data(self, message: str, **kwargs: Any) -> None:
        level, prefix = self.CHANNEL_PREFIXES["data"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def trading(self, message: str, **kwargs: Any) -> None:
        level, prefix = self.CHANNEL_PREFIXES["trading"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def order(self, message: str, **kwargs: Any) -> None:
        level, prefix = self.CHANNEL_PREFIXES["order"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def position(self, message: str, **kwargs: Any) -> None:
        level, prefix = self.CHANNEL_PREFIXES["position"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def cache(self, message: str, **kwargs: Any) -> None:
        level, prefix = self.CHANNEL_PREFIXES["cache"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def storage(self, message: str, **kwargs: Any) -> None:
        level, prefix = self.CHANNEL_PREFIXES["storage"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def network(self, message: str, **kwargs: Any) -> None:
        level, prefix = self.CHANNEL_PREFIXES["network"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def analysis(self, message: str, **kwargs: Any) -> None:
        level, prefix = self.CHANNEL_PREFIXES["analysis"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def ml(self, message: str, **kwargs: Any) -> None:
        level, prefix = self.CHANNEL_PREFIXES["ml"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def print_section(self, title: str, char: str = "=", width: int = 50) -> None:
        separator = char * width
        self._emit_console(f"\n{separator}\n{title}\n{separator}")

    def print_table(self, headers: list[str], rows: list[list[str]]) -> None:
        if not rows:
            self._emit_console("No data available")
            return

        col_widths = [len(header) for header in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                col_widths[idx] = max(col_widths[idx], len(str(cell)))

        header_row = " | ".join(header.ljust(col_widths[idx]) for idx, header in enumerate(headers))
        separator = "-+-".join("-" * width for width in col_widths)
        self._emit_console(header_row)
        self._emit_console(separator)

        for row in rows:
            row_text = " | ".join(str(cell).ljust(col_widths[idx]) for idx, cell in enumerate(row))
            self._emit_console(row_text)

    def printKeyValue(self, key: str, value: Any, indent: int = 0) -> None:
        padding = " " * indent
        self._emit_console(f"{padding}{key}: {value}")


StructuredLogger = UnifiedLogger


__all__ = ["UnifiedLogger", "StructuredLogger", "LOG_FIELDS", "RESERVED_LOG_RECORD_ATTRS"]
