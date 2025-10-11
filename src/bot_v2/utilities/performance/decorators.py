"""Predefined performance monitoring decorators."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from .timing import measure_performance_decorator

T = TypeVar("T")


def monitor_trading_operation(
    operation_name: str,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    return measure_performance_decorator(
        operation_name=f"trading.{operation_name}",
        tags={"component": "trading"},
    )


def monitor_database_operation(
    operation_name: str,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    return measure_performance_decorator(
        operation_name=f"database.{operation_name}",
        tags={"component": "database"},
    )


def monitor_api_operation(operation_name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    return measure_performance_decorator(
        operation_name=f"api.{operation_name}",
        tags={"component": "api"},
    )


__all__ = [
    "monitor_trading_operation",
    "monitor_database_operation",
    "monitor_api_operation",
]
