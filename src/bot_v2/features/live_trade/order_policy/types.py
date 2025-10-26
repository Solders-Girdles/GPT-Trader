"""Typed dictionary helpers for order policy operations."""

from __future__ import annotations

from typing import TypedDict


class OrderConfig(TypedDict, total=False):
    order_type: str
    tif: str
    post_only: bool
    reduce_only: bool
    use_market: bool
    fallback_reason: str
    error: str


class SupportedOrderConfig(TypedDict):
    order_type: str
    tif: str
    post_only: bool
    reduce_only: bool


__all__ = ["OrderConfig", "SupportedOrderConfig"]
