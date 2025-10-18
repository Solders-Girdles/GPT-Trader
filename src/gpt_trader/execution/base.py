"""Broker execution abstractions."""

from __future__ import annotations

from typing import Protocol


class Broker(Protocol):
    """Minimal interface for submitting orders."""

    def submit(self, order: object) -> str:
        """Submit an order and return a broker-specific identifier."""
        raise NotImplementedError


__all__ = ["Broker"]
