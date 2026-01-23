"""Curated optional imports aligned with supported dependency groups."""

from __future__ import annotations

from .optional import optional_import

# Optional dependencies (extras or optional import patterns).
pandas = optional_import("pandas")
aiohttp = optional_import("aiohttp")
websocket = optional_import("websocket")
psutil = optional_import("psutil")
opentelemetry = optional_import("opentelemetry")


__all__ = [
    "pandas",
    "aiohttp",
    "websocket",
    "psutil",
    "opentelemetry",
]
