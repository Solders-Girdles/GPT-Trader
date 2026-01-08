"""Observability package for metrics and tracing."""

from __future__ import annotations

from gpt_trader.observability.tracing import init_tracing, trace_span

__all__ = ["init_tracing", "trace_span"]
