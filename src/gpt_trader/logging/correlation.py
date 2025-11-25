"""Correlation ID management for tracking requests across the system."""

from __future__ import annotations

import contextvars
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# Context variable to store the current correlation ID
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)

# Context variable to store domain-specific context
domain_context_var: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "domain_context", default={}
)


def get_correlation_id() -> str:
    """Get the current correlation ID from the context."""
    return correlation_id_var.get("")


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID in the current context."""
    correlation_id_var.set(correlation_id)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def get_domain_context() -> dict[str, Any]:
    """Get the current domain context from the context."""
    return domain_context_var.get({})


def set_domain_context(context: dict[str, Any]) -> None:
    """Set the domain context in the current context."""
    domain_context_var.set(context)


def update_domain_context(**kwargs: Any) -> None:
    """Update the domain context with additional key-value pairs."""
    current_context = get_domain_context()
    updated_context = {**current_context, **kwargs}
    set_domain_context(updated_context)


def add_domain_field(key: str, value: Any) -> None:
    """Add a single field to the domain context."""
    current_context = get_domain_context()
    current_context[key] = value
    set_domain_context(current_context)


@contextmanager
def correlation_context(correlation_id: str | None = None, **domain_fields: Any) -> Iterator[None]:
    """Context manager for setting correlation ID and domain fields.

    Args:
        correlation_id: Optional correlation ID. If None, a new one will be generated.
        **domain_fields: Domain-specific fields to include in the context.

    Yields:
        None
    """
    # Store the current context to restore later
    token_correlation = correlation_id_var.set(correlation_id or generate_correlation_id())

    # Store and update domain context
    current_domain = get_domain_context()
    updated_domain = {**current_domain, **domain_fields}
    token_domain = domain_context_var.set(updated_domain)

    try:
        yield
    finally:
        # Restore the previous context
        correlation_id_var.reset(token_correlation)
        domain_context_var.reset(token_domain)


@contextmanager
def symbol_context(symbol: str, **additional_fields: Any) -> Iterator[None]:
    """Context manager for setting symbol-specific context.

    Args:
        symbol: Trading symbol
        **additional_fields: Additional domain fields

    Yields:
        None
    """
    add_domain_field("symbol", symbol)
    if additional_fields:
        update_domain_context(**additional_fields)

    try:
        yield
    finally:
        # Note: We don't clean up the symbol context as it might be needed
        # for the duration of the operation
        pass


@contextmanager
def order_context(
    order_id: str, symbol: str | None = None, **additional_fields: Any
) -> Iterator[None]:
    """Context manager for setting order-specific context.

    Args:
        order_id: Order ID
        symbol: Optional trading symbol
        **additional_fields: Additional domain fields

    Yields:
        None
    """
    fields = {"order_id": order_id}
    if symbol:
        fields["symbol"] = symbol
    fields.update(additional_fields)

    update_domain_context(**fields)

    try:
        yield
    finally:
        # Note: We don't clean up the order context as it might be needed
        # for the duration of the operation
        pass


def get_log_context() -> dict[str, Any]:
    """Get the complete log context including correlation ID and domain fields.

    Returns:
        Dictionary with correlation_id and domain fields
    """
    context = {}

    correlation_id = get_correlation_id()
    if correlation_id:
        context["correlation_id"] = correlation_id

    domain_context = get_domain_context()
    if domain_context:
        context.update(domain_context)

    return context
