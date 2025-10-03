"""Utilities for re-binding orchestration services to their owning bot instance."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for typing
    from bot_v2.orchestration.perps_bot import PerpsBot

logger = logging.getLogger(__name__)


def rebind_bot_services(bot: PerpsBot, attributes: Iterable[str] | None = None) -> None:
    """Ensure coordinator-style services reference the provided bot.

    Many orchestration services cache a `_bot` attribute during construction. When the
    PerpsBot builder hydrates a temporary instance and we later adopt its state, those
    cached references can still point at the throwaway object. This helper walks the
    selected attributes (or every attribute on the bot when ``attributes`` is omitted)
    and re-assigns any `_bot` field so background loops observe the live runtime bot.

    Args:
        bot: The real bot instance that services should reference.
        attributes: Optional collection of attribute names to inspect. When omitted
            we will iterate over ``bot.__dict__`` which covers dynamically-added
            services from tests or future refactors.
    """
    if attributes is None:
        items = tuple(bot.__dict__.items())
    else:
        items = tuple((name, getattr(bot, name, None)) for name in attributes)

    visited: set[int] = set()
    for name, candidate in items:
        if candidate is None:
            continue
        marker = id(candidate)
        if marker in visited:
            continue
        visited.add(marker)

        if not hasattr(candidate, "_bot"):
            continue

        try:
            setattr(candidate, "_bot", bot)
        except Exception:  # pragma: no cover - defensive guard for odd descriptors
            logger.debug("Unable to rebind _bot for %s", name, exc_info=True)
