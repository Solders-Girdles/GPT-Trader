"""Shared abstractions for legacy coordinator facades."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from bot_v2.orchestration.context_builder import build_coordinator_context
from bot_v2.orchestration.coordinators.base import CoordinatorContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bot_v2.orchestration.perps_bot import PerpsBot

__all__ = [
    "BaseCoordinatorFacade",
    "ContextPreservingCoordinator",
]

P = ParamSpec("P")
T = TypeVar("T")


class BaseCoordinatorFacade:
    """Provide common plumbing for facades wrapping coordinator implementations."""

    __slots__ = ("_bot",)

    def _setup_facade(
        self,
        bot: PerpsBot | None,
        *,
        overrides: dict[str, Any] | None = None,
        placeholder_factory: Callable[[], CoordinatorContext] | None = None,
    ) -> CoordinatorContext:
        """Initialise facade state and return the coordinator context to bootstrap with."""

        self._bot = bot
        if bot is None:
            if placeholder_factory is None:
                raise ValueError("Facade requires a placeholder context when bot is None.")
            return placeholder_factory()
        return self._build_context(bot, overrides=overrides)

    # ------------------------------------------------------------------
    def _context_overrides(self, bot: PerpsBot) -> dict[str, Any]:
        """Return coordinator context overrides to enforce for this facade."""

        return {}

    def _build_context(
        self,
        bot: PerpsBot,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> CoordinatorContext:
        """Construct a CoordinatorContext for the provided bot."""

        facade_overrides = dict(self._context_overrides(bot))
        if overrides:
            facade_overrides.update(overrides)
        return build_coordinator_context(bot, overrides=facade_overrides)

    def _apply_context_overrides(self, context: CoordinatorContext) -> CoordinatorContext:
        """Ensure the current context includes required overrides."""

        bot = getattr(self, "_bot", None)
        if bot is None:
            return context

        bot_config = getattr(bot, "config", None)
        if bot_config is not None and getattr(context, "config", None) is not bot_config:
            context = context.with_updates(config=bot_config)

        normalized_symbols: tuple[str, ...] | None = None
        bot_symbols = getattr(bot, "symbols", None)
        if bot_symbols is not None:
            try:
                normalized_symbols = tuple(bot_symbols)
            except TypeError:
                normalized_symbols = None
        if normalized_symbols is None and bot_config is not None:
            config_symbols = getattr(bot_config, "symbols", None)
            if config_symbols is not None:
                try:
                    normalized_symbols = tuple(config_symbols)
                except TypeError:
                    normalized_symbols = None
        if normalized_symbols is not None and normalized_symbols != getattr(
            context, "symbols", None
        ):
            context = context.with_updates(symbols=normalized_symbols)

        overrides = self._context_overrides(bot)
        if not overrides:
            return context

        replacements: dict[str, Any] = {}
        for key, value in overrides.items():
            if value is None:
                continue
            current = getattr(context, key, None)
            if current is not value:
                replacements[key] = value

        if replacements:
            return context.with_updates(**replacements)
        return context

    def _facade_update_context(
        self,
        context: CoordinatorContext,
        *,
        sync: bool = True,
    ) -> None:
        """Update the coordinator context while enforcing facade overrides."""

        context = self._apply_context_overrides(context)
        super().update_context(context)  # type: ignore[misc]
        if sync and getattr(self, "_bot", None) is not None:
            self._sync_bot(context)

    def update_context(self, context: CoordinatorContext) -> None:  # type: ignore[override]
        """Override update_context to keep bot state in sync."""

        self._facade_update_context(context)

    def _refresh_context_from_bot(
        self,
        *,
        overrides: dict[str, Any] | None = None,
        sync: bool = False,
    ) -> CoordinatorContext:
        """Rebuild coordinator context from the bot and update the facade."""

        bot = getattr(self, "_bot", None)
        if bot is None:
            return self.context  # type: ignore[attr-defined]

        context = self._build_context(bot, overrides=overrides)
        self._facade_update_context(context, sync=sync)
        return self.context  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    def _sync_bot(self, context: CoordinatorContext) -> None:
        """Propagate coordinator context changes back to the bot."""

        raise NotImplementedError


class ContextPreservingCoordinator:
    """Mixin providing decorators for context refresh and sync orchestration."""

    @classmethod
    def context_action(
        cls,
        *,
        sync_after: bool = False,
        pass_context: bool = False,
        overrides: Callable[[Any], dict[str, Any] | None] | None = None,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """Wrap a method to refresh context before execution and optionally sync after."""

        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            if inspect.iscoroutinefunction(func):

                @wraps(func)
                async def async_wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
                    override_values = overrides(self) if overrides else None
                    self._refresh_context_from_bot(overrides=override_values, sync=False)
                    context = self.context  # type: ignore[attr-defined]

                    call_args = args
                    if pass_context:
                        call_args = (context, *args)

                    result = await func(self, *call_args, **kwargs)  # type: ignore[arg-type]
                    if sync_after:
                        self._sync_bot(self.context)  # type: ignore[attr-defined]
                    return result

                return async_wrapper  # type: ignore[return-value]

            @wraps(func)
            def sync_wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> T:
                override_values = overrides(self) if overrides else None
                self._refresh_context_from_bot(overrides=override_values, sync=False)
                context = self.context  # type: ignore[attr-defined]

                call_args = args
                if pass_context:
                    call_args = (context, *args)

                result = func(self, *call_args, **kwargs)  # type: ignore[arg-type]
                if sync_after:
                    self._sync_bot(self.context)  # type: ignore[attr-defined]
                return result

            return sync_wrapper  # type: ignore[return-value]

        return decorator

    async def _schedule_with_context(
        self,
        coroutine: Callable[[], Awaitable[T]],
        *,
        overrides: dict[str, Any] | None = None,
        sync_after: bool = False,
    ) -> T:
        """Helper for async call-sites needing explicit control."""

        self._refresh_context_from_bot(overrides=overrides, sync=False)
        result = await coroutine()
        if sync_after:
            self._sync_bot(self.context)  # type: ignore[attr-defined]
        return result
