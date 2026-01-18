"""Strategy initialization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from gpt_trader.config.types import Profile
from gpt_trader.features.live_trade.orchestrator.spot_profile_service import SpotProfileService
from gpt_trader.features.live_trade.strategies.perps_baseline import (
    BaselinePerpsStrategy,
    PerpsStrategy,
    PerpsStrategyConfig,
    SpotStrategy,
    SpotStrategyConfig,
)

from .logging_utils import logger  # naming: allow

if TYPE_CHECKING:  # pragma: no cover
    from gpt_trader.features.live_trade.bot import TradingBot


class StrategyInitializationMixin:
    """Handle strategy initialization for spot and derivatives profiles."""

    def __init__(
        self,
        bot: TradingBot,
        spot_profile_service: SpotProfileService | None = None,
        **_: Any,
    ) -> None:
        self._bot = bot
        self._spot_profiles = spot_profile_service or SpotProfileService(config=bot.config)

    def init_strategy(self) -> None:
        bot = self._bot
        state = bot.runtime_state
        assert state is not None, "Runtime state not initialized"
        derivatives_enabled = bot.config.derivatives_enabled
        strategy_config = getattr(bot.config, "strategy", None)
        risk_config = getattr(bot.config, "risk", None)

        try:
            default_short = int(getattr(strategy_config, "short_ma_period", 5))
        except (TypeError, ValueError):
            default_short = 5
        try:
            default_long = int(getattr(strategy_config, "long_ma_period", 20))
        except (TypeError, ValueError):
            default_long = 20

        strategy_trailing_stop = getattr(strategy_config, "trailing_stop_pct", None)
        trailing_stop_pct: float | None
        if strategy_trailing_stop is not None:
            try:
                trailing_stop_pct = float(strategy_trailing_stop)
            except (TypeError, ValueError):
                trailing_stop_pct = None
        else:
            trailing_stop_pct = None

        if trailing_stop_pct is None:
            risk_trailing_stop = getattr(risk_config, "trailing_stop_pct", None)
            if risk_trailing_stop is not None:
                try:
                    trailing_stop_pct = float(risk_trailing_stop)
                except (TypeError, ValueError):
                    trailing_stop_pct = None

        try:
            target_leverage = int(getattr(risk_config, "target_leverage", 1))
        except (TypeError, ValueError):
            target_leverage = 1
        if bot.config.profile == Profile.SPOT:
            rules = self._spot_profiles.load(bot.config.symbols or [])
            for symbol in bot.config.symbols or []:
                rule = rules.get(symbol, {})
                short = int(rule.get("short_window", default_short))
                long = int(rule.get("long_window", default_long))
                strategy_kwargs = {
                    "short_ma_period": short,
                    "long_ma_period": long,
                    # target_leverage is a read-only property in SpotStrategyConfig (always 1)
                    "trailing_stop_pct": trailing_stop_pct,
                    # enable_shorts defaults to False in SpotStrategyConfig
                    "force_entry_on_trend": True,  # Ignition Phase: Allow trend entry
                }
                fraction_override = rule.get("position_fraction")
                if fraction_override is None:
                    fraction_override = bot.config.perps_position_fraction
                if fraction_override is not None:
                    try:
                        strategy_kwargs["position_fraction"] = float(fraction_override)
                    except (TypeError, ValueError):
                        logger.warning(
                            "Invalid position_fraction=%s for %s; using default",
                            fraction_override,
                            symbol,
                            operation="strategy_init",
                            stage="spot_fraction",
                            symbol=symbol,
                        )
                state.symbol_strategies[symbol] = SpotStrategy(
                    # Dict spread from dynamic config; mypy can't infer types
                    config=SpotStrategyConfig(**strategy_kwargs),  # type: ignore[arg-type]
                    risk_manager=bot.risk_manager,
                )
        else:
            strategy_kwargs = {
                "short_ma_period": default_short,
                "long_ma_period": default_long,
                "target_leverage": target_leverage if derivatives_enabled else 1,
                "trailing_stop_pct": trailing_stop_pct,
                "enable_shorts": bot.config.active_enable_shorts if derivatives_enabled else False,
                "force_entry_on_trend": True,  # Ignition Phase: Allow trend entry
            }
            fraction_override = bot.config.perps_position_fraction
            if fraction_override is not None:
                try:
                    strategy_kwargs["position_fraction"] = float(fraction_override)
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid PERPS_POSITION_FRACTION=%s; using default",
                        fraction_override,
                        operation="strategy_init",
                        stage="perps_fraction",
                    )

            state.strategy = PerpsStrategy(
                # Dict spread from dynamic config; mypy can't infer types
                config=PerpsStrategyConfig(**strategy_kwargs),  # type: ignore[arg-type]
                risk_manager=bot.risk_manager,
            )

    def get_strategy(self, symbol: str) -> BaselinePerpsStrategy:
        bot = self._bot
        state = bot.runtime_state
        assert state is not None, "Runtime state not initialized"
        if bot.config.profile == Profile.SPOT:
            strat = state.symbol_strategies.get(symbol)
            if strat is None:
                # Fallback: Create SpotStrategy for missing symbol
                strat = SpotStrategy(risk_manager=bot.risk_manager)
                state.symbol_strategies[symbol] = strat
            return cast(BaselinePerpsStrategy, strat)
        return cast(BaselinePerpsStrategy, state.strategy)


__all__ = ["StrategyInitializationMixin"]
