"""Strategy initialization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bot_v2.features.live_trade.strategies.perps_baseline import (
    BaselinePerpsStrategy,
    StrategyConfig,
)
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.spot_profile_service import SpotProfileService

from .logging_utils import logger

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.orchestration.perps_bot import PerpsBot


class StrategyInitializationMixin:
    """Handle strategy initialization for spot and derivatives profiles."""

    def __init__(
        self,
        bot: PerpsBot,
        spot_profile_service: SpotProfileService | None = None,
        **_: Any,
    ) -> None:
        self._bot = bot
        self._spot_profiles = spot_profile_service or SpotProfileService()

    def init_strategy(self) -> None:
        bot = self._bot
        state = bot.runtime_state
        derivatives_enabled = bot.config.derivatives_enabled
        if bot.config.profile == Profile.SPOT:
            rules = self._spot_profiles.load(bot.config.symbols or [])
            for symbol in bot.config.symbols or []:
                rule = rules.get(symbol, {})
                short = int(rule.get("short_window", bot.config.short_ma))
                long = int(rule.get("long_window", bot.config.long_ma))
                strategy_kwargs = {
                    "short_ma_period": short,
                    "long_ma_period": long,
                    "target_leverage": 1,
                    "trailing_stop_pct": bot.config.trailing_stop_pct,
                    "enable_shorts": False,
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
                state.symbol_strategies[symbol] = BaselinePerpsStrategy(
                    config=StrategyConfig(**strategy_kwargs),
                    risk_manager=bot.risk_manager,
                )
        else:
            strategy_kwargs = {
                "short_ma_period": bot.config.short_ma,
                "long_ma_period": bot.config.long_ma,
                "target_leverage": bot.config.target_leverage if derivatives_enabled else 1,
                "trailing_stop_pct": bot.config.trailing_stop_pct,
                "enable_shorts": bot.config.enable_shorts if derivatives_enabled else False,
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

            state.strategy = BaselinePerpsStrategy(
                config=StrategyConfig(**strategy_kwargs),
                risk_manager=bot.risk_manager,
            )

    def get_strategy(self, symbol: str) -> BaselinePerpsStrategy:
        bot = self._bot
        state = bot.runtime_state
        if bot.config.profile == Profile.SPOT:
            strat = state.symbol_strategies.get(symbol)
            if strat is None:
                strat = BaselinePerpsStrategy(risk_manager=bot.risk_manager)
                state.symbol_strategies[symbol] = strat
            return strat
        return state.strategy  # type: ignore[return-value]


__all__ = ["StrategyInitializationMixin"]
