from __future__ import annotations

import asyncio
import logging
import time as _time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from bot_v2.features.brokerages.core.interfaces import Balance, Position
from bot_v2.features.live_trade.indicators import mean_decimal as _mean_decimal
from bot_v2.features.live_trade.indicators import (
    relative_strength_index as _rsi_from_closes,
)
from bot_v2.features.live_trade.indicators import to_decimal as _to_decimal
from bot_v2.features.live_trade.indicators import true_range as _true_range
from bot_v2.features.live_trade.risk_runtime import CircuitBreakerAction
from bot_v2.features.live_trade.strategies.perps_baseline import (
    Action,
    BaselinePerpsStrategy,
    Decision,
    StrategyConfig,
)
from bot_v2.monitoring.system import get_logger as _get_plog
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.spot_profile_service import SpotProfileService
from bot_v2.utilities.quantities import quantity_from

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.perps_bot import PerpsBot

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SymbolProcessingContext:
    symbol: str
    balances: Sequence[Balance]
    equity: Decimal
    positions: dict[str, Position]
    position_state: dict[str, Any] | None
    position_quantity: Decimal
    marks: list[Decimal]
    product: Any | None


class StrategyOrchestrator:
    """Encapsulates strategy initialization and decision execution per symbol."""

    requires_context = True

    def __init__(
        self,
        bot: PerpsBot,
        spot_profile_service: SpotProfileService | None = None,
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
                        )
                state.symbol_strategies[symbol] = BaselinePerpsStrategy(
                    config=StrategyConfig(**strategy_kwargs),  # type: ignore[arg-type]
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
                        "Invalid PERPS_POSITION_FRACTION=%s; using default", fraction_override
                    )

            state.strategy = BaselinePerpsStrategy(
                config=StrategyConfig(**strategy_kwargs),  # type: ignore[arg-type]
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
        return state.strategy  # type: ignore[attr-defined,return-value]

    async def process_symbol(
        self,
        symbol: str,
        balances: Sequence[Balance] | None = None,
        position_map: dict[str, Position] | None = None,
    ) -> None:
        bot = self._bot
        try:
            context = await self._prepare_context(symbol, balances, position_map)
            if context is None:
                return

            decision = await self._resolve_decision(context)
            self._record_decision(symbol, decision)

            if decision.action in {Action.BUY, Action.SELL, Action.CLOSE}:
                await bot.execute_decision(
                    symbol,
                    decision,
                    context.marks[-1],
                    context.product,
                    context.position_state,
                )
        except Exception as exc:
            logger.error("Error processing %s: %s", symbol, exc, exc_info=True)

    async def _prepare_context(
        self,
        symbol: str,
        balances: Sequence[Balance] | None,
        position_map: dict[str, Position] | None,
    ) -> SymbolProcessingContext | None:
        balances = await self._ensure_balances(balances)
        equity = self._extract_equity(balances)
        if self._kill_switch_engaged():
            return None

        positions_lookup = await self._ensure_positions(position_map)
        position_state, position_quantity = self._build_position_state(symbol, positions_lookup)

        marks = self._get_marks(symbol)
        if not marks:
            return None

        adjusted_equity = self._adjust_equity(equity, position_quantity, marks, symbol)
        if adjusted_equity == Decimal("0"):
            logger.error("No equity info for %s", symbol)
            return None

        product = None
        try:
            product = self._bot.get_product(symbol)
        except Exception:
            logger.debug("Failed to fetch product metadata for %s", symbol, exc_info=True)

        context = SymbolProcessingContext(
            symbol=symbol,
            balances=balances,
            equity=adjusted_equity,
            positions=positions_lookup,
            position_state=position_state,
            position_quantity=position_quantity,
            marks=list(marks),
            product=product,
        )

        if not self._run_risk_gates(context):
            return None

        return context

    async def _resolve_decision(self, context: SymbolProcessingContext) -> Decision:
        strategy = self.get_strategy(context.symbol)
        decision = self._evaluate_strategy(
            strategy,
            context.symbol,
            context.marks,
            context.position_state,
            context.equity,
            context.product,
        )

        if self._bot.config.profile == Profile.SPOT:
            decision = await self._apply_spot_filters(context, decision)
        return decision

    async def _ensure_balances(self, balances: Sequence[Balance] | None) -> Sequence[Balance]:
        if balances is not None:
            return balances
        return await asyncio.to_thread(self._bot.broker.list_balances)

    def _extract_equity(self, balances: Sequence[Balance]) -> Decimal:
        cash_assets = {"USD", "USDC"}
        usd_balance = next(
            (b for b in balances if getattr(b, "asset", "").upper() in cash_assets),
            None,
        )
        return cast(Decimal, usd_balance.total) if usd_balance is not None else Decimal("0")

    def _kill_switch_engaged(self) -> bool:
        bot = self._bot
        if getattr(bot.risk_manager.config, "kill_switch_enabled", False):
            logger.warning("Kill switch enabled - skipping trading loop")
            return True
        return False

    async def _ensure_positions(
        self, position_map: dict[str, Position] | None
    ) -> dict[str, Position]:
        if position_map is not None:
            return position_map
        positions = await asyncio.to_thread(self._bot.broker.list_positions)
        return {p.symbol: p for p in positions if hasattr(p, "symbol")}

    def _build_position_state(
        self, symbol: str, positions_lookup: dict[str, Position]
    ) -> tuple[dict[str, Any] | None, Decimal]:
        if symbol not in positions_lookup:
            return None, Decimal("0")

        pos = positions_lookup[symbol]
        quantity_val = quantity_from(pos, default=Decimal("0"))
        state = {
            "quantity": quantity_val,
            "side": getattr(pos, "side", "long"),
            "entry": getattr(pos, "entry_price", None),
        }
        try:
            quantity = Decimal(str(quantity_val))
        except Exception:
            quantity = Decimal("0")
        return state, quantity

    def _get_marks(self, symbol: str) -> list[Decimal]:
        raw_marks = self._bot.mark_windows.get(symbol, [])
        marks = [Decimal(str(mark)) for mark in raw_marks]
        if not marks:
            logger.warning(f"No marks for {symbol}")
        return marks

    def _adjust_equity(
        self, equity: Decimal, position_quantity: Decimal, marks: Sequence[Decimal], symbol: str
    ) -> Decimal:
        if position_quantity and marks:
            try:
                equity += abs(position_quantity) * marks[-1]
            except Exception as exc:
                logger.debug(
                    "Failed to adjust equity for %s position: %s", symbol, exc, exc_info=True
                )
        return equity

    def _run_risk_gates(self, context: SymbolProcessingContext) -> bool:
        bot = self._bot
        try:
            window = context.marks[-max(bot.config.long_ma, 20) :]
            cb = bot.risk_manager.check_volatility_circuit_breaker(context.symbol, list(window))
            if cb.triggered and cb.action is CircuitBreakerAction.KILL_SWITCH:
                logger.warning("Kill switch tripped by volatility CB for %s", context.symbol)
                return False
        except Exception as exc:
            logger.debug(
                "Volatility circuit breaker check failed for %s: %s",
                context.symbol,
                exc,
                exc_info=True,
            )

        try:
            if bot.risk_manager.check_mark_staleness(context.symbol):
                logger.warning("Skipping %s due to stale market data", context.symbol)
                return False
        except Exception as exc:
            logger.debug(
                "Mark staleness check failed for %s: %s", context.symbol, exc, exc_info=True
            )

        return True

    def _evaluate_strategy(
        self,
        strategy: BaselinePerpsStrategy,
        symbol: str,
        marks: Sequence[Decimal],
        position_state: dict[str, Any] | None,
        equity: Decimal,
        product: Any | None,
    ) -> Decision:
        _t0 = _time.perf_counter()
        decision = strategy.decide(
            symbol=symbol,
            current_mark=marks[-1],
            position_state=position_state,
            recent_marks=list(marks[:-1]) if len(marks) > 1 else [],
            equity=equity,
            product=product,
        )
        _dt_ms = (_time.perf_counter() - _t0) * 1000.0
        try:
            _get_plog().log_strategy_duration(strategy=type(strategy).__name__, duration_ms=_dt_ms)
        except Exception as exc:
            logger.debug("Failed to log strategy duration: %s", exc, exc_info=True)
        return decision

    def _record_decision(self, symbol: str, decision: Decision) -> None:
        self._bot.last_decisions[symbol] = decision
        logger.info(f"{symbol} Decision: {decision.action.value} - {decision.reason}")

    async def _apply_spot_filters(
        self, context: SymbolProcessingContext, decision: Decision
    ) -> Decision:
        rules = self._spot_profiles.get(context.symbol)
        if not rules or decision.action != Action.BUY:
            return decision
        if context.position_state:
            position_qty_raw = quantity_from(context.position_state, default=Decimal("0"))
            position_qty = (
                position_qty_raw if isinstance(position_qty_raw, Decimal) else Decimal("0")
            )
            if position_qty != Decimal("0"):
                return decision

        needs_data = False
        max_window = 0

        vol_config = rules.get("volatility_filter") if isinstance(rules, dict) else None
        if isinstance(vol_config, dict):
            window = int(vol_config.get("window", 0))
            if window > 0:
                needs_data = True
                max_window = max(max_window, window)

        volma_config = rules.get("volume_filter") if isinstance(rules, dict) else None
        if isinstance(volma_config, dict):
            window = int(volma_config.get("window", 0))
            if window > 0:
                needs_data = True
                max_window = max(max_window, window)

        momentum_config = rules.get("momentum_filter") if isinstance(rules, dict) else None
        if isinstance(momentum_config, dict):
            window = int(momentum_config.get("window", 0))
            if window > 0:
                needs_data = True
                max_window = max(max_window, window)

        trend_config = rules.get("trend_filter") if isinstance(rules, dict) else None
        if isinstance(trend_config, dict):
            window = int(trend_config.get("window", 0))
            if window > 0:
                needs_data = True
                max_window = max(max_window, window)

        if not needs_data:
            return decision

        symbol = context.symbol
        candles = await self._fetch_spot_candles(symbol, max_window)
        if not candles:
            logger.debug("Insufficient candle data for %s; deferring entry", symbol)
            return Decision(action=Action.HOLD, reason="indicator_data_unavailable")

        closes = [Decimal(str(_to_decimal(getattr(c, "close", 0)))) for c in candles]
        volumes = [Decimal(str(_to_decimal(getattr(c, "volume", 0)))) for c in candles]
        highs = [Decimal(str(_to_decimal(getattr(c, "high", 0)))) for c in candles]
        lows = [Decimal(str(_to_decimal(getattr(c, "low", 0)))) for c in candles]

        if isinstance(volma_config, dict):
            window = int(volma_config.get("window", 0))
            multiplier = _to_decimal(volma_config.get("multiplier", 1))
            if window > 0:
                if len(volumes) < window + 1:
                    return Decision(action=Action.HOLD, reason="volume_filter_wait")
                recent = volumes[-(window + 1) : -1]
                avg_vol = _mean_decimal(recent)
                latest_vol = volumes[-1]
                if avg_vol <= Decimal("0") or latest_vol < avg_vol * multiplier:
                    logger.info("%s entry blocked by volume filter", context.symbol)
                    return Decision(action=Action.HOLD, reason="volume_filter_blocked")

        if isinstance(momentum_config, dict):
            window = int(momentum_config.get("window", 0))
            _overbought = _to_decimal(momentum_config.get("overbought", 70))
            oversold = _to_decimal(momentum_config.get("oversold", 30))
            if window > 0:
                if len(closes) < window + 1:
                    return Decision(action=Action.HOLD, reason="momentum_filter_wait")
                rsi = _rsi_from_closes(closes[-(window + 1) :])
                if rsi > oversold:
                    logger.info(
                        "%s entry blocked by momentum filter (RSI=%.2f)",
                        context.symbol,
                        float(rsi),
                    )
                    return Decision(action=Action.HOLD, reason="momentum_filter_blocked")

        if isinstance(trend_config, dict):
            window = int(trend_config.get("window", 0))
            min_slope = _to_decimal(trend_config.get("min_slope", 0))
            if window > 0:
                if len(closes) < window + 1:
                    return Decision(action=Action.HOLD, reason="trend_filter_wait")
                current_ma = _mean_decimal(closes[-window:])
                prev_ma = _mean_decimal(closes[-(window + 1) : -1])
                slope = (current_ma - prev_ma) / Decimal(window)
                if slope < min_slope:
                    logger.info(
                        "%s entry blocked by trend filter (slope=%.6f)",
                        context.symbol,
                        float(slope),
                    )
                    return Decision(action=Action.HOLD, reason="trend_filter_blocked")

        if isinstance(vol_config := rules.get("volatility_filter"), dict):
            window = int(vol_config.get("window", 0))
            min_vol = _to_decimal(vol_config.get("min_vol", 0))
            max_vol = _to_decimal(vol_config.get("max_vol", 1))
            if window > 0:
                if len(closes) < window + 1:
                    return Decision(action=Action.HOLD, reason="volatility_filter_wait")
                atr_values: list[Decimal] = []
                prev_close: Decimal | None = None
                start_idx = max(len(closes) - window - 1, 0)
                for idx in range(start_idx, len(closes)):
                    if prev_close is None and idx > 0:
                        prev_close = closes[idx - 1]
                    tr = _true_range(highs[idx], lows[idx], prev_close)
                    atr_values.append(tr)
                    prev_close = closes[idx]
                atr = _mean_decimal(atr_values[-window:])
                if atr <= Decimal("0"):
                    return Decision(action=Action.HOLD, reason="volatility_filter_blocked")
                vol_pct = atr / closes[-1]
                if vol_pct < min_vol or vol_pct > max_vol:
                    logger.info(
                        "%s entry blocked by volatility filter (%.6f)",
                        context.symbol,
                        float(vol_pct),
                    )
                    return Decision(action=Action.HOLD, reason="volatility_filter_blocked")

        return decision

    async def _fetch_spot_candles(self, symbol: str, window: int) -> list[Any]:
        bot = self._bot
        limit = max(window + 2, 10)
        try:
            candles = await asyncio.to_thread(
                bot.broker.get_candles,
                symbol,
                "ONE_HOUR",
                limit,
            )
        except Exception as exc:
            logger.debug("Failed to fetch candles for %s: %s", symbol, exc, exc_info=True)
            return []
        if not candles:
            return []
        return sorted(
            candles,
            key=lambda c: getattr(c, "ts", getattr(c, "timestamp", datetime.utcnow())),
        )
