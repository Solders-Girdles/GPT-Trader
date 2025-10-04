from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot_v2.features.brokerages.core.interfaces import Balance, Position
from bot_v2.features.live_trade.indicators import mean_decimal as _mean_decimal
from bot_v2.features.live_trade.indicators import (
    relative_strength_index as _rsi_from_closes,
)
from bot_v2.features.live_trade.indicators import to_decimal as _to_decimal
from bot_v2.features.live_trade.indicators import true_range as _true_range
from bot_v2.features.live_trade.strategies.perps_baseline import (
    Action,
    BaselinePerpsStrategy,
    Decision,
)
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.equity_calculator import EquityCalculator
from bot_v2.orchestration.risk_gate_validator import RiskGateValidator
from bot_v2.orchestration.spot_profile_service import SpotProfileService
from bot_v2.orchestration.strategy_executor import StrategyExecutor
from bot_v2.orchestration.strategy_registry import StrategyRegistry
from bot_v2.utilities.quantities import quantity_from

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.perps_bot import PerpsBot

logger = logging.getLogger(__name__)


class StrategyOrchestrator:
    """Encapsulates strategy initialization and decision execution per symbol."""

    def __init__(
        self,
        bot: PerpsBot,
        spot_profile_service: SpotProfileService | None = None,
        equity_calculator: EquityCalculator | None = None,
        risk_gate_validator: RiskGateValidator | None = None,
        strategy_registry: StrategyRegistry | None = None,
        strategy_executor: StrategyExecutor | None = None,
    ) -> None:
        self._bot = bot
        self._spot_profiles = spot_profile_service or SpotProfileService()
        self.equity_calculator = equity_calculator or EquityCalculator()
        self._risk_gate_validator = risk_gate_validator
        self._strategy_registry = strategy_registry
        self._strategy_executor = strategy_executor

    @property
    def risk_gate_validator(self) -> RiskGateValidator:
        """Get or create risk gate validator (lazy initialization)."""
        if self._risk_gate_validator is None:
            self._risk_gate_validator = RiskGateValidator(self._bot.risk_manager)
        return self._risk_gate_validator

    @property
    def strategy_registry(self) -> StrategyRegistry:
        """Get or create strategy registry (lazy initialization)."""
        if self._strategy_registry is None:
            self._strategy_registry = StrategyRegistry(
                self._bot.config,
                self._bot.risk_manager,
                self._spot_profiles,
            )
        return self._strategy_registry

    @property
    def strategy_executor(self) -> StrategyExecutor:
        """Get or create strategy executor (lazy initialization)."""
        if self._strategy_executor is None:
            self._strategy_executor = StrategyExecutor(self._bot)
        return self._strategy_executor

    def init_strategy(self) -> None:
        """Initialize strategies via strategy registry."""
        self.strategy_registry.initialize()

        # Sync strategies to bot for backward compatibility
        bot = self._bot
        if bot.config.profile == Profile.SPOT:
            bot._symbol_strategies = self.strategy_registry.symbol_strategies
        else:
            bot.strategy = self.strategy_registry.default_strategy

    def get_strategy(self, symbol: str) -> BaselinePerpsStrategy:
        """Get strategy for symbol via strategy registry."""
        return self.strategy_registry.get_strategy(symbol)

    async def process_symbol(
        self,
        symbol: str,
        balances: Sequence[Balance] | None = None,
        position_map: dict[str, Position] | None = None,
    ) -> None:
        bot = self._bot
        try:
            balances = await self._ensure_balances(balances)
            if self._kill_switch_engaged():
                return

            positions_lookup = await self._ensure_positions(position_map)
            position_state, position_quantity = self._build_position_state(symbol, positions_lookup)

            marks = self._get_marks(symbol)
            if not marks:
                return

            # Calculate total equity (cash + position value)
            equity = self.equity_calculator.calculate(
                balances=balances,
                position_quantity=position_quantity,
                current_mark=marks[-1] if marks else None,
                symbol=symbol,
            )
            if equity == Decimal("0"):
                logger.error(f"No equity info for {symbol}")
                return

            if not self.risk_gate_validator.validate_gates(
                symbol, marks, lookback_window=max(bot.config.long_ma, 20)
            ):
                return

            strategy_obj = self.get_strategy(symbol)
            decision = self.strategy_executor.evaluate(
                strategy_obj, symbol, marks, position_state, equity
            )

            if bot.config.profile == Profile.SPOT:
                decision = await self._apply_spot_filters(symbol, decision, position_state)

            self.strategy_executor.record_decision(symbol, decision)

            if decision.action in {Action.BUY, Action.SELL, Action.CLOSE}:
                await bot.execute_decision(
                    symbol, decision, marks[-1], bot.get_product(symbol), position_state
                )
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    async def _ensure_balances(self, balances: Sequence[Balance] | None) -> Sequence[Balance]:
        if balances is not None:
            return balances
        return await asyncio.to_thread(self._bot.broker.list_balances)

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
        marks = self._bot.mark_windows.get(symbol, [])
        if not marks:
            logger.warning(f"No marks for {symbol}")
        return marks

    async def _apply_spot_filters(
        self, symbol: str, decision: Decision, position_state: dict[str, Any] | None
    ) -> Decision:
        rules = self._spot_profiles.get(symbol)
        if not rules or decision.action != Action.BUY:
            return decision
        if position_state and quantity_from(position_state) != Decimal("0"):
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

        candles = await self._fetch_spot_candles(symbol, max_window)
        if not candles:
            logger.debug("Insufficient candle data for %s; deferring entry", symbol)
            return Decision(action=Action.HOLD, reason="indicator_data_unavailable")

        closes = [_to_decimal(getattr(c, "close", 0)) for c in candles]
        volumes = [_to_decimal(getattr(c, "volume", 0)) for c in candles]
        highs = [_to_decimal(getattr(c, "high", 0)) for c in candles]
        lows = [_to_decimal(getattr(c, "low", 0)) for c in candles]

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
                    logger.info("%s entry blocked by volume filter", symbol)
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
                        "%s entry blocked by momentum filter (RSI=%.2f)", symbol, float(rsi)
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
                        "%s entry blocked by trend filter (slope=%.6f)", symbol, float(slope)
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
                        "%s entry blocked by volatility filter (%.6f)", symbol, float(vol_pct)
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
