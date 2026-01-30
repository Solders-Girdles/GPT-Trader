"""Research backtesting adapter onto the canonical engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from gpt_trader.backtesting.simulation.broker import SimulatedBroker
from gpt_trader.backtesting.simulation.fee_calculator import FeeCalculator
from gpt_trader.backtesting.simulation.fill_model import OrderFillModel
from gpt_trader.backtesting.types import SimulationConfig
from gpt_trader.core import Action, Candle, Decision
from gpt_trader.core.account import Order
from gpt_trader.core.trading import OrderSide, OrderStatus, OrderType, TimeInForce
from gpt_trader.features.live_trade.strategies.base import MarketDataContext
from gpt_trader.features.research.backtesting.data_loader import HistoricalDataPoint
from gpt_trader.features.research.backtesting.simulator import (
    BacktestConfig,
    BacktestResult,
    Position,
    SimulatedOrder,
    SimulatedTrade,
)
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="backtest_adapter")


class StrategyProtocol(Protocol):
    """Protocol for strategies compatible with backtesting."""

    def decide(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: dict[str, object] | None,
        recent_marks: list[Decimal],
        equity: Decimal,
        product: object | None,
        market_data: MarketDataContext | None = None,
    ) -> Decision:
        """Generate a trading decision."""
        ...


@dataclass
class PendingOrder:
    """Pending order scheduled for a future bar."""

    due_index: int
    order: SimulatedOrder
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    price: Decimal | None = None
    stop_price: Decimal | None = None
    tif: TimeInForce | str | None = None
    reduce_only: bool | None = None


class FlatFeeCalculator:
    """Flat fee calculator for legacy fee_rate_bps support."""

    def __init__(self, fee_rate_bps: float) -> None:
        self._fee_rate_bps = Decimal(str(fee_rate_bps))

    def calculate(self, notional_usd: Decimal, is_maker: bool) -> Decimal:  # noqa: ARG002
        return notional_usd * self._fee_rate_bps / Decimal("10000")


class BacktestSimulator:
    """Adapter simulator that delegates execution to the canonical broker."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self._orders: list[SimulatedOrder] = []
        self._trades: list[SimulatedTrade] = []
        self._pending_orders: list[PendingOrder] = []
        self._recent_marks: list[Decimal] = []
        self._order_counter = 0
        self._broker: SimulatedBroker | None = None
        self._fill_model: OrderFillModel | None = None
        self._fee_calculator: FlatFeeCalculator | FeeCalculator
        self._broker_order_map: dict[str, SimulatedOrder] = {}
        self._broker_trade_recorded: set[str] = set()
        if self.config.use_tiered_fees:
            self._fee_calculator = FeeCalculator(tier=self.config.fee_tier)
        else:
            self._fee_calculator = FlatFeeCalculator(self.config.fee_rate_bps)

    def run(
        self,
        strategy: StrategyProtocol,
        data_points: list[HistoricalDataPoint],
        symbol: str | None = None,
    ) -> BacktestResult:
        if not data_points:
            return BacktestResult(
                trades=[],
                orders=[],
                final_equity=self.config.initial_equity,
                final_position=Position(symbol=symbol or "UNKNOWN"),
            )

        if symbol is None:
            symbol = data_points[0].symbol

        self._reset_state()
        broker = self._build_broker(symbol, data_points[0].timestamp, data_points[-1].timestamp)
        self._broker = broker
        self._fill_model = broker._fill_model

        logger.info(
            "Starting backtest",
            symbol=symbol,
            data_points=len(data_points),
            initial_equity=str(self.config.initial_equity),
        )

        for i, point in enumerate(data_points):
            self._recent_marks.append(point.mark_price)
            if len(self._recent_marks) > 100:
                self._recent_marks = self._recent_marks[-100:]

            position_before = broker.get_position(point.symbol)

            candle = self._build_candle(point)
            broker.update_bar(symbol, candle)
            self._sync_broker_orders(broker)

            self._process_pending_orders(i, point, broker)
            self._process_funding(point, broker)

            market_data = MarketDataContext(
                orderbook_snapshot=point.orderbook_snapshot,
                trade_volume_stats=point.trade_flow_stats,
                spread_bps=Decimal(str(point.spread_bps)) if point.spread_bps else None,
            )

            decision = strategy.decide(
                symbol=point.symbol,
                current_mark=point.mark_price,
                position_state=self._position_state(broker, point.symbol),
                recent_marks=list(self._recent_marks[:-1]),
                equity=broker.equity,
                product=None,
                market_data=market_data,
            )

            self._execute_decision(decision, point, broker, i)
            if position_before is None and broker.get_position(point.symbol):
                self._process_funding(point, broker)
            self._sync_broker_orders(broker)
            broker.update_equity_curve()

        self._close_at_end(data_points[-1], broker)

        final_position = self._position_from_broker(broker, symbol)

        return BacktestResult(
            trades=self._trades,
            orders=self._orders,
            final_equity=broker.get_equity(),
            final_position=final_position,
            equity_curve=broker.get_equity_curve(),
            start_time=data_points[0].timestamp,
            end_time=data_points[-1].timestamp,
            data_points_processed=len(data_points),
        )

    def _reset_state(self) -> None:
        self._orders = []
        self._trades = []
        self._pending_orders = []
        self._recent_marks = []
        self._order_counter = 0
        self._broker = None
        self._fill_model = None
        self._broker_order_map = {}
        self._broker_trade_recorded = set()

    def _build_broker(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> SimulatedBroker:
        slippage_bps = {symbol: Decimal(str(self.config.slippage_bps))}
        spread_impact = Decimal("0.5") if self.config.use_spread_slippage else Decimal("0")
        sim_config = SimulationConfig(
            start_date=start_time,
            end_date=end_time,
            granularity="CUSTOM",
            initial_equity_usd=self.config.initial_equity,
            fee_tier=self.config.fee_tier,
            slippage_bps=slippage_bps,
            spread_impact_pct=spread_impact,
            enable_funding_pnl=self.config.enable_funding_pnl,
            funding_accrual_hours=self.config.funding_accrual_hours,
            funding_settlement_hours=self.config.funding_settlement_hours,
            funding_rates_8h=self.config.funding_rates_8h,
        )
        broker = SimulatedBroker(
            initial_equity_usd=self.config.initial_equity,
            fee_tier=self.config.fee_tier,
            config=sim_config,
        )
        broker._fee_calculator = self._fee_calculator
        broker.connect()
        return broker

    def _build_candle(self, point: HistoricalDataPoint) -> Candle:
        spread = Decimal("0")
        if point.spread_bps:
            spread = point.mark_price * Decimal(str(point.spread_bps)) / Decimal("10000")
        high = point.mark_price + (spread * Decimal("2"))
        low = point.mark_price - (spread * Decimal("2"))
        return Candle(
            ts=point.timestamp,
            open=point.mark_price,
            high=high,
            low=low,
            close=point.mark_price,
            volume=Decimal("1000000"),
        )

    def _position_state(
        self,
        broker: SimulatedBroker,
        symbol: str,
    ) -> dict[str, object] | None:
        position = broker.get_position(symbol)
        if not position:
            return None
        return {
            "symbol": symbol,
            "side": position.side,
            "quantity": abs(position.quantity),
            "entry_price": position.entry_price,
        }

    def _position_from_broker(self, broker: SimulatedBroker, symbol: str) -> Position:
        position = broker.get_position(symbol)
        if not position or position.quantity == 0:
            return Position(symbol=symbol)
        return Position(
            symbol=symbol,
            side=position.side,
            quantity=abs(position.quantity),
            entry_price=position.entry_price,
            entry_time=None,
            unrealized_pnl=position.unrealized_pnl,
        )

    def _execute_decision(
        self,
        decision: Decision,
        point: HistoricalDataPoint,
        broker: SimulatedBroker,
        index: int,
    ) -> None:
        action = decision.action
        if action == Action.HOLD:
            return

        order_type, price, stop_price, tif, reduce_only = self._resolve_order_fields(decision)
        has_pending = bool(self._pending_orders)
        if has_pending:
            if action == Action.CLOSE:
                self._cancel_pending_orders(point.symbol, "close_signal", point.timestamp)
                return
            if action in (Action.BUY, Action.SELL):
                if self.config.cancel_pending_on_new_signal:
                    self._cancel_pending_orders(point.symbol, "new_signal", point.timestamp)
                else:
                    return

        if action == Action.CLOSE:
            self._close_position(
                point,
                broker,
                index,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
                tif=tif,
                reduce_only=reduce_only,
            )
            return

        if action == Action.BUY:
            if self._close_if_short(point, broker, index):
                return
            self._open_position(
                point,
                broker,
                index,
                side="buy",
                reason=decision.reason,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
                tif=tif,
                reduce_only=reduce_only,
            )
            return

        if action == Action.SELL:
            if self._close_if_long(point, broker, index):
                return
            self._open_position(
                point,
                broker,
                index,
                side="sell",
                reason=decision.reason,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
                tif=tif,
                reduce_only=reduce_only,
            )

    def _close_if_short(
        self,
        point: HistoricalDataPoint,
        broker: SimulatedBroker,
        index: int,
    ) -> bool:
        position = broker.get_position(point.symbol)
        if not position or position.side != "short":
            return False
        self._schedule_order(
            point=point,
            index=index,
            side="buy",
            quantity=abs(position.quantity),
            reason="close_before_buy",
            intent="close_short",
            order_type=OrderType.MARKET,
            price=None,
            stop_price=None,
            tif=None,
            reduce_only=None,
        )
        return self.config.order_fill_delay_bars != 0

    def _close_if_long(
        self,
        point: HistoricalDataPoint,
        broker: SimulatedBroker,
        index: int,
    ) -> bool:
        position = broker.get_position(point.symbol)
        if not position or position.side != "long":
            return False
        self._schedule_order(
            point=point,
            index=index,
            side="sell",
            quantity=abs(position.quantity),
            reason="close_before_sell",
            intent="close_long",
            order_type=OrderType.MARKET,
            price=None,
            stop_price=None,
            tif=None,
            reduce_only=None,
        )
        return self.config.order_fill_delay_bars != 0

    def _open_position(
        self,
        point: HistoricalDataPoint,
        broker: SimulatedBroker,
        index: int,
        side: str,
        reason: str,
        order_type: OrderType,
        price: Decimal | None,
        stop_price: Decimal | None,
        tif: TimeInForce | str | None,
        reduce_only: bool | None,
    ) -> None:
        quantity = self._calculate_order_quantity(point, broker, side)
        intent = "open_long" if side == "buy" else "open_short"
        self._schedule_order(
            point=point,
            index=index,
            side=side,
            quantity=quantity,
            reason=reason,
            intent=intent,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            tif=tif,
            reduce_only=reduce_only,
        )

    def _close_position(
        self,
        point: HistoricalDataPoint,
        broker: SimulatedBroker,
        index: int,
        order_type: OrderType,
        price: Decimal | None,
        stop_price: Decimal | None,
        tif: TimeInForce | str | None,
        reduce_only: bool | None,
    ) -> None:
        position = broker.get_position(point.symbol)
        if not position or position.side == "flat" or position.quantity == 0:
            return
        side = "sell" if position.side == "long" else "buy"
        intent = "close_long" if position.side == "long" else "close_short"
        self._schedule_order(
            point=point,
            index=index,
            side=side,
            quantity=abs(position.quantity),
            reason="close_signal",
            intent=intent,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            tif=tif,
            reduce_only=reduce_only,
        )

    def _close_at_end(self, point: HistoricalDataPoint, broker: SimulatedBroker) -> None:
        position = broker.get_position(point.symbol)
        if not position or position.quantity == 0:
            return
        side = "sell" if position.side == "long" else "buy"
        result = broker.place_order(
            symbol=point.symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=abs(position.quantity),
        )

        if result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            fill_qty = result.filled_quantity if result.filled_quantity else abs(position.quantity)
            fill_price = result.avg_fill_price or point.mark_price
            fee = self.calculate_fee(fill_qty * fill_price)
            self._trades.append(
                SimulatedTrade(
                    timestamp=result.updated_at or point.timestamp,
                    symbol=point.symbol,
                    side=side,
                    quantity=fill_qty,
                    price=fill_price,
                    fee=fee,
                    reason="end_of_backtest",
                )
            )

    def _calculate_order_quantity(
        self,
        point: HistoricalDataPoint,
        broker: SimulatedBroker,
        side: str,
    ) -> Decimal:
        position_value = broker.equity * Decimal(str(self.config.position_size_pct))
        max_value = broker.equity * Decimal(str(self.config.max_position_pct))
        position_value = min(position_value, max_value)
        fill_price = self._calculate_fill_price(point, side)
        return position_value / fill_price

    def _calculate_fill_price(self, point: HistoricalDataPoint, side: str) -> Decimal:
        fill_model = self._fill_model or self._build_fill_model(point.symbol)

        candle = self._build_candle(point)
        spread = (candle.high - candle.low) / Decimal("4")
        mid = candle.close
        best_bid = mid - spread / Decimal("2")
        best_ask = mid + spread / Decimal("2")

        order = Order(
            id="preview",
            symbol=point.symbol,
            side=OrderSide.BUY if side in ("buy", "long") else OrderSide.SELL,
            type=OrderType.MARKET,
            quantity=Decimal("1"),
            status=OrderStatus.PENDING,
            submitted_at=point.timestamp,
            created_at=point.timestamp,
        )

        result = fill_model.fill_market_order(
            order=order,
            current_bar=candle,
            best_bid=best_bid,
            best_ask=best_ask,
        )
        return result.fill_price or point.mark_price

    def _resolve_order_fields(
        self,
        decision: Decision,
    ) -> tuple[OrderType, Decimal | None, Decimal | None, TimeInForce | str | None, bool | None]:
        indicators = decision.indicators or {}
        order_type_raw = indicators.get("order_type") or indicators.get("orderType")
        price_raw = indicators.get("price") or indicators.get("limit_price")
        stop_raw = indicators.get("stop_price")
        tif = indicators.get("tif") or indicators.get("time_in_force")
        reduce_only = indicators.get("reduce_only")

        order_type = self._coerce_order_type(order_type_raw, price_raw, stop_raw)
        price = self._to_decimal(price_raw)
        stop_price = self._to_decimal(stop_raw)

        return order_type, price, stop_price, tif, reduce_only

    def _coerce_order_type(
        self,
        order_type_raw: object | None,
        price: object | None,
        stop_price: object | None,
    ) -> OrderType:
        if isinstance(order_type_raw, OrderType):
            return order_type_raw
        if isinstance(order_type_raw, str) and order_type_raw:
            key = order_type_raw.upper()
            try:
                return OrderType[key]
            except KeyError:
                try:
                    return OrderType(order_type_raw)
                except ValueError:
                    pass

        if stop_price is not None and price is not None:
            return OrderType.STOP_LIMIT
        if stop_price is not None:
            return OrderType.STOP
        if price is not None:
            return OrderType.LIMIT
        return OrderType.MARKET

    def _to_decimal(self, value: object | None) -> Decimal | None:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    def _apply_broker_result(
        self,
        order: SimulatedOrder,
        broker_order: Order,
        point: HistoricalDataPoint | None,
        *,
        track_open: bool,
    ) -> None:
        order.status = broker_order.status
        order.filled_at = broker_order.updated_at
        order.fill_price = broker_order.avg_fill_price

        if broker_order.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            self._record_trade_from_broker(order, broker_order, point)
            return

        if broker_order.status == OrderStatus.CANCELLED:
            order.cancel_reason = "cancelled"
            order.cancelled_at = broker_order.updated_at or (point.timestamp if point else None)
            return

        if broker_order.status == OrderStatus.REJECTED:
            order.cancel_reason = "rejected"
            order.cancelled_at = broker_order.updated_at or (point.timestamp if point else None)
            return

        if track_open and broker_order.status in (OrderStatus.SUBMITTED, OrderStatus.PENDING):
            self._broker_order_map[broker_order.id] = order

    def _record_trade_from_broker(
        self,
        order: SimulatedOrder,
        broker_order: Order,
        point: HistoricalDataPoint | None,
    ) -> None:
        if broker_order.id in self._broker_trade_recorded:
            return

        fill_qty = broker_order.filled_quantity if broker_order.filled_quantity else order.quantity
        fill_price = broker_order.avg_fill_price or order.fill_price
        if fill_price is None:
            fill_price = point.mark_price if point else Decimal("0")
        fee = self.calculate_fee(fill_qty * fill_price)
        timestamp = broker_order.updated_at or order.filled_at
        if timestamp is None and point is not None:
            timestamp = point.timestamp
        self._trades.append(
            SimulatedTrade(
                timestamp=timestamp or datetime.now(),
                symbol=order.symbol,
                side=order.side,
                quantity=fill_qty,
                price=fill_price,
                fee=fee,
                reason=order.reason,
            )
        )
        self._broker_trade_recorded.add(broker_order.id)

    def _sync_broker_orders(self, broker: SimulatedBroker) -> None:
        if not self._broker_order_map:
            return

        to_remove: list[str] = []
        for broker_id, order in self._broker_order_map.items():
            if broker_id in broker._filled_orders:
                self._apply_broker_result(
                    order,
                    broker._filled_orders[broker_id],
                    None,
                    track_open=False,
                )
                to_remove.append(broker_id)
            elif broker_id in broker._cancelled_orders:
                self._apply_broker_result(
                    order,
                    broker._cancelled_orders[broker_id],
                    None,
                    track_open=False,
                )
                to_remove.append(broker_id)

        for broker_id in to_remove:
            self._broker_order_map.pop(broker_id, None)

    def _schedule_order(
        self,
        *,
        point: HistoricalDataPoint,
        index: int,
        side: str,
        quantity: Decimal,
        reason: str,
        intent: str,
        order_type: OrderType,
        price: Decimal | None,
        stop_price: Decimal | None,
        tif: TimeInForce | str | None,
        reduce_only: bool | None,
        force_immediate: bool = False,
    ) -> None:
        self._order_counter += 1
        order_type_value = order_type.value.lower()
        order = SimulatedOrder(
            id=f"order-{self._order_counter}",
            symbol=point.symbol,
            side=side,
            order_type=order_type_value,
            quantity=quantity,
            status=OrderStatus.PENDING,
            submitted_at=point.timestamp,
            reason=reason,
            intent=intent,
        )
        self._orders.append(order)

        if order_type != OrderType.MARKET:
            self._execute_scheduled_order(
                order,
                side,
                quantity,
                point,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
                tif=tif,
                reduce_only=reduce_only,
            )
            return

        delay = 0 if force_immediate else self.config.order_fill_delay_bars
        if delay <= 0:
            self._execute_scheduled_order(
                order,
                side,
                quantity,
                point,
                order_type=order_type,
                price=price,
                stop_price=stop_price,
                tif=tif,
                reduce_only=reduce_only,
            )
            return

        due_index = index + delay
        pending = PendingOrder(
            due_index=due_index,
            order=order,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            tif=tif,
            reduce_only=reduce_only,
        )
        self._pending_orders.append(pending)

    def _process_pending_orders(
        self,
        index: int,
        point: HistoricalDataPoint,
        broker: SimulatedBroker,
    ) -> None:
        if not self._pending_orders:
            return

        remaining: list[PendingOrder] = []
        for pending in self._pending_orders:
            if pending.order.status != OrderStatus.PENDING:
                continue
            if index >= pending.due_index:
                self._execute_scheduled_order(
                    pending.order,
                    "buy" if pending.side == OrderSide.BUY else "sell",
                    pending.quantity,
                    point,
                    broker=broker,
                    order_type=pending.order_type,
                    price=pending.price,
                    stop_price=pending.stop_price,
                    tif=pending.tif,
                    reduce_only=pending.reduce_only,
                )
            else:
                remaining.append(pending)

        self._pending_orders = remaining

    def _execute_scheduled_order(
        self,
        order: SimulatedOrder,
        side: str,
        quantity: Decimal,
        point: HistoricalDataPoint,
        broker: SimulatedBroker | None = None,
        order_type: OrderType = OrderType.MARKET,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tif: TimeInForce | str | None = None,
        reduce_only: bool | None = None,
    ) -> None:
        broker = broker or self._broker
        if broker is None:
            return

        result = broker.place_order(
            symbol=order.symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            tif=tif,
            reduce_only=reduce_only,
        )

        self._apply_broker_result(order, result, point, track_open=True)

    def _cancel_pending_orders(
        self,
        symbol: str,
        reason: str,
        timestamp: datetime,
    ) -> None:
        if not self._pending_orders:
            return

        remaining: list[PendingOrder] = []
        for pending in self._pending_orders:
            if pending.order.symbol != symbol:
                remaining.append(pending)
                continue
            if pending.order.status == OrderStatus.PENDING:
                pending.order.status = OrderStatus.CANCELLED
                pending.order.cancel_reason = reason
                pending.order.cancelled_at = timestamp

        self._pending_orders = remaining

    def calculate_fee(self, notional: Decimal) -> Decimal:
        return self._fee_calculator.calculate(notional_usd=notional, is_maker=False)

    def _process_funding(self, point: HistoricalDataPoint, broker: SimulatedBroker) -> None:
        if not self.config.enable_funding_pnl:
            return
        if not self.config.funding_rates_8h:
            return

        funding_rate = self.config.funding_rates_8h.get(point.symbol)
        if funding_rate is None:
            return

        broker.process_funding(point.symbol, funding_rate)

    def _build_fill_model(self, symbol: str) -> OrderFillModel:
        slippage_bps = {symbol: Decimal(str(self.config.slippage_bps))}
        spread_impact = Decimal("0.5") if self.config.use_spread_slippage else Decimal("0")
        return OrderFillModel(slippage_bps=slippage_bps, spread_impact_pct=spread_impact)
