from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from ..logging import get_logger
from .base import Account, Broker, Order, Position

logger = get_logger("alpaca_paper")


class AlpacaPaperBroker(Broker):
    """Alpaca paper trading broker implementation."""

    def __init__(
        self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"
    ) -> None:
        """Initialize the Alpaca paper trading client."""
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.base_url = base_url
        logger.info("Alpaca paper trading client initialized")

    def get_account(self) -> Account:
        """Get current account information."""
        try:
            account = self.trading_client.get_account()
            return Account(
                id=account.id,
                account_number=account.account_number,
                status=account.status,
                crypto_status=account.crypto_status,
                currency=account.currency,
                buying_power=float(account.buying_power),
                regt_buying_power=float(account.regt_buying_power),
                daytrading_buying_power=float(account.daytrading_buying_power),
                non_marginable_buying_power=float(account.non_marginable_buying_power),
                cash=float(account.cash),
                accrued_fees=float(account.accrued_fees),
                pending_transfer_out=float(account.pending_transfer_out),
                pending_transfer_in=float(account.pending_transfer_in),
                portfolio_value=float(account.portfolio_value),
                pattern_day_trader=account.pattern_day_trader,
                trading_blocked=account.trading_blocked,
                transfers_blocked=account.transfers_blocked,
                account_blocked=account.account_blocked,
                created_at=account.created_at,
                trade_suspended_by_user=account.trade_suspended_by_user,
                multiplier=account.multiplier,
                shorting_enabled=account.shorting_enabled,
                equity=float(account.equity),
                last_equity=float(account.last_equity),
                long_market_value=float(account.long_market_value),
                short_market_value=float(account.short_market_value),
                initial_margin=float(account.initial_margin),
                maintenance_margin=float(account.maintenance_margin),
                last_maintenance_margin=float(account.last_maintenance_margin),
                sma=float(account.sma),
                daytrade_count=account.daytrade_count,
            )
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            raise

    def get_positions(self) -> list[Position]:
        """Get current positions."""
        try:
            positions = self.trading_client.get_all_positions()
            return [
                Position(
                    symbol=pos.symbol,
                    qty=int(pos.qty),
                    avg_price=float(pos.avg_entry_price),
                    market_value=float(pos.market_value),
                    unrealized_pl=float(pos.unrealized_pl),
                    unrealized_plpc=float(pos.unrealized_plpc),
                    current_price=float(pos.current_price),
                    timestamp=datetime.now(),
                )
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a specific symbol."""
        try:
            position = self.trading_client.get_position(symbol)
            return Position(
                symbol=position.symbol,
                qty=int(position.qty),
                avg_price=float(position.avg_entry_price),
                market_value=float(position.market_value),
                unrealized_pl=float(position.unrealized_pl),
                unrealized_plpc=float(position.unrealized_plpc),
                current_price=float(position.current_price),
                timestamp=datetime.now(),
            )
        except Exception as e:
            logger.debug(f"No position found for {symbol}: {e}")
            return None

    def submit_market_order(self, symbol: str, side: str, qty: int) -> Order:
        """Submit a market order."""
        try:
            side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=TimeInForce.DAY,
            )

            order = self.trading_client.submit_order(order_request)
            logger.info(f"Submitted market order: {symbol} {side} {qty} shares")

            return self._convert_alpaca_order(order)
        except Exception as e:
            logger.error(f"Failed to submit market order: {e}")
            raise

    def submit_limit_order(self, symbol: str, side: str, qty: int, limit_price: float) -> Order:
        """Submit a limit order."""
        try:
            side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )

            order = self.trading_client.submit_order(order_request)
            logger.info(f"Submitted limit order: {symbol} {side} {qty} shares @ {limit_price}")

            return self._convert_alpaca_order(order)
        except Exception as e:
            logger.error(f"Failed to submit limit order: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Canceled order: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order(self, order_id: str) -> Order | None:
        """Get order details."""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return self._convert_alpaca_order(order)
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def get_orders(self, status: str | None = None, limit: int = 500) -> list[Order]:
        """Get orders with optional status filter."""
        try:
            orders = self.trading_client.get_orders(status=status, limit=limit)
            return [self._convert_alpaca_order(order) for order in orders]
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    def get_bars(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Day") -> Any:
        """Get historical bars for a symbol."""
        try:
            # Convert timeframe string to Alpaca TimeFrame
            tf_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame.Minute5,
                "15Min": TimeFrame.Minute15,
                "30Min": TimeFrame.Minute30,
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day,
            }
            tf = tf_map.get(timeframe, TimeFrame.Day)

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
            )

            bars = self.data_client.get_stock_bars(request)
            return bars
        except Exception as e:
            logger.error(f"Failed to get bars for {symbol}: {e}")
            raise

    def get_latest_bar(self, symbol: str) -> dict[str, Any] | None:
        """Get the latest bar for a symbol."""
        try:
            end = datetime.now()
            start = end - timedelta(days=1)

            bars = self.get_bars(symbol, start, end, "1Day")
            if bars and len(bars.data) > 0:
                bar = (
                    bars.data[symbol][-1]
                    if symbol in bars.data
                    else bars.data[list(bars.data.keys())[0]][-1]
                )
                return {
                    "open": float(bar.o),
                    "high": float(bar.h),
                    "low": float(bar.l),
                    "close": float(bar.c),
                    "volume": int(bar.v),
                    "timestamp": bar.t,
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get latest bar for {symbol}: {e}")
            return None

    def _convert_alpaca_order(self, alpaca_order) -> Order:
        """Convert Alpaca order to our Order dataclass."""
        return Order(
            id=alpaca_order.id,
            symbol=alpaca_order.symbol,
            side=alpaca_order.side.value,
            qty=int(alpaca_order.qty),
            filled_qty=int(alpaca_order.filled_qty),
            filled_avg_price=(
                float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else 0.0
            ),
            status=alpaca_order.status.value,
            order_type=alpaca_order.order_type.value,
            time_in_force=alpaca_order.time_in_force.value,
            created_at=alpaca_order.created_at,
            filled_at=alpaca_order.filled_at,
            canceled_at=alpaca_order.canceled_at,
            limit_price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
            stop_price=float(alpaca_order.stop_price) if alpaca_order.stop_price else None,
        )
