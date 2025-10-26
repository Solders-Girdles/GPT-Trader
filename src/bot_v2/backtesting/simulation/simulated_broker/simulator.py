from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict

from bot_v2.backtesting.simulation.fill_model import OrderFillModel
from bot_v2.backtesting.types import BacktestResult, FeeTier
from bot_v2.features.brokerages.core.interfaces import (
    Balance,
    Candle,
    MarketType,
    Order,
    Position,
    Product,
    Quote,
)

from .market import MarketState
from .orders import OrderEngine
from .portfolio import PortfolioManager


class SimulatedBroker:
    """Simulated broker orchestration composed of modular components."""

    def __init__(
        self,
        initial_equity_usd: Decimal = Decimal("100000"),
        fee_tier: FeeTier = FeeTier.TIER_2,
        slippage_bps: Dict[str, Decimal] | None = None,
        spread_impact_pct: Decimal = Decimal("0.5"),
        enable_funding_pnl: bool = True,
    ) -> None:
        self._market_state = MarketState()
        self._portfolio = PortfolioManager(
            initial_equity_usd=initial_equity_usd,
            fee_tier=fee_tier,
            enable_funding_pnl=enable_funding_pnl,
        )
        fill_model = OrderFillModel(slippage_bps=slippage_bps, spread_impact_pct=spread_impact_pct)
        self._orders = OrderEngine(
            portfolio=self._portfolio,
            market_state=self._market_state,
            fill_model=fill_model,
        )

    # ------------------------------------------------------------------
    def update_market_data(
        self,
        current_time: datetime,
        bars: Dict[str, Candle],
        quotes: Dict[str, Quote] | None = None,
        next_bars: Dict[str, Candle] | None = None,
    ) -> None:
        self._market_state.update(current_time, bars, quotes, next_bars)
        self._portfolio.update_marks(self._market_state.quotes.items())
        self._orders.process_pending_orders()
        self._portfolio.accrue_funding(self._market_state.current_time)
        self._portfolio.record_equity(self._market_state.current_time)

    # Connection APIs ---------------------------------------------------
    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        return None

    def validate_connection(self) -> bool:
        return True

    def get_account_id(self) -> str:
        return "SIMULATED_ACCOUNT"

    # Product + Market Data ---------------------------------------------
    def register_product(self, product: Product) -> None:
        self._portfolio.register_product(product)

    def list_products(self, market: MarketType | None = None) -> list[Product]:
        return self._portfolio.list_products(market)

    def get_product(self, symbol: str) -> Product:
        try:
            return self._portfolio.get_product(symbol)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc

    def get_quote(self, symbol: str) -> Quote:
        try:
            return self._market_state.get_quote(symbol)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc

    def get_candles(self, symbol: str, granularity: str, limit: int = 300) -> list[Candle]:
        bar = self._market_state.get_bar(symbol)
        return [bar] if bar else []

    # Account -----------------------------------------------------------
    def list_balances(self) -> list[Balance]:
        return self._portfolio.list_balances()

    def get_account_info(self) -> dict:
        return {
            "equity": self.get_equity(),
            "cash": self._portfolio.cash_balance,
            "margin_used": self.get_margin_used(),
            "margin_available": self.get_margin_available(),
        }

    def list_positions(self) -> list[Position]:
        return self._portfolio.list_positions()

    # Orders ------------------------------------------------------------
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: str,
        limit_price: str | None = None,
        stop_price: str | None = None,
        time_in_force: str | None = None,
        post_only: bool = False,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> Order:
        return self._orders.place_order(
            current_time=self._market_state.current_time,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            post_only=post_only,
            reduce_only=reduce_only,
            client_order_id=client_order_id,
        )

    def cancel_order(self, order_id: str) -> bool:
        return self._orders.cancel_order(order_id, self._market_state.current_time)

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get_order(order_id)

    def list_orders(self, status: str | None = None, symbol: str | None = None) -> list[Order]:
        return self._orders.list_orders(status=status, symbol=symbol)

    def list_fills(self, symbol: str | None = None, limit: int = 100) -> list[dict]:
        return self._orders.list_fills(symbol=symbol, limit=limit)

    # Portfolio metrics -------------------------------------------------
    def get_equity(self) -> Decimal:
        return self._portfolio.get_equity()

    def get_margin_used(self) -> Decimal:
        return self._portfolio.get_margin_used()

    def get_margin_available(self) -> Decimal:
        return self._portfolio.get_margin_available()

    def generate_report(self) -> BacktestResult:
        return self._portfolio.generate_report()


__all__ = ["SimulatedBroker"]
