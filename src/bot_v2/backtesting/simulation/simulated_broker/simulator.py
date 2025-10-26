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

    # Compatibility ----------------------------------------------------
    def __getattr__(self, name: str) -> Any:  # pragma: no cover - compatibility shim
        """Provide read-only access to legacy private attributes."""
        if name == "_initial_equity":
            return self._portfolio._initial_equity
        if name == "_cash_balance":
            return self._portfolio.cash_balance
        if name == "_positions":
            return self._portfolio._positions
        if name == "_orders":
            return self._orders._orders
        if name == "_open_orders":
            return self._orders._open_orders
        if name == "_products":
            return self._portfolio.products
        if name == "_current_time":
            return self._market_state.current_time
        if name == "_current_bar":
            return self._market_state.bars
        if name == "_current_quotes":
            return self._market_state.quotes
        if name == "_next_bar":
            return self._market_state.next_bars
        if name == "_fee_calculator":
            return self._portfolio.fee_calculator
        if name == "_fill_model":
            return self._orders._fill_model
        if name == "_funding_tracker":
            return self._portfolio.funding_tracker
        if name == "_realized_pnl":
            return self._portfolio._realized_pnl
        if name == "_fees_paid":
            return self._portfolio._fees_paid
        if name == "_total_trades":
            return self._portfolio._total_trades
        if name == "_winning_trades":
            return self._portfolio._winning_trades
        if name == "_losing_trades":
            return self._portfolio._losing_trades
        if name == "_peak_equity":
            return self._portfolio._peak_equity
        if name == "_max_drawdown":
            return self._portfolio._max_drawdown
        if name == "_equity_history":
            return self._portfolio._equity_history

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


__all__ = ["SimulatedBroker"]
