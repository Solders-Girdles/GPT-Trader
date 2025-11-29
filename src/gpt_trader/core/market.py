"""Core market data types used across all slices."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from gpt_trader.core.trading import MarketType


@dataclass
class Candle:
    """OHLCV price bar."""

    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass
class Quote:
    """Current bid/ask/last price snapshot."""

    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    ts: datetime


@dataclass
class Product:
    """Trading instrument specification."""

    symbol: str
    base_asset: str
    quote_asset: str
    market_type: MarketType
    min_size: Decimal
    step_size: Decimal
    min_notional: Decimal | None
    price_increment: Decimal
    leverage_max: int | None
    expiry: datetime | None = None
    contract_size: Decimal | None = None
    funding_rate: Decimal | None = None
    next_funding_time: datetime | None = None
