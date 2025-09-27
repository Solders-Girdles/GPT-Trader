"""
Coinbase adapter implementing the brokerage protocol for perpetuals trading.

Wires REST and WS clients and exposes a broker-agnostic interface with full
market/limit order support, reduce-only enforcement, and WebSocket market data.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .client import CoinbaseClient, CoinbaseAuth
from .cdp_auth import CDPAuth, create_cdp_auth
from .cdp_auth_v2 import CDPAuthV2, create_cdp_auth_v2
from .endpoints import CoinbaseEndpoints
from .models import APIConfig, normalize_symbol, to_candle, to_order, to_product, to_quote
from .utils import (
    ProductCatalog,
    quantize_to_increment,
    enforce_perp_rules,
    MarkCache,
    FundingCalculator,
    PositionState
)
from .market_data_utils import RollingWindow
from .ws import CoinbaseWebSocket, WSSubscription, SequenceGuard, normalize_market_message
from .errors import InvalidRequestError
from ....persistence.event_store import EventStore
from ..core.interfaces import (
    Candle,
    IBrokerage,
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Product,
    Quote,
    TimeInForce,
    Balance,
    InsufficientFunds,
    InvalidRequestError,
)
from ....errors import ValidationError

import logging
logger = logging.getLogger(__name__)


class CoinbaseBrokerage(IBrokerage):
    """
    Production-ready Coinbase perpetuals adapter.

    Features:
    - Dynamic product discovery for BTC/ETH/SOL/XRP perps
    - Market/limit orders with GTC/IOC support
    - Reduce-only enforcement and client-ID generation
    - WebSocket market data with staleness guards
    - Funding rate tracking and position management
    """

    def __init__(self, config: APIConfig):
        self.config = config
        self._config = config
        self.endpoints = CoinbaseEndpoints(
            mode=config.api_mode,
            sandbox=config.sandbox,
            enable_derivatives=config.enable_derivatives
        )
        self._ws_url = self.endpoints.websocket_url()

        # Initialize REST client
        if config.cdp_api_key and config.cdp_private_key:
            auth = create_cdp_auth_v2(
                api_key_name=config.cdp_api_key,
                private_key_pem=config.cdp_private_key
            )
            logger.info("Using CDP JWT authentication (SDK-compatible)")
        else:
            auth = CoinbaseAuth(
                api_key=config.api_key,
                api_secret=config.api_secret,
                passphrase=config.passphrase,
                api_mode=config.api_mode
            )
            logger.info(f"Using HMAC authentication with {config.api_mode} mode")

        self.client = CoinbaseClient(
            auth=auth,
            base_url=self.endpoints.base_url,
            api_mode=config.api_mode,
            api_version=config.api_version
        )

        # Initialize product catalog with funding support
        self.product_catalog = ProductCatalog(ttl_seconds=900)  # 15min TTL
        self._mark_cache = MarkCache()
        self._funding_calculator = FundingCalculator()
        self._positions: Dict[str, PositionState] = {}
        self._event_store = EventStore()

        # Initialize WebSocket market data (lazy)
        self._ws_client: Optional[CoinbaseWebSocket] = None
        self._market_data: Dict[str, Dict] = {}  # symbol -> market snapshot
        self._rolling_windows: Dict[str, Dict[str, RollingWindow]] = {}  # symbol -> metric -> window
        self._ws_factory_override: Optional[Callable[[], CoinbaseWebSocket]] = None
        self._connected = False
        self._account_id: Optional[str] = None

        logger.info(f"CoinbaseBrokerage initialized - mode: {config.api_mode}, sandbox: {config.sandbox}")

    # ===== PRODUCT DISCOVERY =====

    def list_products(self, market: Optional[MarketType] = None) -> List[Product]:
        """List all products with optional market type filter."""
        try:
            response = self.client.get_products() or {}
            items = response.get('products') or response.get('data') or []
            products: List[Product] = []

            for item in items:
                product = to_product(item)

                # Add funding data for perpetuals
                if product.market_type == MarketType.PERPETUAL:
                    product = self._enrich_with_funding(product)

                # Apply market type filter
                if market is None or product.market_type == market:
                    products.append(product)

            # Cache perpetuals for quick access
            if market == MarketType.PERPETUAL or market is None:
                perps = [p for p in products if p.market_type == MarketType.PERPETUAL]
                logger.info(f"Found {len(perps)} perpetual products")

            return products

        except Exception as e:
            logger.error(f"Failed to list products: {e}")
            return []

    def get_product(self, symbol: str) -> Optional[Product]:
        """Get single product by symbol, preferring cached catalog data."""
        normalized = normalize_symbol(symbol)
        product: Optional[Product] = None

        try:
            if hasattr(self.product_catalog, "get"):
                product = self.product_catalog.get(self.client, normalized)
        except Exception as exc:
            logger.debug("Product catalog lookup failed for %s: %s", normalized, exc, exc_info=True)
            product = None

        if product is None:
            try:
                if hasattr(self.client, "get_product"):
                    response = self.client.get_product(normalized) or {}
                else:
                    response = self.client.get(self.endpoints.get_product(normalized)) or {}
                product = to_product(response)
            except Exception as exc:
                logger.error(f"Failed to get product {symbol}: {exc}")
                return None

        if product.market_type == MarketType.PERPETUAL:
            try:
                funding_rate = None
                next_funding_time = None
                if hasattr(self.product_catalog, "get_funding"):
                    funding_rate, next_funding_time = self.product_catalog.get_funding(self.client, product.symbol)
                if funding_rate is not None:
                    product.funding_rate = Decimal(str(funding_rate))
                    product.next_funding_time = next_funding_time
                else:
                    product = self._enrich_with_funding(product)
            except Exception as exc:
                logger.debug("Funding enrichment via catalog failed for %s: %s", product.symbol, exc, exc_info=True)
                product = self._enrich_with_funding(product)

        return product

    def _enrich_with_funding(self, product: Product) -> Product:
        """Add funding rate data to perpetual product."""
        if not self.endpoints.supports_derivatives():
            return product

        try:
            funding_rate, next_funding_time = self.product_catalog.get_funding(self.client, product.symbol)
            if funding_rate is not None:
                product.funding_rate = Decimal(str(funding_rate))
            if next_funding_time is not None:
                product.next_funding_time = next_funding_time
        except Exception as exc:
            logger.debug(f"Could not fetch funding data for {product.symbol}: {exc}")

        return product

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        try:
            response = self.client.get(self.endpoints.get_order(order_id))
            return to_order(response)
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def list_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Order]:
        """List orders with optional filters."""
        try:
            params = {}
            if symbol:
                params['product_id'] = symbol
            if status:
                params['order_status'] = status

            response = self.client.get(self.endpoints.list_orders(), params=params)

            orders = []
            for item in response.get('orders', []):
                orders.append(to_order(item))

            return orders

        except Exception as e:
            logger.error(f"Failed to list orders: {e}")
            return []

    # ===== POSITION MANAGEMENT =====

    def list_positions(self) -> List[Position]:
        """List all positions."""
        if not self.endpoints.supports_derivatives():
            return []

        try:
            response = self.client.get(self.endpoints.list_positions())

            positions = []
            for item in response.get('positions', []):
                position = self._map_position(item)
                if position:
                    positions.append(position)

            return positions

        except Exception as e:
            logger.error(f"Failed to list positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        if not self.endpoints.supports_derivatives():
            return None

        try:
            response = self.client.get(self.endpoints.get_position(symbol))
            return self._map_position(response)
        except Exception as e:
            logger.debug(f"No position found for {symbol}: {e}")
            return None

    def _map_position(self, data: Dict) -> Optional[Position]:
        """Map API response to Position object."""
        try:
            from ..core.interfaces import Position

            return Position(
                symbol=data.get('product_id', ''),
                quantity=Decimal(str(data.get('size', '0'))),
                entry_price=Decimal(str(data.get('entry_price', '0'))),
                current_price=Decimal(str(data.get('mark_price', data.get('entry_price', '0')))),
                unrealized_pnl=Decimal(str(data.get('unrealized_pnl', '0'))),
                leverage=int(data.get('leverage', 1))
            )
        except Exception as e:
            logger.error(f"Failed to map position: {e}")
            return None

    # ===== MARKET DATA =====

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get current quote for symbol."""
        # Try WebSocket data first if available
        if symbol in self._market_data:
            market_data = self._market_data[symbol]
            if market_data.get('last_update') and \
               (datetime.utcnow() - market_data['last_update']).total_seconds() < 10:
                return Quote(
                    symbol=symbol,
                    bid=market_data.get('bid', Decimal('0')),
                    ask=market_data.get('ask', Decimal('0')),
                    last=market_data.get('last', Decimal('0')),
                    ts=market_data['last_update']
                )

        # Fallback to REST API
        try:
            response = self.client.get(self.endpoints.get_ticker(symbol))
            return to_quote({**response, 'symbol': symbol})
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    def get_market_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data snapshot."""
        if symbol not in self._market_data:
            return {}

        data = self._market_data[symbol].copy()

        # Add rolling metrics if available
        if symbol in self._rolling_windows:
            windows = self._rolling_windows[symbol]
            data.update({
                'vol_1m': windows.get('vol_1m', RollingWindow(60)).sum,
                'vol_5m': windows.get('vol_5m', RollingWindow(300)).sum,
            })

        return data

    # ===== WEBSOCKET INTEGRATION =====

    def start_market_data(self, symbols: List[str]):
        """Start WebSocket market data streams."""
        if not self._ws_client:
            self._ws_client = CoinbaseWebSocket(
                url=self.endpoints.ws_url,
                auth=self.client.auth if hasattr(self.client, 'auth') else None
            )

        # Initialize market data structures
        for symbol in symbols:
            self._market_data[symbol] = {
                'mid': Decimal('0'),
                'spread_bps': 0,
                'depth_l1': Decimal('0'),
                'depth_l10': Decimal('0'),
                'last_update': None
            }

            self._rolling_windows[symbol] = {
                'vol_1m': RollingWindow(60),
                'vol_5m': RollingWindow(300)
            }

        # Subscribe to channels
        subscriptions = []
        for symbol in symbols:
            subscriptions.extend([
                WSSubscription(channel="ticker", product_ids=[symbol]),
                WSSubscription(channel="matches", product_ids=[symbol]),
                WSSubscription(channel="level2", product_ids=[symbol])
            ])

        # Set up message handlers
        self._ws_client.on_message = self._handle_ws_message
        self._ws_client.subscribe(subscriptions)

        logger.info(f"Started WebSocket market data for {len(symbols)} symbols")

    def _handle_ws_message(self, message: Dict):
        """Handle WebSocket market data messages with schema normalization."""
        try:
            # Normalize message schema for resilience
            message = self._normalize_ws_message(message)
            if not message:
                return

            channel = message.get('type')
            symbol = message.get('product_id')

            if not symbol or symbol not in self._market_data:
                return

            market_data = self._market_data[symbol]
            now = datetime.utcnow()

            if channel == 'ticker':
                # Update ticker data
                if 'best_bid' in message and 'best_ask' in message:
                    bid = Decimal(str(message['best_bid']))
                    ask = Decimal(str(message['best_ask']))

                    market_data['bid'] = bid
                    market_data['ask'] = ask
                    market_data['mid'] = (bid + ask) / 2

                    if bid > 0:
                        market_data['spread_bps'] = float((ask - bid) / bid * 10000)

                if 'price' in message:
                    market_data['last'] = Decimal(str(message['price']))

                market_data['last_update'] = now

            elif channel == 'match':
                # Update trade data for volume calculation
                if 'size' in message:
                    size = Decimal(str(message['size']))

                    # Update rolling volume windows
                    if symbol in self._rolling_windows:
                        for window in self._rolling_windows[symbol].values():
                            window.add(float(size), now)

            elif channel in ['l2update', 'level2', 'l2']:
                # Update order book depth (handle multiple channel names)
                self._update_depth_from_l2(symbol, message)

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    def _normalize_ws_message(self, message: Dict) -> Optional[Dict]:
        """Normalize WebSocket message schema for resilience.

        Handles common variations in channel names and field keys.
        """
        if not message:
            return None

        # Normalize channel/type field
        channel = message.get('type', message.get('channel', message.get('event', '')))

        # Map channel variants to standard names
        channel_map = {
            'ticker': ['ticker', 'tickers', 'ticker_batch', 'tick'],
            'match': ['match', 'matches', 'trade', 'trades', 'executed_trade'],
            'l2update': ['l2update', 'level2', 'l2', 'level2_batch', 'orderbook', 'book']
        }

        normalized_channel = None
        for standard, variants in channel_map.items():
            if any(v in channel.lower() for v in variants):
                normalized_channel = standard
                break

        if not normalized_channel:
            return None

        message['type'] = normalized_channel

        # Normalize product_id/symbol field
        if 'product_id' not in message:
            message['product_id'] = message.get('symbol', message.get('instrument', ''))

        # Normalize ticker fields
        if normalized_channel == 'ticker':
            # Handle price field variants
            if 'price' not in message:
                message['price'] = message.get('last_price', message.get('last', message.get('close', '')))

            # Handle bid/ask field variants
            if 'best_bid' not in message:
                message['best_bid'] = message.get('bid', message.get('bid_price', '0'))
            if 'best_ask' not in message:
                message['best_ask'] = message.get('ask', message.get('ask_price', '0'))

        # Normalize trade/match fields
        elif normalized_channel == 'match':
            # Handle size field variants
            if 'size' not in message:
                message['size'] = message.get('quantity', message.get('qty', message.get('amount', '0')))

            # Handle price field variants
            if 'price' not in message:
                message['price'] = message.get('trade_price', message.get('execution_price', '0'))

        # Normalize orderbook fields
        elif normalized_channel == 'l2update':
            # Handle changes/updates field variants
            if 'changes' not in message:
                if 'updates' in message:
                    message['changes'] = message['updates']
                elif 'bids' in message and 'asks' in message:
                    # Convert full book snapshot to changes format
                    changes = []
                    for bid in message.get('bids', [])[:10]:
                        if len(bid) >= 2:
                            changes.append(['buy', str(bid[0]), str(bid[1])])
                    for ask in message.get('asks', [])[:10]:
                        if len(ask) >= 2:
                            changes.append(['sell', str(ask[0]), str(ask[1])])
                    message['changes'] = changes
                elif 'data' in message and isinstance(message['data'], list):
                    message['changes'] = message['data']

        return message

    def _update_depth_from_l2(self, symbol: str, message: Dict):
        """Update depth metrics from L2 order book data."""
        # This is a simplified implementation
        # In production, you'd maintain a full order book
        try:
            changes = message.get('changes', [])
            if not changes:
                return

            # Calculate USD notional depth (price * size)
            bid_depth_usd = Decimal('0')
            ask_depth_usd = Decimal('0')
            bid_depth_l1_usd = Decimal('0')
            ask_depth_l1_usd = Decimal('0')

            bid_count = 0
            ask_count = 0

            for change in changes[:10]:  # Top 10 levels
                side, price_str, size_str = change
                price = Decimal(price_str) if price_str else Decimal('0')
                size = Decimal(size_str) if size_str != '0' else Decimal('0')

                # Calculate USD notional value
                notional = price * size

                if side == 'buy':
                    bid_depth_usd += notional
                    # Track L1 (best bid)
                    if bid_count == 0:
                        bid_depth_l1_usd = notional
                    bid_count += 1
                elif side == 'sell':
                    ask_depth_usd += notional
                    # Track L1 (best ask)
                    if ask_count == 0:
                        ask_depth_l1_usd = notional
                    ask_count += 1

            market_data = self._market_data[symbol]
            # L1 depth: sum of best bid and best ask notional
            market_data['depth_l1'] = bid_depth_l1_usd + ask_depth_l1_usd
            # L10 depth: sum of all levels notional
            market_data['depth_l10'] = bid_depth_usd + ask_depth_usd

        except Exception as e:
            logger.debug(f"Error updating depth for {symbol}: {e}")

    # ===== UTILITY METHODS =====

    def is_stale(self, symbol: str, threshold_seconds: int = 10) -> bool:
        """Check if market data is stale."""
        if symbol not in self._market_data:
            return True

        last_update = self._market_data[symbol].get('last_update')
        if not last_update:
            return True

        return (datetime.utcnow() - last_update).total_seconds() > threshold_seconds

    def get_perpetuals(self) -> List[Product]:
        """Get all available perpetual products."""
        return self.list_products(market_type=MarketType.PERPETUAL)

    # Connectivity
    def connect(self) -> bool:
        """Connect and validate authentication by fetching accounts."""
        try:
            # Test connection with a lightweight authenticated endpoint
            data = self.client.get_accounts()
            if data:
                # Successfully authenticated
                self._connected = True
                # Use first account ID if available
                accounts = data.get("accounts") or data.get("data") or []
                if accounts and len(accounts) > 0:
                    self._account_id = accounts[0].get("uuid") or accounts[0].get("id") or "COINBASE"
                else:
                    self._account_id = "COINBASE"

                import logging
                logging.getLogger(__name__).info(f"Connected to Coinbase (account: {self._account_id})")
                return True
            else:
                self._connected = False
                return False
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Connection failed: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        self._connected = False

    def validate_connection(self) -> bool:
        return self._connected

    # Accounts and balances
    def get_account_id(self) -> str:
        return self._account_id or ""

    def get_portfolio_balances(self) -> List[Balance]:
        """Get complete portfolio balances including USD from portfolio breakdown."""
        try:
            # First, get the portfolio ID from accounts
            accounts_data = self.client.get_accounts() or {}
            accounts = accounts_data.get("accounts", [])

            if not accounts:
                return self.list_balances()  # Fallback to account balances

            # Get portfolio ID from first account
            portfolio_id = accounts[0].get("retail_portfolio_id")
            if not portfolio_id:
                return self.list_balances()  # Fallback to account balances

            # Get portfolio breakdown
            breakdown = self.client.get_portfolio_breakdown(portfolio_id)
            if not breakdown:
                return self.list_balances()  # Fallback to account balances

            balances: List[Balance] = []
            breakdown_data = breakdown.get("breakdown", {})

            # Process spot positions (includes USD)
            spot_positions = breakdown_data.get("spot_positions", [])
            for pos in spot_positions:
                asset = pos.get("asset", "")

                # Get balance amount
                balance_crypto = pos.get("total_balance_crypto", 0)
                if isinstance(balance_crypto, dict):
                    amount = Decimal(str(balance_crypto.get("value", 0)))
                else:
                    amount = Decimal(str(balance_crypto)) if balance_crypto else Decimal("0")

                # Get hold amount
                hold_crypto = pos.get("hold", 0)
                if isinstance(hold_crypto, dict):
                    hold = Decimal(str(hold_crypto.get("value", 0)))
                else:
                    hold = Decimal(str(hold_crypto)) if hold_crypto else Decimal("0")

                # Available is total minus hold
                available = amount - hold

                if amount > 0 or asset in ['USD', 'USDC', 'EUR', 'GBP']:
                    balances.append(
                        Balance(
                            asset=asset,
                            total=amount,
                            available=available,
                            hold=hold,
                        )
                    )

            return balances

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not get portfolio breakdown, falling back to account balances: {e}")
            return self.list_balances()

    def list_balances(self) -> List[Balance]:
        """Get balances from accounts, properly parsing nested balance objects."""
        raw = self.client.get_accounts() or {}
        if isinstance(raw, list):
            accounts = raw
        else:
            accounts = raw.get("accounts") or raw.get("data") or []
        balances: List[Balance] = []

        for a in accounts:
            try:
                currency = a.get("currency") or a.get("asset") or ""

                # Parse available_balance (nested object with 'value' field)
                available_balance = a.get("available_balance", {})
                if isinstance(available_balance, dict) and available_balance:
                    available = available_balance.get("value", "0")
                else:
                    fallback_available = a.get("available")
                    if fallback_available is not None:
                        available = str(fallback_available)
                    else:
                        available = str(available_balance) if available_balance else "0"

                # Parse hold (nested object with 'value' field)
                hold_data = a.get("hold", {})
                if isinstance(hold_data, dict) and hold_data:
                    hold = hold_data.get("value", "0")
                else:
                    hold = str(hold_data) if hold_data else str(a.get("hold", "0"))

                total_field = a.get("balance") or a.get("total_balance")
                if isinstance(total_field, dict):
                    total = total_field.get("value", "0")
                elif total_field is not None:
                    total = str(total_field)
                else:
                    # Derive total from available + hold when not provided explicitly
                    total = str(Decimal(str(available)) + Decimal(str(hold)))

                # Convert to Decimal, handling any format issues
                try:
                    total_decimal = Decimal(str(total))
                    available_decimal = Decimal(str(available))
                    hold_decimal = Decimal(str(hold))

                    # Only add if there's a balance or it's a fiat currency
                    if total_decimal > 0 or currency in ['USD', 'USDC', 'EUR', 'GBP']:
                        balances.append(
                            Balance(
                                asset=str(currency),
                                total=total_decimal,
                                available=available_decimal,
                                hold=hold_decimal,
                            )
                        )
                except (ValueError, InvalidOperation) as e:
                    import logging
                    logger.warning(f"Could not parse balance for {currency}: {e}")
                    continue

            except Exception as e:
                import logging
                logger.warning(f"Error processing account {a.get('currency', 'unknown')}: {e}")
                continue

        return balances

    # Account metadata helpers
    def get_key_permissions(self) -> Dict[str, Any]:
        data = self.client.get_key_permissions() or {}
        return data.get("key_permissions") or data

    def get_fee_schedule(self) -> Dict[str, Any]:
        return self.client.get_fees() or {}

    def get_account_limits(self) -> Dict[str, Any]:
        data = self.client.get_limits() or {}
        return data.get("limits") or data

    def get_transaction_summary(self) -> Dict[str, Any]:
        return self.client.get_transaction_summary() or {}

    def list_payment_methods(self) -> List[Dict[str, Any]]:
        data = self.client.list_payment_methods() or {}
        return list(data.get("payment_methods") or data.get("data") or [])

    def get_payment_method(self, payment_method_id: str) -> Dict[str, Any]:
        data = self.client.get_payment_method(payment_method_id) or {}
        return data.get("payment_method") or data

    def list_portfolios(self) -> List[Dict[str, Any]]:
        data = self.client.list_portfolios() or {}
        return list(data.get("portfolios") or data.get("data") or [])

    def get_portfolio(self, portfolio_uuid: str) -> Dict[str, Any]:
        data = self.client.get_portfolio(portfolio_uuid) or {}
        return data.get("portfolio") or data

    def get_portfolio_breakdown(self, portfolio_uuid: str) -> Dict[str, Any]:
        data = self.client.get_portfolio_breakdown(portfolio_uuid) or {}
        return data.get("breakdown") or data

    def move_portfolio_funds(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.move_funds(payload) or {}

    # Convert helpers
    def create_convert_quote(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.convert_quote(payload) or {}

    def commit_convert_trade(self, trade_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.client.commit_convert_trade(trade_id, payload or {}) or {}

    def get_convert_trade(self, trade_id: str) -> Dict[str, Any]:
        return self.client.get_convert_trade(trade_id) or {}

    # Products and market data
    def list_products(self, market: Optional[MarketType] = None) -> List[to_product.__annotations__["return"]]:
        data = self.client.get_products() or {}
        items = data.get("products") or data.get("data") or []
        products = [to_product(p) for p in items]
        if market:
            products = [p for p in products if p.market_type == market]
        return products

    def get_quote(self, symbol: str) -> Quote:
        pid = normalize_symbol(symbol)
        data = self.client.get_ticker(pid)
        return to_quote({"product_id": pid, **(data or {})})

    def get_candles(self, symbol: str, granularity: str, limit: int = 200) -> List[Candle]:
        pid = normalize_symbol(symbol)
        data = self.client.get_candles(pid, granularity, limit) or {}
        items = data.get("candles") or data.get("data") or []
        return [to_candle(c) for c in items]

    # Orders
    def _build_order_payload(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: Decimal,
        price: Optional[Decimal],
        stop_price: Optional[Decimal],
        tif: TimeInForce,
        client_id: Optional[str],
        reduce_only: Optional[bool],
        leverage: Optional[int],
        post_only: bool = False,
        include_client_id: bool = True,
    ) -> Dict[str, object]:
        pid = normalize_symbol(symbol)
        product = self.product_catalog.get(self.client, pid)

        def _coerce_enum(enum_cls, raw, field: str):
            if isinstance(raw, enum_cls):
                return raw
            if raw is None:
                raise ValidationError(f"{field} is required", field=field)
            token = str(raw)
            try:
                return enum_cls[token.upper()]
            except (KeyError, AttributeError):
                for member in enum_cls:  # type: ignore[arg-type]
                    if token.lower() == str(member.value).lower():
                        return member
            raise ValidationError(f"Unsupported {field}: {token}", field=field)

        side_enum = _coerce_enum(OrderSide, side, "side")
        order_type_enum = _coerce_enum(OrderType, order_type, "order_type")

        adjusted_qty = quantize_to_increment(qty, product.step_size)

        adjusted_price = price
        if order_type_enum in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if adjusted_price is None:
                raise ValidationError("price is required for limit orders", field="price")
            adjusted_price = quantize_to_increment(adjusted_price, product.price_increment)

        adjusted_stop = stop_price
        if adjusted_stop is not None:
            adjusted_stop = quantize_to_increment(adjusted_stop, product.price_increment)

        if adjusted_qty < product.min_size:
            raise ValidationError(
                f"qty {adjusted_qty} is below min_size {product.min_size}", field="qty", value=str(adjusted_qty)
            )

        gtd_requested = False
        if isinstance(tif, TimeInForce):
            tif_enum = tif
        else:
            token = str(tif).upper()
            if token in {"GTD", "GOOD_TILL_DATE"}:
                tif_enum = TimeInForce.GTC
                gtd_requested = True
            else:
                tif_enum = _coerce_enum(TimeInForce, tif, "time_in_force")

        payload: Dict[str, object] = {
            "product_id": pid,
            "side": side_enum.value.upper(),
        }

        if product.min_notional:
            reference_price = adjusted_price
            if reference_price is None:
                quote = self.get_quote(pid)
                reference_price = quote.last
            notional = adjusted_qty * reference_price
            if notional < product.min_notional:
                raise ValidationError(
                    f"notional {notional} below min_notional {product.min_notional}",
                    field="qty",
                    value=str(adjusted_qty),
                )

        if order_type_enum == OrderType.LIMIT:
            limit_map = {
                TimeInForce.IOC: "limit_limit_ioc",
                TimeInForce.FOK: "limit_limit_fok",
            }
            config_key = limit_map.get(tif_enum, "limit_limit_gtc")
            payload["order_configuration"] = {
                config_key: {
                    "base_size": str(adjusted_qty),
                    "limit_price": str(adjusted_price) if adjusted_price is not None else "0",
                }
            }
        elif order_type_enum == OrderType.MARKET:
            payload["order_configuration"] = {
                "market_market_ioc": {
                    "base_size": str(adjusted_qty)
                }
            }
        elif order_type_enum == OrderType.STOP_LIMIT and adjusted_stop is not None:
            payload["order_configuration"] = {
                "stop_limit_stop_limit_gtc": {
                    "base_size": str(adjusted_qty),
                    "limit_price": str(adjusted_price) if adjusted_price is not None else "0",
                    "stop_price": str(adjusted_stop),
                }
            }
        else:
            payload["type"] = order_type_enum.value
            payload["size"] = str(adjusted_qty)
            payload["time_in_force"] = tif_enum.value
            if adjusted_price is not None:
                payload["price"] = str(adjusted_price)
            if adjusted_stop is not None:
                payload["stop_price"] = str(adjusted_stop)

        if "order_configuration" in payload:
            payload.setdefault("type", order_type_enum.value)
            payload.setdefault("size", str(adjusted_qty))
            payload.setdefault("time_in_force", tif_enum.value)
            if adjusted_price is not None:
                payload.setdefault("price", str(adjusted_price))
            if adjusted_stop is not None:
                payload.setdefault("stop_price", str(adjusted_stop))
            if gtd_requested:
                config = next(iter(payload["order_configuration"].values()))
                config["time_in_force"] = "GOOD_TILL_DATE"
                config["end_time"] = (datetime.utcnow() + timedelta(minutes=2)).isoformat()
                payload["time_in_force"] = "GOOD_TILL_DATE"
        elif gtd_requested:
            payload["time_in_force"] = "GOOD_TILL_DATE"

        if include_client_id and client_id:
            payload["client_order_id"] = client_id
        if reduce_only is not None:
            payload["reduce_only"] = reduce_only
        if post_only and order_type_enum == OrderType.LIMIT:
            payload["post_only"] = True
        if leverage is not None:
            payload["leverage"] = leverage

        return payload

    def _execute_order_payload(
        self,
        symbol: str,
        payload: Dict[str, object],
        client_id: Optional[str],
    ) -> Order:
        if os.getenv("ORDER_PREVIEW_ENABLED"):
            try:
                self.client.preview_order(payload)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - preview optional
                logger.debug(f"Order preview failed (ignored): {exc}")

        try:
            data = self.client.place_order(payload)
            return to_order(data or {})
        except InsufficientFunds as exc:
            logger.error(f"Insufficient funds for {payload.get('product_id')}: {exc}")
            raise
        except ValidationError as exc:
            logger.error(f"Order validation failed for {payload.get('product_id')}: {exc}")
            raise
        except InvalidRequestError as exc:
            if client_id and "duplicate" in str(exc).lower():
                existing = self._find_existing_order_by_client_id(payload.get("product_id", ""), client_id)
                if existing:
                    logger.info(
                        "Resolved duplicate client_order_id for %s via order %s",
                        payload.get("product_id"),
                        existing.id,
                    )
                    return existing
            logger.error(f"Order placement failed for {payload.get('product_id')}: {exc}")
            raise
        except Exception as exc:
            logger.error(
                f"Order placement failed for {payload.get('product_id')}: {exc.__class__.__name__}: {exc}"
            )
            raise

    def _find_existing_order_by_client_id(self, product_id: str, client_id: str) -> Optional[Order]:
        if not client_id:
            return None

        try:
            data = self.client.list_orders(product_id=product_id) or {}
        except Exception as exc:
            logger.debug(f"Could not resolve duplicate order for {product_id}: {exc}")
            return None

        items = data.get("orders") or data.get("data") or []
        matches = [o for o in items if str(o.get("client_order_id")) == client_id]
        if not matches:
            return None

        def order_timestamp(order: Dict[str, Any]) -> datetime:
            raw = order.get("created_at") or order.get("submitted_at")
            if not raw:
                return datetime.min
            try:
                normalized = str(raw).replace("Z", "+00:00")
                return datetime.fromisoformat(normalized)
            except Exception:
                return datetime.min

        best = max(matches, key=order_timestamp)
        return to_order(best)

    def preview_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        tif: TimeInForce = TimeInForce.GTC,
        reduce_only: Optional[bool] = None,
        leverage: Optional[int] = None,
        post_only: bool = False,
    ) -> Dict[str, Any]:
        payload = self._build_order_payload(
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            price=price,
            stop_price=stop_price,
            tif=tif,
            client_id=None,
            reduce_only=reduce_only,
            leverage=leverage,
            post_only=post_only,
            include_client_id=False,
        )
        return self.client.preview_order(payload)  # type: ignore[attr-defined]

    def edit_order_preview(
        self,
        *,
        order_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        tif: TimeInForce = TimeInForce.GTC,
        new_client_id: Optional[str] = None,
        reduce_only: Optional[bool] = None,
        leverage: Optional[int] = None,
        post_only: bool = False,
    ) -> Dict[str, Any]:
        payload = self._build_order_payload(
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            price=price,
            stop_price=stop_price,
            tif=tif,
            client_id=None,
            reduce_only=reduce_only,
            leverage=leverage,
            post_only=post_only,
            include_client_id=False,
        )
        payload["order_id"] = order_id
        if new_client_id:
            payload["new_client_order_id"] = new_client_id
        return self.client.edit_order_preview(payload)  # type: ignore[attr-defined]

    def edit_order(self, order_id: str, preview_id: str) -> Order:
        payload = {"order_id": order_id, "preview_id": preview_id}
        data = self.client.edit_order(payload)  # type: ignore[attr-defined]
        return to_order(data or {})

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        tif: TimeInForce = TimeInForce.GTC,
        client_id: Optional[str] = None,
        reduce_only: Optional[bool] = None,
        leverage: Optional[int] = None,
        post_only: bool = False,
    ) -> Order:
        final_client_id = client_id or f"perps_{uuid.uuid4().hex[:12]}"

        payload = self._build_order_payload(
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            price=price,
            stop_price=stop_price,
            tif=tif,
            client_id=final_client_id,
            reduce_only=reduce_only,
            leverage=leverage,
            post_only=post_only,
        )

        return self._execute_order_payload(symbol, payload, final_client_id)

    def cancel_order(self, order_id: str) -> bool:
        res = self.client.cancel_orders([order_id]) or {}
        results = res.get("results") or res.get("data") or []
        for r in results:
            if str(r.get("order_id")) == order_id and r.get("success") is True:
                return True
        # Some APIs return just list of cancelled ids
        ids = res.get("cancelled_order_ids") or []
        return order_id in ids

    # PnL Methods (Phase 4)
    def _update_position_metrics(self, symbol: str) -> None:
        """Update position metrics with current mark price."""
        if symbol not in self._positions:
            return

        mark = self._mark_cache.get_mark(symbol)
        if mark is None:
            return  # No fresh mark available

        position = self._positions[symbol]
        unrealized_pnl = position.get_unrealized_pnl(mark)

        # Check for funding accrual
        funding_rate, next_funding_time = self.product_catalog.get_funding(self.client, symbol)
        funding_delta = self._funding_calculator.accrue_if_due(
            symbol=symbol,
            position_size=position.qty,
            position_side=position.side,
            mark_price=mark,
            funding_rate=funding_rate,
            next_funding_time=next_funding_time
        )

        if funding_delta != 0:
            position.realized_pnl += funding_delta
            # Record funding event
            self._event_store.append_metric(
                bot_id="coinbase_perps",
                metrics={
                    "type": "funding",
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "side": position.side,
                    "qty": str(position.qty),
                    "funding_rate": str(funding_rate or Decimal("0")),
                    "mark_price": str(mark),
                    "funding_amount": str(funding_delta)
                }
            )

        # Record metric snapshot
        position_value = position.qty * mark

        self._event_store.append_position(
            bot_id="coinbase_perps",
            position={
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "side": position.side,
                "qty": str(position.qty),
                "entry_price": str(position.entry_price),
                "mark_price": str(mark),
                "unrealized_pnl": str(unrealized_pnl),
                "realized_pnl": str(position.realized_pnl),
                "position_value": str(position_value)
            }
        )

    def _process_fill_for_pnl(self, fill: Dict) -> None:
        """Process a fill event to update position PnL."""
        symbol = fill.get('product_id')
        if not symbol:
            return

        # Extract fill details
        fill_qty = Decimal(str(fill.get('size', '0')))
        fill_price = Decimal(str(fill.get('price', '0')))
        fill_side = str(fill.get('side', '')).lower()  # 'buy' or 'sell'

        if fill_qty == 0 or fill_price == 0:
            return

        # Get or create position
        if symbol not in self._positions:
            # New position
            position_side = 'long' if fill_side == 'buy' else 'short'
            self._positions[symbol] = PositionState(
                symbol=symbol,
                side=position_side,
                qty=fill_qty,
                entry_price=fill_price
            )
        else:
            # Update existing position
            position = self._positions[symbol]
            realized_delta = position.update_from_fill(fill_qty, fill_price, fill_side)

            if realized_delta != 0:
                logger.info(f"Realized PnL for {symbol}: {realized_delta}")

        # Update metrics with latest state
        self._update_position_metrics(symbol)

    def get_position_pnl(self, symbol: str) -> Dict[str, Any]:
        """Get current PnL for a position.

        Returns:
            Dict with uPnL, rPnL, funding_accrued, mark, entry, qty, side
        """
        if symbol not in self._positions:
            return {
                'symbol': symbol,
                'qty': Decimal('0'),
                'side': None,
                'entry': None,
                'mark': None,
                'unrealized_pnl': Decimal('0'),
                'realized_pnl': Decimal('0'),
                'funding_accrued': Decimal('0')
            }

        position = self._positions[symbol]
        mark = self._mark_cache.get_mark(symbol)
        unrealized_pnl = position.get_unrealized_pnl(mark)

        # Get funding history for this symbol from tail events
        events = self._event_store.tail(bot_id="coinbase_perps", limit=100, types=["metric"])
        funding_events = [e for e in events if e.get("type") == "funding" and e.get("symbol") == symbol]
        total_funding = sum(Decimal(e.get("funding_amount", "0")) for e in funding_events)

        return {
            'symbol': symbol,
            'qty': position.qty,
            'side': position.side,
            'entry': position.entry_price,
            'mark': mark,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': position.realized_pnl,
            'funding_accrued': total_funding
        }

    def get_portfolio_pnl(self) -> Dict[str, Any]:
        """Get aggregated PnL across all positions.

        Returns:
            Dict with total uPnL, rPnL, and per-symbol breakdown
        """
        total_unrealized = Decimal('0')
        total_realized = Decimal('0')
        total_funding = Decimal('0')
        positions = {}

        for symbol in self._positions:
            pnl_data = self.get_position_pnl(symbol)
            positions[symbol] = pnl_data
            total_unrealized += pnl_data['unrealized_pnl']
            total_realized += pnl_data['realized_pnl']
            total_funding += pnl_data['funding_accrued']

        return {
            'total_unrealized_pnl': total_unrealized,
            'total_realized_pnl': total_realized,
            'total_funding': total_funding,
            'total_pnl': total_unrealized + total_realized,
            'positions': positions
        }

    def close_position(self, symbol: str, qty: Optional[Decimal] = None, reduce_only: bool = True) -> Order:
        """Close a position using the native endpoint, falling back to market order."""
        product_id = normalize_symbol(symbol)
        # Determine position context for close
        position_qty: Decimal | None = qty
        close_side: OrderSide | None = None
        if position_qty is None or close_side is None:
            position = next((p for p in self.list_positions() if p.symbol == product_id), None)
            if not position:
                raise
            current_qty = getattr(position, "qty", getattr(position, "quantity", Decimal("0")))
            if current_qty == 0:
                raise
            position_qty = abs(current_qty)
            close_side = OrderSide.SELL if current_qty > 0 else OrderSide.BUY
        else:
            close_side = OrderSide.SELL if qty and qty > 0 else OrderSide.BUY

        payload: Dict[str, Any] = {
            "product_id": product_id,
            "reduce_only": reduce_only,
            "size": str(position_qty),
            "side": close_side.value.upper(),
        }

        try:
            data = self.client.close_position(payload) or {}
            return to_order(data.get("order") or data)
        except Exception as exc:
            logger.warning(f"close_position fallback triggered for {symbol}: {exc}")

            order = self.place_order(
                symbol=product_id,
                side=close_side,
                order_type=OrderType.MARKET,
                qty=position_qty,
                reduce_only=reduce_only,
            )
            return order

    def get_order(self, order_id: str) -> Order:
        data = self.client.get_order_historical(order_id) or {}
        return to_order(data.get("order") or data)

    def list_orders(self, status: Optional[OrderStatus] = None, symbol: Optional[str] = None) -> List[Order]:
        params: Dict[str, str] = {}
        if status:
            params["order_status"] = status.value
        if symbol:
            params["product_id"] = normalize_symbol(symbol)
        try:
            data = self.client.list_orders(**params) or {}
        except Exception as exc:
            logger.error(f"Failed to list orders: {exc}")
            return []
        items = data.get("orders") or data.get("data") or []
        return [to_order(o) for o in items]

    # Positions and fills
    def list_positions(self) -> List[Position]:
        """List positions from CFM endpoints when available.

        Returns an empty list for spot-only configs (enable_derivatives=False).
        """
        if not self.config.enable_derivatives:
            return []
        try:
            data = self.client.list_positions() or {}
            items = data.get("positions") or data.get("data") or []
            from .models import to_position
            return [to_position(p) for p in items]
        except Exception as e:
            # Log error but don't crash - return empty list
            import logging
            logging.getLogger(__name__).warning(f"Failed to fetch positions: {e}")
            return []

    def list_fills(self, symbol: Optional[str] = None, limit: int = 200) -> List[Dict]:
        params: Dict[str, str] = {"limit": str(limit)}
        if symbol:
            params["product_id"] = normalize_symbol(symbol)
        try:
            data = self.client.list_fills(**params) or {}
        except Exception as exc:
            logger.error(f"Failed to list fills: {exc}")
            return []
        return data.get("fills") or data.get("data") or []

    # Streaming
    def stream_trades(self, symbols: Sequence[str]) -> Iterable[Dict]:
        """Stream market trades for given symbols with normalized Decimal prices/sizes."""
        ws = self._create_ws()
        sub = WSSubscription(channels=["market_trades"], product_ids=[normalize_symbol(s) for s in symbols])
        ws.subscribe(sub)
        for msg in ws.stream_messages():
            # Normalize market data (Decimal conversion, timestamp)
            normalized = normalize_market_message(msg)
            # Ensure product_id is present
            if 'product_id' not in normalized and 'symbol' in normalized:
                normalized['product_id'] = normalized['symbol']

            # Update mark cache with trade price for perpetuals
            if 'product_id' in normalized and 'price' in normalized:
                symbol = normalized['product_id']
                price = normalized['price']
                if isinstance(price, Decimal):
                    # Check if this is a perpetual product
                    try:
                        product = self.product_catalog.get(self.client, symbol)
                        if product.market_type == MarketType.PERPETUAL:
                            self._mark_cache.set_mark(symbol, price)
                            # Update position metrics with new mark
                            self._update_position_metrics(symbol)
                    except Exception:
                        pass  # Not a known product or error accessing catalog

            yield normalized

    def stream_orderbook(self, symbols: Sequence[str], level: int = 1) -> Iterable[Dict]:
        """Stream orderbook updates. Uses 'level2' for level>=2, else 'ticker' for L1."""
        ws = self._create_ws()
        channel = "level2" if level >= 2 else "ticker"
        sub = WSSubscription(channels=[channel], product_ids=[normalize_symbol(s) for s in symbols])
        ws.subscribe(sub)
        for msg in ws.stream_messages():
            # Normalize market data for orderbook
            normalized = normalize_market_message(msg)

            # For ticker channel, update mark cache with mid price for perpetuals
            if channel == "ticker" and 'product_id' in normalized:
                symbol = normalized['product_id']
                bid = normalized.get('best_bid') or normalized.get('bid')
                ask = normalized.get('best_ask') or normalized.get('ask')

                if bid and ask and isinstance(bid, Decimal) and isinstance(ask, Decimal):
                    # Use mid price as mark
                    mid_price = (bid + ask) / 2
                    try:
                        product = self.product_catalog.get(self.client, symbol)
                        if product.market_type == MarketType.PERPETUAL:
                            self._mark_cache.set_mark(symbol, mid_price)
                            # Update position metrics with new mark
                            self._update_position_metrics(symbol)
                    except Exception:
                        pass  # Not a known product or error accessing catalog

            yield normalized

    # Internal factory for test injection
    def _create_ws(self) -> CoinbaseWebSocket:
        if self._ws_factory_override:
            return self._ws_factory_override()

        # Create WebSocket with auth provider if available
        ws_auth_provider = None
        auth_obj = getattr(self.client, "auth", None)

        if os.getenv("COINBASE_WS_USER_AUTH") and hasattr(auth_obj, "generate_jwt"):
            def ws_auth_provider():
                try:
                    generator = getattr(auth_obj, "generate_jwt")
                    try:
                        token = generator("GET", "/users/self")
                    except TypeError:
                        token = generator()
                    return {"jwt": token}
                except Exception as exc:
                    logger.warning("Failed to generate WS auth via client auth object: %s", exc)
                    return None
        elif self._config.enable_derivatives and self._config.auth_type == "JWT":
            # Create a provider that generates JWT auth data for WS
            def ws_auth_provider():
                try:
                    from .auth import CDPAuthV2

                    cdp_auth = CDPAuthV2(
                        api_key_name=self._config.cdp_api_key,
                        private_key=self._config.cdp_private_key
                    )
                    token = cdp_auth.generate_jwt()
                    return {"jwt": token}
                except Exception as e:
                    logger.warning(f"Failed to generate WS auth: {e}")
                    return None

        return CoinbaseWebSocket(
            url=self._ws_url,
            ws_auth_provider=ws_auth_provider
        )

    # Testing helpers (no network)
    def set_http_transport_for_testing(self, transport) -> None:
        self.client.set_transport_for_testing(transport)

    def set_ws_factory_for_testing(self, factory) -> None:
        self._ws_factory_override = factory

    # Coinbase-specific: stream authenticated user events (orders/fills)
    def stream_user_events(self, product_ids: Optional[Sequence[str]] = None) -> Iterable[Dict]:
        """Stream authenticated user events (orders/fills) with sequence gap detection."""
        ws = self._create_ws()
        sub = WSSubscription(channels=["user"], product_ids=[normalize_symbol(s) for s in (product_ids or [])])
        ws.subscribe(sub)
        guard = SequenceGuard()
        for msg in ws.stream_messages():
            # Apply sequence guard for gap detection
            annotated = guard.annotate(msg)

            # Process fills for PnL tracking
            if annotated.get('type') == 'fill' or annotated.get('event_type') == 'fill':
                self._process_fill_for_pnl(annotated)

            yield annotated
