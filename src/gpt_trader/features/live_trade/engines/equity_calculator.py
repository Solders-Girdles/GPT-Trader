"""
Equity calculation for trading engine.

Extracted from TradingEngine to separate concerns:
- Calculate total equity from balances and positions
- Handle multi-asset valuation with price lookups
- Track calculation metrics and broker failures
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from gpt_trader.monitoring.metrics_collector import record_histogram
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.core import Position
    from gpt_trader.features.live_trade.degradation import DegradationState

logger = get_logger(__name__, component="equity_calculator")


class EquityCalculator:
    """
    Calculates total account equity from balances and positions.

    Handles:
    - Fetching and logging balance summaries
    - USD/USDC as direct cash collateral
    - Converting non-USD assets using ticker prices
    - Adding unrealized PnL from open positions
    - Tracking broker failures for degradation
    """

    # Stable coins treated as 1:1 with USD for valuation
    STABLE_QUOTES = frozenset({"USD", "USDC"})

    def __init__(
        self,
        config: Any,
        degradation: DegradationState,
        risk_manager: Any,
        price_history: dict[str, deque[Decimal]],
        broker_calls: Any | None = None,
    ) -> None:
        """
        Initialize equity calculator.

        Args:
            config: Configuration (must have coinbase_default_quote, read_only attributes)
            degradation: Degradation state for tracking broker failures
            risk_manager: Risk manager for degradation config (must have .config)
            price_history: Cached price history by product_id
        """
        self._config = config
        self._degradation = degradation
        self._risk_manager = risk_manager
        self._price_history = price_history
        self._known_products: set[str] | None = None
        self._known_products_last_refresh: float | None = None
        self._known_products_ttl_seconds = int(
            getattr(self._config, "product_catalog_ttl_seconds", 3600)
        )
        self._broker_calls = (
            broker_calls
            if broker_calls is not None
            and asyncio.iscoroutinefunction(getattr(broker_calls, "__call__", None))
            else None
        )

    async def _call_broker(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        broker_calls = self._broker_calls
        if broker_calls is not None:
            return await broker_calls(func, *args, **kwargs)
        return await asyncio.to_thread(func, *args, **kwargs)

    async def calculate_total_equity(
        self,
        broker: Any,
        positions: dict[str, Position],
    ) -> Decimal | None:
        """
        Calculate total equity = collateral + unrealized PnL.

        Args:
            broker: Broker for balance/ticker fetches
            positions: Current open positions

        Returns:
            Total equity in quote currency, or None on failure
        """
        start_time = time.perf_counter()
        result = "ok"
        try:
            balances = await self._call_broker(broker.list_balances)

            self._log_balance_summary(balances)

            cash_collateral, converted_collateral, diagnostics = await self._calculate_collateral(
                broker, balances
            )
            collateral = cash_collateral + converted_collateral

            self._log_collateral_details(cash_collateral, collateral, diagnostics)

            # Add unrealized PnL from open positions
            unrealized_pnl = sum(
                (p.unrealized_pnl for p in positions.values()),
                Decimal("0"),
            )
            logger.info(f"Unrealized PnL: ${unrealized_pnl}")

            total_equity = collateral + unrealized_pnl
            logger.info(
                f"Total equity calculated: ${total_equity} "
                f"(collateral=${collateral} + unrealized_pnl=${unrealized_pnl})"
            )

            self._log_zero_equity_warning(total_equity)

            # Success: reset broker failure counter
            self._degradation.reset_broker_failures()
            return total_equity

        except Exception as e:
            result = "error"
            logger.error(
                f"Failed to fetch balances: {e}",
                error_type=type(e).__name__,
                operation="fetch_total_equity",
                exc_info=True,
            )
            logger.error(
                "Unable to calculate equity. Check: "
                "1) Network connectivity, "
                "2) API credentials validity, "
                "3) Broker service health"
            )
            # Track broker failure for degradation
            config = self._risk_manager.config if self._risk_manager else None
            if config is not None:
                self._degradation.record_broker_failure(config)
            return None
        finally:
            duration = time.perf_counter() - start_time
            record_histogram(
                "gpt_trader_equity_computation_seconds",
                duration,
                labels={"result": result},
            )

    def _log_balance_summary(self, balances: list[Any]) -> None:
        """Log summary of all assets returned from broker."""
        if balances:
            all_assets = [b.asset for b in balances]
            non_zero_assets = [(b.asset, b.available, b.total) for b in balances if b.total > 0]

            logger.info(f"Fetched {len(balances)} balances from broker")
            logger.info(
                f"All assets in response: {', '.join(all_assets) if all_assets else 'NONE'}"
            )

            if non_zero_assets:
                logger.info(f"Assets with non-zero balances: {len(non_zero_assets)}")
                for asset, avail, total in non_zero_assets:
                    logger.info(f"  {asset}: available={avail}, total={total}")
            else:
                logger.warning(
                    "All balances are zero - this may indicate an API "
                    "permission or portfolio scoping issue"
                )
        else:
            logger.warning("Received empty balance list from broker - check API configuration")

    async def _calculate_collateral(
        self, broker: Any, balances: list[Any]
    ) -> tuple[Decimal, Decimal, dict[str, list[str]]]:
        """
        Calculate cash and converted collateral from balances.

        Args:
            broker: Broker for ticker fetches
            balances: List of balance objects

        Returns:
            Tuple of (cash_collateral, converted_collateral, diagnostics)
        """
        cash_collateral = Decimal("0")
        converted_collateral = Decimal("0")

        diagnostics: dict[str, list[str]] = {
            "usd_usdc_found": [],
            "other_assets_found": [],
            "priced_assets": [],
            "unpriced_assets": [],
        }

        quote = str(getattr(self._config, "coinbase_default_quote", None) or "USD").upper()
        use_total_balance = bool(getattr(self._config, "read_only", False))
        valuation_quotes = self._build_valuation_quotes(quote)
        known_products: set[str] | None = None
        known_products_checked = False

        for balance in balances:
            logger.debug(
                f"Balance: {balance.asset} = {balance.available} available, {balance.total} total"
            )
            asset = str(balance.asset or "").upper()
            amount = balance.total if use_total_balance else balance.available

            # Handle USD/USDC directly
            if asset in ("USD", "USDC"):
                cash_collateral += amount
                if amount > 0:
                    diagnostics["usd_usdc_found"].append(f"{asset}=${amount}")
                continue

            if balance.total > 0:
                diagnostics["other_assets_found"].append(f"{asset}={balance.total}")

            if amount <= 0:
                continue

            # Handle quote currency directly
            if asset == quote:
                cash_collateral += amount
                diagnostics["priced_assets"].append(f"{asset}=${amount}")
                continue

            # Handle stable-to-stable conversion
            if asset in self.STABLE_QUOTES and quote in self.STABLE_QUOTES:
                cash_collateral += amount
                diagnostics["priced_assets"].append(f"{asset}≈${amount}")
                continue

            # Convert non-USD assets using ticker prices
            if not known_products_checked:
                known_products = await self._get_known_products(broker)
                known_products_checked = True
            usd_value = await self._value_asset(
                broker,
                asset,
                amount,
                valuation_quotes,
                diagnostics,
                known_products,
            )
            if usd_value is not None:
                converted_collateral += usd_value

        return cash_collateral, converted_collateral, diagnostics

    def _build_valuation_quotes(self, quote: str) -> list[str]:
        """Build ordered list of quote currencies for valuation."""
        valuation_quotes: list[str] = [quote]
        # Add stable coins as fallbacks
        for stable in ("USD", "USDC"):
            if stable not in valuation_quotes:
                valuation_quotes.append(stable)
        return valuation_quotes

    async def _get_known_products(self, broker: Any) -> set[str] | None:
        """Fetch and cache known product ids to avoid invalid ticker requests."""
        now = time.time()
        if self._known_products and self._known_products_last_refresh is not None:
            if now - self._known_products_last_refresh < self._known_products_ttl_seconds:
                return self._known_products

        product_catalog = getattr(broker, "product_catalog", None)
        cache = getattr(product_catalog, "_cache", None)
        if isinstance(cache, dict) and cache:
            self._known_products = {str(key).upper() for key in cache.keys()}
            self._known_products_last_refresh = now
            return self._known_products

        list_products = getattr(broker, "list_products", None)
        if callable(list_products):
            try:
                products = await self._call_broker(list_products)
            except Exception as exc:
                logger.debug("Failed to list products for valuation: %s", exc)
                return None
            product_ids: set[str] = set()
            for product in products:
                symbol = None
                if isinstance(product, dict):
                    symbol = product.get("product_id") or product.get("id")
                else:
                    symbol = getattr(product, "symbol", None)
                if symbol:
                    product_ids.add(str(symbol).upper())
            if product_ids:
                self._known_products = product_ids
                self._known_products_last_refresh = now
                return self._known_products

        return None

    async def _value_asset(
        self,
        broker: Any,
        asset: str,
        amount: Decimal,
        valuation_quotes: list[str],
        diagnostics: dict[str, list[str]],
        known_products: set[str] | None,
    ) -> Decimal | None:
        """
        Value a non-USD asset using ticker prices.

        Returns:
            USD value of asset, or None if unable to value
        """
        last_price: Decimal | None = None
        used_pair: str | None = None

        for quote in valuation_quotes:
            product_id = f"{asset}-{quote}"
            history = self._price_history.get(product_id)
            if history:
                last_price = history[-1]
            else:
                last_price = None

            try:
                if last_price is None:
                    if known_products is not None and product_id.upper() not in known_products:
                        continue
                    ticker = await self._call_broker(broker.get_ticker, product_id)
                    last_price = Decimal(str(ticker.get("price", 0)))
                if last_price and last_price > 0:
                    used_pair = product_id
                    break
            except Exception as exc:
                logger.debug(
                    "Unable to value %s via %s: %s",
                    asset,
                    product_id,
                    exc,
                )
                continue

        if last_price and last_price > 0 and used_pair:
            usd_value = amount * last_price
            diagnostics["priced_assets"].append(
                f"{asset}={amount} @ {used_pair}≈{usd_value.quantize(Decimal('0.01'))}"
            )
            return usd_value
        else:
            diagnostics["unpriced_assets"].append(asset)
            return None

    def _log_collateral_details(
        self,
        cash_collateral: Decimal,
        total_collateral: Decimal,
        diagnostics: dict[str, list[str]],
    ) -> None:
        """Log detailed collateral breakdown."""
        quote = str(getattr(self._config, "coinbase_default_quote", None) or "USD").upper()
        use_total_balance = bool(getattr(self._config, "read_only", False))

        cash_label = "Total cash holdings" if use_total_balance else "Available cash collateral"
        logger.info(
            "%s (%s): $%s",
            cash_label,
            quote,
            cash_collateral.quantize(Decimal("0.01")),
        )

        if diagnostics["usd_usdc_found"]:
            logger.info(f"USD/USDC assets counted: {', '.join(diagnostics['usd_usdc_found'])}")
        else:
            collateral_scope = "total holdings" if use_total_balance else "available collateral"
            logger.warning(
                "No USD/USDC balances found in %s; valuing non-USD assets using %s tickers",
                collateral_scope,
                quote,
            )
            if diagnostics["other_assets_found"]:
                logger.info(
                    "Non-USD assets detected: %s",
                    ", ".join(diagnostics["other_assets_found"]),
                )

        if diagnostics["priced_assets"]:
            logger.info(
                "Included non-USD assets in equity: %s",
                "; ".join(diagnostics["priced_assets"]),
            )

        if diagnostics["unpriced_assets"]:
            logger.warning(
                "Could not value these assets in %s: %s",
                quote,
                ", ".join(sorted(set(diagnostics["unpriced_assets"]))),
            )

    def _log_zero_equity_warning(self, total_equity: Decimal) -> None:
        """Log diagnostic warning if equity is zero."""
        if total_equity == 0:
            logger.warning(
                "Total equity is $0.00. This typically means: "
                "1) No USD/USDC in account (only crypto assets), "
                "2) Wrong portfolio selected (check portfolio_uuid), "
                "3) API permission issue, or "
                "4) No funds in account"
            )
