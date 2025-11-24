"""Streaming helpers for the Coinbase data provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="coinbase_provider")

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.coinbase.market_data_service import (
        CoinbaseTickerService,
        TickerCache,
    )
    from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket
    from gpt_trader.config.runtime_settings import RuntimeSettings


class CoinbaseStreamingMixin:
    """Mixin that encapsulates Coinbase WebSocket streaming behaviour."""

    enable_streaming: bool
    cache_ttl: int
    ticker_service: "CoinbaseTickerService | None"
    ticker_cache: "TickerCache | None"
    _settings: "RuntimeSettings"

    def _initialize_streaming(self, enable_streaming: bool, cache_ttl: int) -> None:
        self.enable_streaming = enable_streaming
        self.cache_ttl = cache_ttl
        self.ticker_service = None
        self.ticker_cache = None
        if self.enable_streaming:
            self._setup_streaming()

    def _setup_streaming(self) -> None:
        """Setup WebSocket streaming for real-time data."""
        try:
            from gpt_trader.features.brokerages.coinbase.market_data_service import (
                CoinbaseTickerService,
                TickerCache,
            )
            from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to import streaming dependencies: %s",
                exc,
                operation="streaming_setup",
                status="import_error",
            )
            self.enable_streaming = False
            return

        try:
            self.ticker_cache = TickerCache()

            def ws_factory() -> "CoinbaseWebSocket":
                ws = CoinbaseWebSocket(
                    url="wss://advanced-trade-ws.coinbase.com",
                    settings=self._settings,
                )
                ws.connect()
                return ws

            self.ticker_service = CoinbaseTickerService(
                websocket_factory=ws_factory,
                symbols=[],
                cache=self.ticker_cache,
                on_update=self._on_ticker_update,
            )

            logger.info(
                "WebSocket streaming setup complete",
                operation="streaming_setup",
                status="success",
            )
        except Exception as exc:
            logger.warning(
                "Failed to setup streaming, falling back to REST",
                operation="streaming_setup",
                status="fallback",
                error=str(exc),
            )
            self.enable_streaming = False
            self.ticker_service = None
            self.ticker_cache = None

    def _on_ticker_update(self, symbol: str, ticker: Any) -> None:
        """Callback for ticker updates from WebSocket."""
        logger.debug(
            "Received ticker update",
            symbol=symbol,
            bid=getattr(ticker, "bid", None),
            ask=getattr(ticker, "ask", None),
            last=getattr(ticker, "last", None),
        )

    def _subscribe_streaming(self, symbols: list[str]) -> None:
        if not self.enable_streaming or not self.ticker_service:
            return
        try:
            self.ticker_service.set_symbols(symbols)
            self.ticker_service.ensure_started()
            logger.info(
                "Subscribed to WebSocket updates",
                symbols=len(symbols),
                operation="streaming_subscribe",
                status="success",
            )
        except Exception as exc:
            logger.warning(
                "Failed to subscribe to WebSocket",
                operation="streaming_subscribe",
                status="error",
                error=str(exc),
            )

    def start_streaming(self) -> None:
        """Start WebSocket streaming if configured."""
        if self.enable_streaming and self.ticker_service:
            self.ticker_service.start()
            logger.info(
                "WebSocket streaming started",
                operation="streaming",
                status="started",
            )

    def stop_streaming(self) -> None:
        """Stop WebSocket streaming."""
        if self.ticker_service:
            self.ticker_service.stop()
            logger.info(
                "WebSocket streaming stopped",
                operation="streaming",
                status="stopped",
            )


__all__ = ["CoinbaseStreamingMixin"]
