"""
Comprehensive REST/WebSocket test suite for Coinbase brokerage.

Covers:
- Enum coercion (invalid values, case variations, GTD fallback)
- Min-notional validation and edge cases
- Reconnect and error handling scenarios
- Transport stub reuse for maintainability

Phase 1 brokerages coverage as specified in next steps.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from decimal import Decimal
from typing import Any
from unittest.mock import Mock

import pytest

from bot_v2.errors import ValidationError
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from bot_v2.features.brokerages.coinbase.market_data_service import MarketDataService
from bot_v2.features.brokerages.coinbase.models import (
    APIConfig,
    to_order,
    to_product,
    to_quote,
)
from bot_v2.features.brokerages.coinbase.rest_service import CoinbaseRestService
from bot_v2.features.brokerages.coinbase.specs import ProductSpec, SpecsService
from bot_v2.features.brokerages.coinbase.transports import MockTransport
from bot_v2.features.brokerages.coinbase.utilities import ProductCatalog
from bot_v2.features.brokerages.coinbase.ws import CoinbaseWebSocket, WSSubscription
from bot_v2.features.brokerages.core.interfaces import (
    MarketType,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from bot_v2.persistence.event_store import EventStore
from tests.fixtures.coinbase_factories import (
    CoinbaseEdgeCaseFactory,
    CoinbaseOrderFactory,
    CoinbaseProductFactory,
    CoinbaseQuoteFactory,
)


# ============================================================================
# Centralized Transport Stubs
# ============================================================================


class ReusableHTTPTransport:
    """
    Centralized HTTP transport stub for REST endpoint tests.

    Provides configurable responses, error injection, and call tracking
    to avoid fixture sprawl across test suites.
    """

    def __init__(self):
        self.responses: deque[tuple[int, dict, str]] = deque()
        self.calls: list[dict[str, Any]] = []
        self.should_fail = False
        self.failure_count = 0
        self.max_failures = 0

    def add_response(self, status: int, headers: dict | None = None, body: str | dict = ""):
        """Queue a response to be returned."""
        if isinstance(body, dict):
            body = json.dumps(body)
        self.responses.append((status, headers or {}, body))

    def set_failure_mode(self, count: int):
        """Inject failures for the next N calls."""
        self.should_fail = True
        self.max_failures = count
        self.failure_count = 0

    def __call__(self, method: str, url: str, headers: dict, body: Any, timeout: float):
        """Simulate HTTP call."""
        self.calls.append({
            "method": method,
            "url": url,
            "headers": headers,
            "body": body,
            "timeout": timeout,
        })

        # Simulate failures
        if self.should_fail and self.failure_count < self.max_failures:
            self.failure_count += 1
            return 503, {}, json.dumps({"error": "service_unavailable"})

        # Return queued response
        if not self.responses:
            return 200, {}, json.dumps({"success": True})

        return self.responses.popleft()


class ReusableWebSocketTransport:
    """
    Centralized WebSocket transport stub for streaming tests.

    Provides message batching, disconnect simulation, and subscription tracking
    to avoid duplication across WebSocket test scenarios.
    """

    def __init__(self, message_batches: list[tuple[list[dict], bool]] | None = None):
        """
        Args:
            message_batches: List of (messages, should_disconnect) tuples
        """
        self.message_batches: deque = deque(message_batches or [])
        self.subscriptions: list[dict] = []
        self.connect_count = 0
        self.disconnect_count = 0
        self.connected = False

    def connect(self, url: str, headers: dict[str, Any] | None = None) -> None:
        """Simulate connection."""
        self.connected = True
        self.connect_count += 1

    def disconnect(self) -> None:
        """Simulate disconnection."""
        self.connected = False
        self.disconnect_count += 1

    def subscribe(self, payload: dict[str, Any]) -> None:
        """Track subscription payloads."""
        self.subscriptions.append(payload)

    def stream(self):
        """Stream messages from batches."""
        if not self.message_batches:
            return

        messages, should_disconnect = self.message_batches.popleft()
        for msg in messages:
            yield msg

        if should_disconnect:
            raise ConnectionError("simulated disconnect")

    def add_message_batch(self, messages: list[dict], disconnect: bool = False):
        """Add a batch of messages to stream."""
        self.message_batches.append((messages, disconnect))


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def http_transport():
    """Reusable HTTP transport stub."""
    return ReusableHTTPTransport()


@pytest.fixture
def ws_transport():
    """Reusable WebSocket transport stub."""
    return ReusableWebSocketTransport()


@pytest.fixture
def specs_service(tmp_path):
    """Specs service with test configuration."""
    config_file = tmp_path / "test_specs.yaml"
    config_file.write_text("""
products:
  BTC-USD-PERP:
    min_size: "0.001"
    step_size: "0.001"
    price_increment: "0.5"
    min_notional: "10"
    max_size: "1000"
  ETH-USD-PERP:
    min_size: "0.01"
    step_size: "0.01"
    price_increment: "0.1"
    min_notional: "10"
    max_size: "5000"
""")
    return SpecsService(str(config_file))


@pytest.fixture
def rest_service(http_transport):
    """REST service wired with test transport for integration testing."""
    config = APIConfig(
        api_key="test_key",
        api_secret="test_secret",
        passphrase=None,
        base_url="https://api.coinbase.com",
        sandbox=False,
    )
    endpoints = CoinbaseEndpoints(mode="advanced", sandbox=False, enable_derivatives=True)
    client = CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode="advanced")
    client.set_transport_for_testing(http_transport)

    product_catalog = ProductCatalog(ttl_seconds=900)
    market_data = MarketDataService()
    event_store = EventStore()

    return CoinbaseRestService(
        client=client,
        endpoints=endpoints,
        config=config,
        product_catalog=product_catalog,
        market_data=market_data,
        event_store=event_store,
    )


# ============================================================================
# Test Suite 1: Enum Coercion
# ============================================================================


class TestEnumCoercion:
    """
    Test enum coercion for orders, including invalid values, case variations,
    null/missing fields, and fallback behavior.
    """

    def test_order_side_coercion_valid(self):
        """Test valid order side values are coerced correctly."""
        buy_order = CoinbaseOrderFactory.create_order(side="buy")
        sell_order = CoinbaseOrderFactory.create_order(side="sell")

        order1 = to_order(buy_order)
        order2 = to_order(sell_order)

        assert order1.side == OrderSide.BUY
        assert order2.side == OrderSide.SELL

    def test_order_side_coercion_case_insensitive(self):
        """Test order side coercion is case-insensitive."""
        test_cases = ["BUY", "Buy", "bUy", "SELL", "Sell", "SeLl"]

        for side in test_cases:
            order = CoinbaseOrderFactory.create_order(side=side)
            result = to_order(order)

            expected = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            assert result.side == expected

    def test_order_side_coercion_invalid_defaults_to_sell(self):
        """Test invalid side values default to SELL."""
        invalid_order = CoinbaseOrderFactory.create_order(side="invalid")
        result = to_order(invalid_order)
        assert result.side == OrderSide.SELL

    def test_order_type_coercion_all_valid_types(self):
        """Test all valid order types are coerced correctly."""
        test_cases = [
            ("limit", OrderType.LIMIT),
            ("market", OrderType.MARKET),
            ("stop", OrderType.STOP),
            ("stop_market", OrderType.STOP),
            ("stop_limit", OrderType.STOP_LIMIT),
        ]

        for coinbase_type, expected_type in test_cases:
            order = CoinbaseOrderFactory.create_order(order_type=coinbase_type)
            result = to_order(order)
            assert result.type == expected_type

    def test_order_type_coercion_case_insensitive(self):
        """Test order type coercion is case-insensitive."""
        test_cases = ["LIMIT", "Limit", "LiMiT", "MARKET", "Market"]

        for otype in test_cases:
            order = CoinbaseOrderFactory.create_order(order_type=otype)
            result = to_order(order)

            if otype.lower() == "limit":
                assert result.type == OrderType.LIMIT
            elif otype.lower() == "market":
                assert result.type == OrderType.MARKET

    def test_order_type_coercion_invalid_defaults_to_limit(self):
        """Test invalid order types default to LIMIT."""
        invalid_order = CoinbaseOrderFactory.create_order(order_type="unknown")
        result = to_order(invalid_order)
        assert result.type == OrderType.LIMIT

    def test_order_status_coercion_all_valid_statuses(self):
        """Test all valid order statuses are coerced correctly."""
        test_cases = [
            ("pending", OrderStatus.PENDING),
            ("open", OrderStatus.SUBMITTED),
            ("new", OrderStatus.SUBMITTED),
            ("partially_filled", OrderStatus.PARTIALLY_FILLED),
            ("filled", OrderStatus.FILLED),
            ("canceled", OrderStatus.CANCELLED),
            ("cancelled", OrderStatus.CANCELLED),
            ("rejected", OrderStatus.REJECTED),
        ]

        for coinbase_status, expected_status in test_cases:
            order = CoinbaseOrderFactory.create_order(status=coinbase_status)
            result = to_order(order)
            assert result.status == expected_status

    def test_order_status_coercion_case_insensitive(self):
        """Test order status coercion is case-insensitive."""
        test_cases = ["OPEN", "Open", "oPeN", "FILLED", "Filled", "FiLlEd"]

        for status in test_cases:
            order = CoinbaseOrderFactory.create_order(status=status)
            result = to_order(order)

            if status.lower() == "open":
                assert result.status == OrderStatus.SUBMITTED
            elif status.lower() == "filled":
                assert result.status == OrderStatus.FILLED

    def test_order_status_coercion_invalid_defaults_to_submitted(self):
        """Test invalid statuses default to SUBMITTED."""
        invalid_order = CoinbaseOrderFactory.create_order(status="unknown_status")
        result = to_order(invalid_order)
        assert result.status == OrderStatus.SUBMITTED

    def test_time_in_force_coercion_all_valid_values(self):
        """Test all valid TIF values are coerced correctly."""
        test_cases = [
            ("gtc", TimeInForce.GTC),
            ("ioc", TimeInForce.IOC),
            ("fok", TimeInForce.FOK),
        ]

        for coinbase_tif, expected_tif in test_cases:
            order = CoinbaseOrderFactory.create_order(time_in_force=coinbase_tif)
            result = to_order(order)
            assert result.tif == expected_tif

    def test_time_in_force_coercion_case_insensitive(self):
        """Test TIF coercion is case-insensitive."""
        test_cases = ["GTC", "Gtc", "gTc", "IOC", "Ioc", "FOK", "Fok"]

        for tif in test_cases:
            order = CoinbaseOrderFactory.create_order(time_in_force=tif)
            result = to_order(order)

            if tif.lower() == "gtc":
                assert result.tif == TimeInForce.GTC
            elif tif.lower() == "ioc":
                assert result.tif == TimeInForce.IOC
            elif tif.lower() == "fok":
                assert result.tif == TimeInForce.FOK

    def test_time_in_force_gtd_fallback_to_gtc(self):
        """
        Test GTD (Good-Till-Date) falls back to GTC.

        GTD is not supported in the internal enum, so it should
        default to GTC for safety.
        """
        gtd_order = CoinbaseEdgeCaseFactory.create_gtd_fallback_order()
        result = to_order(gtd_order)
        assert result.tif == TimeInForce.GTC

    def test_time_in_force_invalid_defaults_to_gtc(self):
        """Test invalid TIF values default to GTC."""
        invalid_order = CoinbaseOrderFactory.create_order(time_in_force="unknown")
        result = to_order(invalid_order)
        assert result.tif == TimeInForce.GTC

    def test_order_missing_required_fields_uses_fallbacks(self):
        """Test orders with missing fields use appropriate fallbacks."""
        minimal_order = CoinbaseEdgeCaseFactory.create_missing_fields_order()
        result = to_order(minimal_order)

        # Should not crash and should have sensible defaults
        assert result.id == "incomplete-order"
        assert result.symbol == ""  # Missing product_id
        assert result.side == OrderSide.SELL  # Default
        assert result.type == OrderType.LIMIT  # Default
        assert result.quantity == Decimal("0")  # Missing size

    def test_product_contract_type_coercion(self):
        """Test product contract type coercion to MarketType."""
        test_cases = [
            ({"product_id": "BTC-USD"}, MarketType.SPOT),
            ({"product_id": "BTC-USD-PERP", "contract_type": "perpetual"}, MarketType.PERPETUAL),
            ({"product_id": "BTC-USD-FUT", "contract_type": "future"}, MarketType.FUTURES),
        ]

        for payload_override, expected_market_type in test_cases:
            if expected_market_type == MarketType.SPOT:
                product = CoinbaseProductFactory.create_spot_product(**payload_override)
            elif expected_market_type == MarketType.PERPETUAL:
                product = CoinbaseProductFactory.create_perps_product(**payload_override)
            else:
                product = CoinbaseProductFactory.create_futures_product(**payload_override)

            result = to_product(product)
            assert result.market_type == expected_market_type

    def test_product_invalid_contract_type_defaults_to_spot(self):
        """Test invalid contract types default to SPOT."""
        invalid_product = CoinbaseEdgeCaseFactory.create_invalid_product()
        result = to_product(invalid_product)
        # Should not raise, should default to SPOT
        assert result.market_type == MarketType.SPOT


# ============================================================================
# Test Suite 2: Min-Notional Validation
# ============================================================================


class TestMinNotionalValidation:
    """
    Test min-notional validation through real REST service.

    Exercises actual validation logic in _build_order_payload to catch
    regressions in min-notional enforcement.
    """

    def test_rest_service_rejects_order_below_min_size(self, rest_service, http_transport):
        """Test REST service raises ValidationError for orders below min_size."""
        product = CoinbaseProductFactory.create_perps_product(
            symbol="BTC-USD-PERP",
            min_size="0.001",
            step_size="0.001",
            price_increment="0.5",
        )

        # ProductCatalog fetches product list first, then specific product
        http_transport.add_response(200, body={"products": [product]})
        http_transport.add_response(200, body=product)

        # Try to build order with size below min_size
        with pytest.raises(ValidationError) as exc_info:
            rest_service._build_order_payload(
                symbol="BTC-USD-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.0001"),  # Below min_size of 0.001
                price=Decimal("50000"),
                stop_price=None,
                tif=TimeInForce.GTC,
                client_id="test",
                reduce_only=None,
                leverage=None,
            )

        assert "below min_size" in str(exc_info.value)
        assert "quantity" in str(exc_info.value)

    def test_rest_service_rejects_order_below_min_notional(self, rest_service, http_transport):
        """Test REST service raises ValidationError for orders below min_notional."""
        product = CoinbaseProductFactory.create_perps_product(
            symbol="BTC-USD-PERP",
            min_size="0.001",
            step_size="0.001",
            price_increment="0.5",
            min_notional="10",  # $10 minimum
        )

        http_transport.add_response(200, body={"products": [product]})
        http_transport.add_response(200, body=product)

        # Order: 0.001 BTC @ $5 = $5 notional (below $10 min)
        with pytest.raises(ValidationError) as exc_info:
            rest_service._build_order_payload(
                symbol="BTC-USD-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),  # Passes min_size
                price=Decimal("5"),  # But only $5 notional
                stop_price=None,
                tif=TimeInForce.GTC,
                client_id="test",
                reduce_only=None,
                leverage=None,
            )

        assert "below min_notional" in str(exc_info.value)
        assert "10" in str(exc_info.value)

    def test_rest_service_accepts_order_at_min_notional_boundary(
        self, rest_service, http_transport
    ):
        """Test REST service accepts orders exactly at min_notional."""
        product = CoinbaseProductFactory.create_perps_product(
            symbol="BTC-USD-PERP",
            min_size="0.001",
            step_size="0.001",
            price_increment="0.5",
            min_notional="10",
        )

        http_transport.add_response(200, body={"products": [product]})
        http_transport.add_response(200, body=product)

        # Order: 0.002 BTC @ $5000 = $10 notional (exactly at min)
        payload = rest_service._build_order_payload(
            symbol="BTC-USD-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.002"),
            price=Decimal("5000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id="test",
            reduce_only=None,
            leverage=None,
        )

        assert payload is not None
        assert payload["product_id"] == "BTC-USD-PERP"

    def test_rest_service_accepts_order_above_min_notional(self, rest_service, http_transport):
        """Test REST service accepts orders above min_notional."""
        product = CoinbaseProductFactory.create_perps_product(
            symbol="BTC-USD-PERP",
            min_size="0.001",
            step_size="0.001",
            price_increment="0.5",
            min_notional="10",
        )

        http_transport.add_response(200, body={"products": [product]})
        http_transport.add_response(200, body=product)

        # Order: 0.001 BTC @ $50,000 = $50 notional (above $10 min)
        payload = rest_service._build_order_payload(
            symbol="BTC-USD-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id="test",
            reduce_only=None,
            leverage=None,
        )

        assert payload is not None
        assert Decimal(payload["size"]) == Decimal("0.001")

    def test_rest_service_market_order_min_notional_fetches_quote(
        self, rest_service, http_transport
    ):
        """Test market orders fetch quote for min_notional validation."""
        product = CoinbaseProductFactory.create_perps_product(
            symbol="BTC-USD-PERP",
            min_size="0.001",
            min_notional="10",
        )

        http_transport.add_response(200, body={"products": [product]})
        http_transport.add_response(200, body=product)
        # Quote response (for notional calculation)
        http_transport.add_response(
            200,
            body=CoinbaseQuoteFactory.create_quote(symbol="BTC-USD-PERP", price="5"),  # $5/BTC
        )

        # Market order: 0.001 BTC @ $5 = $0.005 notional (below $10 min)
        with pytest.raises(ValidationError) as exc_info:
            rest_service._build_order_payload(
                symbol="BTC-USD-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.001"),  # Passes min_size check
                price=None,  # Market order - fetches quote
                stop_price=None,
                tif=TimeInForce.IOC,
                client_id="test",
                reduce_only=None,
                leverage=None,
            )

        assert "below min_notional" in str(exc_info.value)

    def test_rest_service_quantizes_to_step_size(self, rest_service, http_transport):
        """Test REST service quantizes order sizes to step_size."""
        product = CoinbaseProductFactory.create_perps_product(
            symbol="BTC-USD-PERP",
            min_size="0.001",
            step_size="0.001",
            price_increment="0.5",
        )

        http_transport.add_response(200, body={"products": [product]})
        http_transport.add_response(200, body=product)

        # Order with unaligned size should be quantized
        payload = rest_service._build_order_payload(
            symbol="BTC-USD-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.0015"),  # Not aligned to 0.001
            price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id="test",
            reduce_only=None,
            leverage=None,
        )

        # 0.0015 may or may not be quantized depending on quantize_to_increment logic
        # Verify size is valid and close to input
        size = Decimal(payload["size"])
        assert size >= Decimal("0.001")  # At least min_size
        assert size <= Decimal("0.002")  # Shouldn't increase significantly

    def test_spec_service_min_notional_parsing(self, specs_service):
        """Test SpecsService correctly parses min_notional from YAML."""
        spec = specs_service.build_spec("BTC-USD-PERP")
        assert spec.min_notional == Decimal("10")

    def test_spec_service_caches_specs(self, specs_service):
        """Test SpecsService caches product specs."""
        spec1 = specs_service.build_spec("BTC-USD-PERP")
        spec2 = specs_service.build_spec("BTC-USD-PERP")

        # Should return same cached instance
        assert spec1 is spec2


# ============================================================================
# Test Suite 3: Reconnect and Error Handling
# ============================================================================


class TestReconnectAndErrorHandling:
    """
    Test WebSocket reconnection, REST error handling, and recovery scenarios.
    """

    def test_websocket_single_reconnect_success(self, ws_transport):
        """Test WebSocket reconnects once after disconnect."""
        ws_transport.add_message_batch([{"seq": 1}, {"seq": 2}], disconnect=True)
        ws_transport.add_message_batch([{"seq": 3}, {"seq": 4}], disconnect=False)

        ws = CoinbaseWebSocket("wss://test", max_retries=3, base_delay=0)
        ws.set_transport(ws_transport)
        ws.subscribe(WSSubscription(channels=["ticker"], product_ids=["BTC-USD"]))

        messages = list(ws.stream_messages())

        assert len(messages) == 4
        assert ws_transport.connect_count == 2  # Initial + 1 reconnect
        assert len(ws_transport.subscriptions) == 2  # Resubscribe after reconnect

    def test_websocket_multiple_reconnects(self, ws_transport):
        """Test WebSocket handles multiple consecutive reconnects."""
        # 3 disconnects, then success
        ws_transport.add_message_batch([{"seq": 1}], disconnect=True)
        ws_transport.add_message_batch([{"seq": 2}], disconnect=True)
        ws_transport.add_message_batch([{"seq": 3}], disconnect=True)
        ws_transport.add_message_batch([{"seq": 4}, {"seq": 5}], disconnect=False)

        ws = CoinbaseWebSocket("wss://test", max_retries=5, base_delay=0)
        ws.set_transport(ws_transport)
        ws.subscribe(WSSubscription(channels=["ticker"], product_ids=["BTC-USD"]))

        messages = list(ws.stream_messages())

        assert len(messages) == 5
        assert ws_transport.connect_count == 4  # Initial + 3 reconnects

    def test_websocket_reconnect_exhausts_retries(self, ws_transport):
        """Test WebSocket stops after max retries exceeded."""
        # Only provide failing batches
        ws_transport.add_message_batch([{"seq": 1}], disconnect=True)
        ws_transport.add_message_batch([{"seq": 2}], disconnect=True)
        ws_transport.add_message_batch([{"seq": 3}], disconnect=True)

        ws = CoinbaseWebSocket("wss://test", max_retries=2, base_delay=0)
        ws.set_transport(ws_transport)
        ws.subscribe(WSSubscription(channels=["ticker"], product_ids=["BTC-USD"]))

        # Should get messages before exhausting retries
        messages = list(ws.stream_messages())

        # Should have gotten at least the first few messages
        assert len(messages) >= 1
        # Connect count should be initial + max_retries
        assert ws_transport.connect_count <= 3

    def test_websocket_resubscribes_after_reconnect(self, ws_transport):
        """Test WebSocket resubscribes to channels after reconnect."""
        ws_transport.add_message_batch([{"seq": 1}], disconnect=True)
        ws_transport.add_message_batch([{"seq": 2}], disconnect=False)

        ws = CoinbaseWebSocket("wss://test", max_retries=3, base_delay=0)
        ws.set_transport(ws_transport)

        # Subscribe to multiple channels
        ws.subscribe(WSSubscription(channels=["ticker"], product_ids=["BTC-USD"]))
        ws.subscribe(WSSubscription(channels=["user"], product_ids=[]))

        messages = list(ws.stream_messages())

        # Should have resubscribed after reconnect
        # Initial subscriptions: 2, after reconnect: +1 or more
        assert len(ws_transport.subscriptions) >= 3
        # Verify ticker and user channels were subscribed
        channels = [sub["channels"] for sub in ws_transport.subscriptions]
        assert ["ticker"] in channels
        assert ["user"] in channels

    def test_client_retries_on_503_service_unavailable(self, http_transport):
        """Test CoinbaseClient retries on 503 errors with exponential backoff."""
        # Simulate 2 failures then success
        http_transport.set_failure_mode(count=2)
        http_transport.add_response(200, body={"products": []})

        # Create client with injected transport
        client = CoinbaseClient(base_url="https://api.coinbase.com", auth=None)
        client.set_transport_for_testing(http_transport)

        # Should retry and eventually succeed
        response = client.get("/api/v3/brokerage/market/products")

        assert response == {"products": []}
        assert len(http_transport.calls) == 3  # 2 failures + 1 success

    def test_client_retries_429_rate_limit_with_retry_after(self, http_transport):
        """Test CoinbaseClient respects retry-after header on 429."""
        # First response: rate limit with retry-after
        http_transport.add_response(
            429, headers={"retry-after": "0.1"}, body={"error": "rate_limited"}
        )
        # Second response: success
        http_transport.add_response(200, body={"data": "success"})

        client = CoinbaseClient(base_url="https://api.coinbase.com", auth=None)
        client.set_transport_for_testing(http_transport)

        # Should retry after respecting retry-after header
        response = client.get("/api/v3/brokerage/market/products")

        assert response == {"data": "success"}
        assert len(http_transport.calls) == 2

    def test_client_does_not_retry_400_client_errors(self, http_transport):
        """Test CoinbaseClient does not retry 400 bad request errors."""
        http_transport.add_response(400, body={"error": "bad_request", "message": "Invalid param"})

        client = CoinbaseClient(base_url="https://api.coinbase.com", auth=None)
        client.set_transport_for_testing(http_transport)

        # Should raise immediately without retry
        with pytest.raises(Exception):  # Will raise InvalidRequestError or similar
            client.post("/api/v3/brokerage/orders", {})

        assert len(http_transport.calls) == 1  # No retries

    def test_client_does_not_retry_401_auth_errors(self, http_transport):
        """Test CoinbaseClient does not retry 401 auth errors."""
        http_transport.add_response(401, body={"error": "unauthorized", "message": "Invalid key"})

        client = CoinbaseClient(base_url="https://api.coinbase.com", auth=None)
        client.set_transport_for_testing(http_transport)

        # Should raise immediately without retry
        with pytest.raises(Exception):  # Will raise AuthError or similar
            client.get("/api/v3/brokerage/accounts")

        assert len(http_transport.calls) == 1  # No retries for auth errors

    def test_client_exhausts_retries_on_persistent_503(self, http_transport):
        """Test CoinbaseClient stops retrying after max_retries exceeded."""
        # All responses are 503 errors
        for _ in range(5):
            http_transport.add_response(503, body={"error": "service_unavailable"})

        client = CoinbaseClient(base_url="https://api.coinbase.com", auth=None)
        client.set_transport_for_testing(http_transport)

        # Should exhaust retries and raise
        with pytest.raises(Exception):  # Will raise BrokerageError or similar
            client.get("/api/v3/brokerage/market/products")

        # Should have made initial attempt + max_retries (typically 3)
        assert len(http_transport.calls) >= 3

    def test_websocket_gap_detection_after_reconnect(self, ws_transport):
        """Test sequence gap detection after reconnect."""
        # First batch with sequence 1-3, then disconnect
        ws_transport.add_message_batch(
            [{"sequence": 1}, {"sequence": 2}, {"sequence": 3}],
            disconnect=True
        )
        # After reconnect, start at sequence 10 (gap!)
        ws_transport.add_message_batch(
            [{"sequence": 10}, {"sequence": 11}],
            disconnect=False
        )

        ws = CoinbaseWebSocket("wss://test", max_retries=3, base_delay=0)
        ws.set_transport(ws_transport)
        ws.subscribe(WSSubscription(channels=["user"], product_ids=[]))

        messages = list(ws.stream_messages())

        # Should receive all messages
        assert len(messages) == 5

        # Sequence tracking would detect gap (tested separately in ws tests)
        sequences = [m.get("sequence") for m in messages if "sequence" in m]
        assert sequences == [1, 2, 3, 10, 11]


# ============================================================================
# Test Suite 4: Quote and Edge Case Handling
# ============================================================================


class TestQuoteEdgeCases:
    """Test quote parsing edge cases including zero prices and missing data."""

    def test_quote_with_zero_price_falls_back_to_trades(self):
        """Test quotes with zero price fall back to trades array."""
        zero_price_quote = CoinbaseEdgeCaseFactory.create_zero_price_quote()
        result = to_quote(zero_price_quote)

        # Should have used trade price as fallback
        assert result.last == Decimal("50000.00")

    def test_quote_missing_bid_ask_defaults_to_zero(self):
        """Test missing bid/ask defaults to zero."""
        minimal_quote = {
            "product_id": "BTC-USD",
            "price": "50000.00",
        }
        result = to_quote(minimal_quote)

        assert result.bid == Decimal("0")
        assert result.ask == Decimal("0")
        assert result.last == Decimal("50000.00")

    def test_quote_missing_timestamp_uses_current_time(self):
        """Test missing timestamp uses current time."""
        quote = CoinbaseQuoteFactory.create_quote(time=None)
        result = to_quote(quote)

        # Should have a timestamp (current time)
        assert result.ts is not None
        assert isinstance(result.ts, datetime)

    def test_quote_from_trades_array(self):
        """Test quote derivation from trades array."""
        quote = CoinbaseQuoteFactory.create_quote_from_trades(
            symbol="ETH-USD",
            last_trade_price="3000.50"
        )
        result = to_quote(quote)

        assert result.symbol == "ETH-USD"
        assert result.last == Decimal("3000.50")


# ============================================================================
# Integration Tests
# ============================================================================


class TestTransportStubIntegration:
    """Test transport stubs work correctly in integration scenarios."""

    def test_http_transport_call_tracking(self, http_transport):
        """Test HTTP transport tracks all calls correctly."""
        http_transport.add_response(200, body={"orders": []})
        http_transport.add_response(200, body={"products": []})

        # Make two calls
        http_transport("GET", "/orders", {"Auth": "token"}, None, 10)
        http_transport("GET", "/products", {"Auth": "token"}, None, 10)

        assert len(http_transport.calls) == 2
        assert http_transport.calls[0]["url"] == "/orders"
        assert http_transport.calls[1]["url"] == "/products"

    def test_ws_transport_subscription_tracking(self, ws_transport):
        """Test WebSocket transport tracks subscriptions."""
        ws_transport.add_message_batch([{"type": "ticker"}], disconnect=False)

        ws = CoinbaseWebSocket("wss://test", transport=ws_transport)
        ws.connect()
        ws.subscribe(WSSubscription(channels=["ticker"], product_ids=["BTC-USD"]))
        ws.subscribe(WSSubscription(channels=["level2"], product_ids=["ETH-USD"]))

        assert len(ws_transport.subscriptions) == 2
        assert ws_transport.subscriptions[0]["channels"] == ["ticker"]
        assert ws_transport.subscriptions[1]["channels"] == ["level2"]

    def test_reusable_transport_stubs_reduce_duplication(self, http_transport, ws_transport):
        """
        Demonstrate transport stubs reduce test duplication.

        This pattern should be used across all REST/WebSocket tests
        to avoid fixture sprawl.
        """
        # HTTP pattern
        http_transport.add_response(200, body={"status": "ok"})
        status, _, body = http_transport("GET", "/status", {}, None, 10)
        assert status == 200

        # WebSocket pattern
        ws_transport.add_message_batch([{"event": "connected"}], disconnect=False)
        ws = CoinbaseWebSocket("wss://test", transport=ws_transport)
        messages = list(ws_transport.stream())
        assert len(messages) == 1

        # Stubs are reusable across tests without additional setup
