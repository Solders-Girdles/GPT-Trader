from __future__ import annotations

from decimal import Decimal

import pytest

from tests.unit.gpt_trader.features.brokerages.coinbase.rest.order_service_test_base import (
    OrderServiceTestBase,
    assert_cancel_order_raises_for_rejection_or_missing_order,
    assert_cancel_order_re_raises_brokerage_error,
    assert_cancel_order_success,
    assert_cancel_order_wraps_unexpected_exception,
    assert_close_position_fallback_on_exception,
    assert_close_position_finds_correct_symbol_among_multiple,
    assert_close_position_no_fallback_raises,
    assert_close_position_no_open_position_raises,
    assert_close_position_success,
    assert_close_position_with_client_order_id,
    assert_get_order_handles_exception,
    assert_get_order_not_found_error_returns_none,
    assert_get_order_re_raises_brokerage_error,
    assert_get_order_returns_none_when_order_missing,
    assert_get_order_returns_order,
    assert_list_fills_empty_response_returns_empty_list,
    assert_list_fills_handles_exception_logs_and_raises,
    assert_list_fills_pagination,
    assert_list_fills_passes_filters,
    assert_list_fills_re_raises_brokerage_error,
    assert_list_fills_returns_fills,
    assert_list_orders_empty_response_returns_empty_list,
    assert_list_orders_handles_exception,
    assert_list_orders_pagination,
    assert_list_orders_passes_filters_and_limit,
    assert_list_orders_re_raises_brokerage_error,
    assert_list_orders_returns_orders,
    assert_place_order_builds_payload,
    assert_place_order_executes_payload,
    assert_place_order_market_order_includes_price_none,
    assert_place_order_passes_all_parameters,
    mock_position,
)


class TestPlaceOrder(OrderServiceTestBase):
    def test_place_order_builds_payload(self, order_service, mock_payload_builder) -> None:
        assert_place_order_builds_payload(order_service, mock_payload_builder)

    def test_place_order_executes_payload(
        self, order_service, mock_payload_builder, mock_payload_executor
    ) -> None:
        assert_place_order_executes_payload(
            order_service, mock_payload_builder, mock_payload_executor
        )

    def test_place_order_passes_all_parameters(self, order_service, mock_payload_builder) -> None:
        assert_place_order_passes_all_parameters(order_service, mock_payload_builder)

    def test_place_order_market_order_includes_price_none(
        self, order_service, mock_payload_builder
    ) -> None:
        assert_place_order_market_order_includes_price_none(order_service, mock_payload_builder)


class TestCancelOrder(OrderServiceTestBase):
    def test_cancel_order_success(self, order_service, mock_client) -> None:
        assert_cancel_order_success(order_service, mock_client)

    @pytest.mark.parametrize(
        ("response", "match"),
        [
            (
                {"results": [{"order_id": "order-123", "success": False}]},
                "Cancellation rejected",
            ),
            (
                {"results": [{"order_id": "other-order", "success": True}]},
                "not found in cancellation response",
            ),
            ({"results": []}, "not found in cancellation response"),
            ({}, "not found in cancellation response"),
        ],
    )
    def test_cancel_order_raises_for_rejection_or_missing_order(
        self, order_service, mock_client, response, match
    ) -> None:
        assert_cancel_order_raises_for_rejection_or_missing_order(
            order_service, mock_client, response, match
        )

    def test_cancel_order_re_raises_brokerage_error(self, order_service, mock_client) -> None:
        assert_cancel_order_re_raises_brokerage_error(order_service, mock_client)

    def test_cancel_order_wraps_unexpected_exception(self, order_service, mock_client) -> None:
        assert_cancel_order_wraps_unexpected_exception(order_service, mock_client)


class TestListOrders(OrderServiceTestBase):
    def test_list_orders_returns_orders(
        self, order_service, mock_client, sample_order_response
    ) -> None:
        assert_list_orders_returns_orders(order_service, mock_client, sample_order_response)

    def test_list_orders_empty_response_returns_empty_list(
        self, order_service, mock_client
    ) -> None:
        assert_list_orders_empty_response_returns_empty_list(order_service, mock_client)

    @pytest.mark.parametrize(
        ("kwargs", "expected_call_kwargs"),
        [
            ({"product_id": "BTC-USD"}, {"product_id": "BTC-USD"}),
            ({"status": ["PENDING", "OPEN"]}, {"order_status": ["PENDING", "OPEN"]}),
            ({"limit": 50}, {"limit": 50}),
        ],
    )
    def test_list_orders_passes_filters_and_limit(
        self, order_service, mock_client, kwargs, expected_call_kwargs
    ) -> None:
        assert_list_orders_passes_filters_and_limit(
            order_service, mock_client, kwargs, expected_call_kwargs
        )

    def test_list_orders_pagination(
        self, order_service, mock_client, sample_order_response
    ) -> None:
        assert_list_orders_pagination(order_service, mock_client, sample_order_response)

    def test_list_orders_handles_exception(self, order_service, mock_client) -> None:
        assert_list_orders_handles_exception(order_service, mock_client)

    def test_list_orders_re_raises_brokerage_error(self, order_service, mock_client) -> None:
        assert_list_orders_re_raises_brokerage_error(order_service, mock_client)


class TestGetOrder(OrderServiceTestBase):
    def test_get_order_returns_order(
        self, order_service, mock_client, sample_order_response
    ) -> None:
        assert_get_order_returns_order(order_service, mock_client, sample_order_response)

    def test_get_order_returns_none_when_order_missing(self, order_service, mock_client) -> None:
        assert_get_order_returns_none_when_order_missing(order_service, mock_client)

    def test_get_order_not_found_error_returns_none(self, order_service, mock_client) -> None:
        assert_get_order_not_found_error_returns_none(order_service, mock_client)

    def test_get_order_handles_exception(self, order_service, mock_client) -> None:
        assert_get_order_handles_exception(order_service, mock_client)

    def test_get_order_re_raises_brokerage_error(self, order_service, mock_client) -> None:
        assert_get_order_re_raises_brokerage_error(order_service, mock_client)


class TestListFills(OrderServiceTestBase):
    def test_list_fills_returns_fills(self, order_service, mock_client) -> None:
        assert_list_fills_returns_fills(order_service, mock_client)

    def test_list_fills_empty_response_returns_empty_list(self, order_service, mock_client) -> None:
        assert_list_fills_empty_response_returns_empty_list(order_service, mock_client)

    @pytest.mark.parametrize(
        ("kwargs", "expected_call_kwargs"),
        [
            ({"product_id": "ETH-USD"}, {"product_id": "ETH-USD"}),
            ({"order_id": "order-456"}, {"order_id": "order-456"}),
        ],
    )
    def test_list_fills_passes_filters(
        self, order_service, mock_client, kwargs, expected_call_kwargs
    ) -> None:
        assert_list_fills_passes_filters(order_service, mock_client, kwargs, expected_call_kwargs)

    def test_list_fills_pagination(self, order_service, mock_client) -> None:
        assert_list_fills_pagination(order_service, mock_client)

    def test_list_fills_handles_exception_logs_and_raises(
        self, order_service, mock_client, monkeypatch
    ) -> None:
        assert_list_fills_handles_exception_logs_and_raises(order_service, mock_client, monkeypatch)

    def test_list_fills_re_raises_brokerage_error(self, order_service, mock_client) -> None:
        assert_list_fills_re_raises_brokerage_error(order_service, mock_client)


class TestClosePosition(OrderServiceTestBase):
    def test_close_position_success(
        self, order_service, mock_client, mock_position_provider, sample_order_response
    ) -> None:
        assert_close_position_success(
            order_service, mock_client, mock_position_provider, sample_order_response
        )

    @pytest.mark.parametrize(
        "positions",
        [
            [],
            [mock_position("BTC-PERP", Decimal("0"))],
            [mock_position("ETH-PERP", Decimal("1.0"))],
        ],
    )
    def test_close_position_no_open_position_raises(
        self, order_service, mock_position_provider, positions
    ) -> None:
        assert_close_position_no_open_position_raises(
            order_service, mock_position_provider, positions
        )

    def test_close_position_with_client_order_id(
        self, order_service, mock_client, mock_position_provider, sample_order_response
    ) -> None:
        assert_close_position_with_client_order_id(
            order_service, mock_client, mock_position_provider, sample_order_response
        )

    def test_close_position_fallback_on_exception(
        self, order_service, mock_client, mock_position_provider
    ) -> None:
        assert_close_position_fallback_on_exception(
            order_service, mock_client, mock_position_provider
        )

    def test_close_position_no_fallback_raises(
        self, order_service, mock_client, mock_position_provider
    ) -> None:
        assert_close_position_no_fallback_raises(order_service, mock_client, mock_position_provider)

    def test_close_position_finds_correct_symbol_among_multiple(
        self, order_service, mock_client, mock_position_provider, sample_order_response
    ) -> None:
        assert_close_position_finds_correct_symbol_among_multiple(
            order_service, mock_client, mock_position_provider, sample_order_response
        )
