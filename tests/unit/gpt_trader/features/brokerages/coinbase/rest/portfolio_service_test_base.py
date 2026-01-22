"""Shared fixtures for `PortfolioService` tests."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.core import Balance, InvalidRequestError
from gpt_trader.core import Position as CorePosition
from gpt_trader.core.account import CFMBalance
from gpt_trader.core.account import Position as AccountPosition
from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.rest.portfolio_service import PortfolioService
from gpt_trader.persistence.event_store import EventStore


class PortfolioServiceTestBase:
    @pytest.fixture
    def mock_client(self) -> Mock:
        """Create a mock Coinbase REST client."""
        return Mock()  # Don't use spec to allow dynamic method mocking

    @pytest.fixture
    def mock_endpoints(self) -> Mock:
        """Create a mock CoinbaseEndpoints."""
        return Mock(spec=CoinbaseEndpoints)

    @pytest.fixture
    def mock_event_store(self) -> Mock:
        """Create a mock EventStore."""
        return Mock(spec=EventStore)

    @pytest.fixture
    def portfolio_service(
        self,
        mock_client: Mock,
        mock_endpoints: Mock,
        mock_event_store: Mock,
    ) -> PortfolioService:
        """Create a PortfolioService instance with mocked dependencies."""
        return PortfolioService(
            client=mock_client,
            endpoints=mock_endpoints,
            event_store=mock_event_store,
        )


def assert_service_init(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
    mock_event_store: Mock,
) -> None:
    assert portfolio_service._client == mock_client
    assert portfolio_service._endpoints == mock_endpoints
    assert portfolio_service._event_store == mock_event_store


def assert_list_balances_returns_balances(
    portfolio_service: PortfolioService,
    mock_client: Mock,
) -> None:
    mock_client.get_accounts.return_value = {
        "accounts": [
            {
                "uuid": "acc_1",
                "currency": "BTC",
                "available_balance": {"value": "1.5"},
                "hold": {"value": "0.1"},
                "balance": {"value": "1.6"},
            },
            {
                "uuid": "acc_2",
                "currency": "USD",
                "available_balance": {"value": "10000.00"},
                "hold": {"value": "500.00"},
                "balance": {"value": "10500.00"},
            },
        ]
    }
    result = portfolio_service.list_balances()
    assert len(result) == 2
    assert result[0].asset == "BTC"
    assert result[0].available == Decimal("1.5")
    assert result[0].hold == Decimal("0.1")
    assert result[0].total == Decimal("1.6")


def assert_list_balances_handles_list_response(
    portfolio_service: PortfolioService,
    mock_client: Mock,
) -> None:
    mock_client.get_accounts.return_value = [
        {
            "uuid": "acc_1",
            "currency": "BTC",
            "available": "1.5",
            "hold": "0.1",
            "balance": "1.6",
        }
    ]
    result = portfolio_service.list_balances()
    assert len(result) == 1
    assert result[0].asset == "BTC"


def assert_list_balances_calculates_total_when_missing(
    portfolio_service: PortfolioService,
    mock_client: Mock,
) -> None:
    mock_client.get_accounts.return_value = {
        "accounts": [
            {
                "uuid": "acc_1",
                "currency": "ETH",
                "available_balance": {"value": "5.0"},
                "hold": {"value": "1.0"},
            }
        ]
    }
    result = portfolio_service.list_balances()
    assert result[0].total == Decimal("6.0")


def assert_list_balances_handles_exception(
    portfolio_service: PortfolioService,
    mock_client: Mock,
) -> None:
    mock_client.get_accounts.side_effect = Exception("API error")
    result = portfolio_service.list_balances()
    assert result == []


def assert_list_balances_skips_invalid_entries(
    portfolio_service: PortfolioService,
    mock_client: Mock,
) -> None:
    mock_client.get_accounts.return_value = {
        "accounts": [
            {
                "uuid": "acc_1",
                "currency": "BTC",
                "available_balance": {"value": "1.5"},
                "hold": {"value": "0.1"},
                "balance": {"value": "1.6"},
            },
            {"uuid": "acc_2", "currency": None},
        ]
    }
    result = portfolio_service.list_balances()
    assert len(result) >= 1


def assert_get_portfolio_balances_delegates_to_list_balances(
    portfolio_service: PortfolioService,
    mock_client: Mock,
) -> None:
    mock_client.get_accounts.return_value = {"accounts": []}
    result = portfolio_service.get_portfolio_balances()
    assert result == []
    mock_client.get_accounts.assert_called_once()


def assert_list_positions_returns_positions(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = True
    mock_client.list_positions.return_value = [
        CorePosition(
            symbol="BTC-PERP",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("100.00"),
            mark_price=Decimal("51000.00"),
            realized_pnl=Decimal("0.00"),
            side="LONG",
        )
    ]
    result = portfolio_service.list_positions()
    assert len(result) == 1
    assert result[0].symbol == "BTC-PERP"


def assert_list_positions_returns_empty_when_derivatives_not_supported(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = False
    result = portfolio_service.list_positions()
    assert result == []
    mock_client.list_positions.assert_not_called()


def assert_list_positions_handles_exception(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = True
    mock_client.list_positions.side_effect = Exception("API error")
    result = portfolio_service.list_positions()
    assert result == []


def assert_get_position_returns_position(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = True
    mock_client.get_cfm_position.return_value = {
        "product_id": "BTC-PERP",
        "side": "LONG",
        "contracts": "0.5",
        "entry_price": "50000.00",
        "unrealized_pnl": "100.00",
        "realized_pnl": "100.00",
    }
    result = portfolio_service.get_position("BTC-PERP")
    assert result is not None
    assert result.symbol == "BTC-PERP"


def assert_get_position_returns_none_when_not_supported(
    portfolio_service: PortfolioService,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = False
    result = portfolio_service.get_position("BTC-PERP")
    assert result is None


def assert_get_cfm_balance_summary_returns_summary(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
    mock_event_store: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = True
    mock_client.cfm_balance_summary.return_value = {
        "balance_summary": {"total_balance": "50000.00", "available_balance": "45000.00"}
    }
    result = portfolio_service.get_cfm_balance_summary()
    assert result["total_balance"] == Decimal("50000.00")
    mock_event_store.append_metric.assert_called_once()


def assert_get_cfm_balance_summary_returns_empty_when_not_supported(
    portfolio_service: PortfolioService,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = False
    result = portfolio_service.get_cfm_balance_summary()
    assert result == {}


def assert_get_cfm_balance_summary_normalises_decimals_and_emits_metric(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
    mock_event_store: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = True
    mock_client.cfm_balance_summary.return_value = {
        "balance_summary": {
            "portfolio_value": "100.50",
            "available_margin": "25.75",
            "timestamp": "2024-05-01T00:00:00Z",
        }
    }
    summary = portfolio_service.get_cfm_balance_summary()
    assert summary["portfolio_value"] == Decimal("100.50")
    assert summary["available_margin"] == Decimal("25.75")
    mock_event_store.append_metric.assert_called_once()
    metrics_payload = mock_event_store.append_metric.call_args.kwargs["metrics"]
    assert metrics_payload["event_type"] == "cfm_balance_summary"


def assert_list_cfm_sweeps_returns_sweeps(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = True
    mock_client.cfm_sweeps.return_value = {
        "sweeps": [
            {"sweep_id": "sweep_1", "amount": "100.00"},
            {"sweep_id": "sweep_2", "amount": "200.00"},
        ]
    }
    result = portfolio_service.list_cfm_sweeps()
    assert len(result) == 2
    assert result[0]["amount"] == Decimal("100.00")


def assert_list_cfm_sweeps_returns_empty_when_derivatives_disabled(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = False
    sweeps = portfolio_service.list_cfm_sweeps()
    assert sweeps == []
    mock_client.cfm_sweeps.assert_not_called()


def assert_list_cfm_sweeps_normalises_entries(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
    mock_event_store: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = True
    mock_client.cfm_sweeps.return_value = {
        "sweeps": [
            {"sweep_id": "sw1", "amount": "10.0"},
            {"sweep_id": "sw2", "amount": "5.5"},
        ]
    }
    sweeps = portfolio_service.list_cfm_sweeps()
    assert sweeps[0]["amount"] == Decimal("10.0")
    assert sweeps[1]["amount"] == Decimal("5.5")
    assert mock_event_store.append_metric.called


def assert_get_cfm_sweeps_schedule_returns_schedule(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = True
    mock_client.cfm_sweeps_schedule.return_value = {
        "schedule": {"frequency": "daily", "time": "00:00"}
    }
    result = portfolio_service.get_cfm_sweeps_schedule()
    assert result["frequency"] == "daily"


def assert_get_cfm_margin_window_returns_window(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = True
    mock_client.cfm_intraday_current_margin_window.return_value = {
        "margin_window": "INTRADAY",
        "leverage": "10",
    }
    result = portfolio_service.get_cfm_margin_window()
    assert result["margin_window"] == "INTRADAY"


def assert_get_cfm_margin_window_handles_errors(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = True
    mock_client.cfm_intraday_current_margin_window.side_effect = RuntimeError("boom")
    result = portfolio_service.get_cfm_margin_window()
    assert result == {}


def assert_update_cfm_margin_window_success(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
    mock_event_store: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = True
    mock_client.cfm_intraday_margin_setting.return_value = {
        "margin_window": "OVERNIGHT",
        "leverage": "5",
    }
    result = portfolio_service.update_cfm_margin_window("OVERNIGHT")
    assert result["leverage"] == Decimal("5")
    mock_event_store.append_metric.assert_called_once()


def assert_update_cfm_margin_window_raises_when_not_supported(
    portfolio_service: PortfolioService,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = False
    with pytest.raises(InvalidRequestError, match="Derivatives not supported"):
        portfolio_service.update_cfm_margin_window("OVERNIGHT")


def assert_update_cfm_margin_window_enforces_derivatives(
    portfolio_service: PortfolioService,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = False
    with pytest.raises(InvalidRequestError):
        portfolio_service.update_cfm_margin_window("INTRADAY_STANDARD")


def assert_update_cfm_margin_window_calls_client_and_emits(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
    mock_event_store: Mock,
) -> None:
    mock_endpoints.supports_derivatives.return_value = True
    mock_client.cfm_intraday_margin_setting.return_value = {
        "status": "accepted",
        "leverage": "3",
    }
    response = portfolio_service.update_cfm_margin_window(
        "INTRADAY_STANDARD", effective_time="2024-05-01T00:00:00Z"
    )
    mock_client.cfm_intraday_margin_setting.assert_called_once()
    payload = mock_client.cfm_intraday_margin_setting.call_args.args[0]
    assert payload["margin_window"] == "INTRADAY_STANDARD"
    assert payload["effective_time"] == "2024-05-01T00:00:00Z"
    assert response["leverage"] == Decimal("3")
    metrics_payload = mock_event_store.append_metric.call_args.kwargs["metrics"]
    assert metrics_payload["event_type"] == "cfm_margin_setting"


def assert_intx_allocate_requires_advanced_mode(
    portfolio_service: PortfolioService,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.mode = "exchange"
    with pytest.raises(InvalidRequestError, match="advanced mode"):
        portfolio_service.intx_allocate({"amount": "1000"})


def assert_intx_allocate_success(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
    mock_event_store: Mock,
) -> None:
    mock_endpoints.mode = "advanced"
    mock_client.intx_allocate.return_value = {
        "allocated_amount": "1000.00",
        "source_amount": "1000.00",
    }
    result = portfolio_service.intx_allocate({"amount": "1000"})
    assert result["allocated_amount"] == Decimal("1000.00")
    mock_event_store.append_metric.assert_called_once()


def assert_intx_allocate_normalises_and_emits_metric(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
    mock_event_store: Mock,
) -> None:
    mock_endpoints.mode = "advanced"
    mock_client.intx_allocate.return_value = {"status": "ok", "allocated_amount": "10.5"}
    response = portfolio_service.intx_allocate({"allocated_amount": "10.5"})
    assert response["allocated_amount"] == Decimal("10.5")
    mock_event_store.append_metric.assert_called_once()
    metrics_payload = mock_event_store.append_metric.call_args.kwargs["metrics"]
    assert metrics_payload["event_type"] == "intx_allocation"


def assert_get_intx_balances_returns_empty_when_not_advanced(
    portfolio_service: PortfolioService,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.mode = "exchange"
    result = portfolio_service.get_intx_balances("portfolio_123")
    assert result == []


def assert_get_intx_balances_returns_balances(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.mode = "advanced"
    mock_client.get_intx_portfolio.return_value = {
        "balances": [
            {"currency": "BTC", "amount": "1.5", "hold": "0.1"},
            {"currency": "USD", "amount": "10000.00", "hold": "0"},
        ]
    }
    result = portfolio_service.get_intx_balances("portfolio_123")
    assert len(result) == 2
    assert result[0]["amount"] == Decimal("1.5")


def assert_get_intx_balances_normalises_entries(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
    mock_event_store: Mock,
) -> None:
    mock_endpoints.mode = "advanced"
    mock_client.get_intx_portfolio.return_value = {
        "balances": [
            {"asset": "USD", "amount": "100.5", "hold": "0"},
            {"asset": "BTC", "amount": "0.25", "hold": "0"},
        ]
    }
    balances = portfolio_service.get_intx_balances("pf-1")
    assert balances[0]["amount"] == Decimal("100.5")
    assert balances[1]["amount"] == Decimal("0.25")
    assert mock_event_store.append_metric.called


def assert_get_intx_balances_handles_errors(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
    mock_event_store: Mock,
) -> None:
    mock_endpoints.mode = "advanced"
    mock_client.get_intx_portfolio.side_effect = RuntimeError("boom")
    balances = portfolio_service.get_intx_balances("pf-1")
    assert balances == []
    mock_event_store.append_metric.assert_not_called()


def assert_get_intx_portfolio_returns_empty_when_not_advanced(
    portfolio_service: PortfolioService,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.mode = "exchange"
    result = portfolio_service.get_intx_portfolio("portfolio_123")
    assert result == {}


def assert_get_intx_portfolio_success(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.mode = "advanced"
    mock_client.get_intx_portfolio.return_value = {
        "portfolio_id": "portfolio_123",
        "portfolio_value": "50000.00",
    }
    result = portfolio_service.get_intx_portfolio("portfolio_123")
    assert result["portfolio_value"] == Decimal("50000.00")


def assert_get_intx_portfolio_returns_normalised_dict(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.mode = "advanced"
    mock_client.get_intx_portfolio.return_value = {"portfolio_value": "2500.75"}
    portfolio = portfolio_service.get_intx_portfolio("pf-1")
    assert portfolio["portfolio_value"] == Decimal("2500.75")


def assert_list_intx_positions_returns_positions(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.mode = "advanced"
    mock_client.list_intx_positions.return_value = {
        "positions": [
            {
                "product_id": "BTC-PERP",
                "side": "LONG",
                "number_of_contracts": "0.5",
                "entry_vwap": {"value": "50000.00"},
            }
        ]
    }
    result = portfolio_service.list_intx_positions("portfolio_123")
    assert len(result) == 1


def assert_list_intx_positions_returns_normalised_list(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.mode = "advanced"
    mock_client.list_intx_positions.return_value = {
        "positions": [
            {
                "product_id": "BTC-USD",
                "quantity": "1.5",
                "side": "long",
                "entry_price": "100",
                "mark_price": "101",
                "unrealized_pnl": "1",
                "realized_pnl": "1",
            }
        ]
    }
    positions = portfolio_service.list_intx_positions("pf-1")
    assert positions[0].quantity == Decimal("1.5")


def assert_get_intx_position_handles_missing(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
) -> None:
    mock_endpoints.mode = "advanced"
    mock_client.get_intx_position.side_effect = RuntimeError("no position")
    position = portfolio_service.get_intx_position("pf-1", "BTC-USD")
    assert position is None


def assert_get_intx_multi_asset_collateral_emits_metric(
    portfolio_service: PortfolioService,
    mock_client: Mock,
    mock_endpoints: Mock,
    mock_event_store: Mock,
) -> None:
    mock_endpoints.mode = "advanced"
    mock_client.get_intx_multi_asset_collateral.return_value = {"total_usd_value": "5000.25"}
    collateral = portfolio_service.get_intx_multi_asset_collateral()
    assert collateral["total_usd_value"] == Decimal("5000.25")
    assert mock_event_store.append_metric.called
    metrics_payload = mock_event_store.append_metric.call_args.kwargs["metrics"]
    assert metrics_payload["event_type"] == "intx_multi_asset_collateral"


def _make_edge_service() -> tuple[PortfolioService, Mock, Mock]:
    client = Mock()
    endpoints = Mock(spec=CoinbaseEndpoints)
    event_store = Mock(spec=EventStore)
    service = PortfolioService(client=client, endpoints=endpoints, event_store=event_store)
    return service, client, endpoints


def assert_list_cfm_positions_invalid_expiry_sets_none() -> None:
    service, client, endpoints = _make_edge_service()
    endpoints.supports_derivatives.return_value = True
    client.cfm_positions.return_value = {
        "positions": [
            {
                "product_id": "BTC-2025",
                "number_of_contracts": "1",
                "avg_entry_price": "100",
                "current_price": "110",
                "unrealized_pnl": "5",
                "daily_realized_pnl": "1",
                "side": "LONG",
                "expiration_time": "not-a-date",
            }
        ]
    }
    positions = service.list_cfm_positions()
    assert len(positions) == 1
    assert positions[0].contract_expiry is None


def assert_list_spot_positions_skips_usd_and_zero() -> None:
    service, _client, _endpoints = _make_edge_service()
    service.list_balances = Mock(
        return_value=[
            Balance(asset="USD", total=Decimal("100"), available=Decimal("80")),
            Balance(asset="BTC", total=Decimal("0"), available=Decimal("0")),
            Balance(asset="ETH", total=Decimal("2"), available=Decimal("2")),
        ]
    )
    positions = service.list_spot_positions_as_core()
    assert len(positions) == 1
    assert positions[0].symbol == "ETH-USD"
    assert positions[0].quantity == Decimal("2")


def assert_get_cfm_balance_missing_or_empty_summary() -> None:
    service, client, endpoints = _make_edge_service()
    endpoints.supports_derivatives.return_value = True
    client.cfm_balance_summary.return_value = {}
    assert service.get_cfm_balance() is None
    client.cfm_balance_summary.return_value = {"balance_summary": {}}
    assert service.get_cfm_balance() is None


def assert_get_cfm_balance_parses_nested_values() -> None:
    service, client, endpoints = _make_edge_service()
    endpoints.supports_derivatives.return_value = True
    client.cfm_balance_summary.return_value = {
        "balance_summary": {
            "futures_buying_power": {"value": "100"},
            "total_usd_balance": {"value": "200"},
            "available_margin": {"value": "50"},
            "initial_margin": {"value": "25"},
            "unrealized_pnl": {"value": "10"},
            "daily_realized_pnl": {"value": "5"},
            "liquidation_threshold": {"value": "150"},
            "liquidation_buffer_amount": {"value": "20"},
            "liquidation_buffer_percentage": "60.5",
        }
    }
    balance = service.get_cfm_balance()
    assert balance is not None
    assert balance.futures_buying_power == Decimal("100")
    assert balance.total_usd_balance == Decimal("200")
    assert balance.available_margin == Decimal("50")
    assert balance.liquidation_buffer_percentage == 60.5


def assert_has_cfm_access_false_without_summary() -> None:
    service, client, endpoints = _make_edge_service()
    endpoints.supports_derivatives.return_value = True
    client.cfm_balance_summary.return_value = {"status": "ok"}
    assert service.has_cfm_access() is False


def assert_get_unified_balance_combines_spot_and_cfm() -> None:
    service, _client, _endpoints = _make_edge_service()
    service.list_balances = Mock(
        return_value=[
            Balance(asset="USD", total=Decimal("100"), available=Decimal("75")),
            Balance(asset="BTC", total=Decimal("1"), available=Decimal("1")),
        ]
    )
    service.get_cfm_balance = Mock(
        return_value=CFMBalance(
            futures_buying_power=Decimal("300"),
            total_usd_balance=Decimal("200"),
            available_margin=Decimal("50"),
            initial_margin=Decimal("25"),
            unrealized_pnl=Decimal("10"),
            daily_realized_pnl=Decimal("5"),
            liquidation_threshold=Decimal("150"),
            liquidation_buffer_amount=Decimal("20"),
            liquidation_buffer_percentage=60.0,
        )
    )
    balance = service.get_unified_balance()
    assert balance.spot_balance == Decimal("75")
    assert balance.cfm_balance == Decimal("200")
    assert balance.cfm_available_margin == Decimal("50")
    assert balance.cfm_buying_power == Decimal("300")
    assert balance.total_equity == Decimal("275")


def assert_get_unified_balance_without_usd_spot() -> None:
    service, _client, _endpoints = _make_edge_service()
    service.list_balances = Mock(
        return_value=[Balance(asset="BTC", total=Decimal("1"), available=Decimal("1"))]
    )
    service.get_cfm_balance = Mock(
        return_value=CFMBalance(
            futures_buying_power=Decimal("300"),
            total_usd_balance=Decimal("200"),
            available_margin=Decimal("50"),
            initial_margin=Decimal("25"),
            unrealized_pnl=Decimal("10"),
            daily_realized_pnl=Decimal("5"),
            liquidation_threshold=Decimal("150"),
            liquidation_buffer_amount=Decimal("20"),
            liquidation_buffer_percentage=60.0,
        )
    )
    balance = service.get_unified_balance()
    assert balance.spot_balance == Decimal("0")
    assert balance.cfm_balance == Decimal("200")
    assert balance.total_equity == Decimal("200")


def assert_list_all_positions_merges_spot_and_cfm() -> None:
    service, _client, endpoints = _make_edge_service()
    endpoints.supports_derivatives.return_value = True
    service.list_spot_positions_as_core = Mock(
        return_value=[
            AccountPosition(
                symbol="BTC-USD",
                quantity=Decimal("1"),
                entry_price=Decimal("0"),
                mark_price=Decimal("0"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
                leverage=1,
                product_type="SPOT",
            )
        ]
    )
    service.list_cfm_positions = Mock(
        return_value=[
            AccountPosition(
                symbol="BTC-2025",
                quantity=Decimal("2"),
                entry_price=Decimal("100"),
                mark_price=Decimal("110"),
                unrealized_pnl=Decimal("5"),
                realized_pnl=Decimal("1"),
                side="long",
                leverage=None,
                product_type="FUTURE",
            )
        ]
    )
    positions = service.list_all_positions()
    product_types = {position.product_type for position in positions}
    assert product_types == {"SPOT", "FUTURE"}


def assert_list_all_positions_spot_only_when_no_derivatives() -> None:
    service, _client, endpoints = _make_edge_service()
    endpoints.supports_derivatives.return_value = False
    service.list_spot_positions_as_core = Mock(return_value=[])
    service.list_cfm_positions = Mock(return_value=[])
    positions = service.list_all_positions()
    assert positions == []
    service.list_cfm_positions.assert_not_called()
