from __future__ import annotations

from tests.unit.gpt_trader.features.brokerages.coinbase.rest import (
    portfolio_service_test_base as helpers,
)


class TestPortfolioServiceInit(helpers.PortfolioServiceTestBase):
    def test_service_init(
        self, portfolio_service, mock_client, mock_endpoints, mock_event_store
    ) -> None:
        helpers.assert_service_init(
            portfolio_service, mock_client, mock_endpoints, mock_event_store
        )


class TestPortfolioServiceBalances(helpers.PortfolioServiceTestBase):
    def test_balances(self, portfolio_service, mock_client) -> None:
        helpers.assert_get_portfolio_balances_delegates_to_list_balances(
            portfolio_service, mock_client
        )
        helpers.assert_list_balances_returns_balances(portfolio_service, mock_client)
        helpers.assert_list_balances_handles_list_response(portfolio_service, mock_client)
        helpers.assert_list_balances_calculates_total_when_missing(portfolio_service, mock_client)
        helpers.assert_list_balances_skips_invalid_entries(portfolio_service, mock_client)
        helpers.assert_list_balances_handles_exception(portfolio_service, mock_client)


class TestPortfolioServicePositions(helpers.PortfolioServiceTestBase):
    def test_positions(self, portfolio_service, mock_client, mock_endpoints) -> None:
        helpers.assert_list_positions_returns_empty_when_derivatives_not_supported(
            portfolio_service, mock_client, mock_endpoints
        )
        helpers.assert_list_positions_returns_positions(
            portfolio_service, mock_client, mock_endpoints
        )
        helpers.assert_list_positions_handles_exception(
            portfolio_service, mock_client, mock_endpoints
        )
        helpers.assert_get_position_returns_position(portfolio_service, mock_client, mock_endpoints)
        helpers.assert_get_position_returns_none_when_not_supported(
            portfolio_service, mock_endpoints
        )


class TestCfmBalanceSummary(helpers.PortfolioServiceTestBase):
    def test_cfm_balance_summary(
        self, portfolio_service, mock_client, mock_endpoints, mock_event_store
    ) -> None:
        helpers.assert_get_cfm_balance_summary_returns_empty_when_not_supported(
            portfolio_service, mock_endpoints
        )
        helpers.assert_get_cfm_balance_summary_returns_summary(
            portfolio_service, mock_client, mock_endpoints, mock_event_store
        )
        mock_event_store.append_metric.reset_mock()
        helpers.assert_get_cfm_balance_summary_normalises_decimals_and_emits_metric(
            portfolio_service, mock_client, mock_endpoints, mock_event_store
        )


class TestCfmSweeps(helpers.PortfolioServiceTestBase):
    def test_cfm_sweeps(
        self, portfolio_service, mock_client, mock_endpoints, mock_event_store
    ) -> None:
        helpers.assert_list_cfm_sweeps_returns_empty_when_derivatives_disabled(
            portfolio_service, mock_client, mock_endpoints
        )
        helpers.assert_list_cfm_sweeps_returns_sweeps(
            portfolio_service, mock_client, mock_endpoints
        )
        helpers.assert_list_cfm_sweeps_normalises_entries(
            portfolio_service, mock_client, mock_endpoints, mock_event_store
        )
        helpers.assert_get_cfm_sweeps_schedule_returns_schedule(
            portfolio_service, mock_client, mock_endpoints
        )


class TestCfmMarginWindow(helpers.PortfolioServiceTestBase):
    def test_cfm_margin_window(
        self, portfolio_service, mock_client, mock_endpoints, mock_event_store
    ) -> None:
        helpers.assert_get_cfm_margin_window_returns_window(
            portfolio_service, mock_client, mock_endpoints
        )
        helpers.assert_get_cfm_margin_window_handles_errors(
            portfolio_service, mock_client, mock_endpoints
        )
        helpers.assert_update_cfm_margin_window_raises_when_not_supported(
            portfolio_service, mock_endpoints
        )
        helpers.assert_update_cfm_margin_window_enforces_derivatives(
            portfolio_service, mock_endpoints
        )
        helpers.assert_update_cfm_margin_window_success(
            portfolio_service, mock_client, mock_endpoints, mock_event_store
        )
        mock_client.cfm_intraday_margin_setting.reset_mock()
        mock_event_store.append_metric.reset_mock()
        helpers.assert_update_cfm_margin_window_calls_client_and_emits(
            portfolio_service, mock_client, mock_endpoints, mock_event_store
        )


class TestIntxAllocate(helpers.PortfolioServiceTestBase):
    def test_intx_allocate(
        self, portfolio_service, mock_client, mock_endpoints, mock_event_store
    ) -> None:
        helpers.assert_intx_allocate_requires_advanced_mode(portfolio_service, mock_endpoints)
        helpers.assert_intx_allocate_success(
            portfolio_service, mock_client, mock_endpoints, mock_event_store
        )
        mock_event_store.append_metric.reset_mock()
        helpers.assert_intx_allocate_normalises_and_emits_metric(
            portfolio_service, mock_client, mock_endpoints, mock_event_store
        )


class TestIntxBalances(helpers.PortfolioServiceTestBase):
    def test_intx_balances(
        self, portfolio_service, mock_client, mock_endpoints, mock_event_store
    ) -> None:
        helpers.assert_get_intx_balances_returns_empty_when_not_advanced(
            portfolio_service, mock_endpoints
        )
        helpers.assert_get_intx_balances_returns_balances(
            portfolio_service, mock_client, mock_endpoints
        )
        helpers.assert_get_intx_balances_normalises_entries(
            portfolio_service, mock_client, mock_endpoints, mock_event_store
        )
        mock_event_store.append_metric.reset_mock()
        helpers.assert_get_intx_balances_handles_errors(
            portfolio_service, mock_client, mock_endpoints, mock_event_store
        )


class TestIntxPortfolio(helpers.PortfolioServiceTestBase):
    def test_intx_portfolio(self, portfolio_service, mock_client, mock_endpoints) -> None:
        helpers.assert_get_intx_portfolio_returns_empty_when_not_advanced(
            portfolio_service, mock_endpoints
        )
        helpers.assert_get_intx_portfolio_success(portfolio_service, mock_client, mock_endpoints)
        helpers.assert_get_intx_portfolio_returns_normalised_dict(
            portfolio_service, mock_client, mock_endpoints
        )


class TestIntxPositions(helpers.PortfolioServiceTestBase):
    def test_intx_positions(self, portfolio_service, mock_client, mock_endpoints) -> None:
        helpers.assert_list_intx_positions_returns_positions(
            portfolio_service, mock_client, mock_endpoints
        )
        helpers.assert_list_intx_positions_returns_normalised_list(
            portfolio_service, mock_client, mock_endpoints
        )
        helpers.assert_get_intx_position_handles_missing(
            portfolio_service, mock_client, mock_endpoints
        )


class TestIntxCollateral(helpers.PortfolioServiceTestBase):
    def test_intx_multi_asset_collateral(
        self, portfolio_service, mock_client, mock_endpoints, mock_event_store
    ) -> None:
        helpers.assert_get_intx_multi_asset_collateral_emits_metric(
            portfolio_service, mock_client, mock_endpoints, mock_event_store
        )


class TestCfmPositionEdges:
    def test_list_cfm_positions_invalid_expiry_sets_none(self) -> None:
        helpers.assert_list_cfm_positions_invalid_expiry_sets_none()


class TestSpotPositionEdges:
    def test_list_spot_positions_skips_usd_and_zero(self) -> None:
        helpers.assert_list_spot_positions_skips_usd_and_zero()


class TestCfmBalanceEdges:
    def test_cfm_balance_edges(self) -> None:
        helpers.assert_get_cfm_balance_missing_or_empty_summary()
        helpers.assert_get_cfm_balance_parses_nested_values()
        helpers.assert_has_cfm_access_false_without_summary()


class TestUnifiedBalanceEdges:
    def test_unified_balance_edges(self) -> None:
        helpers.assert_get_unified_balance_combines_spot_and_cfm()
        helpers.assert_get_unified_balance_without_usd_spot()


class TestAllPositionsEdges:
    def test_all_positions_edges(self) -> None:
        helpers.assert_list_all_positions_merges_spot_and_cfm()
        helpers.assert_list_all_positions_spot_only_when_no_derivatives()
