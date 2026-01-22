from tests.unit.gpt_trader.features.brokerages.coinbase.rest.contract_suite_test_base import (
    CoinbaseRestContractSuiteBase,
    assert_order_service_contracts,
    assert_pnl_service_contracts,
    assert_portfolio_service_contracts,
)


class TestCoinbaseRestContractSuite(CoinbaseRestContractSuiteBase):
    def test_order_service_contracts(
        self,
        order_service,
        service_core,
        portfolio_service,
        mock_product_catalog,
        mock_product,
        mock_client,
        monkeypatch,
    ) -> None:
        assert_order_service_contracts(
            order_service,
            service_core,
            portfolio_service,
            mock_product_catalog,
            mock_product,
            mock_client,
            monkeypatch,
        )

    def test_pnl_service_contracts(self, pnl_service, mock_market_data) -> None:
        assert_pnl_service_contracts(pnl_service, mock_market_data)

    def test_portfolio_service_contracts(
        self, portfolio_service, mock_client, mock_endpoints, monkeypatch
    ) -> None:
        assert_portfolio_service_contracts(
            portfolio_service, mock_client, mock_endpoints, monkeypatch
        )
