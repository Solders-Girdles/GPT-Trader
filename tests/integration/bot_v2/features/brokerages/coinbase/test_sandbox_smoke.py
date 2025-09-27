import os
import time
import pytest


def _should_run():
    return (
        os.getenv("COINBASE_API_KEY")
        and os.getenv("COINBASE_API_SECRET")
        and os.getenv("COINBASE_SANDBOX") == "1"
    )


@pytest.mark.integration
def test_products_and_accounts_smoke(monkeypatch):
    if not _should_run():
        pytest.skip("Coinbase sandbox creds not set; skipping")

    # Ensure factory selects coinbase
    monkeypatch.setenv("BROKER", "coinbase")

    from bot_v2.orchestration.broker_factory import create_brokerage

    broker = create_brokerage()
    assert broker.connect() is True

    # Products (public)
    prods = broker.list_products()
    assert isinstance(prods, list) and len(prods) > 0

    # Accounts (auth)
    balances = broker.list_balances()
    assert isinstance(balances, list)


@pytest.mark.integration
def test_place_and_cancel_limit_order_sandbox(monkeypatch):
    if not _should_run() or os.getenv("COINBASE_RUN_ORDER_TESTS") != "1":
        pytest.skip("Order tests disabled; set COINBASE_RUN_ORDER_TESTS=1 to enable")

    monkeypatch.setenv("BROKER", "coinbase")

    from decimal import Decimal
    from bot_v2.orchestration.broker_factory import create_brokerage
    from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce

    broker = create_brokerage()
    assert broker.connect() is True

    # Use a low size and a far price to avoid fills
    symbol = os.getenv("COINBASE_ORDER_SYMBOL", "BTC-USD")
    price = Decimal(os.getenv("COINBASE_TEST_LIMIT_PRICE", "10"))
    qty = Decimal(os.getenv("COINBASE_TEST_QTY", "0.001"))

    order = broker.place_order(
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=qty,
        price=price,
        tif=TimeInForce.GTC,
        client_id=f"gpt-trader-test-{int(time.time())}",
        reduce_only=False,
    )
    assert order.id

    # Cancel immediately
    assert broker.cancel_order(order.id) is True

