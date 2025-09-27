from bot_v2.features.brokerages.coinbase.endpoints import ENDPOINTS
from bot_v2.features.brokerages.coinbase import client as client_mod


def test_registry_non_empty_and_methods_exist():
    assert len(ENDPOINTS) >= 6
    # Ensure client has corresponding method names where obvious
    method_map = {
        "list_products": "get_products",
        "get_product": "get_product",
        "get_product_ticker": "get_ticker",
        "get_product_candles": "get_candles",
        "get_product_book": "get_product_book",
        "list_accounts": "get_accounts",
        "get_account": "get_account",
        "place_order": "place_order",
        "cancel_orders": "cancel_orders",
        "get_order": "get_order",
        "list_orders": "list_orders",
        "list_orders_batch": "list_orders_batch",
        "list_fills": "list_fills",
        "get_fees": "get_fees",
        "get_limits": "get_limits",
        "create_convert": "create_convert",
        "get_convert": "get_convert",
    }

    for ep in ENDPOINTS:
        meth_name = method_map.get(ep.name)
        if meth_name is None:
            # Not all endpoints must map 1:1 to client methods, but most should
            continue
        assert hasattr(client_mod.CoinbaseClient, meth_name), f"Missing {meth_name} for {ep.name}"
