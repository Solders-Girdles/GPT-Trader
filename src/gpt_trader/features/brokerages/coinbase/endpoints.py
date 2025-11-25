"""
Coinbase Endpoints helper.
"""

from gpt_trader.features.brokerages.coinbase.models import APIConfig


class CoinbaseEndpoints:
    def __init__(self, config: APIConfig):
        self.config = config
        self.mode = config.api_mode
        self.endpoints = {
            "exchange": {
                "accounts": "/accounts",
                "account": "/accounts/{account_id}",
                "orders": "/orders",
                "order": "/orders/{order_id}",
                "products": "/products",
                "product": "/products/{product_id}",
                "candles": "/products/{product_id}/candles",
                "ticker": "/products/{product_id}/ticker",
                "trades": "/products/{product_id}/trades",
                "order_book": "/products/{product_id}/book",
                "time": "/time",
            },
        }

    def supports_derivatives(self) -> bool:
        return self.config.enable_derivatives
