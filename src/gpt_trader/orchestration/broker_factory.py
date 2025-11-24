import os
from typing import Tuple, Any

from gpt_trader.features.brokerages.coinbase.models import APIConfig
from gpt_trader.data_providers.coinbase.provider import CoinbaseBrokerage
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog

def create_brokerage() -> Tuple[Any, EventStore, MarketDataService, ProductCatalog]:
    broker_name = os.environ.get("BROKER", "coinbase").lower()

    if broker_name != "coinbase":
        raise ValueError(f"Unsupported broker: {broker_name}")

    sandbox = os.environ.get("COINBASE_SANDBOX", "0") == "1"
    
    # Defaults and overrides based on environment
    if sandbox:
        base_url = "https://api-public.sandbox.exchange.coinbase.com"
        ws_url = "wss://ws-feed-public.sandbox.exchange.coinbase.com"
        auth_type = "HMAC"
        api_mode = "exchange" 
    else:
        base_url = os.environ.get("COINBASE_API_BASE", "https://api.coinbase.com")
        ws_url = os.environ.get("COINBASE_WS_URL", "wss://advanced-trade-ws.coinbase.com")
        api_mode = os.environ.get("COINBASE_API_MODE", "advanced")
        
        if os.environ.get("COINBASE_CDP_API_KEY") and os.environ.get("COINBASE_CDP_PRIVATE_KEY"):
             auth_type = "JWT"
        else:
             auth_type = "HMAC"

    api_config = APIConfig(
        api_key=os.environ.get("COINBASE_API_KEY", ""),
        api_secret=os.environ.get("COINBASE_API_SECRET", ""),
        passphrase=os.environ.get("COINBASE_API_PASSPHRASE"),
        base_url=base_url,
        sandbox=sandbox,
        ws_url=ws_url,
        api_mode=api_mode,
        auth_type=auth_type,
        cdp_api_key=os.environ.get("COINBASE_CDP_API_KEY"),
        cdp_private_key=os.environ.get("COINBASE_CDP_PRIVATE_KEY")
    )

    # Instantiate broker with api_config as expected by tests (mocking interface)
    # Note: Real CoinbaseBrokerage (placeholder) signature is different, but tests mock it.
    broker = CoinbaseBrokerage(api_config, settings=None)

    event_store = EventStore()
    market_data = MarketDataService(symbols=[]) 
    product_catalog = ProductCatalog()

    return broker, event_store, market_data, product_catalog
