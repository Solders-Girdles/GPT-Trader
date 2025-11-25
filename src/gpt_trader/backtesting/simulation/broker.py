from decimal import Decimal

from gpt_trader.backtesting.types import FeeTier
from gpt_trader.features.brokerages.core.interfaces import Balance, Position, Product


class SimulatedBroker:
    def __init__(
        self, initial_equity_usd: Decimal = Decimal("100000"), fee_tier: FeeTier = FeeTier.TIER_2
    ):
        self.equity = initial_equity_usd
        self.fee_tier = fee_tier
        self.products: dict[str, Product] = {}
        self.balances: dict[str, Balance] = {
            "USDC": Balance(asset="USDC", total=initial_equity_usd, available=initial_equity_usd)
        }
        self.positions: dict[str, Position] = {}
        self.connected = False

    def register_product(self, product: Product) -> None:
        self.products[product.symbol] = product

    def get_product(self, symbol: str) -> Product | None:
        return self.products.get(symbol)

    def connect(self) -> bool:
        self.connected = True
        return True

    def validate_connection(self) -> bool:
        return self.connected

    def get_account_id(self) -> str:
        return "SIMULATED_ACCOUNT"

    def disconnect(self) -> None:
        self.connected = False

    def list_balances(self) -> list[Balance]:
        return list(self.balances.values())

    def get_equity(self) -> Decimal:
        return self.equity

    def get_account_info(self) -> dict[str, Decimal]:
        return {"cash": self.equity}

    def list_positions(self) -> list[Position]:
        return list(self.positions.values())
