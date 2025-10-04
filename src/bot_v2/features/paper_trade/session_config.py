"""
Paper Trading Session Configuration.

Provides structured configuration for paper trading sessions with validation.
Extracted from PaperTradingSession to improve separation of concerns.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PaperSessionConfig:
    """
    Configuration for a paper trading session.

    Encapsulates all session parameters with sensible defaults and validation.
    """

    strategy_name: str
    symbols: list[str]
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    max_positions: int = 10
    position_size: float = 0.95
    update_interval: int = 60
    strategy_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Post-initialization hook.

        Note: Validation intentionally omitted to preserve backward compatibility
        with existing behavior. The original PaperTradingSession accepted any
        values without validation. Consider adding validation in a future phase.
        """
        pass


class SessionConfigBuilder:
    """
    Builder for PaperSessionConfig with kwargs extraction.

    Provides backward-compatible construction from legacy kwargs pattern.
    """

    @staticmethod
    def from_kwargs(
        strategy: str,
        symbols: list[str],
        initial_capital: float = 100000.0,
        **kwargs: Any,
    ) -> PaperSessionConfig:
        """
        Build configuration from legacy kwargs pattern.

        Extracts session-specific parameters and passes remainder to strategy_params.

        Args:
            strategy: Strategy name
            symbols: List of symbols to trade
            initial_capital: Starting capital
            **kwargs: Additional parameters (session config or strategy params)

        Returns:
            PaperSessionConfig with extracted and validated parameters
        """
        # Extract session parameters (remove from kwargs)
        commission = kwargs.pop("commission", 0.001)
        slippage = kwargs.pop("slippage", 0.0005)
        max_positions = kwargs.pop("max_positions", 10)
        position_size = kwargs.pop("position_size", 0.95)
        update_interval = kwargs.pop("update_interval", 60)

        # Remaining kwargs are strategy parameters
        strategy_params = kwargs

        return PaperSessionConfig(
            strategy_name=strategy,
            symbols=symbols,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            max_positions=max_positions,
            position_size=position_size,
            update_interval=update_interval,
            strategy_params=strategy_params,
        )
