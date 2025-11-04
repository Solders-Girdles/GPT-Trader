"""Kelly calculations from trade statistics."""

from __future__ import annotations

from bot_v2.features.position_sizing.types import TradeStatistics

from .calculations import fractional_kelly


def kelly_from_statistics(stats: TradeStatistics, fraction: float = 0.25) -> float:
    """Calculate fractional Kelly from trade statistics."""
    if stats.total_trades < 10:
        return 0.0

    return fractional_kelly(
        win_rate=stats.win_rate,
        avg_win=stats.avg_win_return,
        avg_loss=stats.avg_loss_return,
        fraction=fraction,
    )


__all__ = ["kelly_from_statistics"]
