from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from bot_v2.features.strategies.interfaces import StrategyContext, IStrategy
from bot_v2.orchestration.execution import PaperExecutionEngine
from bot_v2.orchestration.registry import StrategyRegistry
from bot_v2.persistence.event_store import EventStore


@dataclass
class RiskConfig:
    max_positions: int = 5
    max_position_size: float = 0.2
    stop_loss: float = 0.05
    take_profit: float = 0.10
    commission: float = 0.006
    slippage: float = 0.001


@dataclass
class BotConfig:
    bot_id: str
    name: str
    symbols: List[str]
    strategy: str
    strategy_params: Dict[str, object] = field(default_factory=dict)
    capital: float = 10_000.0
    mode: str = "paper"  # future: 'live'
    risk: RiskConfig = field(default_factory=RiskConfig)
    loop_sleep: float = 6.0
    auto_start: bool = False


@dataclass
class BotMetrics:
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_update: Optional[datetime] = None
    trades: int = 0
    signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0
    execution_rate: float = 0.0
    trades_per_hour: float = 0.0
    signals_per_hour: float = 0.0
    equity: float = 0.0


class BotStatus:
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ManagedBot:
    config: BotConfig
    status: str = BotStatus.CREATED
    metrics: BotMetrics = field(default_factory=BotMetrics)
    error: Optional[str] = None


class BotManager:
    """Minimal bot manager with lifecycle and paper execution."""

    def __init__(self):
        self._bots: Dict[str, ManagedBot] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._stops: Dict[str, threading.Event] = {}
        self._events = EventStore()

    # CRUD
    def add_bot(self, cfg: BotConfig) -> None:
        if cfg.bot_id in self._bots:
            raise ValueError(f"Bot {cfg.bot_id} already exists")
        self._bots[cfg.bot_id] = ManagedBot(config=cfg)

    def list_bots(self) -> List[ManagedBot]:
        return list(self._bots.values())

    def get_bot(self, bot_id: str) -> ManagedBot:
        if bot_id not in self._bots:
            raise KeyError(bot_id)
        return self._bots[bot_id]

    # Lifecycle
    def start(self, bot_id: str) -> None:
        bot = self.get_bot(bot_id)
        if bot.status == BotStatus.RUNNING:
            return
        stop_event = threading.Event()
        self._stops[bot_id] = stop_event
        t = threading.Thread(target=self._run_bot, args=(bot_id, stop_event), daemon=True)
        self._threads[bot_id] = t
        bot.status = BotStatus.RUNNING
        t.start()

    def stop(self, bot_id: str) -> None:
        ev = self._stops.get(bot_id)
        if ev:
            ev.set()
        t = self._threads.get(bot_id)
        if t:
            t.join(timeout=2)
        bot = self._bots.get(bot_id)
        if bot:
            bot.status = BotStatus.STOPPED

    # Worker
    def _run_bot(self, bot_id: str, stop_event: threading.Event) -> None:
        bot = self._bots[bot_id]
        cfg = bot.config
        try:
            # Init strategy and execution
            strat: IStrategy = StrategyRegistry.create(cfg.strategy, **cfg.strategy_params)
            exec_engine = PaperExecutionEngine(
                commission=cfg.risk.commission,
                slippage=cfg.risk.slippage,
                initial_capital=cfg.capital,
            )
            exec_engine.connect()

            ctx = StrategyContext(symbols=cfg.symbols)
            last_prices: Dict[str, float] = {}

            while not stop_event.is_set() and bot.status == BotStatus.RUNNING:
                # Update prices
                for sym in cfg.symbols:
                    mid = exec_engine.get_mid(sym)
                    if mid is None:
                        continue
                    last_prices[sym] = mid
                    strat.update_price(sym, mid)

                # Generate signals and apply simple risk controls
                signals = strat.get_signals(ctx)
                bot.metrics.signals += len(signals)
                for s in signals:
                    if s.side == "buy":
                        bot.metrics.buy_signals += 1
                    elif s.side == "sell":
                        bot.metrics.sell_signals += 1
                    else:
                        bot.metrics.hold_signals += 1

                # Execute: position limit and sizing by confidence
                # Position value = confidence * max_position_size * equity
                equity = exec_engine.equity()
                bot.metrics.equity = equity
                open_syms = set(exec_engine.positions.keys())
                for s in signals:
                    if s.side == "buy":
                        if len(open_syms) >= cfg.risk.max_positions:
                            continue
                        amount = equity * cfg.risk.max_position_size * max(0.2, min(1.0, s.confidence))
                        tr = exec_engine.buy(s.symbol, amount, reason=f"{cfg.strategy} signal")
                        if tr:
                            bot.metrics.trades += 1
                            open_syms.add(s.symbol)
                            self._events.append_trade(
                                bot.config.bot_id,
                                {
                                    'symbol': tr.symbol,
                                    'side': tr.side,
                                    'quantity': tr.quantity,
                                    'price': tr.price,
                                    'value': tr.value,
                                    'commission': tr.commission,
                                    'reason': tr.reason,
                                },
                            )
                    elif s.side == "sell":
                        tr = exec_engine.sell(s.symbol, None, reason=f"{cfg.strategy} signal")
                        if tr:
                            bot.metrics.trades += 1
                            open_syms.discard(s.symbol)
                            self._events.append_trade(
                                bot.config.bot_id,
                                {
                                    'symbol': tr.symbol,
                                    'side': tr.side,
                                    'quantity': tr.quantity,
                                    'price': tr.price,
                                    'value': tr.value,
                                    'commission': tr.commission,
                                    'pnl': tr.pnl,
                                    'reason': tr.reason,
                                },
                            )

                # Stops/TP
                for sym, pos in list(exec_engine.positions.items()):
                    px = exec_engine.get_mid(sym)
                    if px is None:
                        continue
                    change = (px - pos.entry_price) / pos.entry_price if pos.entry_price else 0.0
                    if change <= -cfg.risk.stop_loss or change >= cfg.risk.take_profit:
                        tr = exec_engine.sell(sym, None, reason="risk-exit")
                        if tr:
                            bot.metrics.trades += 1
                            self._events.append_trade(
                                bot.config.bot_id,
                                {
                                    'symbol': tr.symbol,
                                    'side': tr.side,
                                    'quantity': tr.quantity,
                                    'price': tr.price,
                                    'value': tr.value,
                                    'commission': tr.commission,
                                    'pnl': tr.pnl,
                                    'reason': tr.reason,
                                },
                            )

                # Update metrics
                bot.metrics.last_update = datetime.utcnow()
                elapsed_h = max((bot.metrics.last_update - bot.metrics.start_time).total_seconds() / 3600.0, 1e-6)
                bot.metrics.trades_per_hour = bot.metrics.trades / elapsed_h
                bot.metrics.signals_per_hour = bot.metrics.signals / elapsed_h
                bot.metrics.execution_rate = (bot.metrics.trades / bot.metrics.signals * 100.0) if bot.metrics.signals else 0.0

                # Persist metrics snapshot
                self._events.append_metric(
                    bot.config.bot_id,
                    {
                        'equity': bot.metrics.equity,
                        'trades': bot.metrics.trades,
                        'signals': bot.metrics.signals,
                        'trades_per_hour': bot.metrics.trades_per_hour,
                        'signals_per_hour': bot.metrics.signals_per_hour,
                        'execution_rate': bot.metrics.execution_rate,
                    },
                )

                time.sleep(max(0.5, cfg.loop_sleep))

        except Exception as e:
            bot.status = BotStatus.ERROR
            bot.error = f"{e.__class__.__name__}: {e}"
            try:
                self._events.append_error(bot.config.bot_id, bot.error)
            except Exception:
                pass
