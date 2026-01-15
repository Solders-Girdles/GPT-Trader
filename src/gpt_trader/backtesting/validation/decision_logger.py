"""Decision logging for golden-path validation.

This module captures every strategy decision for later comparison
between live and simulated execution paths.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from gpt_trader.utilities.datetime_helpers import utc_now


@dataclass
class StrategyDecision:
    """
    Represents a single strategy decision point.

    Captures all inputs and outputs of a strategy decision for
    deterministic replay and comparison.
    """

    # Identification
    decision_id: str
    cycle_id: str
    timestamp: datetime

    # Context
    symbol: str
    equity: Decimal
    position_quantity: Decimal
    position_side: str | None  # "long", "short", or None

    # Market state
    mark_price: Decimal
    recent_marks: list[Decimal]  # Window of recent prices
    bid: Decimal | None = None
    ask: Decimal | None = None
    volume: Decimal | None = None

    # Strategy inputs
    strategy_name: str = ""
    strategy_params: dict[str, Any] = field(default_factory=dict)

    # Decision output
    action: str = "HOLD"  # "BUY", "SELL", "HOLD"
    target_quantity: Decimal = Decimal("0")
    target_price: Decimal | None = None
    order_type: str = "MARKET"
    reason: str = ""

    # Risk checks
    risk_checks_passed: bool = True
    risk_check_failures: list[str] = field(default_factory=list)

    # Execution result (filled in after order execution)
    order_id: str | None = None
    fill_price: Decimal | None = None
    fill_quantity: Decimal | None = None
    slippage_bps: Decimal | None = None

    @classmethod
    def create(
        cls,
        cycle_id: str,
        symbol: str,
        equity: Decimal,
        position_quantity: Decimal,
        position_side: str | None,
        mark_price: Decimal,
        recent_marks: list[Decimal],
    ) -> StrategyDecision:
        """Create a new decision with generated ID and timestamp."""
        return cls(
            decision_id=str(uuid.uuid4())[:12],
            cycle_id=cycle_id,
            timestamp=utc_now(),
            symbol=symbol,
            equity=equity,
            position_quantity=position_quantity,
            position_side=position_side,
            mark_price=mark_price,
            recent_marks=recent_marks,
        )

    def with_market_data(
        self,
        bid: Decimal | None = None,
        ask: Decimal | None = None,
        volume: Decimal | None = None,
    ) -> StrategyDecision:
        """Add market data to decision."""
        self.bid = bid
        self.ask = ask
        self.volume = volume
        return self

    def with_strategy(
        self,
        name: str,
        params: dict[str, Any] | None = None,
    ) -> StrategyDecision:
        """Add strategy info to decision."""
        self.strategy_name = name
        self.strategy_params = params or {}
        return self

    def with_action(
        self,
        action: str,
        quantity: Decimal = Decimal("0"),
        price: Decimal | None = None,
        order_type: str = "MARKET",
        reason: str = "",
    ) -> StrategyDecision:
        """Record strategy decision."""
        self.action = action
        self.target_quantity = quantity
        self.target_price = price
        self.order_type = order_type
        self.reason = reason
        return self

    def with_risk_result(
        self,
        passed: bool,
        failures: list[str] | None = None,
    ) -> StrategyDecision:
        """Record risk check results."""
        self.risk_checks_passed = passed
        self.risk_check_failures = failures or []
        return self

    def with_execution(
        self,
        order_id: str,
        fill_price: Decimal,
        fill_quantity: Decimal,
        slippage_bps: Decimal | None = None,
    ) -> StrategyDecision:
        """Record execution result."""
        self.order_id = order_id
        self.fill_price = fill_price
        self.fill_quantity = fill_quantity
        self.slippage_bps = slippage_bps
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert Decimals to strings for JSON serialization
        for key, value in data.items():
            if isinstance(value, Decimal):
                data[key] = str(value)
            elif isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, list) and value and isinstance(value[0], Decimal):
                data[key] = [str(v) for v in value]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyDecision:
        """Create from dictionary."""
        # Convert strings back to Decimals
        decimal_fields = [
            "equity",
            "position_quantity",
            "mark_price",
            "bid",
            "ask",
            "volume",
            "target_quantity",
            "target_price",
            "fill_price",
            "fill_quantity",
            "slippage_bps",
        ]
        for field_name in decimal_fields:
            if data.get(field_name) is not None:
                data[field_name] = Decimal(str(data[field_name]))

        # Convert recent_marks
        if data.get("recent_marks"):
            data["recent_marks"] = [Decimal(str(m)) for m in data["recent_marks"]]

        # Convert timestamp
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        return cls(**data)


class DecisionLogger:
    """
    Logs strategy decisions for golden-path validation.

    Records every decision to enable:
    1. Deterministic replay of backtest scenarios
    2. Comparison between live and simulated decisions
    3. Debugging of strategy behavior
    """

    def __init__(
        self,
        storage_path: Path | str | None = None,
        max_memory_decisions: int = 10000,
    ):
        """
        Initialize decision logger.

        Args:
            storage_path: Optional path to persist decisions to disk
            max_memory_decisions: Max decisions to keep in memory
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_memory_decisions = max_memory_decisions
        self._decisions: list[StrategyDecision] = []
        self._current_cycle_id: str | None = None

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

    def start_cycle(self, cycle_id: str | None = None) -> str:
        """
        Start a new decision cycle.

        Args:
            cycle_id: Optional cycle ID (generated if not provided)

        Returns:
            Cycle ID
        """
        self._current_cycle_id = cycle_id or str(uuid.uuid4())[:12]
        return self._current_cycle_id

    def log_decision(self, decision: StrategyDecision) -> None:
        """
        Log a strategy decision.

        Args:
            decision: Decision to log
        """
        self._decisions.append(decision)

        # Trim if over limit
        if len(self._decisions) > self.max_memory_decisions:
            self._decisions = self._decisions[-self.max_memory_decisions :]

        # Optionally persist to disk
        if self.storage_path:
            self._persist_decision(decision)

    def _persist_decision(self, decision: StrategyDecision) -> None:
        """Persist a decision to disk."""
        if not self.storage_path:
            return

        filename = f"{decision.timestamp.strftime('%Y%m%d')}_{decision.cycle_id}.jsonl"
        filepath = self.storage_path / filename

        with open(filepath, "a") as f:
            json.dump(decision.to_dict(), f)
            f.write("\n")

    def get_decisions(
        self,
        cycle_id: str | None = None,
        symbol: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[StrategyDecision]:
        """
        Retrieve logged decisions with optional filtering.

        Args:
            cycle_id: Filter by cycle ID
            symbol: Filter by symbol
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            List of matching decisions
        """
        decisions = self._decisions

        if cycle_id:
            decisions = [d for d in decisions if d.cycle_id == cycle_id]

        if symbol:
            decisions = [d for d in decisions if d.symbol == symbol]

        if start_time:
            decisions = [d for d in decisions if d.timestamp >= start_time]

        if end_time:
            decisions = [d for d in decisions if d.timestamp < end_time]

        return decisions

    def get_decision_by_id(self, decision_id: str) -> StrategyDecision | None:
        """Get a specific decision by ID."""
        for decision in self._decisions:
            if decision.decision_id == decision_id:
                return decision
        return None

    def clear(self) -> None:
        """Clear all logged decisions."""
        self._decisions = []

    def export_to_json(self, filepath: Path | str) -> int:
        """
        Export all decisions to a JSON file.

        Args:
            filepath: Output file path

        Returns:
            Number of decisions exported
        """
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            data = [d.to_dict() for d in self._decisions]
            json.dump(data, f, indent=2)
        return len(self._decisions)

    def import_from_json(self, filepath: Path | str) -> int:
        """
        Import decisions from a JSON file.

        Args:
            filepath: Input file path

        Returns:
            Number of decisions imported
        """
        filepath = Path(filepath)
        with open(filepath) as f:
            data = json.load(f)
            for item in data:
                decision = StrategyDecision.from_dict(item)
                self._decisions.append(decision)
        return len(data)

    @property
    def current_cycle_id(self) -> str | None:
        """Get current cycle ID."""
        return self._current_cycle_id

    @property
    def decision_count(self) -> int:
        """Get total number of logged decisions."""
        return len(self._decisions)
