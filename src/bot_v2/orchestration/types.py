"""
Type definitions for the orchestration layer
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime


class TradingMode(Enum):
    """Trading execution modes"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    OPTIMIZE = "optimize"


@dataclass
class OrchestratorConfig:
    """Configuration for the trading orchestrator"""
    mode: TradingMode = TradingMode.BACKTEST
    symbols: List[str] = None
    strategies: List[str] = None
    capital: float = 10000.0
    risk_limits: Dict[str, float] = None
    enable_ml_strategy: bool = True
    enable_regime_detection: bool = True
    enable_adaptive_portfolio: bool = True
    risk_tolerance: float = 0.02
    min_confidence: float = 0.6
    max_position_pct: float = 0.2
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['AAPL', 'GOOGL', 'MSFT']
        if self.risk_limits is None:
            self.risk_limits = {
                'max_position': 0.2,
                'max_daily_loss': 0.02,
                'max_drawdown': 0.1
            }


@dataclass
class OrchestrationResult:
    """Result from orchestrator execution"""
    timestamp: datetime
    mode: TradingMode
    symbol: str
    success: bool
    data: Dict[str, Any]
    errors: List[str]
    metrics: Dict[str, float]
    execution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'mode': self.mode.value,
            'symbol': self.symbol,
            'success': self.success,
            'data': self.data,
            'errors': self.errors,
            'metrics': self.metrics,
            'execution_time': self.execution_time
        }