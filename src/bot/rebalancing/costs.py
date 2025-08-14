"""
Transaction cost modeling for portfolio rebalancing
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime


@dataclass
class CostParameters:
    """Parameters for transaction cost modeling"""

    # Fixed costs
    commission_per_trade: float = 0.0  # Dollar amount per trade
    commission_rate: float = 0.0  # Percentage of trade value

    # Variable costs
    bid_ask_spread: float = 0.001  # Default 10 bps
    market_impact_linear: float = 0.0001  # Linear impact coefficient
    market_impact_sqrt: float = 0.001  # Square-root impact coefficient

    # Slippage
    slippage_rate: float = 0.0005  # Default 5 bps

    # Tax rates (if applicable)
    short_term_tax_rate: float = 0.0
    long_term_tax_rate: float = 0.0

    # Minimum thresholds
    min_trade_size: float = 100.0  # Minimum trade size in dollars

    # Market conditions
    volatility_adjustment: bool = True  # Adjust costs based on volatility
    liquidity_adjustment: bool = True  # Adjust costs based on liquidity


class TransactionCostModel:
    """Model for estimating and tracking transaction costs"""

    def __init__(self,
                 parameters: Optional[CostParameters] = None,
                 db_manager=None):
        """Initialize transaction cost model

        Args:
            parameters: Cost parameters
            db_manager: Database manager for storing cost data
        """
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters or CostParameters()
        self.db_manager = db_manager

        # Cost tracking
        self.total_costs = 0.0
        self.cost_history = []
