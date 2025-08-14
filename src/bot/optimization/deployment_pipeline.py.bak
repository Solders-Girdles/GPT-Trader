"""
Automated deployment pipeline for selecting and deploying optimized strategies.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from bot.backtest.engine_portfolio import run_backtest
from bot.config import get_config
from bot.exec.alpaca_paper import AlpacaPaperBroker
from bot.live.trading_engine import LiveTradingEngine
from bot.portfolio.allocator import PortfolioRules
from bot.strategy.trend_breakout import TrendBreakoutParams, TrendBreakoutStrategy
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class DeploymentConfig(BaseModel):
    """Configuration for automated deployment."""

    # Strategy selection criteria
    min_sharpe: float = Field(1.0, description="Minimum Sharpe ratio for deployment")
    max_drawdown: float = Field(0.15, description="Maximum drawdown allowed")
    min_trades: int = Field(20, description="Minimum number of trades required")
    min_cagr: float = Field(0.05, description="Minimum CAGR required")

    # Robustness requirements
    min_walk_forward_windows: int = Field(3, description="Minimum walk-forward windows")
    max_sharpe_std: float = Field(0.5, description="Maximum Sharpe ratio standard deviation")

    # Deployment settings
    deployment_budget: float = Field(10000.0, description="Total deployment budget")
    max_concurrent_strategies: int = Field(3, description="Maximum concurrent strategies")
    risk_per_strategy: float = Field(0.02, description="Risk per strategy as fraction of budget")

    # Validation settings
    validation_period_days: int = Field(30, description="Days to validate before full deployment")
    min_validation_sharpe: float = Field(0.5, description="Minimum Sharpe during validation")

    # Paper trading settings
    symbols: list[str] = Field(default_factory=list)
    rebalance_interval: int = Field(300, description="Rebalance interval in seconds")
    max_positions: int = Field(10, description="Maximum positions per strategy")

    @field_validator("min_sharpe")
    def validate_min_sharpe(cls, v):
        if v <= 0:
            raise ValueError("min_sharpe must be positive")
        return v

    @field_validator("max_drawdown")
    def validate_max_drawdown(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("max_drawdown must be between 0 and 1")
        return v

    @field_validator("min_trades")
    def validate_min_trades(cls, v):
        if v <= 0:
            raise ValueError("min_trades must be positive")
        return v

    @field_validator("deployment_budget")
    def validate_deployment_budget(cls, v):
        if v <= 0:
            raise ValueError("deployment_budget must be positive")
        return v

    @field_validator("risk_per_strategy")
    def validate_risk_per_strategy(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("risk_per_strategy must be between 0 and 1")
        return v


class StrategyCandidate(BaseModel):
    """A strategy candidate for deployment."""

    strategy_name: str = "trend_breakout"
    parameters: dict[str, Any]
    performance_metrics: dict[str, float] = {}
    robustness_metrics: dict[str, float] = {}
    walk_forward_results: list[dict[str, float]] = []
    rank_score: float = 0.0
    deployment_ready: bool = False

    # Additional fields for backward compatibility with tests
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    total_return: float | None = None
    n_trades: int | None = None
    cagr: float | None = None
    walk_forward_sharpe_mean: float | None = None
    walk_forward_sharpe_std: float | None = None
    walk_forward_windows: int | None = None

    def __init__(self, **data) -> None:
        super().__init__(**data)

        # Populate performance_metrics from individual fields if provided
        if self.performance_metrics == {} and any(
            [
                self.sharpe_ratio is not None,
                self.max_drawdown is not None,
                self.total_return is not None,
                self.n_trades is not None,
                self.cagr is not None,
            ]
        ):
            self.performance_metrics = {
                "sharpe": self.sharpe_ratio or 0.0,
                "max_drawdown": self.max_drawdown or 0.0,
                "total_return": self.total_return or 0.0,
                "n_trades": self.n_trades or 0,
                "cagr": self.cagr or 0.0,
            }

        # Populate robustness_metrics from walk-forward fields if provided
        if self.robustness_metrics == {} and any(
            [
                self.walk_forward_sharpe_mean is not None,
                self.walk_forward_sharpe_std is not None,
                self.walk_forward_windows is not None,
            ]
        ):
            self.robustness_metrics = {
                "walk_forward_sharpe_mean": self.walk_forward_sharpe_mean or 0.0,
                "walk_forward_sharpe_std": self.walk_forward_sharpe_std or 0.0,
                "walk_forward_windows": self.walk_forward_windows or 0,
            }


class DeploymentPipeline:
    """Automated pipeline for strategy deployment."""

    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.candidates: list[StrategyCandidate] = []
        self.deployed_strategies: list[dict[str, Any]] = []

    def load_optimization_results(self, results_path: str) -> None:
        """Load and analyze optimization results."""
        logger.info(f"Loading optimization results from {results_path}")

        # Load results
        results_df = pd.read_csv(results_path)

        # Filter and rank candidates
        self._filter_candidates(results_df)
        self._rank_candidates()
        self._validate_robustness()

        logger.info(f"Found {len(self.candidates)} deployment candidates")

    def _filter_candidates(self, results_df: pd.DataFrame) -> None:
        """Filter candidates based on performance criteria."""
        filtered = results_df[
            (results_df["sharpe"] >= self.config.min_sharpe)
            & (results_df["max_drawdown"] <= self.config.max_drawdown)
            & (results_df["n_trades"] >= self.config.min_trades)
            & (results_df["cagr"] >= self.config.min_cagr)
        ].copy()

        for _, row in filtered.iterrows():
            candidate = StrategyCandidate(
                strategy_name="trend_breakout",
                parameters=self._extract_parameters(row),
                performance_metrics={
                    "sharpe": row["sharpe"],
                    "cagr": row["cagr"],
                    "max_drawdown": row["max_drawdown"],
                    "total_return": row.get("total_return", 0),
                    "n_trades": row["n_trades"],
                },
                robustness_metrics={},
                walk_forward_results=[],
                rank_score=0.0,
            )
            self.candidates.append(candidate)

    def _extract_parameters(self, row: pd.Series) -> dict[str, Any]:
        """Extract strategy parameters from result row."""
        params = {}
        for col in row.index:
            if col.startswith("param_"):
                param_name = col.replace("param_", "")
                params[param_name] = row[col]
        return params

    def _rank_candidates(self) -> None:
        """Rank candidates using composite scoring."""
        for candidate in self.candidates:
            # Composite score based on multiple metrics
            sharpe_score = min(candidate.performance_metrics["sharpe"] / 2.0, 1.0)
            cagr_score = min(candidate.performance_metrics["cagr"] / 0.5, 1.0)
            drawdown_score = 1.0 - (candidate.performance_metrics["max_drawdown"] / 0.2)
            trade_score = min(candidate.performance_metrics["n_trades"] / 100, 1.0)

            # Weighted composite score
            candidate.rank_score = (
                0.4 * sharpe_score + 0.3 * cagr_score + 0.2 * drawdown_score + 0.1 * trade_score
            )

        # Sort by rank score
        self.candidates.sort(key=lambda x: x.rank_score, reverse=True)

    def _validate_robustness(self) -> None:
        """Validate strategy robustness using walk-forward results."""
        for candidate in self.candidates:
            # Check if we have walk-forward data
            if hasattr(candidate, "walk_forward_results") and candidate.walk_forward_results:
                sharpe_values = [wf["sharpe"] for wf in candidate.walk_forward_results]
                sharpe_std = pd.Series(sharpe_values).std()

                candidate.robustness_metrics = {
                    "sharpe_std": sharpe_std,
                    "sharpe_mean": pd.Series(sharpe_values).mean(),
                    "n_windows": len(sharpe_values),
                }

                # Mark as deployment ready if robust
                candidate.deployment_ready = (
                    len(sharpe_values) >= self.config.min_walk_forward_windows
                    and sharpe_std <= self.config.max_sharpe_std
                )
            else:
                # No walk-forward data, use basic criteria
                candidate.deployment_ready = candidate.rank_score >= 0.7

    def validate_strategy(self, candidate: StrategyCandidate) -> bool:
        """Validate a strategy candidate with recent data."""
        logger.info(f"Validating strategy: {candidate.parameters}")

        try:
            # Create strategy
            strategy = TrendBreakoutStrategy(TrendBreakoutParams(**candidate.parameters))

            # Create portfolio rules
            rules = PortfolioRules(
                per_trade_risk_pct=self.config.risk_per_strategy,
                atr_k=candidate.parameters.get("atr_k", 2.0),
                max_positions=self.config.max_positions,
                cost_bps=5.0,
            )

            # Run validation backtest on recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.validation_period_days)

            validation_results = []
            for symbol in self.config.symbols[:3]:  # Test on subset
                result = run_backtest(
                    symbol=symbol,
                    start=start_date,
                    end=end_date,
                    strategy=strategy,
                    rules=rules,
                    return_summary=True,
                    quiet_mode=True,
                    write_portfolio_csv=False,
                    write_trades_csv=False,
                    write_summary_csv=False,
                    make_plot=False,
                )
                if result:
                    validation_results.append(result["summary"])

            if not validation_results:
                logger.warning("No validation results obtained")
                return False

            # Aggregate validation metrics
            avg_sharpe = sum(r.get("sharpe", 0) for r in validation_results) / len(
                validation_results
            )

            logger.info(f"Validation Sharpe: {avg_sharpe:.3f}")
            return avg_sharpe >= self.config.min_validation_sharpe

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def deploy_strategies(self) -> list[dict[str, Any]]:
        """Deploy the best strategies to paper trading."""
        deployed = []

        # Get top candidates
        top_candidates = [
            c
            for c in self.candidates
            if c.deployment_ready and len(deployed) < self.config.max_concurrent_strategies
        ]

        for candidate in top_candidates:
            if self.validate_strategy(candidate):
                try:
                    deployment = self._deploy_single_strategy(candidate)
                    deployed.append(deployment)
                    logger.info(f"Successfully deployed strategy: {candidate.parameters}")
                except Exception as e:
                    logger.error(f"Failed to deploy strategy: {e}")

        return deployed

    def _deploy_single_strategy(self, candidate: StrategyCandidate) -> dict[str, Any]:
        """Deploy a single strategy to paper trading."""
        # Create strategy
        strategy = TrendBreakoutStrategy(TrendBreakoutParams(**candidate.parameters))

        # Create portfolio rules
        rules = PortfolioRules(
            per_trade_risk_pct=self.config.risk_per_strategy,
            atr_k=candidate.parameters.get("atr_k", 2.0),
            max_positions=self.config.max_positions,
            cost_bps=5.0,
        )

        # Initialize broker
        config = get_config()
        broker = AlpacaPaperBroker(
            api_key=config.alpaca.api_key_id,
            secret_key=config.alpaca.api_secret_key,
            base_url=config.alpaca.paper_base_url,
        )

        # Create trading engine
        engine = LiveTradingEngine(
            broker=broker,
            strategy=strategy,
            rules=rules,
            symbols=self.config.symbols,
            rebalance_interval=self.config.rebalance_interval,
            max_positions=self.config.max_positions,
        )

        deployment = {
            "strategy": strategy,
            "rules": rules,
            "engine": engine,
            "candidate": candidate,
            "deployment_time": datetime.now(),
            "status": "deployed",
        }

        return deployment

    def generate_deployment_report(self, output_path: str) -> None:
        """Generate a deployment report."""
        report = {
            "deployment_config": self.config.dict(),
            "candidates_analyzed": len(self.candidates),
            "candidates_deployment_ready": sum(1 for c in self.candidates if c.deployment_ready),
            "strategies_deployed": len(self.deployed_strategies),
            "top_candidates": [
                {
                    "rank": i + 1,
                    "parameters": c.parameters,
                    "performance": c.performance_metrics,
                    "robustness": c.robustness_metrics,
                    "rank_score": c.rank_score,
                    "deployment_ready": c.deployment_ready,
                }
                for i, c in enumerate(self.candidates[:10])
            ],
            "deployed_strategies": [
                {
                    "parameters": d.get("candidate", {}).get("parameters", d.get("parameters", {})),
                    "performance": d.get("candidate", {}).get(
                        "performance_metrics", d.get("performance", {})
                    ),
                    "deployment_time": (
                        d.get("deployment_time", datetime.now()).isoformat()
                        if hasattr(d.get("deployment_time", datetime.now()), "isoformat")
                        else str(d.get("deployment_time", datetime.now()))
                    ),
                    "status": d.get("status", "unknown"),
                }
                for d in self.deployed_strategies
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Deployment report saved to {output_path}")


def run_deployment_pipeline(
    optimization_results_path: str, config: DeploymentConfig, output_dir: str = "data/deployment"
) -> None:
    """Run the complete deployment pipeline."""
    logger.info("Starting automated deployment pipeline")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    pipeline = DeploymentPipeline(config)

    # Load and analyze results
    pipeline.load_optimization_results(optimization_results_path)

    # Deploy strategies
    deployed = pipeline.deploy_strategies()

    # Generate report
    report_path = (
        Path(output_dir) / f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    pipeline.generate_deployment_report(str(report_path))

    logger.info(f"Deployment pipeline complete. Deployed {len(deployed)} strategies.")
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    # Example usage
    config = DeploymentConfig(
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        min_sharpe=1.2,
        max_drawdown=0.12,
        min_trades=30,
        deployment_budget=10000.0,
        max_concurrent_strategies=2,
    )

    run_deployment_pipeline(
        optimization_results_path="data/optimization/all_results.csv", config=config
    )
