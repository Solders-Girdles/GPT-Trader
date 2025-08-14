"""
Performance monitoring and alerting system for deployed strategies.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any

from bot.config import get_config

settings = get_config()
from bot.exec.alpaca_paper import AlpacaPaperBroker
from bot.intelligence.selection_metrics import SelectionAccuracyCalculator
from bot.live.portfolio_manager import LivePortfolioManager
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PerformanceThresholds(BaseModel):
    """Performance thresholds for monitoring."""

    # Sharpe ratio thresholds
    min_sharpe: float = Field(0.5, description="Minimum acceptable Sharpe ratio")
    sharpe_decline_threshold: float = Field(0.3, description="Maximum Sharpe decline from baseline")

    # Drawdown thresholds
    max_drawdown: float = Field(0.15, description="Maximum acceptable drawdown")
    drawdown_increase_threshold: float = Field(
        0.05, description="Maximum drawdown increase from baseline"
    )

    # Return thresholds
    min_cagr: float = Field(0.05, description="Minimum acceptable CAGR")
    return_decline_threshold: float = Field(
        0.02, description="Maximum return decline from baseline"
    )

    # Volatility thresholds
    max_volatility: float = Field(0.25, description="Maximum acceptable volatility")

    # Trade frequency thresholds
    min_trades_per_month: int = Field(5, description="Minimum trades per month")
    max_trades_per_month: int = Field(100, description="Maximum trades per month")

    # Position thresholds
    max_position_concentration: float = Field(0.3, description="Maximum position concentration")
    min_diversification: int = Field(3, description="Minimum number of positions")

    # Transition quality thresholds (Phase 1/2)
    min_transition_smoothness: float | None = Field(
        None, description="Minimum acceptable transition smoothness (0-1). If None, disabled."
    )


class AlertConfig(BaseModel):
    """Configuration for alerts."""

    # Alert channels
    email_enabled: bool = Field(False, description="Enable email alerts")
    slack_enabled: bool = Field(False, description="Enable Slack alerts")
    webhook_enabled: bool = Field(False, description="Enable webhook alerts")

    # Alert settings
    alert_cooldown_hours: int = Field(24, description="Hours between repeated alerts")
    alert_recipients: list[str] = Field(default_factory=list, description="Alert recipients")

    # Webhook settings
    webhook_url: str | None = Field(None, description="Webhook URL for alerts")


class PerformanceAlert(BaseModel):
    """A performance alert."""

    strategy_id: str
    alert_type: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    current_value: float
    threshold_value: float
    baseline_value: float | None = None
    timestamp: datetime
    acknowledged: bool = False


class StrategyPerformance(BaseModel):
    """Performance metrics for a strategy."""

    strategy_id: str
    timestamp: datetime

    # Current metrics
    sharpe_ratio: float
    cagr: float
    max_drawdown: float
    volatility: float
    total_return: float

    # Trade metrics
    n_trades: int
    win_rate: float
    avg_trade_return: float

    # Position metrics
    n_positions: int
    position_concentration: float
    diversification_score: float

    # Risk metrics
    var_95: float
    expected_shortfall: float

    # Comparison to baseline
    sharpe_vs_baseline: float
    return_vs_baseline: float
    drawdown_vs_baseline: float


class PerformanceMonitor:
    """Monitors performance of deployed strategies."""

    def __init__(
        self,
        broker: AlpacaPaperBroker,
        thresholds: PerformanceThresholds,
        alert_config: AlertConfig,
    ) -> None:
        self.broker = broker
        self.thresholds = thresholds
        self.alert_config = alert_config

        # Performance tracking
        self.performance_history: dict[str, list[StrategyPerformance]] = {}
        self.baseline_metrics: dict[str, dict[str, Any]] = {}
        self.alerts: list[PerformanceAlert] = []

        # Portfolio manager with default rules for monitoring
        from bot.portfolio.allocator import PortfolioRules

        default_rules = PortfolioRules()
        self.portfolio_manager = LivePortfolioManager(broker, default_rules)

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_interval = 300  # 5 minutes

        # Phase 1 selection metrics tracking
        self._selection_metrics_calc = SelectionAccuracyCalculator(k=3)
        self._selection_metrics_history: list[dict[str, float]] = []
        # Turnover tracking (rolling)
        self._recent_turnover: list[float] = []

    def record_selection_metrics(
        self,
        predicted_ranks: list[str],
        actual_performance: dict[str, float],
        selected_strategies: list[str],
        optimal_strategies: list[str] | None = None,
    ) -> dict[str, float]:
        """Record selection/transition effectiveness metrics.

        Returns the computed metrics snapshot.
        """
        try:
            topk = float(
                self._selection_metrics_calc.calculate_top_k_accuracy(
                    predicted_ranks, actual_performance
                )
            )
            rho = float(
                self._selection_metrics_calc.calculate_rank_correlation(
                    predicted_ranks, actual_performance
                )
            )
            if optimal_strategies is None:
                # Default optimal: top-k by actual performance
                sorted_actual = sorted(actual_performance.items(), key=lambda x: x[1], reverse=True)
                optimal_strategies = [k for k, _ in sorted_actual[: self._selection_metrics_calc.k]]
            regret = float(
                self._selection_metrics_calc.calculate_regret(
                    selected_strategies, actual_performance, optimal_strategies
                )
            )
            snapshot = {"top_k_accuracy": topk, "rank_correlation": rho, "regret": regret}
            self._selection_metrics_history.append(snapshot)
            # keep last 100
            if len(self._selection_metrics_history) > 100:
                self._selection_metrics_history = self._selection_metrics_history[-100:]
            return snapshot
        except Exception:
            return {"top_k_accuracy": 0.0, "rank_correlation": 0.0, "regret": 0.0}

    async def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.is_monitoring:
            logger.warning("Performance monitoring already running")
            return

        self.is_monitoring = True
        logger.info("Starting performance monitoring")

        try:
            while self.is_monitoring:
                await self._monitoring_cycle()
                await asyncio.sleep(self.monitoring_interval)
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
            self.is_monitoring = False
            raise

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.is_monitoring = False
        logger.info("Stopped performance monitoring")

    async def _monitoring_cycle(self) -> None:
        """Run a single monitoring cycle."""
        try:
            # Refresh portfolio state
            await self.portfolio_manager.refresh_state()

            # Get current performance metrics
            performance = await self._calculate_current_performance()

            # Store performance history
            for strategy_perf in performance:
                strategy_id = strategy_perf.strategy_id
                if strategy_id not in self.performance_history:
                    self.performance_history[strategy_id] = []
                self.performance_history[strategy_id].append(strategy_perf)

                # Keep only last 30 days of history
                cutoff = datetime.now() - timedelta(days=30)
                self.performance_history[strategy_id] = [
                    p for p in self.performance_history[strategy_id] if p.timestamp > cutoff
                ]

            # Check for alerts
            await self._check_alerts(performance)

            # Update baseline metrics if needed
            self._update_baselines(performance)

        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")

    async def _calculate_current_performance(self) -> list[StrategyPerformance]:
        """Calculate current performance metrics for all strategies."""
        performance_list = []

        try:
            # Get account information
            account = self.broker.get_account()

            # Get current positions and convert to dict format
            positions_raw = self.broker.get_positions()
            positions = [
                {
                    "symbol": pos.symbol,
                    "qty": pos.qty,
                    "market_value": pos.market_value,
                    "current_price": pos.current_price,
                    "avg_price": pos.avg_price,
                }
                for pos in positions_raw
            ]

            # Get portfolio value history (last 30 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                account, positions, start_date, end_date
            )

            # Create performance object with proper field mapping
            performance = StrategyPerformance(
                strategy_id="portfolio",
                timestamp=datetime.now(),
                sharpe_ratio=portfolio_metrics.get("sharpe_ratio", 0.0),
                cagr=portfolio_metrics.get("cagr", 0.0),
                max_drawdown=portfolio_metrics.get("max_drawdown", 0.0),
                volatility=portfolio_metrics.get("volatility", 0.0),
                total_return=portfolio_metrics.get("total_return", 0.0),
                n_trades=int(portfolio_metrics.get("n_trades", 0)),
                win_rate=portfolio_metrics.get("win_rate", 0.0),
                avg_trade_return=portfolio_metrics.get("avg_trade_return", 0.0),
                n_positions=int(portfolio_metrics.get("n_positions", 0)),
                position_concentration=portfolio_metrics.get("position_concentration", 0.0),
                diversification_score=portfolio_metrics.get("diversification_score", 0.0),
                var_95=portfolio_metrics.get("var_95", 0.0),
                expected_shortfall=portfolio_metrics.get("expected_shortfall", 0.0),
                sharpe_vs_baseline=portfolio_metrics.get("sharpe_vs_baseline", 0.0),
                return_vs_baseline=portfolio_metrics.get("return_vs_baseline", 0.0),
                drawdown_vs_baseline=portfolio_metrics.get("drawdown_vs_baseline", 0.0),
            )

            performance_list.append(performance)

        except Exception as e:
            logger.error(f"Error calculating performance: {e}")

        return performance_list

    def _calculate_portfolio_metrics(
        self, account: Any, positions: list[dict], start_date: datetime, end_date: datetime
    ) -> dict[str, float]:
        """Calculate portfolio performance metrics."""
        try:
            # Get historical data for portfolio value calculation
            # This would need to be implemented based on your data structure
            # For now, return placeholder metrics

            # Calculate basic metrics
            total_value = account.portfolio_value
            cash = account.cash
            total_value - cash

            # Calculate position metrics
            n_positions = len(positions)
            position_values = [p.get("market_value", 0) for p in positions]
            total_position_value = sum(position_values)

            if total_position_value > 0:
                position_concentration = max(position_values) / total_position_value
            else:
                position_concentration = 0.0

            # Calculate diversification score (0-1, higher is better)
            if n_positions > 1:
                diversification_score = 1.0 - (position_concentration - 1.0 / n_positions)
            else:
                diversification_score = 0.0

            # Placeholder metrics (would need real historical data)
            return {
                "sharpe_ratio": 1.2,  # Placeholder
                "cagr": 0.15,  # Placeholder
                "max_drawdown": 0.08,  # Placeholder
                "volatility": 0.18,  # Placeholder
                "total_return": 0.12,  # Placeholder
                "n_trades": 25,  # Placeholder
                "win_rate": 0.65,  # Placeholder
                "avg_trade_return": 0.002,  # Placeholder
                "n_positions": n_positions,
                "position_concentration": position_concentration,
                "diversification_score": diversification_score,
                "var_95": 0.02,  # Placeholder
                "expected_shortfall": 0.03,  # Placeholder
                "sharpe_vs_baseline": 0.0,  # Will be calculated later
                "return_vs_baseline": 0.0,  # Will be calculated later
                "drawdown_vs_baseline": 0.0,  # Will be calculated later
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {
                "sharpe_ratio": 0.0,
                "cagr": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "total_return": 0.0,
                "n_trades": 0,
                "win_rate": 0.0,
                "avg_trade_return": 0.0,
                "n_positions": 0,
                "position_concentration": 0.0,
                "diversification_score": 0.0,
                "var_95": 0.0,
                "expected_shortfall": 0.0,
                "sharpe_vs_baseline": 0.0,
                "return_vs_baseline": 0.0,
                "drawdown_vs_baseline": 0.0,
            }

    async def _check_alerts(self, performance: list[StrategyPerformance]) -> None:
        """Check for performance alerts."""
        for perf in performance:
            strategy_id = perf.strategy_id

            # Check Sharpe ratio
            if perf.sharpe_ratio < self.thresholds.min_sharpe:
                await self._create_alert(
                    strategy_id,
                    "low_sharpe",
                    "medium",
                    f"Sharpe ratio {perf.sharpe_ratio:.3f} below threshold {self.thresholds.min_sharpe}",
                    perf.sharpe_ratio,
                    self.thresholds.min_sharpe,
                )

            # Check Sharpe decline
            if strategy_id in self.baseline_metrics:
                baseline_sharpe = self.baseline_metrics[strategy_id].get("sharpe_ratio", 0)
                sharpe_decline = baseline_sharpe - perf.sharpe_ratio
                if sharpe_decline > self.thresholds.sharpe_decline_threshold:
                    await self._create_alert(
                        strategy_id,
                        "sharpe_decline",
                        "high",
                        f"Sharpe ratio declined by {sharpe_decline:.3f} from baseline {baseline_sharpe:.3f}",
                        perf.sharpe_ratio,
                        baseline_sharpe,
                    )

            # Check drawdown
            if perf.max_drawdown > self.thresholds.max_drawdown:
                await self._create_alert(
                    strategy_id,
                    "high_drawdown",
                    "high",
                    f"Drawdown {perf.max_drawdown:.3f} above threshold {self.thresholds.max_drawdown}",
                    perf.max_drawdown,
                    self.thresholds.max_drawdown,
                )

            # Check position concentration
            if perf.position_concentration > self.thresholds.max_position_concentration:
                await self._create_alert(
                    strategy_id,
                    "high_concentration",
                    "medium",
                    f"Position concentration {perf.position_concentration:.3f} above threshold {self.thresholds.max_position_concentration}",
                    perf.position_concentration,
                    self.thresholds.max_position_concentration,
                )

            # Check diversification
            if perf.n_positions < self.thresholds.min_diversification:
                await self._create_alert(
                    strategy_id,
                    "low_diversification",
                    "low",
                    f"Only {perf.n_positions} positions, below minimum {self.thresholds.min_diversification}",
                    perf.n_positions,
                    self.thresholds.min_diversification,
                )

    async def _create_alert(
        self,
        strategy_id: str,
        alert_type: str,
        severity: str,
        message: str,
        current_value: float,
        threshold_value: float,
        baseline_value: float | None = None,
    ) -> None:
        """Create and send a performance alert."""
        # Check cooldown
        cutoff = datetime.now() - timedelta(hours=self.alert_config.alert_cooldown_hours)
        recent_alerts = [
            a
            for a in self.alerts
            if a.strategy_id == strategy_id and a.alert_type == alert_type and a.timestamp > cutoff
        ]

        if recent_alerts:
            logger.debug(f"Alert {alert_type} for {strategy_id} in cooldown period")
            return

        # Create alert
        alert = PerformanceAlert(
            strategy_id=strategy_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            baseline_value=baseline_value,
            timestamp=datetime.now(),
        )

        self.alerts.append(alert)

        # Send alert
        await self._send_alert(alert)

        logger.warning(f"Performance alert: {message}")

    async def _send_alert(self, alert: PerformanceAlert) -> None:
        """Send alert through configured channels."""
        alert_data = {
            "strategy_id": alert.strategy_id,
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "message": alert.message,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
            "baseline_value": alert.baseline_value,
            "timestamp": alert.timestamp.isoformat(),
        }

        # Send to webhook if configured
        if self.alert_config.webhook_enabled and self.alert_config.webhook_url:
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.alert_config.webhook_url, json=alert_data
                    ) as response:
                        if response.status != 200:
                            logger.error(f"Webhook alert failed: {response.status}")
            except Exception as e:
                logger.error(f"Error sending webhook alert: {e}")

        # Send email if configured
        if self.alert_config.email_enabled:
            await self._send_email_alert(alert)

        # Send Slack if configured
        if self.alert_config.slack_enabled:
            await self._send_slack_alert(alert)

    async def _send_email_alert(self, alert: PerformanceAlert) -> None:
        """Send email alert."""
        # Implementation would depend on your email service
        logger.info(f"Email alert would be sent: {alert.message}")

    async def _send_slack_alert(self, alert: PerformanceAlert) -> None:
        """Send Slack alert."""
        # Implementation would depend on your Slack integration
        logger.info(f"Slack alert would be sent: {alert.message}")

    def _update_baselines(self, performance: list[StrategyPerformance]) -> None:
        """Update baseline metrics for strategies."""
        for perf in performance:
            strategy_id = perf.strategy_id

            # Update baseline if not exists or if performance is better
            if strategy_id not in self.baseline_metrics:
                self.baseline_metrics[strategy_id] = {
                    "sharpe_ratio": perf.sharpe_ratio,
                    "cagr": perf.cagr,
                    "max_drawdown": perf.max_drawdown,
                    "total_return": perf.total_return,
                    "timestamp": perf.timestamp,
                }
            else:
                baseline = self.baseline_metrics[strategy_id]
                # Update if current performance is better
                if perf.sharpe_ratio > baseline.get("sharpe_ratio", 0):
                    baseline["sharpe_ratio"] = perf.sharpe_ratio
                    baseline["timestamp"] = perf.timestamp

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary for all strategies."""
        summary: dict[str, Any] = {
            "monitoring_active": self.is_monitoring,
            "strategies": {},
            "alerts": {
                "total": len(self.alerts),
                "unacknowledged": len([a for a in self.alerts if not a.acknowledged]),
                "by_severity": {},
            },
            "selection_metrics": (
                self._selection_metrics_history[-1] if self._selection_metrics_history else {}
            ),
            "recent_turnover": (self._recent_turnover[-1] if self._recent_turnover else 0.0),
        }

        # Turnover rolling statistics (mean and 95th percentile over recent window)
        try:
            if self._recent_turnover:
                data = list(self._recent_turnover)
                n = len(data)
                mean_val = float(sum(data) / n)
                # Compute p95 without numpy to avoid heavy deps here
                sorted_data = sorted(data)
                # p95 index: ceil(0.95*n) - 1, clipped
                import math

                idx = max(0, min(n - 1, int(math.ceil(0.95 * n) - 1)))
                p95_val = float(sorted_data[idx])
                summary["turnover_stats"] = {
                    "count": n,
                    "mean": mean_val,
                    "p95": p95_val,
                }
            else:
                summary["turnover_stats"] = {"count": 0, "mean": 0.0, "p95": 0.0}
        except Exception:
            # Be resilient; omit stats on failure
            summary["turnover_stats"] = {"count": 0, "mean": 0.0, "p95": 0.0}

        # Strategy performance
        for strategy_id, history in self.performance_history.items():
            if history:
                latest = history[-1]
                summary["strategies"][strategy_id] = {
                    "current_sharpe": latest.sharpe_ratio,
                    "current_cagr": latest.cagr,
                    "current_drawdown": latest.max_drawdown,
                    "n_positions": latest.n_positions,
                    "last_update": latest.timestamp.isoformat(),
                    "baseline_sharpe": self.baseline_metrics.get(strategy_id, {}).get(
                        "sharpe_ratio", 0
                    ),
                }

        # Alert summary
        for alert in self.alerts:
            severity = alert.severity
            if severity not in summary["alerts"]["by_severity"]:
                summary["alerts"]["by_severity"][severity] = 0
            summary["alerts"]["by_severity"][severity] += 1

        return summary

    # Lightweight turnover recorder used by orchestrator
    def record_turnover(self, turnover_value: float) -> None:
        try:
            t = float(turnover_value)
        except Exception:
            return
        self._recent_turnover.append(t)
        if len(self._recent_turnover) > 100:
            self._recent_turnover = self._recent_turnover[-100:]

    def save_performance_data(self, output_path: str) -> None:
        """Save performance data to file."""
        data = {
            "performance_history": {
                strategy_id: [p.dict() for p in history]
                for strategy_id, history in self.performance_history.items()
            },
            "baseline_metrics": self.baseline_metrics,
            "alerts": [a.dict() for a in self.alerts],
            "export_timestamp": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Performance data saved to {output_path}")


async def run_performance_monitor(
    broker: AlpacaPaperBroker, thresholds: PerformanceThresholds, alert_config: AlertConfig
) -> None:
    """Run the performance monitor."""
    monitor = PerformanceMonitor(broker, thresholds, alert_config)

    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Received interrupt, stopping monitor")
    finally:
        await monitor.stop_monitoring()


if __name__ == "__main__":
    # Example usage
    thresholds = PerformanceThresholds(
        min_sharpe=0.8,
        max_drawdown=0.12,
        min_cagr=0.08,
        sharpe_decline_threshold=0.3,
        drawdown_increase_threshold=0.05,
        return_decline_threshold=0.02,
        max_volatility=0.25,
        min_trades_per_month=5,
        max_trades_per_month=100,
        max_position_concentration=0.3,
        min_diversification=3,
        min_transition_smoothness=0.7,
    )

    alert_config = AlertConfig(
        email_enabled=False,
        slack_enabled=False,
        webhook_enabled=True,
        webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        alert_cooldown_hours=24,
    )

    # Initialize broker
    broker = AlpacaPaperBroker(
        api_key=settings.alpaca.api_key_id or "test_key",
        secret_key=settings.alpaca.api_secret_key or "test_secret",
        base_url=settings.alpaca.paper_base_url,
    )

    # Run monitor
    asyncio.run(run_performance_monitor(broker, thresholds, alert_config))
