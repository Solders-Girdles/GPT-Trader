"""
Auto-scaling System Based on Market Conditions

Dynamic resource allocation for trading systems:
- Market volatility-based scaling
- Volume-triggered resource allocation
- Event-driven scaling (earnings, news)
- Cost-optimized resource management
- Predictive scaling using ML
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class MarketCondition(Enum):
    """Market condition categories"""

    QUIET = "quiet"  # Low volatility, low volume
    NORMAL = "normal"  # Average conditions
    ACTIVE = "active"  # High activity
    VOLATILE = "volatile"  # High volatility
    EXTREME = "extreme"  # Extreme conditions (circuit breakers, crashes)


class ResourceType(Enum):
    """Types of resources to scale"""

    COMPUTE = "compute"  # CPU cores
    MEMORY = "memory"  # RAM
    STORAGE = "storage"  # Disk space
    NETWORK = "network"  # Bandwidth
    GPU = "gpu"  # GPU resources
    WORKERS = "workers"  # Worker processes/threads


class ScalingAction(Enum):
    """Scaling actions"""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ResourceConfig:
    """Resource configuration"""

    resource_type: ResourceType
    min_units: int
    max_units: int
    current_units: int
    unit_cost: float  # Cost per unit per hour
    scale_increment: int = 1
    cooldown_seconds: int = 300

    def can_scale_up(self) -> bool:
        return self.current_units + self.scale_increment <= self.max_units

    def can_scale_down(self) -> bool:
        return self.current_units - self.scale_increment >= self.min_units


@dataclass
class MarketMetrics:
    """Real-time market metrics"""

    timestamp: datetime
    volatility: float  # Annualized volatility
    volume: float  # Trading volume
    spread: float  # Bid-ask spread
    trade_frequency: float  # Trades per second
    price_change: float  # % change from open
    vix: float | None = None  # VIX if available
    news_sentiment: float | None = None  # -1 to 1

    def get_intensity_score(self) -> float:
        """Calculate market intensity score (0-1)"""
        # Normalize each metric to 0-1 range
        vol_score = min(self.volatility / 100, 1.0)  # Cap at 100% volatility
        volume_score = min(self.volume / 1e9, 1.0)  # Normalize to billions
        spread_score = min(self.spread / 0.01, 1.0)  # 1% spread = max
        freq_score = min(self.trade_frequency / 100, 1.0)  # 100 trades/sec = max
        change_score = min(abs(self.price_change) / 10, 1.0)  # 10% change = max

        # Weighted average
        weights = [0.3, 0.2, 0.1, 0.2, 0.2]
        scores = [vol_score, volume_score, spread_score, freq_score, change_score]

        intensity = sum(w * s for w, s in zip(weights, scores, strict=False))

        # Add VIX if available
        if self.vix is not None:
            vix_score = min(self.vix / 50, 1.0)  # VIX 50 = max
            intensity = intensity * 0.8 + vix_score * 0.2

        return intensity


@dataclass
class ScalingPolicy:
    """Policy for auto-scaling decisions"""

    name: str
    condition_threshold: float  # Market intensity threshold
    scale_up_threshold: float  # Utilization % to scale up
    scale_down_threshold: float  # Utilization % to scale down
    min_duration_seconds: int  # Minimum time in condition
    max_cost_per_hour: float  # Maximum cost constraint
    priority: int = 0  # Higher priority policies override
    enabled: bool = True

    def should_scale_up(self, utilization: float, intensity: float) -> bool:
        return intensity >= self.condition_threshold and utilization >= self.scale_up_threshold

    def should_scale_down(self, utilization: float, intensity: float) -> bool:
        return intensity < self.condition_threshold and utilization < self.scale_down_threshold


class MarketAnalyzer:
    """
    Analyzes market conditions for scaling decisions.

    Features:
    - Real-time volatility calculation
    - Volume analysis
    - Event detection
    - Condition classification
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.price_history = deque(maxlen=window_size)
        self.volume_history = deque(maxlen=window_size)
        self.metrics_history = deque(maxlen=window_size)
        self.logger = logging.getLogger(__name__)

    def update(self, price: float, volume: float, timestamp: datetime | None = None) -> None:
        """Update with new market data"""
        timestamp = timestamp or datetime.utcnow()

        self.price_history.append((timestamp, price))
        self.volume_history.append((timestamp, volume))

        # Calculate metrics if enough data
        if len(self.price_history) >= 20:
            metrics = self._calculate_metrics(timestamp)
            self.metrics_history.append(metrics)

    def _calculate_metrics(self, timestamp: datetime) -> MarketMetrics:
        """Calculate current market metrics"""
        prices = [p for _, p in self.price_history]
        volumes = [v for _, v in self.volume_history]

        # Calculate returns
        returns = np.diff(prices) / prices[:-1]

        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252 * 24 * 60)  # Assuming minute data

        # Volume
        total_volume = sum(volumes)

        # Spread estimate (simplified)
        price_range = max(prices) - min(prices)
        spread = price_range / np.mean(prices)

        # Trade frequency
        time_range = (self.price_history[-1][0] - self.price_history[0][0]).total_seconds()
        trade_frequency = len(prices) / max(1, time_range)

        # Price change
        price_change = (prices[-1] - prices[0]) / prices[0]

        return MarketMetrics(
            timestamp=timestamp,
            volatility=volatility * 100,  # Convert to percentage
            volume=total_volume,
            spread=spread,
            trade_frequency=trade_frequency,
            price_change=price_change * 100,  # Convert to percentage
        )

    def get_current_condition(self) -> MarketCondition:
        """Classify current market condition"""
        if not self.metrics_history:
            return MarketCondition.NORMAL

        latest = self.metrics_history[-1]
        intensity = latest.get_intensity_score()

        if intensity < 0.2:
            return MarketCondition.QUIET
        elif intensity < 0.4:
            return MarketCondition.NORMAL
        elif intensity < 0.6:
            return MarketCondition.ACTIVE
        elif intensity < 0.8:
            return MarketCondition.VOLATILE
        else:
            return MarketCondition.EXTREME

    def get_prediction(self, horizon_minutes: int = 30) -> dict[str, float]:
        """Predict future market conditions"""
        if len(self.metrics_history) < 10:
            return {"intensity": 0.5, "confidence": 0.0}

        # Simple trend-based prediction
        intensities = [m.get_intensity_score() for m in self.metrics_history]

        # Calculate trend
        x = np.arange(len(intensities))
        coeffs = np.polyfit(x, intensities, 1)
        trend = coeffs[0]

        # Project forward
        future_steps = horizon_minutes  # Assuming minute data
        predicted_intensity = intensities[-1] + trend * future_steps
        predicted_intensity = np.clip(predicted_intensity, 0, 1)

        # Confidence based on trend stability
        residuals = np.array(intensities) - np.polyval(coeffs, x)
        confidence = max(0, 1 - np.std(residuals))

        return {"intensity": predicted_intensity, "confidence": confidence, "trend": trend}


class ResourceManager:
    """
    Manages resource allocation and scaling.

    Features:
    - Multi-resource management
    - Cost optimization
    - Scaling history
    - Resource utilization tracking
    """

    def __init__(self) -> None:
        self.resources = {}  # type -> ResourceConfig
        self.scaling_history = deque(maxlen=1000)
        self.last_scale_time = {}  # resource_type -> timestamp
        self.logger = logging.getLogger(__name__)

        # Simulated utilization (in production, would query actual systems)
        self.utilization = {}

    def add_resource(self, config: ResourceConfig) -> None:
        """Add resource to manage"""
        self.resources[config.resource_type] = config
        self.utilization[config.resource_type] = 0.5  # Start at 50%
        self.last_scale_time[config.resource_type] = datetime.utcnow()

    def scale_resource(self, resource_type: ResourceType, action: ScalingAction) -> bool:
        """Scale a resource up or down"""
        if resource_type not in self.resources:
            return False

        config = self.resources[resource_type]

        # Check cooldown
        if resource_type in self.last_scale_time:
            elapsed = (datetime.utcnow() - self.last_scale_time[resource_type]).total_seconds()
            if elapsed < config.cooldown_seconds:
                self.logger.info(
                    f"Resource {resource_type.value} in cooldown ({elapsed:.0f}s/{config.cooldown_seconds}s)"
                )
                return False

        # Perform scaling
        if action == ScalingAction.SCALE_UP:
            if config.can_scale_up():
                old_units = config.current_units
                config.current_units += config.scale_increment
                self._record_scaling(resource_type, action, old_units, config.current_units)
                self.last_scale_time[resource_type] = datetime.utcnow()
                self.logger.info(
                    f"Scaled up {resource_type.value}: {old_units} -> {config.current_units}"
                )
                return True

        elif action == ScalingAction.SCALE_DOWN:
            if config.can_scale_down():
                old_units = config.current_units
                config.current_units -= config.scale_increment
                self._record_scaling(resource_type, action, old_units, config.current_units)
                self.last_scale_time[resource_type] = datetime.utcnow()
                self.logger.info(
                    f"Scaled down {resource_type.value}: {old_units} -> {config.current_units}"
                )
                return True

        return False

    def _record_scaling(
        self, resource_type: ResourceType, action: ScalingAction, old_units: int, new_units: int
    ) -> None:
        """Record scaling action"""
        self.scaling_history.append(
            {
                "timestamp": datetime.utcnow(),
                "resource_type": resource_type.value,
                "action": action.value,
                "old_units": old_units,
                "new_units": new_units,
            }
        )

    def get_utilization(self, resource_type: ResourceType) -> float:
        """Get current resource utilization"""
        # In production, would query actual metrics
        # For demo, simulate based on units
        if resource_type in self.utilization:
            # Add some random variation
            base = self.utilization[resource_type]
            variation = np.random.uniform(-0.1, 0.1)
            return np.clip(base + variation, 0, 1)
        return 0.5

    def update_utilization(self, resource_type: ResourceType, value: float) -> None:
        """Update resource utilization"""
        self.utilization[resource_type] = np.clip(value, 0, 1)

    def get_total_cost(self) -> float:
        """Calculate total hourly cost"""
        total = 0
        for config in self.resources.values():
            total += config.current_units * config.unit_cost
        return total

    def get_resource_summary(self) -> dict[str, Any]:
        """Get summary of all resources"""
        return {
            resource_type.value: {
                "current_units": config.current_units,
                "min_units": config.min_units,
                "max_units": config.max_units,
                "utilization": self.get_utilization(resource_type),
                "hourly_cost": config.current_units * config.unit_cost,
            }
            for resource_type, config in self.resources.items()
        }


class AutoScaler:
    """
    Main auto-scaling orchestrator.

    Features:
    - Policy-based scaling
    - Predictive scaling
    - Cost optimization
    - Multi-resource coordination
    """

    def __init__(self) -> None:
        self.market_analyzer = MarketAnalyzer()
        self.resource_manager = ResourceManager()
        self.policies = []  # List of ScalingPolicy
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.scaling_thread = None

        # Metrics
        self.scaling_decisions = deque(maxlen=1000)

        # Initialize default resources
        self._init_default_resources()

        # Initialize default policies
        self._init_default_policies()

    def _init_default_resources(self) -> None:
        """Initialize default resource configurations"""
        default_resources = [
            ResourceConfig(
                resource_type=ResourceType.COMPUTE,
                min_units=2,
                max_units=16,
                current_units=4,
                unit_cost=0.10,
                scale_increment=2,
            ),
            ResourceConfig(
                resource_type=ResourceType.MEMORY,
                min_units=4,
                max_units=64,
                current_units=8,
                unit_cost=0.05,
                scale_increment=4,
            ),
            ResourceConfig(
                resource_type=ResourceType.WORKERS,
                min_units=1,
                max_units=10,
                current_units=2,
                unit_cost=0.20,
                scale_increment=1,
            ),
        ]

        for resource in default_resources:
            self.resource_manager.add_resource(resource)

    def _init_default_policies(self) -> None:
        """Initialize default scaling policies"""
        default_policies = [
            ScalingPolicy(
                name="quiet_market",
                condition_threshold=0.2,
                scale_up_threshold=0.8,
                scale_down_threshold=0.3,
                min_duration_seconds=300,
                max_cost_per_hour=10.0,
                priority=1,
            ),
            ScalingPolicy(
                name="normal_market",
                condition_threshold=0.4,
                scale_up_threshold=0.7,
                scale_down_threshold=0.4,
                min_duration_seconds=180,
                max_cost_per_hour=20.0,
                priority=2,
            ),
            ScalingPolicy(
                name="active_market",
                condition_threshold=0.6,
                scale_up_threshold=0.6,
                scale_down_threshold=0.5,
                min_duration_seconds=60,
                max_cost_per_hour=50.0,
                priority=3,
            ),
            ScalingPolicy(
                name="extreme_market",
                condition_threshold=0.8,
                scale_up_threshold=0.5,
                scale_down_threshold=0.7,
                min_duration_seconds=30,
                max_cost_per_hour=100.0,
                priority=4,
            ),
        ]

        self.policies = default_policies

    def start(self) -> None:
        """Start auto-scaling"""
        if self.is_running:
            return

        self.is_running = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()

        self.logger.info("Auto-scaler started")

    def stop(self) -> None:
        """Stop auto-scaling"""
        self.is_running = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5)

        self.logger.info("Auto-scaler stopped")

    def _scaling_loop(self) -> None:
        """Main scaling decision loop"""
        condition_start_time = {}  # condition -> start_time

        while self.is_running:
            try:
                # Get current market condition
                condition = self.market_analyzer.get_current_condition()

                # Track condition duration
                if condition not in condition_start_time:
                    condition_start_time[condition] = datetime.utcnow()

                condition_duration = (
                    datetime.utcnow() - condition_start_time[condition]
                ).total_seconds()

                # Get market intensity
                if self.market_analyzer.metrics_history:
                    intensity = self.market_analyzer.metrics_history[-1].get_intensity_score()
                else:
                    intensity = 0.5

                # Make scaling decisions for each resource
                for resource_type in self.resource_manager.resources:
                    decision = self._make_scaling_decision(
                        resource_type, intensity, condition_duration
                    )

                    if decision != ScalingAction.NO_ACTION:
                        success = self.resource_manager.scale_resource(resource_type, decision)

                        self._record_decision(resource_type, decision, intensity, success)

                # Predictive scaling
                prediction = self.market_analyzer.get_prediction(30)
                if prediction["confidence"] > 0.7:
                    self._handle_predictive_scaling(prediction)

                # Sleep before next check
                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Scaling loop error: {e}")
                time.sleep(30)

    def _make_scaling_decision(
        self, resource_type: ResourceType, intensity: float, condition_duration: float
    ) -> ScalingAction:
        """Make scaling decision for a resource"""
        utilization = self.resource_manager.get_utilization(resource_type)

        # Find applicable policy
        applicable_policy = None
        for policy in sorted(self.policies, key=lambda p: p.priority, reverse=True):
            if not policy.enabled:
                continue

            if intensity >= policy.condition_threshold:
                if condition_duration >= policy.min_duration_seconds:
                    applicable_policy = policy
                    break

        if not applicable_policy:
            return ScalingAction.NO_ACTION

        # Check cost constraint
        current_cost = self.resource_manager.get_total_cost()
        if current_cost >= applicable_policy.max_cost_per_hour:
            return ScalingAction.SCALE_DOWN if utilization < 0.5 else ScalingAction.NO_ACTION

        # Make decision based on policy
        if applicable_policy.should_scale_up(utilization, intensity):
            return ScalingAction.SCALE_UP
        elif applicable_policy.should_scale_down(utilization, intensity):
            return ScalingAction.SCALE_DOWN
        else:
            return ScalingAction.NO_ACTION

    def _handle_predictive_scaling(self, prediction: dict[str, float]) -> None:
        """Handle predictive scaling"""
        predicted_intensity = prediction["intensity"]

        # Pre-scale if expecting high intensity
        if predicted_intensity > 0.7:
            self.logger.info(f"Predictive scaling: expecting intensity {predicted_intensity:.2f}")

            # Pre-emptively scale up critical resources
            for resource_type in [ResourceType.COMPUTE, ResourceType.WORKERS]:
                utilization = self.resource_manager.get_utilization(resource_type)
                if utilization > 0.5:  # Only if already moderate utilization
                    self.resource_manager.scale_resource(resource_type, ScalingAction.SCALE_UP)

    def _record_decision(
        self, resource_type: ResourceType, decision: ScalingAction, intensity: float, success: bool
    ) -> None:
        """Record scaling decision"""
        self.scaling_decisions.append(
            {
                "timestamp": datetime.utcnow(),
                "resource_type": resource_type.value,
                "decision": decision.value,
                "intensity": intensity,
                "success": success,
            }
        )

    def update_market_data(self, price: float, volume: float) -> None:
        """Update with new market data"""
        self.market_analyzer.update(price, volume)

        # Simulate utilization changes based on market activity
        if self.market_analyzer.metrics_history:
            intensity = self.market_analyzer.metrics_history[-1].get_intensity_score()

            # Update utilization based on intensity
            for resource_type in self.resource_manager.resources:
                # Higher intensity = higher utilization
                base_util = 0.3 + intensity * 0.5
                self.resource_manager.update_utilization(resource_type, base_util)

    def get_status(self) -> dict[str, Any]:
        """Get auto-scaler status"""
        condition = self.market_analyzer.get_current_condition()

        return {
            "market_condition": condition.value,
            "market_intensity": (
                self.market_analyzer.metrics_history[-1].get_intensity_score()
                if self.market_analyzer.metrics_history
                else 0
            ),
            "resources": self.resource_manager.get_resource_summary(),
            "total_cost_per_hour": self.resource_manager.get_total_cost(),
            "recent_decisions": list(self.scaling_decisions)[-10:],
            "prediction": self.market_analyzer.get_prediction(),
        }


def demo_autoscaling() -> None:
    """Demo auto-scaling system"""
    print("🚀 Auto-scaling Demo")
    print("=" * 50)

    # Create auto-scaler
    scaler = AutoScaler()

    # Start auto-scaling
    scaler.start()
    print("📊 Auto-scaler started")

    # Simulate market data
    print("\n📈 Simulating market conditions...")

    for i in range(20):
        # Generate market data with varying intensity
        if i < 5:
            # Quiet market
            price = 100 + np.random.normal(0, 0.1)
            volume = np.random.uniform(1000, 5000)
        elif i < 10:
            # Normal market
            price = 100 + np.random.normal(0, 0.5)
            volume = np.random.uniform(5000, 20000)
        elif i < 15:
            # Active market
            price = 100 + np.random.normal(0, 2.0)
            volume = np.random.uniform(20000, 100000)
        else:
            # Extreme market
            price = 100 + np.random.normal(0, 5.0)
            volume = np.random.uniform(100000, 500000)

        scaler.update_market_data(price, volume)

        if i % 5 == 0:
            status = scaler.get_status()
            print(f"\n⏱️  Step {i}:")
            print(
                f"   Market: {status['market_condition']} (intensity: {status['market_intensity']:.2f})"
            )
            print(f"   Cost: ${status['total_cost_per_hour']:.2f}/hour")

            print("   Resources:")
            for resource, data in status["resources"].items():
                print(
                    f"      {resource}: {data['current_units']} units ({data['utilization']:.1%} utilized)"
                )

        time.sleep(1)

    # Final status
    final_status = scaler.get_status()
    print("\n📊 Final Status:")
    print(f"   Market condition: {final_status['market_condition']}")
    print(f"   Total cost: ${final_status['total_cost_per_hour']:.2f}/hour")
    print(
        f"   Prediction: {final_status['prediction']['intensity']:.2f} intensity "
        f"({final_status['prediction']['confidence']:.1%} confidence)"
    )

    # Stop auto-scaler
    scaler.stop()
    print("\n✅ Auto-scaler stopped")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

    # Run demo
    demo_autoscaling()
