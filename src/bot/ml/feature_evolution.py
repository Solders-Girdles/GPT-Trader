"""
Feature Evolution Tracking System
Phase 3, Week 6: ADAPT-025 to ADAPT-032
Feature importance tracking, obsolescence detection, and adaptive engineering
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)


class FeatureStatus(Enum):
    """Status of features in the system"""

    ACTIVE = "active"
    DECLINING = "declining"
    OBSOLETE = "obsolete"
    EMERGING = "emerging"
    STABLE = "stable"
    VOLATILE = "volatile"


class FeatureType(Enum):
    """Types of features"""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    ENGINEERED = "engineered"
    INTERACTION = "interaction"
    EMBEDDING = "embedding"


@dataclass
class FeatureMetadata:
    """Metadata for individual features"""

    name: str
    feature_type: FeatureType
    created_at: datetime

    # Importance metrics
    current_importance: float = 0.0
    avg_importance: float = 0.0
    importance_trend: float = 0.0
    importance_volatility: float = 0.0

    # Usage statistics
    usage_count: int = 0
    last_used: datetime | None = None
    models_using: set[str] = field(default_factory=set)

    # Performance metrics
    correlation_with_target: float = 0.0
    mutual_information: float = 0.0
    contribution_score: float = 0.0

    # Evolution tracking
    importance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    status: FeatureStatus = FeatureStatus.ACTIVE
    obsolescence_score: float = 0.0

    # Dependencies
    depends_on: list[str] = field(default_factory=list)
    derived_features: list[str] = field(default_factory=list)


@dataclass
class FeatureEvolutionReport:
    """Report on feature evolution"""

    timestamp: datetime
    total_features: int
    active_features: int
    obsolete_features: int
    emerging_features: int

    # Top features
    top_important: list[tuple[str, float]]
    top_declining: list[tuple[str, float]]
    top_emerging: list[tuple[str, float]]

    # Recommendations
    features_to_remove: list[str]
    features_to_add: list[str]
    features_to_monitor: list[str]

    # Statistics
    avg_importance: float
    importance_concentration: float  # Gini coefficient
    feature_stability: float


class FeatureImportanceTracker:
    """Track feature importance over time"""

    def __init__(self, window_size: int = 50, decline_threshold: float = 0.3):
        """
        Initialize importance tracker.

        Args:
            window_size: Window for rolling statistics
            decline_threshold: Threshold for declining importance
        """
        self.window_size = window_size
        self.decline_threshold = decline_threshold

        # Storage
        self.importance_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.timestamp_history: deque = deque(maxlen=window_size)

    def update_importance(
        self, feature_importance: dict[str, float], timestamp: datetime | None = None
    ) -> None:
        """
        Update feature importance.

        Args:
            feature_importance: Current importance scores
            timestamp: Timestamp of update
        """
        timestamp = timestamp or datetime.now()
        self.timestamp_history.append(timestamp)

        for feature, importance in feature_importance.items():
            self.importance_history[feature].append(importance)

    def calculate_importance_trend(self, feature: str) -> float:
        """
        Calculate importance trend.

        Args:
            feature: Feature name

        Returns:
            Trend coefficient (positive = increasing)
        """
        history = list(self.importance_history[feature])

        if len(history) < 2:
            return 0.0

        # Linear regression trend
        x = np.arange(len(history))
        y = np.array(history)

        # Handle edge cases
        if np.std(y) == 0:
            return 0.0

        trend = np.polyfit(x, y, 1)[0]

        # Normalize by mean importance
        mean_importance = np.mean(y)
        if mean_importance > 0:
            normalized_trend = trend / mean_importance
        else:
            normalized_trend = 0.0

        return normalized_trend

    def calculate_importance_volatility(self, feature: str) -> float:
        """
        Calculate importance volatility.

        Args:
            feature: Feature name

        Returns:
            Volatility score
        """
        history = list(self.importance_history[feature])

        if len(history) < 2:
            return 0.0

        # Calculate coefficient of variation
        mean_importance = np.mean(history)
        std_importance = np.std(history)

        if mean_importance > 0:
            volatility = std_importance / mean_importance
        else:
            volatility = 0.0

        return volatility

    def detect_declining_features(self) -> list[tuple[str, float]]:
        """
        Detect features with declining importance.

        Returns:
            List of (feature, decline_score) tuples
        """
        declining = []

        for feature, history in self.importance_history.items():
            if len(history) < self.window_size // 2:
                continue

            # Calculate trend
            trend = self.calculate_importance_trend(feature)

            # Check if declining
            if trend < -self.decline_threshold:
                # Calculate decline score
                recent_avg = np.mean(list(history)[-10:])
                historical_avg = np.mean(list(history))

                if historical_avg > 0:
                    decline_score = 1 - (recent_avg / historical_avg)
                else:
                    decline_score = 0.0

                declining.append((feature, decline_score))

        return sorted(declining, key=lambda x: x[1], reverse=True)


class FeatureObsolescenceDetector:
    """Detect obsolete features"""

    def __init__(self, obsolescence_threshold: float = 0.7, min_history: int = 20):
        """
        Initialize obsolescence detector.

        Args:
            obsolescence_threshold: Threshold for marking obsolete
            min_history: Minimum history required
        """
        self.obsolescence_threshold = obsolescence_threshold
        self.min_history = min_history

    def calculate_obsolescence_score(
        self, feature_metadata: FeatureMetadata, current_time: datetime | None = None
    ) -> float:
        """
        Calculate obsolescence score for feature.

        Args:
            feature_metadata: Feature metadata
            current_time: Current timestamp

        Returns:
            Obsolescence score (0-1, higher = more obsolete)
        """
        current_time = current_time or datetime.now()
        scores = []

        # 1. Importance decline
        if len(feature_metadata.importance_history) >= self.min_history:
            recent_importance = np.mean(list(feature_metadata.importance_history)[-10:])
            historical_importance = np.mean(list(feature_metadata.importance_history))

            if historical_importance > 0:
                importance_ratio = recent_importance / historical_importance
                importance_score = 1 - min(importance_ratio, 1.0)
            else:
                importance_score = 1.0

            scores.append(importance_score * 0.4)  # 40% weight

        # 2. Usage recency
        if feature_metadata.last_used:
            days_since_use = (current_time - feature_metadata.last_used).days
            recency_score = min(days_since_use / 30, 1.0)  # Max at 30 days
            scores.append(recency_score * 0.3)  # 30% weight
        else:
            scores.append(0.3)  # Never used

        # 3. Model coverage
        if feature_metadata.models_using:
            # Assume we want features used by multiple models
            coverage_score = 1 - min(len(feature_metadata.models_using) / 3, 1.0)
            scores.append(coverage_score * 0.2)  # 20% weight
        else:
            scores.append(0.2)  # Not used by any model

        # 4. Correlation decay
        correlation_score = 1 - abs(feature_metadata.correlation_with_target)
        scores.append(correlation_score * 0.1)  # 10% weight

        # Combine scores
        obsolescence_score = sum(scores)

        return min(obsolescence_score, 1.0)

    def identify_obsolete_features(
        self, features_metadata: dict[str, FeatureMetadata]
    ) -> list[str]:
        """
        Identify obsolete features.

        Args:
            features_metadata: All feature metadata

        Returns:
            List of obsolete feature names
        """
        obsolete = []

        for feature_name, metadata in features_metadata.items():
            obsolescence_score = self.calculate_obsolescence_score(metadata)
            metadata.obsolescence_score = obsolescence_score

            if obsolescence_score > self.obsolescence_threshold:
                obsolete.append(feature_name)
                metadata.status = FeatureStatus.OBSOLETE

        return obsolete


class AdaptiveFeatureEngineer:
    """Adaptive feature engineering based on evolution"""

    def __init__(self, max_features: int = 200, interaction_threshold: float = 0.3):
        """
        Initialize adaptive engineer.

        Args:
            max_features: Maximum number of features
            interaction_threshold: Threshold for interaction features
        """
        self.max_features = max_features
        self.interaction_threshold = interaction_threshold

        # Feature generation history
        self.generated_features: dict[str, dict] = {}
        self.feature_performance: dict[str, float] = {}

    def suggest_new_features(
        self, data: pd.DataFrame, target: pd.Series, current_features: list[str]
    ) -> list[dict[str, Any]]:
        """
        Suggest new features based on current performance.

        Args:
            data: Current feature data
            target: Target variable
            current_features: Current feature list

        Returns:
            List of suggested features
        """
        suggestions = []

        # 1. Polynomial features for top performers
        top_features = self._identify_top_features(data, target, current_features)

        for feature in top_features[:5]:
            if feature in data.columns and pd.api.types.is_numeric_dtype(data[feature]):
                suggestions.append(
                    {
                        "name": f"{feature}_squared",
                        "type": "polynomial",
                        "formula": f"{feature}**2",
                        "base_feature": feature,
                    }
                )
                suggestions.append(
                    {
                        "name": f"{feature}_log",
                        "type": "transform",
                        "formula": f"np.log1p(abs({feature}))",
                        "base_feature": feature,
                    }
                )

        # 2. Interaction features
        interactions = self._find_interactions(data, target, top_features)

        for (feat1, feat2), score in interactions[:5]:
            suggestions.append(
                {
                    "name": f"{feat1}_x_{feat2}",
                    "type": "interaction",
                    "formula": f"{feat1} * {feat2}",
                    "base_features": [feat1, feat2],
                    "interaction_score": score,
                }
            )

        # 3. Rolling statistics for temporal features
        temporal_features = self._identify_temporal_features(data)

        for feature in temporal_features[:3]:
            suggestions.append(
                {
                    "name": f"{feature}_rolling_mean_7",
                    "type": "rolling",
                    "formula": f"rolling_mean({feature}, 7)",
                    "base_feature": feature,
                }
            )
            suggestions.append(
                {
                    "name": f"{feature}_rolling_std_7",
                    "type": "rolling",
                    "formula": f"rolling_std({feature}, 7)",
                    "base_feature": feature,
                }
            )

        # 4. Ratio features
        ratio_pairs = self._find_ratio_candidates(data, target)

        for (num, denom), score in ratio_pairs[:3]:
            suggestions.append(
                {
                    "name": f"{num}_div_{denom}",
                    "type": "ratio",
                    "formula": f"{num} / ({denom} + 1e-8)",
                    "base_features": [num, denom],
                    "ratio_score": score,
                }
            )

        return suggestions

    def _identify_top_features(
        self, data: pd.DataFrame, target: pd.Series, features: list[str]
    ) -> list[str]:
        """Identify top performing features"""
        scores = {}

        for feature in features:
            if feature in data.columns:
                if pd.api.types.is_numeric_dtype(data[feature]):
                    # Use mutual information
                    mi_score = mutual_info_regression(
                        data[[feature]].fillna(0), target, random_state=42
                    )[0]
                    scores[feature] = mi_score

        # Sort by score
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [f for f, _ in sorted_features]

    def _find_interactions(
        self, data: pd.DataFrame, target: pd.Series, features: list[str]
    ) -> list[tuple[tuple[str, str], float]]:
        """Find promising interaction features"""
        interactions = []

        # Only check top features to limit combinations
        check_features = features[:10]

        for i, feat1 in enumerate(check_features):
            if feat1 not in data.columns:
                continue

            for feat2 in check_features[i + 1 :]:
                if feat2 not in data.columns:
                    continue

                if pd.api.types.is_numeric_dtype(data[feat1]) and pd.api.types.is_numeric_dtype(
                    data[feat2]
                ):
                    # Create interaction
                    interaction = data[feat1] * data[feat2]

                    # Calculate correlation with target
                    corr = abs(interaction.corr(target))

                    if corr > self.interaction_threshold:
                        interactions.append(((feat1, feat2), corr))

        return sorted(interactions, key=lambda x: x[1], reverse=True)

    def _identify_temporal_features(self, data: pd.DataFrame) -> list[str]:
        """Identify features that might benefit from rolling statistics"""
        temporal = []

        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Check for autocorrelation
                try:
                    series = data[col].fillna(method="ffill").fillna(0)
                    if len(series) > 10:
                        autocorr = series.autocorr(lag=1)
                        if abs(autocorr) > 0.3:  # Significant autocorrelation
                            temporal.append(col)
                except:
                    pass

        return temporal

    def _find_ratio_candidates(
        self, data: pd.DataFrame, target: pd.Series
    ) -> list[tuple[tuple[str, str], float]]:
        """Find candidates for ratio features"""
        ratios = []

        numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]

        # Only check a subset to limit combinations
        check_cols = numeric_cols[:15]

        for i, num in enumerate(check_cols):
            for denom in check_cols[i + 1 :]:
                # Avoid division by zero
                if (data[denom] != 0).any():
                    ratio = data[num] / (data[denom] + 1e-8)

                    # Check if ratio is informative
                    corr = abs(ratio.corr(target))

                    if corr > 0.2:  # Threshold for ratio features
                        ratios.append(((num, denom), corr))

        return sorted(ratios, key=lambda x: x[1], reverse=True)

    def create_feature(self, data: pd.DataFrame, feature_spec: dict[str, Any]) -> pd.Series:
        """
        Create a new feature based on specification.

        Args:
            data: Input data
            feature_spec: Feature specification

        Returns:
            New feature series
        """
        feature_type = feature_spec["type"]

        if feature_type == "polynomial":
            base = data[feature_spec["base_feature"]]
            if feature_spec["formula"].endswith("**2"):
                return base**2
            elif "log" in feature_spec["formula"]:
                return np.log1p(np.abs(base))

        elif feature_type == "interaction":
            feat1, feat2 = feature_spec["base_features"]
            return data[feat1] * data[feat2]

        elif feature_type == "rolling":
            base = data[feature_spec["base_feature"]]
            window = int(feature_spec["formula"].split(",")[1].strip(")"))
            if "mean" in feature_spec["formula"]:
                return base.rolling(window, min_periods=1).mean()
            elif "std" in feature_spec["formula"]:
                return base.rolling(window, min_periods=1).std()

        elif feature_type == "ratio":
            num, denom = feature_spec["base_features"]
            return data[num] / (data[denom] + 1e-8)

        return pd.Series()


class FeatureEvolutionSystem:
    """Complete feature evolution tracking system"""

    def __init__(self, max_features: int = 200, obsolescence_threshold: float = 0.7):
        """
        Initialize feature evolution system.

        Args:
            max_features: Maximum number of features
            obsolescence_threshold: Threshold for obsolescence
        """
        self.max_features = max_features

        # Components
        self.importance_tracker = FeatureImportanceTracker()
        self.obsolescence_detector = FeatureObsolescenceDetector(
            obsolescence_threshold=obsolescence_threshold
        )
        self.adaptive_engineer = AdaptiveFeatureEngineer(max_features=max_features)

        # Feature registry
        self.features_metadata: dict[str, FeatureMetadata] = {}

        # Evolution history
        self.evolution_reports: list[FeatureEvolutionReport] = []

    def register_feature(
        self, name: str, feature_type: FeatureType, metadata: FeatureMetadata | None = None
    ) -> None:
        """
        Register a new feature.

        Args:
            name: Feature name
            feature_type: Type of feature
            metadata: Optional metadata
        """
        if metadata is None:
            metadata = FeatureMetadata(
                name=name, feature_type=feature_type, created_at=datetime.now()
            )

        self.features_metadata[name] = metadata
        logger.info(f"Registered feature: {name}")

    def update_feature_importance(self, importance_scores: dict[str, float], model_id: str) -> None:
        """
        Update feature importance from model.

        Args:
            importance_scores: Feature importance scores
            model_id: Model identifier
        """
        # Update tracker
        self.importance_tracker.update_importance(importance_scores)

        # Update metadata
        for feature_name, importance in importance_scores.items():
            if feature_name in self.features_metadata:
                metadata = self.features_metadata[feature_name]

                # Update current importance
                metadata.current_importance = importance
                metadata.importance_history.append(importance)

                # Update average
                metadata.avg_importance = np.mean(list(metadata.importance_history))

                # Update trend
                metadata.importance_trend = self.importance_tracker.calculate_importance_trend(
                    feature_name
                )

                # Update volatility
                metadata.importance_volatility = (
                    self.importance_tracker.calculate_importance_volatility(feature_name)
                )

                # Update usage
                metadata.usage_count += 1
                metadata.last_used = datetime.now()
                metadata.models_using.add(model_id)

                # Update status
                self._update_feature_status(metadata)

    def _update_feature_status(self, metadata: FeatureMetadata) -> None:
        """Update feature status based on metrics"""
        # Check trend
        if metadata.importance_trend > 0.2:
            metadata.status = FeatureStatus.EMERGING
        elif metadata.importance_trend < -0.2:
            metadata.status = FeatureStatus.DECLINING
        elif metadata.importance_volatility > 0.5:
            metadata.status = FeatureStatus.VOLATILE
        elif metadata.obsolescence_score > 0.7:
            metadata.status = FeatureStatus.OBSOLETE
        else:
            metadata.status = FeatureStatus.STABLE

    def analyze_feature_evolution(self) -> FeatureEvolutionReport:
        """
        Analyze current feature evolution.

        Returns:
            Evolution report
        """
        # Count features by status
        status_counts = defaultdict(int)
        for metadata in self.features_metadata.values():
            status_counts[metadata.status] += 1

        # Find top features
        top_important = sorted(
            [(m.name, m.current_importance) for m in self.features_metadata.values()],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        # Find declining features
        declining = self.importance_tracker.detect_declining_features()

        # Find emerging features
        emerging = [
            (m.name, m.importance_trend)
            for m in self.features_metadata.values()
            if m.status == FeatureStatus.EMERGING
        ]
        emerging = sorted(emerging, key=lambda x: x[1], reverse=True)[:10]

        # Identify obsolete features
        obsolete = self.obsolescence_detector.identify_obsolete_features(self.features_metadata)

        # Calculate statistics
        all_importance = [m.current_importance for m in self.features_metadata.values()]
        avg_importance = np.mean(all_importance) if all_importance else 0

        # Calculate Gini coefficient for concentration
        importance_concentration = self._calculate_gini(all_importance)

        # Calculate stability
        volatilities = [m.importance_volatility for m in self.features_metadata.values()]
        feature_stability = 1 - np.mean(volatilities) if volatilities else 0

        # Create report
        report = FeatureEvolutionReport(
            timestamp=datetime.now(),
            total_features=len(self.features_metadata),
            active_features=status_counts[FeatureStatus.ACTIVE]
            + status_counts[FeatureStatus.STABLE],
            obsolete_features=status_counts[FeatureStatus.OBSOLETE],
            emerging_features=status_counts[FeatureStatus.EMERGING],
            top_important=top_important,
            top_declining=declining[:10],
            top_emerging=emerging,
            features_to_remove=obsolete,
            features_to_add=[],  # Will be populated by suggest_features
            features_to_monitor=[f for f, _ in declining[:5]],
            avg_importance=avg_importance,
            importance_concentration=importance_concentration,
            feature_stability=feature_stability,
        )

        self.evolution_reports.append(report)
        return report

    def _calculate_gini(self, values: list[float]) -> float:
        """Calculate Gini coefficient for concentration"""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)

        return (2 * np.sum(np.arange(1, n + 1) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n

    def suggest_feature_updates(self, data: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
        """
        Suggest feature updates based on evolution.

        Args:
            data: Current feature data
            target: Target variable

        Returns:
            Suggestions dictionary
        """
        suggestions = {"remove": [], "add": [], "modify": [], "monitor": []}

        # Features to remove (obsolete)
        for feature_name, metadata in self.features_metadata.items():
            if metadata.status == FeatureStatus.OBSOLETE:
                suggestions["remove"].append(
                    {
                        "feature": feature_name,
                        "reason": "obsolete",
                        "obsolescence_score": metadata.obsolescence_score,
                    }
                )

        # Features to add (suggested by adaptive engineer)
        current_features = list(self.features_metadata.keys())
        new_features = self.adaptive_engineer.suggest_new_features(data, target, current_features)
        suggestions["add"] = new_features[:10]  # Limit to top 10

        # Features to monitor (declining or volatile)
        for feature_name, metadata in self.features_metadata.items():
            if metadata.status in [FeatureStatus.DECLINING, FeatureStatus.VOLATILE]:
                suggestions["monitor"].append(
                    {
                        "feature": feature_name,
                        "status": metadata.status.value,
                        "trend": metadata.importance_trend,
                        "volatility": metadata.importance_volatility,
                    }
                )

        return suggestions

    def prune_features(self, force: bool = False) -> list[str]:
        """
        Prune obsolete features.

        Args:
            force: Force pruning even if below threshold

        Returns:
            List of pruned features
        """
        pruned = []

        for feature_name, metadata in list(self.features_metadata.items()):
            should_prune = False

            if metadata.status == FeatureStatus.OBSOLETE:
                should_prune = True
            elif force and metadata.obsolescence_score > 0.5:
                should_prune = True

            if should_prune:
                del self.features_metadata[feature_name]
                pruned.append(feature_name)
                logger.info(f"Pruned feature: {feature_name}")

        return pruned

    def get_evolution_summary(self) -> str:
        """
        Get summary of feature evolution.

        Returns:
            Formatted summary
        """
        summary = []
        summary.append("=" * 60)
        summary.append("FEATURE EVOLUTION SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Total Features: {len(self.features_metadata)}")
        summary.append("")

        # Status breakdown
        status_counts = defaultdict(int)
        for metadata in self.features_metadata.values():
            status_counts[metadata.status] += 1

        summary.append("Feature Status:")
        for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
            summary.append(f"  {status.value}: {count}")
        summary.append("")

        # Top features
        top_features = sorted(
            self.features_metadata.values(), key=lambda x: x.current_importance, reverse=True
        )[:5]

        summary.append("Top 5 Features:")
        for i, metadata in enumerate(top_features, 1):
            summary.append(f"  {i}. {metadata.name}")
            summary.append(f"     Importance: {metadata.current_importance:.3f}")
            summary.append(f"     Trend: {metadata.importance_trend:+.3f}")
            summary.append(f"     Status: {metadata.status.value}")

        # Recent report
        if self.evolution_reports:
            latest = self.evolution_reports[-1]
            summary.append("")
            summary.append("Latest Evolution Metrics:")
            summary.append(f"  Feature Stability: {latest.feature_stability:.3f}")
            summary.append(f"  Importance Concentration: {latest.importance_concentration:.3f}")
            summary.append(f"  Emerging Features: {latest.emerging_features}")
            summary.append(f"  Obsolete Features: {latest.obsolete_features}")

        summary.append("")
        summary.append("=" * 60)

        return "\n".join(summary)


if __name__ == "__main__":
    # Example usage
    evolution_system = FeatureEvolutionSystem()

    # Register some features
    for i in range(20):
        evolution_system.register_feature(f"feature_{i}", FeatureType.NUMERICAL)

    # Simulate importance updates
    for _ in range(5):
        importance = {f"feature_{i}": np.random.random() for i in range(20)}
        evolution_system.update_feature_importance(importance, "model_1")

    # Analyze evolution
    report = evolution_system.analyze_feature_evolution()

    # Print summary
    print(evolution_system.get_evolution_summary())
