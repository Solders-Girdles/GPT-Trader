"""
Intelligent Alert System
Phase 3, Week 7: OPS-009 to OPS-016
Alert prioritization, deduplication, correlation, and fatigue prevention
"""

import hashlib
import logging
import re
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""

    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Action required soon
    MEDIUM = "medium"  # Should be investigated
    LOW = "low"  # Informational
    INFO = "info"  # No action needed


class AlertCategory(Enum):
    """Alert categories"""

    PERFORMANCE = "performance"
    RISK = "risk"
    SYSTEM = "system"
    DATA_QUALITY = "data_quality"
    MODEL = "model"
    TRADING = "trading"
    SECURITY = "security"
    COMPLIANCE = "compliance"


class AlertStatus(Enum):
    """Alert status"""

    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"


class DeduplicationStrategy(Enum):
    """Strategies for alert deduplication"""

    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    TIME_WINDOW = "time_window"
    PATTERN_BASED = "pattern_based"
    ML_BASED = "ml_based"


@dataclass
class Alert:
    """Individual alert instance"""

    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    category: AlertCategory

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    component: str = ""

    # Context
    context: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    # Status tracking
    status: AlertStatus = AlertStatus.NEW
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None

    # Correlation
    correlation_id: str | None = None
    parent_alert_id: str | None = None
    child_alert_ids: list[str] = field(default_factory=list)

    # Deduplication
    fingerprint: str | None = None
    occurrence_count: int = 1
    first_seen: datetime | None = None
    last_seen: datetime | None = None

    # Priority
    priority_score: float = 0.0
    business_impact: float = 0.0
    urgency_score: float = 0.0

    # Actions
    recommended_actions: list[str] = field(default_factory=list)
    automated_actions_taken: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "component": self.component,
            "context": self.context,
            "metrics": self.metrics,
            "tags": self.tags,
            "status": self.status.value,
            "priority_score": self.priority_score,
            "occurrence_count": self.occurrence_count,
        }


@dataclass
class AlertRule:
    """Rule for generating alerts"""

    rule_id: str
    name: str
    condition: Callable[[dict[str, Any]], bool]
    severity: AlertSeverity
    category: AlertCategory

    # Configuration
    enabled: bool = True
    threshold: float | None = None
    time_window: timedelta | None = None

    # Rate limiting
    max_alerts_per_hour: int = 10
    cooldown_period: timedelta | None = None

    # Metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: datetime | None = None
    trigger_count: int = 0


@dataclass
class AlertCorrelation:
    """Correlation between alerts"""

    correlation_id: str
    alert_ids: list[str]
    correlation_type: str  # temporal, causal, pattern
    confidence: float
    discovered_at: datetime

    # Correlation metadata
    time_window: timedelta | None = None
    common_attributes: dict[str, Any] = field(default_factory=dict)
    correlation_score: float = 0.0


class AlertPrioritizer:
    """Prioritize alerts based on multiple factors"""

    def __init__(self):
        """Initialize prioritizer"""
        self.severity_weights = {
            AlertSeverity.CRITICAL: 1.0,
            AlertSeverity.HIGH: 0.75,
            AlertSeverity.MEDIUM: 0.5,
            AlertSeverity.LOW: 0.25,
            AlertSeverity.INFO: 0.1,
        }

        self.category_weights = {
            AlertCategory.RISK: 1.0,
            AlertCategory.SECURITY: 0.95,
            AlertCategory.COMPLIANCE: 0.9,
            AlertCategory.TRADING: 0.85,
            AlertCategory.PERFORMANCE: 0.7,
            AlertCategory.MODEL: 0.65,
            AlertCategory.SYSTEM: 0.6,
            AlertCategory.DATA_QUALITY: 0.5,
        }

        # Business impact factors
        self.impact_factors = {
            "revenue_impact": 2.0,
            "user_impact": 1.5,
            "data_loss_risk": 1.8,
            "regulatory_risk": 1.7,
            "reputation_risk": 1.3,
        }

    def calculate_priority(self, alert: Alert) -> float:
        """
        Calculate alert priority score.

        Args:
            alert: Alert to prioritize

        Returns:
            Priority score (0-100)
        """
        # Base score from severity
        severity_score = self.severity_weights[alert.severity] * 30

        # Category importance
        category_score = self.category_weights[alert.category] * 20

        # Business impact
        impact_score = self._calculate_business_impact(alert) * 25

        # Urgency based on time sensitivity
        urgency_score = self._calculate_urgency(alert) * 15

        # Frequency/occurrence boost
        frequency_score = min(10, np.log1p(alert.occurrence_count) * 3)

        # Calculate total
        total_score = (
            severity_score + category_score + impact_score + urgency_score + frequency_score
        )

        # Normalize to 0-100
        priority_score = min(100, max(0, total_score))

        # Update alert
        alert.priority_score = priority_score
        alert.business_impact = impact_score
        alert.urgency_score = urgency_score

        return priority_score

    def _calculate_business_impact(self, alert: Alert) -> float:
        """Calculate business impact score"""
        impact_score = 0.0

        # Check for impact indicators in context
        for factor, weight in self.impact_factors.items():
            if factor in alert.context:
                value = alert.context[factor]
                if isinstance(value, (int, float)):
                    impact_score += min(1.0, value) * weight
                elif isinstance(value, bool) and value:
                    impact_score += weight

        # Check metrics for financial impact
        if "potential_loss" in alert.metrics:
            loss = alert.metrics["potential_loss"]
            if loss > 100000:
                impact_score += 1.0
            elif loss > 10000:
                impact_score += 0.7
            elif loss > 1000:
                impact_score += 0.4

        return min(1.0, impact_score / 5.0)  # Normalize

    def _calculate_urgency(self, alert: Alert) -> float:
        """Calculate urgency based on time factors"""
        urgency = 0.5  # Default urgency

        # Check for time-sensitive indicators
        if "deadline" in alert.context:
            deadline = alert.context["deadline"]
            if isinstance(deadline, datetime):
                time_remaining = (deadline - datetime.now()).total_seconds()
                if time_remaining < 3600:  # Less than 1 hour
                    urgency = 1.0
                elif time_remaining < 86400:  # Less than 1 day
                    urgency = 0.8

        # Market hours urgency
        if alert.category == AlertCategory.TRADING:
            now = datetime.now()
            if 9 <= now.hour < 16:  # Market hours
                urgency *= 1.5

        # Escalation based on age
        age = (datetime.now() - alert.timestamp).total_seconds()
        if alert.severity == AlertSeverity.CRITICAL and age > 300:  # 5 minutes
            urgency *= 1.3

        return min(1.0, urgency)


class AlertDeduplicator:
    """Deduplicate similar alerts"""

    def __init__(
        self,
        strategy: DeduplicationStrategy = DeduplicationStrategy.FUZZY_MATCH,
        time_window: timedelta = timedelta(minutes=5),
    ):
        """
        Initialize deduplicator.

        Args:
            strategy: Deduplication strategy
            time_window: Time window for deduplication
        """
        self.strategy = strategy
        self.time_window = time_window
        self.alert_cache: dict[str, Alert] = {}
        self.fingerprint_index: dict[str, list[str]] = defaultdict(list)

    def generate_fingerprint(self, alert: Alert) -> str:
        """
        Generate fingerprint for alert.

        Args:
            alert: Alert to fingerprint

        Returns:
            Fingerprint string
        """
        if self.strategy == DeduplicationStrategy.EXACT_MATCH:
            # Exact match on key fields
            key_parts = [
                alert.title,
                alert.severity.value,
                alert.category.value,
                alert.source,
                alert.component,
            ]
        elif self.strategy == DeduplicationStrategy.FUZZY_MATCH:
            # Fuzzy match - normalize and extract key terms
            key_parts = [
                self._normalize_text(alert.title),
                alert.severity.value,
                alert.category.value,
                alert.source,
            ]
        elif self.strategy == DeduplicationStrategy.PATTERN_BASED:
            # Extract patterns from message
            patterns = self._extract_patterns(alert.message)
            key_parts = patterns + [alert.category.value]
        else:
            # Time window based - include time bucket
            time_bucket = int(alert.timestamp.timestamp() / self.time_window.total_seconds())
            key_parts = [alert.title, alert.category.value, str(time_bucket)]

        # Generate hash
        fingerprint_str = "|".join(str(p) for p in key_parts)
        fingerprint = hashlib.md5(fingerprint_str.encode()).hexdigest()

        alert.fingerprint = fingerprint
        return fingerprint

    def is_duplicate(self, alert: Alert) -> tuple[bool, Alert | None]:
        """
        Check if alert is duplicate.

        Args:
            alert: Alert to check

        Returns:
            Tuple of (is_duplicate, original_alert)
        """
        fingerprint = self.generate_fingerprint(alert)

        # Check cache
        if fingerprint in self.fingerprint_index:
            # Get recent alerts with same fingerprint
            for alert_id in self.fingerprint_index[fingerprint]:
                cached_alert = self.alert_cache.get(alert_id)
                if cached_alert:
                    # Check if within time window
                    time_diff = alert.timestamp - cached_alert.timestamp
                    if time_diff <= self.time_window:
                        # Update occurrence count
                        cached_alert.occurrence_count += 1
                        cached_alert.last_seen = alert.timestamp
                        return True, cached_alert

        # Not a duplicate - add to cache
        self.alert_cache[alert.alert_id] = alert
        self.fingerprint_index[fingerprint].append(alert.alert_id)
        alert.first_seen = alert.timestamp
        alert.last_seen = alert.timestamp

        # Clean old entries
        self._clean_cache()

        return False, None

    def _normalize_text(self, text: str) -> str:
        """Normalize text for fuzzy matching"""
        # Remove numbers and special characters
        text = re.sub(r"[0-9]+", "NUM", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        # Convert to lowercase and remove extra spaces
        text = " ".join(text.lower().split())
        return text

    def _extract_patterns(self, text: str) -> list[str]:
        """Extract patterns from text"""
        patterns = []

        # Extract error codes
        error_codes = re.findall(r"[A-Z]{2,}-[0-9]{3,}", text)
        patterns.extend(error_codes)

        # Extract key phrases
        key_phrases = re.findall(r"(error|warning|failed|timeout|exception)", text, re.IGNORECASE)
        patterns.extend(key_phrases)

        return patterns

    def _clean_cache(self):
        """Clean old entries from cache"""
        cutoff_time = datetime.now() - self.time_window * 2

        # Remove old alerts
        to_remove = []
        for alert_id, alert in self.alert_cache.items():
            if alert.timestamp < cutoff_time:
                to_remove.append(alert_id)

        for alert_id in to_remove:
            alert = self.alert_cache.pop(alert_id)
            if alert.fingerprint in self.fingerprint_index:
                self.fingerprint_index[alert.fingerprint].remove(alert_id)
                if not self.fingerprint_index[alert.fingerprint]:
                    del self.fingerprint_index[alert.fingerprint]


class AlertCorrelator:
    """Correlate related alerts"""

    def __init__(self, correlation_window: timedelta = timedelta(minutes=10)):
        """
        Initialize correlator.

        Args:
            correlation_window: Time window for correlation
        """
        self.correlation_window = correlation_window
        self.alert_buffer: deque = deque(maxlen=1000)
        self.correlations: list[AlertCorrelation] = []

    def correlate_alerts(self, alerts: list[Alert]) -> list[AlertCorrelation]:
        """
        Find correlations between alerts.

        Args:
            alerts: Alerts to correlate

        Returns:
            List of correlations
        """
        correlations = []

        # Temporal correlation
        temporal_groups = self._find_temporal_correlations(alerts)
        for group in temporal_groups:
            correlation = AlertCorrelation(
                correlation_id=f"temporal_{datetime.now().timestamp()}",
                alert_ids=[a.alert_id for a in group],
                correlation_type="temporal",
                confidence=0.8,
                discovered_at=datetime.now(),
                time_window=self.correlation_window,
            )
            correlations.append(correlation)

        # Causal correlation
        causal_pairs = self._find_causal_correlations(alerts)
        for parent, children in causal_pairs:
            correlation = AlertCorrelation(
                correlation_id=f"causal_{parent.alert_id}",
                alert_ids=[parent.alert_id] + [c.alert_id for c in children],
                correlation_type="causal",
                confidence=0.7,
                discovered_at=datetime.now(),
            )
            correlations.append(correlation)

        # Pattern-based correlation
        pattern_groups = self._find_pattern_correlations(alerts)
        for group in pattern_groups:
            correlation = AlertCorrelation(
                correlation_id=f"pattern_{datetime.now().timestamp()}",
                alert_ids=[a.alert_id for a in group],
                correlation_type="pattern",
                confidence=0.6,
                discovered_at=datetime.now(),
            )
            correlations.append(correlation)

        self.correlations.extend(correlations)
        return correlations

    def _find_temporal_correlations(self, alerts: list[Alert]) -> list[list[Alert]]:
        """Find temporally correlated alerts"""
        groups = []
        sorted_alerts = sorted(alerts, key=lambda a: a.timestamp)

        current_group = []
        for alert in sorted_alerts:
            if not current_group:
                current_group.append(alert)
            else:
                time_diff = alert.timestamp - current_group[-1].timestamp
                if time_diff <= self.correlation_window:
                    current_group.append(alert)
                else:
                    if len(current_group) > 1:
                        groups.append(current_group)
                    current_group = [alert]

        if len(current_group) > 1:
            groups.append(current_group)

        return groups

    def _find_causal_correlations(self, alerts: list[Alert]) -> list[tuple[Alert, list[Alert]]]:
        """Find causal relationships between alerts"""
        causal_pairs = []

        # Look for parent-child relationships
        for i, alert1 in enumerate(alerts):
            children = []
            for j, alert2 in enumerate(alerts):
                if i != j:
                    # Check if alert2 could be caused by alert1
                    if self._is_likely_caused_by(alert2, alert1):
                        children.append(alert2)

            if children:
                causal_pairs.append((alert1, children))

        return causal_pairs

    def _is_likely_caused_by(self, effect: Alert, cause: Alert) -> bool:
        """Check if one alert likely caused another"""
        # Time ordering
        if effect.timestamp <= cause.timestamp:
            return False

        time_diff = (effect.timestamp - cause.timestamp).total_seconds()
        if time_diff > 600:  # More than 10 minutes
            return False

        # Category relationships
        causal_categories = {
            AlertCategory.SYSTEM: [AlertCategory.PERFORMANCE, AlertCategory.DATA_QUALITY],
            AlertCategory.DATA_QUALITY: [AlertCategory.MODEL, AlertCategory.TRADING],
            AlertCategory.MODEL: [AlertCategory.TRADING, AlertCategory.RISK],
        }

        if cause.category in causal_categories:
            if effect.category in causal_categories[cause.category]:
                return True

        # Component relationships
        if cause.component and effect.component:
            if cause.component in effect.context.get("upstream_components", []):
                return True

        return False

    def _find_pattern_correlations(self, alerts: list[Alert]) -> list[list[Alert]]:
        """Find pattern-based correlations"""
        groups = []

        # Group by similar patterns
        pattern_map = defaultdict(list)

        for alert in alerts:
            # Extract key patterns
            patterns = []

            # Similar error types
            if "error_type" in alert.context:
                patterns.append(f"error:{alert.context['error_type']}")

            # Similar components
            if alert.component:
                patterns.append(f"component:{alert.component}")

            # Similar metrics
            for metric, value in alert.metrics.items():
                if isinstance(value, (int, float)):
                    # Quantize metric values
                    quantized = int(value / 10) * 10
                    patterns.append(f"{metric}:{quantized}")

            # Add to pattern map
            pattern_key = "|".join(sorted(patterns))
            if pattern_key:
                pattern_map[pattern_key].append(alert)

        # Create groups from pattern map
        for pattern_alerts in pattern_map.values():
            if len(pattern_alerts) > 1:
                groups.append(pattern_alerts)

        return groups


class AlertFatiguePrevention:
    """Prevent alert fatigue through intelligent suppression"""

    def __init__(self, max_alerts_per_hour: int = 50, suppression_threshold: float = 0.3):
        """
        Initialize fatigue prevention.

        Args:
            max_alerts_per_hour: Maximum alerts per hour
            suppression_threshold: Threshold for suppression
        """
        self.max_alerts_per_hour = max_alerts_per_hour
        self.suppression_threshold = suppression_threshold

        # Tracking
        self.alert_history: deque = deque(maxlen=1000)
        self.suppressed_alerts: list[Alert] = []
        self.alert_rate: dict[str, list[datetime]] = defaultdict(list)

    def should_suppress(self, alert: Alert) -> bool:
        """
        Check if alert should be suppressed.

        Args:
            alert: Alert to check

        Returns:
            True if should be suppressed
        """
        # Check rate limits
        if self._exceeds_rate_limit(alert):
            return True

        # Check if low priority and high volume
        if alert.priority_score < 30 and self._is_high_volume():
            return True

        # Check for noise patterns
        if self._is_likely_noise(alert):
            return True

        # Check for alert storms
        if self._detect_alert_storm():
            # Only allow critical alerts through
            return alert.severity != AlertSeverity.CRITICAL

        return False

    def _exceeds_rate_limit(self, alert: Alert) -> bool:
        """Check if rate limit exceeded"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)

        # Clean old entries
        category_key = f"{alert.category.value}_{alert.severity.value}"
        self.alert_rate[category_key] = [t for t in self.alert_rate[category_key] if t > hour_ago]

        # Check limit
        if len(self.alert_rate[category_key]) >= self.max_alerts_per_hour:
            return True

        # Add current alert
        self.alert_rate[category_key].append(now)
        return False

    def _is_high_volume(self) -> bool:
        """Check if experiencing high alert volume"""
        now = datetime.now()
        recent_alerts = [
            a for a in self.alert_history if (now - a.timestamp).total_seconds() < 3600
        ]
        return len(recent_alerts) > self.max_alerts_per_hour * 0.8

    def _is_likely_noise(self, alert: Alert) -> bool:
        """Check if alert is likely noise"""
        # Low severity info alerts
        if alert.severity == AlertSeverity.INFO and alert.priority_score < 20:
            return True

        # Repeated low-priority alerts
        if alert.occurrence_count > 10 and alert.priority_score < 40:
            return True

        # Known noisy patterns
        noisy_patterns = ["connection reset", "timeout", "rate limit", "temporary failure"]

        message_lower = alert.message.lower()
        for pattern in noisy_patterns:
            if pattern in message_lower and alert.severity == AlertSeverity.LOW:
                return True

        return False

    def _detect_alert_storm(self) -> bool:
        """Detect alert storm condition"""
        now = datetime.now()

        # Count alerts in last 5 minutes
        five_min_ago = now - timedelta(minutes=5)
        recent_count = sum(1 for a in self.alert_history if a.timestamp > five_min_ago)

        # Storm if > 20 alerts in 5 minutes
        return recent_count > 20

    def apply_dynamic_thresholds(self, alerts: list[Alert]) -> list[Alert]:
        """
        Apply dynamic thresholds based on current conditions.

        Args:
            alerts: Alerts to filter

        Returns:
            Filtered alerts
        """
        # Calculate current noise level
        noise_level = self._calculate_noise_level()

        # Adjust threshold based on noise
        if noise_level > 0.7:
            # High noise - be more selective
            min_priority = 60
        elif noise_level > 0.4:
            # Moderate noise
            min_priority = 40
        else:
            # Low noise - allow more through
            min_priority = 20

        # Filter alerts
        filtered = []
        for alert in alerts:
            if alert.priority_score >= min_priority or alert.severity == AlertSeverity.CRITICAL:
                filtered.append(alert)
                self.alert_history.append(alert)
            else:
                alert.status = AlertStatus.SUPPRESSED
                self.suppressed_alerts.append(alert)

        return filtered

    def _calculate_noise_level(self) -> float:
        """Calculate current noise level (0-1)"""
        if len(self.alert_history) < 10:
            return 0.0

        recent_alerts = list(self.alert_history)[-50:]

        # Calculate metrics
        low_priority_ratio = sum(1 for a in recent_alerts if a.priority_score < 30) / len(
            recent_alerts
        )

        duplicate_ratio = sum(1 for a in recent_alerts if a.occurrence_count > 1) / len(
            recent_alerts
        )

        info_ratio = sum(1 for a in recent_alerts if a.severity == AlertSeverity.INFO) / len(
            recent_alerts
        )

        # Combine metrics
        noise_level = low_priority_ratio * 0.4 + duplicate_ratio * 0.3 + info_ratio * 0.3

        return min(1.0, noise_level)


class IntelligentAlertSystem:
    """Main intelligent alert system"""

    def __init__(
        self,
        enable_deduplication: bool = True,
        enable_correlation: bool = True,
        enable_fatigue_prevention: bool = True,
    ):
        """
        Initialize alert system.

        Args:
            enable_deduplication: Enable deduplication
            enable_correlation: Enable correlation
            enable_fatigue_prevention: Enable fatigue prevention
        """
        # Components
        self.prioritizer = AlertPrioritizer()
        self.deduplicator = AlertDeduplicator() if enable_deduplication else None
        self.correlator = AlertCorrelator() if enable_correlation else None
        self.fatigue_prevention = AlertFatiguePrevention() if enable_fatigue_prevention else None

        # Storage
        self.alerts: dict[str, Alert] = {}
        self.rules: dict[str, AlertRule] = {}
        self.alert_queue: deque = deque()

        # Handlers
        self.handlers: dict[AlertSeverity, list[Callable]] = defaultdict(list)

        # Statistics
        self.stats = {
            "total_alerts": 0,
            "suppressed_alerts": 0,
            "deduplicated_alerts": 0,
            "correlated_alerts": 0,
            "alerts_by_severity": defaultdict(int),
            "alerts_by_category": defaultdict(int),
        }

        # Configuration
        self.batch_size = 10
        self.processing_interval = 5  # seconds

        logger.info("Intelligent Alert System initialized")

    def create_alert(
        self, title: str, message: str, severity: AlertSeverity, category: AlertCategory, **kwargs
    ) -> Alert | None:
        """
        Create and process alert.

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            category: Alert category
            **kwargs: Additional alert attributes

        Returns:
            Created alert or None if suppressed
        """
        # Create alert
        alert = Alert(
            alert_id=f"alert_{datetime.now().timestamp()}_{hash(title)}",
            title=title,
            message=message,
            severity=severity,
            category=category,
            source=kwargs.get("source", ""),
            component=kwargs.get("component", ""),
            context=kwargs.get("context", {}),
            metrics=kwargs.get("metrics", {}),
            tags=kwargs.get("tags", []),
        )

        # Calculate priority
        self.prioritizer.calculate_priority(alert)

        # Check deduplication
        if self.deduplicator:
            is_duplicate, original = self.deduplicator.is_duplicate(alert)
            if is_duplicate:
                self.stats["deduplicated_alerts"] += 1
                logger.debug(f"Alert deduplicated: {title}")
                return original

        # Check fatigue prevention
        if self.fatigue_prevention:
            if self.fatigue_prevention.should_suppress(alert):
                alert.status = AlertStatus.SUPPRESSED
                self.stats["suppressed_alerts"] += 1
                logger.debug(f"Alert suppressed: {title}")
                return None

        # Store alert
        self.alerts[alert.alert_id] = alert
        self.alert_queue.append(alert)

        # Update stats
        self.stats["total_alerts"] += 1
        self.stats["alerts_by_severity"][severity] += 1
        self.stats["alerts_by_category"][category] += 1

        # Process immediately if critical
        if severity == AlertSeverity.CRITICAL:
            self._process_alert(alert)

        logger.info(f"Alert created: {alert.alert_id} - {title}")
        return alert

    def add_rule(self, rule: AlertRule) -> None:
        """
        Add alert rule.

        Args:
            rule: Alert rule to add
        """
        self.rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def check_rules(self, data: dict[str, Any]) -> list[Alert]:
        """
        Check rules against data.

        Args:
            data: Data to check

        Returns:
            Generated alerts
        """
        alerts = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            # Check rate limiting
            if rule.last_triggered:
                if rule.cooldown_period:
                    if datetime.now() - rule.last_triggered < rule.cooldown_period:
                        continue

            # Check condition
            try:
                if rule.condition(data):
                    # Create alert from rule
                    alert = self.create_alert(
                        title=rule.name,
                        message=f"Rule triggered: {rule.description}",
                        severity=rule.severity,
                        category=rule.category,
                        context={"rule_id": rule.rule_id, "data": data},
                    )

                    if alert:
                        alerts.append(alert)
                        rule.last_triggered = datetime.now()
                        rule.trigger_count += 1
            except Exception as e:
                logger.error(f"Error checking rule {rule.rule_id}: {e}")

        return alerts

    def process_alerts(self) -> list[Alert]:
        """
        Process queued alerts.

        Returns:
            Processed alerts
        """
        if not self.alert_queue:
            return []

        # Get batch of alerts
        batch = []
        for _ in range(min(self.batch_size, len(self.alert_queue))):
            batch.append(self.alert_queue.popleft())

        # Apply correlation
        if self.correlator and len(batch) > 1:
            correlations = self.correlator.correlate_alerts(batch)
            for correlation in correlations:
                self.stats["correlated_alerts"] += len(correlation.alert_ids)
                # Update alerts with correlation info
                for alert_id in correlation.alert_ids:
                    if alert_id in self.alerts:
                        self.alerts[alert_id].correlation_id = correlation.correlation_id

        # Apply fatigue prevention filtering
        if self.fatigue_prevention:
            batch = self.fatigue_prevention.apply_dynamic_thresholds(batch)

        # Process each alert
        processed = []
        for alert in batch:
            if alert.status != AlertStatus.SUPPRESSED:
                self._process_alert(alert)
                processed.append(alert)

        return processed

    def _process_alert(self, alert: Alert) -> None:
        """Process individual alert"""
        # Call registered handlers
        for handler in self.handlers.get(alert.severity, []):
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

        # Log alert
        logger.info(
            f"Processing alert: {alert.alert_id} - {alert.title} "
            f"[{alert.severity.value}] Priority: {alert.priority_score:.1f}"
        )

    def register_handler(self, severity: AlertSeverity, handler: Callable[[Alert], None]) -> None:
        """
        Register alert handler.

        Args:
            severity: Severity level to handle
            handler: Handler function
        """
        self.handlers[severity].append(handler)

    def get_active_alerts(
        self, severity: AlertSeverity | None = None, category: AlertCategory | None = None
    ) -> list[Alert]:
        """
        Get active alerts.

        Args:
            severity: Filter by severity
            category: Filter by category

        Returns:
            List of active alerts
        """
        active = []

        for alert in self.alerts.values():
            if alert.status in [
                AlertStatus.NEW,
                AlertStatus.ACKNOWLEDGED,
                AlertStatus.INVESTIGATING,
            ]:
                if severity and alert.severity != severity:
                    continue
                if category and alert.category != category:
                    continue
                active.append(alert)

        # Sort by priority
        return sorted(active, key=lambda a: a.priority_score, reverse=True)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge alert.

        Args:
            alert_id: Alert ID

        Returns:
            Success status
        """
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        return False

    def resolve_alert(self, alert_id: str, resolution: str = "") -> bool:
        """
        Resolve alert.

        Args:
            alert_id: Alert ID
            resolution: Resolution description

        Returns:
            Success status
        """
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            if resolution:
                alert.context["resolution"] = resolution
            logger.info(f"Alert resolved: {alert_id}")
            return True
        return False

    def get_statistics(self) -> dict[str, Any]:
        """
        Get system statistics.

        Returns:
            Statistics dictionary
        """
        stats = dict(self.stats)

        # Add current state
        stats["active_alerts"] = len(self.get_active_alerts())
        stats["queued_alerts"] = len(self.alert_queue)
        stats["total_rules"] = len(self.rules)
        stats["enabled_rules"] = sum(1 for r in self.rules.values() if r.enabled)

        # Add rates
        if self.fatigue_prevention:
            stats["noise_level"] = self.fatigue_prevention._calculate_noise_level()

        # Calculate suppression rate
        if stats["total_alerts"] > 0:
            stats["suppression_rate"] = stats["suppressed_alerts"] / stats["total_alerts"]
            stats["deduplication_rate"] = stats["deduplicated_alerts"] / stats["total_alerts"]

        return stats

    def get_summary(self) -> str:
        """
        Get system summary.

        Returns:
            Summary string
        """
        stats = self.get_statistics()
        active_alerts = self.get_active_alerts()

        summary = []
        summary.append("=" * 60)
        summary.append("INTELLIGENT ALERT SYSTEM SUMMARY")
        summary.append("=" * 60)

        summary.append(f"Total Alerts: {stats['total_alerts']}")
        summary.append(f"Active Alerts: {stats['active_alerts']}")
        summary.append(
            f"Suppressed: {stats['suppressed_alerts']} " f"({stats.get('suppression_rate', 0):.1%})"
        )
        summary.append(
            f"Deduplicated: {stats['deduplicated_alerts']} "
            f"({stats.get('deduplication_rate', 0):.1%})"
        )

        summary.append("\nTop Active Alerts:")
        for alert in active_alerts[:5]:
            summary.append(
                f"  - [{alert.severity.value}] {alert.title} "
                f"(Priority: {alert.priority_score:.1f})"
            )

        summary.append("\nAlert Distribution:")
        for severity, count in stats["alerts_by_severity"].items():
            summary.append(f"  {severity.value}: {count}")

        if "noise_level" in stats:
            summary.append(f"\nNoise Level: {stats['noise_level']:.1%}")

        summary.append("=" * 60)

        return "\n".join(summary)


if __name__ == "__main__":
    # Example usage
    system = IntelligentAlertSystem()

    # Add some rules
    rule = AlertRule(
        rule_id="high_cpu",
        name="High CPU Usage",
        condition=lambda d: d.get("cpu_usage", 0) > 80,
        severity=AlertSeverity.HIGH,
        category=AlertCategory.PERFORMANCE,
        description="CPU usage exceeds 80%",
    )
    system.add_rule(rule)

    # Create some alerts
    alert1 = system.create_alert(
        title="Database Connection Failed",
        message="Unable to connect to primary database",
        severity=AlertSeverity.CRITICAL,
        category=AlertCategory.SYSTEM,
        component="database",
        metrics={"downtime_seconds": 120},
    )

    alert2 = system.create_alert(
        title="Model Accuracy Degraded",
        message="Model accuracy dropped below threshold",
        severity=AlertSeverity.HIGH,
        category=AlertCategory.MODEL,
        component="ml_pipeline",
        metrics={"accuracy": 0.52, "threshold": 0.55},
    )

    # Process alerts
    processed = system.process_alerts()

    # Print summary
    print(system.get_summary())
