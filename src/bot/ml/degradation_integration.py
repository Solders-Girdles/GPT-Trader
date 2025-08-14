"""
Integration Module for Advanced Degradation Detection
Phase 3, Week 1: MON-006
Bridges the new Advanced Degradation Detector with existing ModelDegradationMonitor
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

# Import new advanced detector
from .advanced_degradation_detector import (
    AdvancedDegradationDetector,
    DegradationAlert,
)
from .advanced_degradation_detector import (
    DegradationType as AdvancedDegradationType,
)

# Import existing components
from .model_degradation_monitor import (
    DegradationStatus,
    ModelDegradationMonitor,
)

logger = logging.getLogger(__name__)


@dataclass
class IntegratedDegradationReport:
    """Unified degradation report combining both systems"""

    timestamp: datetime
    model_id: str

    # Legacy system metrics
    legacy_status: DegradationStatus
    legacy_accuracy: float
    legacy_trend: float

    # Advanced system metrics
    advanced_status: AdvancedDegradationType
    advanced_score: float
    feature_drift_count: int
    cusum_value: float
    confidence_decay: bool

    # Combined assessment
    overall_status: str  # "healthy", "warning", "degraded", "critical"
    primary_issue: str
    recommended_action: str
    confidence_level: float  # 0-1, confidence in assessment

    # Detailed findings
    findings: list[str]
    alerts: list[DegradationAlert]


class DegradationIntegrator:
    """
    Integrates Advanced Degradation Detector with existing ModelDegradationMonitor
    Provides unified interface and enhanced detection capabilities
    """

    def __init__(
        self,
        legacy_monitor: ModelDegradationMonitor | None = None,
        use_legacy: bool = True,
        use_advanced: bool = True,
    ):
        """
        Initialize the integration layer.

        Args:
            legacy_monitor: Existing ModelDegradationMonitor instance
            use_legacy: Whether to use legacy monitoring
            use_advanced: Whether to use advanced detection
        """
        self.use_legacy = use_legacy
        self.use_advanced = use_advanced

        # Initialize or use provided legacy monitor
        if use_legacy:
            self.legacy_monitor = legacy_monitor or ModelDegradationMonitor()
        else:
            self.legacy_monitor = None

        # Initialize advanced detector
        if use_advanced:
            self.advanced_detector = AdvancedDegradationDetector(
                metrics_window=1000,
                drift_threshold=0.05,
                confidence_threshold=0.6,
                cusum_h=4.0,
                cusum_k=0.5,
            )
        else:
            self.advanced_detector = None

        # Storage for integration
        self.reports = []
        self.last_retraining = None
        self.baseline_set = False

        logger.info(
            f"Initialized DegradationIntegrator (legacy={use_legacy}, advanced={use_advanced})"
        )

    def set_baseline(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidences: np.ndarray,
        model_id: str = "default",
    ) -> None:
        """
        Set baseline for both monitoring systems.

        Args:
            features: Feature values for baseline
            predictions: Model predictions
            actuals: Actual values
            confidences: Prediction confidences
            model_id: Model identifier
        """
        # Set baseline for advanced detector
        if self.advanced_detector:
            self.advanced_detector.update_baseline(features, predictions, actuals, confidences)

        # Set baseline for legacy monitor
        if self.legacy_monitor:
            accuracy = np.mean(predictions == actuals)
            self.legacy_monitor.set_baseline_performance(
                model_id=model_id,
                accuracy=accuracy,
                f1_score=0.0,  # Calculate if needed
                additional_metrics={},
            )

        self.baseline_set = True
        logger.info(
            f"Baseline set for model {model_id} with accuracy {np.mean(predictions == actuals):.3f}"
        )

    def check_degradation(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidences: np.ndarray,
        model_id: str = "default",
    ) -> IntegratedDegradationReport:
        """
        Perform integrated degradation check using both systems.

        Args:
            features: Current feature values
            predictions: Model predictions
            actuals: Actual values
            confidences: Prediction confidences
            model_id: Model identifier

        Returns:
            IntegratedDegradationReport with combined assessment
        """
        if not self.baseline_set:
            logger.warning("Baseline not set, using current batch as baseline")
            self.set_baseline(features, predictions, actuals, confidences, model_id)

        findings = []
        alerts = []

        # Run advanced detection
        advanced_status = AdvancedDegradationType.NONE
        advanced_score = 0.0
        feature_drift_count = 0
        cusum_value = 0.0
        confidence_decay = False

        if self.advanced_detector:
            advanced_metrics = self.advanced_detector.check_degradation(
                features, predictions, actuals, confidences
            )

            advanced_status = advanced_metrics.status
            advanced_score = advanced_metrics.degradation_score

            # Count drifted features
            feature_drift_count = sum(
                1
                for p in advanced_metrics.feature_drift_scores.values()
                if p < self.advanced_detector.drift_threshold
            )

            # Get CUSUM value
            cusum_value = advanced_metrics.cusum_values[1] if advanced_metrics.cusum_values else 0.0

            # Check confidence decay
            confidence_decay = advanced_metrics.confidence_distribution.get("decay_detected", False)

            # Collect findings
            if advanced_status != AdvancedDegradationType.NONE:
                findings.append(f"Advanced detector: {advanced_status.value}")

            if feature_drift_count > 0:
                findings.append(f"{feature_drift_count} features showing drift")

            if confidence_decay:
                findings.append("Model confidence decaying")

            # Get alerts
            alerts = self.advanced_detector.alerts[-5:]  # Last 5 alerts

        # Run legacy detection
        legacy_status = DegradationStatus.HEALTHY
        legacy_accuracy = np.mean(predictions == actuals)
        legacy_trend = 0.0

        if self.legacy_monitor:
            # Update legacy monitor
            self.legacy_monitor.update(
                model_id=model_id,
                predictions=predictions,
                actuals=actuals,
                features=features,
                confidence_scores=confidences,
            )

            # Get legacy assessment
            legacy_report = self.legacy_monitor.get_degradation_report(model_id)
            if legacy_report:
                legacy_status = legacy_report["status"]
                legacy_accuracy = legacy_report["performance_summary"]["current"]
                legacy_trend = legacy_report["performance_summary"]["trend"]

                if legacy_status != DegradationStatus.HEALTHY:
                    findings.append(f"Legacy monitor: {legacy_status.value}")

        # Combine assessments
        overall_status, primary_issue, recommended_action = self._combine_assessments(
            advanced_status,
            advanced_score,
            legacy_status,
            legacy_accuracy,
            feature_drift_count,
            confidence_decay,
        )

        # Calculate confidence in assessment
        confidence_level = self._calculate_confidence(advanced_score, legacy_status, len(alerts))

        # Create integrated report
        report = IntegratedDegradationReport(
            timestamp=datetime.now(),
            model_id=model_id,
            legacy_status=legacy_status,
            legacy_accuracy=legacy_accuracy,
            legacy_trend=legacy_trend,
            advanced_status=advanced_status,
            advanced_score=advanced_score,
            feature_drift_count=feature_drift_count,
            cusum_value=cusum_value,
            confidence_decay=confidence_decay,
            overall_status=overall_status,
            primary_issue=primary_issue,
            recommended_action=recommended_action,
            confidence_level=confidence_level,
            findings=findings,
            alerts=alerts,
        )

        self.reports.append(report)

        # Log assessment
        logger.info(f"Degradation check: {overall_status} (confidence={confidence_level:.2f})")
        if findings:
            logger.info(f"Findings: {'; '.join(findings)}")

        return report

    def _combine_assessments(
        self,
        advanced_status: AdvancedDegradationType,
        advanced_score: float,
        legacy_status: DegradationStatus,
        legacy_accuracy: float,
        feature_drift_count: int,
        confidence_decay: bool,
    ) -> tuple[str, str, str]:
        """
        Combine assessments from both systems into unified decision.

        Returns:
            Tuple of (overall_status, primary_issue, recommended_action)
        """
        # Map statuses to severity levels
        severity_map = {
            DegradationStatus.HEALTHY: 0,
            DegradationStatus.WARNING: 1,
            DegradationStatus.DEGRADED: 2,
            DegradationStatus.CRITICAL: 3,
            DegradationStatus.RETRAINING: 2,
        }

        # Determine overall status
        if advanced_score > 0.8 or legacy_status == DegradationStatus.CRITICAL:
            overall_status = "critical"
            primary_issue = "Severe model degradation detected"
            recommended_action = "Immediate retraining required"

        elif advanced_score > 0.6 or legacy_status == DegradationStatus.DEGRADED:
            overall_status = "degraded"
            primary_issue = self._identify_primary_issue(
                advanced_status, feature_drift_count, confidence_decay
            )
            recommended_action = "Schedule retraining within 24 hours"

        elif advanced_score > 0.4 or legacy_status == DegradationStatus.WARNING:
            overall_status = "warning"
            primary_issue = self._identify_primary_issue(
                advanced_status, feature_drift_count, confidence_decay
            )
            recommended_action = "Monitor closely, prepare for retraining"

        else:
            overall_status = "healthy"
            primary_issue = "No significant issues detected"
            recommended_action = "Continue normal monitoring"

        # Override if both systems agree on critical
        if self.use_legacy and self.use_advanced:
            if advanced_score > 0.7 and severity_map.get(legacy_status, 0) >= 2:
                overall_status = "critical"
                recommended_action = (
                    "Both systems indicate critical degradation - retrain immediately"
                )

        return overall_status, primary_issue, recommended_action

    def _identify_primary_issue(
        self,
        advanced_status: AdvancedDegradationType,
        feature_drift_count: int,
        confidence_decay: bool,
    ) -> str:
        """Identify the primary issue based on detection results"""
        if advanced_status == AdvancedDegradationType.FEATURE_DRIFT:
            return f"Feature drift detected in {feature_drift_count} features"
        elif advanced_status == AdvancedDegradationType.ACCURACY_DRIFT:
            return "Model accuracy degrading"
        elif advanced_status == AdvancedDegradationType.CONFIDENCE_DECAY:
            return "Model confidence decreasing"
        elif advanced_status == AdvancedDegradationType.ERROR_PATTERN_CHANGE:
            return "Error patterns have changed"
        elif advanced_status == AdvancedDegradationType.CONCEPT_DRIFT:
            return "Concept drift detected"
        elif confidence_decay:
            return "Model confidence showing decay"
        elif feature_drift_count > 0:
            return f"Minor drift in {feature_drift_count} features"
        else:
            return "Performance below optimal threshold"

    def _calculate_confidence(
        self, advanced_score: float, legacy_status: DegradationStatus, alert_count: int
    ) -> float:
        """
        Calculate confidence level in the assessment.

        Returns:
            Confidence level between 0 and 1
        """
        confidence = 0.5  # Base confidence

        # Adjust based on advanced score clarity
        if advanced_score > 0.8 or advanced_score < 0.2:
            confidence += 0.2  # Clear signal
        elif 0.4 < advanced_score < 0.6:
            confidence -= 0.1  # Unclear signal

        # Adjust based on legacy status
        if legacy_status in [DegradationStatus.CRITICAL, DegradationStatus.HEALTHY]:
            confidence += 0.1  # Clear status

        # Adjust based on alert consistency
        if alert_count > 3:
            confidence += 0.1  # Multiple alerts
        elif alert_count == 0:
            confidence -= 0.1  # No alerts

        # Adjust if both systems are active and agree
        if self.use_legacy and self.use_advanced:
            if (
                advanced_score > 0.6
                and legacy_status in [DegradationStatus.DEGRADED, DegradationStatus.CRITICAL]
            ) or (advanced_score < 0.3 and legacy_status == DegradationStatus.HEALTHY):
                confidence += 0.2  # Systems agree

        return max(0.0, min(1.0, confidence))

    def should_retrain(
        self, report: IntegratedDegradationReport | None = None, force_check: bool = False
    ) -> tuple[bool, str]:
        """
        Determine if model should be retrained based on integrated assessment.

        Args:
            report: Specific report to check (uses latest if None)
            force_check: Bypass time-based restrictions

        Returns:
            Tuple of (should_retrain, reason)
        """
        if not self.reports and not report:
            return False, "No degradation reports available"

        report = report or self.reports[-1]

        # Check if recently retrained
        if not force_check and self.last_retraining:
            time_since_retraining = datetime.now() - self.last_retraining
            if time_since_retraining < timedelta(hours=6):
                return (
                    False,
                    f"Recently retrained {time_since_retraining.total_seconds()/3600:.1f} hours ago",
                )

        # Decision based on overall status
        if report.overall_status == "critical":
            return True, "Critical degradation detected"

        if report.overall_status == "degraded":
            # Check confidence and specifics
            if report.confidence_level > 0.7:
                return True, f"High confidence degradation: {report.primary_issue}"
            elif report.advanced_score > 0.7:
                return True, "Advanced detector indicates significant degradation"
            elif report.feature_drift_count > 5:
                return True, f"Significant feature drift ({report.feature_drift_count} features)"

        # Check for consistent warnings
        if len(self.reports) >= 5:
            recent_reports = self.reports[-5:]
            warning_count = sum(
                1 for r in recent_reports if r.overall_status in ["warning", "degraded"]
            )
            if warning_count >= 4:
                return True, "Consistent warnings over multiple checks"

        return False, "No retraining needed"

    def get_status_summary(self) -> dict[str, Any]:
        """
        Get summary of current degradation monitoring status.

        Returns:
            Dictionary with status information
        """
        if not self.reports:
            return {
                "status": "no_data",
                "message": "No degradation checks performed yet",
                "baseline_set": self.baseline_set,
            }

        latest = self.reports[-1]

        # Calculate trends if we have enough history
        trend_info = {}
        if len(self.reports) >= 3:
            recent_scores = [r.advanced_score for r in self.reports[-5:]]
            trend_info["score_trend"] = (
                "increasing" if recent_scores[-1] > recent_scores[0] else "decreasing"
            )
            trend_info["avg_score"] = np.mean(recent_scores)

        should_retrain, retrain_reason = self.should_retrain()

        return {
            "status": latest.overall_status,
            "timestamp": latest.timestamp.isoformat(),
            "primary_issue": latest.primary_issue,
            "recommended_action": latest.recommended_action,
            "confidence": latest.confidence_level,
            "metrics": {
                "advanced_score": latest.advanced_score,
                "legacy_accuracy": latest.legacy_accuracy,
                "feature_drift_count": latest.feature_drift_count,
                "confidence_decay": latest.confidence_decay,
            },
            "should_retrain": should_retrain,
            "retrain_reason": retrain_reason,
            "trend": trend_info,
            "total_checks": len(self.reports),
            "recent_alerts": len(latest.alerts),
        }

    def export_report(self, filepath: str, format: str = "json") -> None:
        """
        Export degradation reports to file.

        Args:
            filepath: Path to save report
            format: 'json' or 'csv'
        """
        if not self.reports:
            logger.warning("No reports to export")
            return

        if format == "json":
            report_data = []
            for report in self.reports:
                report_dict = {
                    "timestamp": report.timestamp.isoformat(),
                    "model_id": report.model_id,
                    "overall_status": report.overall_status,
                    "primary_issue": report.primary_issue,
                    "recommended_action": report.recommended_action,
                    "confidence_level": report.confidence_level,
                    "advanced_score": report.advanced_score,
                    "legacy_accuracy": report.legacy_accuracy,
                    "feature_drift_count": report.feature_drift_count,
                    "findings": report.findings,
                }
                report_data.append(report_dict)

            import json

            with open(filepath, "w") as f:
                json.dump(report_data, f, indent=2)

        elif format == "csv":
            # Flatten reports for CSV
            rows = []
            for report in self.reports:
                row = {
                    "timestamp": report.timestamp,
                    "model_id": report.model_id,
                    "overall_status": report.overall_status,
                    "advanced_score": report.advanced_score,
                    "legacy_accuracy": report.legacy_accuracy,
                    "feature_drift_count": report.feature_drift_count,
                    "confidence_level": report.confidence_level,
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)

        logger.info(f"Exported {len(self.reports)} reports to {filepath}")

    def reset(self) -> None:
        """Reset the integrator to initial state"""
        self.reports = []
        self.last_retraining = None
        self.baseline_set = False

        if self.advanced_detector:
            self.advanced_detector = AdvancedDegradationDetector()

        if self.legacy_monitor:
            self.legacy_monitor = ModelDegradationMonitor()

        logger.info("DegradationIntegrator reset to initial state")


# Convenience function for quick integration
def create_integrated_monitor(
    use_legacy: bool = True, use_advanced: bool = True
) -> DegradationIntegrator:
    """
    Create a pre-configured integrated degradation monitor.

    Args:
        use_legacy: Whether to use legacy monitoring
        use_advanced: Whether to use advanced detection

    Returns:
        Configured DegradationIntegrator instance
    """
    return DegradationIntegrator(use_legacy=use_legacy, use_advanced=use_advanced)
