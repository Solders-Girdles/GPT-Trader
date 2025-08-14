"""
Concept Drift Detection for Online Learning Pipeline
Phase 3 - ADAPT-003: Concept drift detector

Implements multiple drift detection algorithms for detecting distribution
changes in incoming data and triggering retraining when necessary.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import warnings
from scipy import stats
from sklearn.metrics import accuracy_score
import json

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of concept drift"""
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    INCREMENTAL = "incremental"
    RECURRING = "recurring"


class DriftSeverity(Enum):
    """Severity levels of detected drift"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftDetectorConfig:
    """Configuration for drift detection algorithms"""
    # ADWIN parameters
    delta: float = 0.002  # Confidence level (smaller = more sensitive)
    
    # Page-Hinkley parameters
    min_instances: int = 30
    threshold: float = 50.0
    alpha: float = 0.9999
    
    # Statistical test parameters
    window_size: int = 1000
    reference_window_size: int = 1000
    p_value_threshold: float = 0.05
    
    # Performance-based detection
    performance_threshold: float = 0.05  # 5% performance drop
    performance_window: int = 100
    
    # Data distribution parameters
    feature_threshold: float = 0.1  # 10% change in feature statistics
    correlation_threshold: float = 0.1  # 10% change in correlations
    
    # General parameters
    warmup_period: int = 100
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.01,
        'medium': 0.05,
        'high': 0.1,
        'critical': 0.2
    })


@dataclass
class DriftDetection:
    """Result of drift detection"""
    is_drift: bool
    drift_type: Optional[DriftType]
    drift_severity: DriftSeverity
    confidence: float
    detected_at: datetime
    drift_score: float
    affected_features: List[str]
    detection_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ADWINDetector:
    """
    ADWIN (Adaptive Windowing) drift detector.
    
    Maintains a variable-length window of recent examples and detects
    changes in the data distribution.
    """
    
    def __init__(self, delta: float = 0.002):
        """Initialize ADWIN detector
        
        Args:
            delta: Confidence level (0 < delta < 1)
        """
        self.delta = delta
        self.window = deque()
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        self.mint = float('inf')
        
    def add_element(self, value: float) -> bool:
        """
        Add a new element and check for drift.
        
        Args:
            value: New data point
            
        Returns:
            True if drift is detected
        """
        self.window.append(value)
        self.width += 1
        
        if self.width == 1:
            self.total = value
            self.variance = 0.0
            self.mint = value
            return False
        
        # Update statistics
        self.total += value
        
        # Calculate variance incrementally
        if self.width > 1:
            mean = self.total / self.width
            self.variance = sum((x - mean) ** 2 for x in self.window) / (self.width - 1)
        
        # Check for drift
        drift_detected = self._detect_change()
        
        if drift_detected:
            self._reset_window()
        
        return drift_detected
    
    def _detect_change(self) -> bool:
        """Detect if change occurred using ADWIN algorithm"""
        if self.width < 2:
            return False
        
        # Calculate cut point
        n = self.width
        mean = self.total / n
        
        for i in range(1, n):
            # Split window at position i
            n0 = i
            n1 = n - i
            
            if n0 < 1 or n1 < 1:
                continue
            
            # Calculate means for both sides
            sum0 = sum(list(self.window)[:i])
            sum1 = sum(list(self.window)[i:])
            
            mean0 = sum0 / n0
            mean1 = sum1 / n1
            
            # Calculate variance for both sides
            var0 = sum((x - mean0) ** 2 for x in list(self.window)[:i]) / max(1, n0 - 1)
            var1 = sum((x - mean1) ** 2 for x in list(self.window)[i:]) / max(1, n1 - 1)
            
            # Calculate epsilon cut
            epsilon_cut = self._calculate_epsilon_cut(n0, n1, var0, var1)
            
            # Check if difference exceeds threshold
            if abs(mean0 - mean1) > epsilon_cut:
                return True
        
        return False
    
    def _calculate_epsilon_cut(self, n0: int, n1: int, var0: float, var1: float) -> float:
        """Calculate epsilon cut threshold"""
        if n0 <= 0 or n1 <= 0:
            return float('inf')
        
        # Hoeffding bound
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        delta_prime = self.delta / n0
        
        epsilon = np.sqrt((var0 + var1) * np.log(2.0 / delta_prime) / (2 * m))
        return epsilon
    
    def _reset_window(self):
        """Reset window after drift detection"""
        # Keep only recent half of the window
        new_size = max(1, self.width // 2)
        self.window = deque(list(self.window)[-new_size:])
        self.width = len(self.window)
        self.total = sum(self.window)
        
        if self.width > 1:
            mean = self.total / self.width
            self.variance = sum((x - mean) ** 2 for x in self.window) / (self.width - 1)
        else:
            self.variance = 0.0


class PageHinkleyDetector:
    """
    Page-Hinkley test for drift detection.
    
    Monitors cumulative sum of differences between observed values
    and their mean to detect changes.
    """
    
    def __init__(self, min_instances: int = 30, threshold: float = 50.0, alpha: float = 0.9999):
        """Initialize Page-Hinkley detector
        
        Args:
            min_instances: Minimum instances before detection
            threshold: Detection threshold
            alpha: Forgetting factor
        """
        self.min_instances = min_instances
        self.threshold = threshold
        self.alpha = alpha
        
        self.sum = 0.0
        self.sum_min = 0.0
        self.x_mean = 0.0
        self.n = 0
        
    def add_element(self, value: float) -> bool:
        """
        Add element and check for drift.
        
        Args:
            value: New data point
            
        Returns:
            True if drift detected
        """
        self.n += 1
        
        # Update mean
        if self.n == 1:
            self.x_mean = value
        else:
            self.x_mean = self.alpha * self.x_mean + (1 - self.alpha) * value
        
        # Update cumulative sum
        self.sum += value - self.x_mean
        
        # Update minimum
        if self.sum < self.sum_min:
            self.sum_min = self.sum
        
        # Check for drift
        if self.n >= self.min_instances:
            ph_value = self.sum - self.sum_min
            if ph_value > self.threshold:
                self._reset()
                return True
        
        return False
    
    def _reset(self):
        """Reset detector after drift"""
        self.sum = 0.0
        self.sum_min = 0.0
        self.n = 0


class ConceptDriftDetector:
    """
    Comprehensive concept drift detector combining multiple algorithms.
    
    Features:
    - Multiple detection algorithms (ADWIN, Page-Hinkley, statistical tests)
    - Performance-based drift detection
    - Feature distribution monitoring
    - Severity assessment
    - Drift type classification
    """
    
    def __init__(self, config: DriftDetectorConfig):
        """Initialize concept drift detector
        
        Args:
            config: Detector configuration
        """
        self.config = config
        
        # Initialize detectors
        self.adwin_detector = ADWINDetector(config.delta)
        self.ph_detector = PageHinkleyDetector(
            config.min_instances, 
            config.threshold, 
            config.alpha
        )
        
        # Data storage
        self.reference_window = deque(maxlen=config.reference_window_size)
        self.current_window = deque(maxlen=config.window_size)
        self.performance_history = deque(maxlen=config.performance_window)
        self.feature_statistics = {}
        self.drift_history: List[DriftDetection] = []
        
        # State tracking
        self.samples_seen = 0
        self.warmup_complete = False
        self.last_drift_time = None
        
        logger.info("Initialized concept drift detector with multiple algorithms")
    
    def add_sample(self, 
                   features: Union[np.ndarray, pd.DataFrame], 
                   target: Optional[float] = None,
                   prediction: Optional[float] = None,
                   performance_metric: Optional[float] = None) -> Optional[DriftDetection]:
        """
        Add new sample and check for concept drift.
        
        Args:
            features: Feature vector or DataFrame
            target: True target value
            prediction: Model prediction
            performance_metric: Performance metric (accuracy, loss, etc.)
            
        Returns:
            DriftDetection object if drift detected, None otherwise
        """
        self.samples_seen += 1
        
        # Convert features to DataFrame if needed
        if isinstance(features, np.ndarray):
            features = pd.DataFrame([features])
        elif isinstance(features, pd.Series):
            features = features.to_frame().T
        
        # Store samples
        if len(self.reference_window) == 0:
            # Initialize reference window
            self.reference_window.extend([features.iloc[0]])
        else:
            self.current_window.append(features.iloc[0])
        
        # Store performance if available
        if performance_metric is not None:
            self.performance_history.append(performance_metric)
        elif target is not None and prediction is not None:
            # Calculate accuracy for binary classification
            acc = 1.0 if abs(target - prediction) < 0.5 else 0.0
            self.performance_history.append(acc)
        
        # Update feature statistics
        self._update_feature_statistics(features.iloc[0])
        
        # Check if warmup period is complete
        if not self.warmup_complete and self.samples_seen >= self.config.warmup_period:
            self.warmup_complete = True
            logger.info("Drift detector warmup period complete")
        
        # Perform drift detection if warmup is complete
        if self.warmup_complete and len(self.current_window) >= 10:
            return self._detect_drift()
        
        return None
    
    def _detect_drift(self) -> Optional[DriftDetection]:
        """Detect concept drift using multiple methods"""
        drift_detections = []
        
        # 1. ADWIN-based detection on performance
        if len(self.performance_history) > 1:
            latest_performance = self.performance_history[-1]
            adwin_drift = self.adwin_detector.add_element(latest_performance)
            if adwin_drift:
                drift_detections.append({
                    'method': 'ADWIN',
                    'confidence': 0.95,
                    'score': 0.8
                })
        
        # 2. Page-Hinkley detection on performance
        if len(self.performance_history) > self.config.min_instances:
            latest_performance = self.performance_history[-1]
            ph_drift = self.ph_detector.add_element(latest_performance)
            if ph_drift:
                drift_detections.append({
                    'method': 'Page-Hinkley',
                    'confidence': 0.9,
                    'score': 0.7
                })
        
        # 3. Statistical tests on feature distributions
        statistical_drift = self._detect_statistical_drift()
        if statistical_drift:
            drift_detections.extend(statistical_drift)
        
        # 4. Performance-based detection
        performance_drift = self._detect_performance_drift()
        if performance_drift:
            drift_detections.append(performance_drift)
        
        # 5. Feature statistics drift
        feature_drift = self._detect_feature_drift()
        if feature_drift:
            drift_detections.extend(feature_drift)
        
        # Combine detections
        if drift_detections:
            return self._combine_drift_detections(drift_detections)
        
        return None
    
    def _detect_statistical_drift(self) -> List[Dict]:
        """Detect drift using statistical tests"""
        if len(self.reference_window) < 30 or len(self.current_window) < 30:
            return []
        
        drift_detections = []
        
        # Convert windows to DataFrames
        ref_df = pd.DataFrame(list(self.reference_window))
        cur_df = pd.DataFrame(list(self.current_window))
        
        # Ensure both DataFrames have the same columns
        common_cols = ref_df.columns.intersection(cur_df.columns)
        if len(common_cols) == 0:
            return []
        
        ref_df = ref_df[common_cols]
        cur_df = cur_df[common_cols]
        
        affected_features = []
        
        for col in common_cols:
            try:
                # Handle numerical columns
                if pd.api.types.is_numeric_dtype(ref_df[col]):
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = stats.ks_2samp(ref_df[col].dropna(), cur_df[col].dropna())
                    
                    if ks_p < self.config.p_value_threshold:
                        affected_features.append(col)
                        drift_detections.append({
                            'method': f'KS-test-{col}',
                            'confidence': 1 - ks_p,
                            'score': ks_stat,
                            'feature': col
                        })
                
                # Handle categorical columns
                else:
                    # Chi-square test for categorical variables
                    ref_counts = ref_df[col].value_counts()
                    cur_counts = cur_df[col].value_counts()
                    
                    # Align categories
                    all_categories = ref_counts.index.union(cur_counts.index)
                    ref_aligned = ref_counts.reindex(all_categories, fill_value=0)
                    cur_aligned = cur_counts.reindex(all_categories, fill_value=0)
                    
                    if len(all_categories) > 1:
                        chi2_stat, chi2_p = stats.chisquare(cur_aligned, ref_aligned)
                        
                        if chi2_p < self.config.p_value_threshold:
                            affected_features.append(col)
                            drift_detections.append({
                                'method': f'Chi2-test-{col}',
                                'confidence': 1 - chi2_p,
                                'score': chi2_stat,
                                'feature': col
                            })
            
            except Exception as e:
                logger.warning(f"Statistical test failed for feature {col}: {e}")
                continue
        
        return drift_detections
    
    def _detect_performance_drift(self) -> Optional[Dict]:
        """Detect drift based on performance degradation"""
        if len(self.performance_history) < self.config.performance_window:
            return None
        
        # Calculate recent vs historical performance
        recent_size = min(20, len(self.performance_history) // 2)
        recent_performance = list(self.performance_history)[-recent_size:]
        historical_performance = list(self.performance_history)[:-recent_size]
        
        if len(historical_performance) < 10:
            return None
        
        recent_mean = np.mean(recent_performance)
        historical_mean = np.mean(historical_performance)
        
        # Check for significant performance drop
        performance_drop = (historical_mean - recent_mean) / historical_mean
        
        if performance_drop > self.config.performance_threshold:
            return {
                'method': 'Performance-degradation',
                'confidence': min(0.95, performance_drop * 2),
                'score': performance_drop,
                'performance_drop': performance_drop
            }
        
        return None
    
    def _detect_feature_drift(self) -> List[Dict]:
        """Detect drift in feature statistics"""
        if not self.feature_statistics or len(self.current_window) < 20:
            return []
        
        drift_detections = []
        current_stats = self._calculate_feature_statistics()
        
        for feature, current_stat in current_stats.items():
            if feature in self.feature_statistics:
                reference_stat = self.feature_statistics[feature]
                
                # Compare means
                if 'mean' in reference_stat and 'mean' in current_stat:
                    mean_change = abs(current_stat['mean'] - reference_stat['mean'])
                    relative_change = mean_change / (abs(reference_stat['mean']) + 1e-8)
                    
                    if relative_change > self.config.feature_threshold:
                        drift_detections.append({
                            'method': f'Feature-mean-{feature}',
                            'confidence': min(0.9, relative_change * 2),
                            'score': relative_change,
                            'feature': feature,
                            'statistic': 'mean'
                        })
                
                # Compare standard deviations
                if 'std' in reference_stat and 'std' in current_stat:
                    std_change = abs(current_stat['std'] - reference_stat['std'])
                    relative_change = std_change / (reference_stat['std'] + 1e-8)
                    
                    if relative_change > self.config.feature_threshold:
                        drift_detections.append({
                            'method': f'Feature-std-{feature}',
                            'confidence': min(0.9, relative_change * 2),
                            'score': relative_change,
                            'feature': feature,
                            'statistic': 'std'
                        })
        
        return drift_detections
    
    def _combine_drift_detections(self, detections: List[Dict]) -> DriftDetection:
        """Combine multiple drift detections into a single result"""
        if not detections:
            return None
        
        # Calculate overall confidence (weighted average)
        total_confidence = sum(d['confidence'] * d['score'] for d in detections)
        total_weight = sum(d['score'] for d in detections)
        overall_confidence = total_confidence / total_weight if total_weight > 0 else 0.0
        
        # Calculate overall drift score
        overall_score = np.mean([d['score'] for d in detections])
        
        # Determine severity
        severity = self._determine_severity(overall_score)
        
        # Determine drift type
        drift_type = self._determine_drift_type(detections)
        
        # Collect affected features
        affected_features = list(set(d.get('feature', '') for d in detections if d.get('feature')))
        
        # Primary detection method
        primary_method = max(detections, key=lambda x: x['confidence'])['method']
        
        detection = DriftDetection(
            is_drift=True,
            drift_type=drift_type,
            drift_severity=severity,
            confidence=overall_confidence,
            detected_at=datetime.now(),
            drift_score=overall_score,
            affected_features=affected_features,
            detection_method=primary_method,
            metadata={
                'num_detectors': len(detections),
                'detection_methods': [d['method'] for d in detections],
                'individual_scores': [d['score'] for d in detections],
                'samples_since_last_drift': self._samples_since_last_drift()
            }
        )
        
        # Update drift history
        self.drift_history.append(detection)
        self.last_drift_time = datetime.now()
        
        # Update reference window after drift detection
        self._update_reference_window()
        
        logger.warning(f"Concept drift detected: {severity.value} severity, "
                      f"confidence={overall_confidence:.3f}, "
                      f"affected_features={affected_features}")
        
        return detection
    
    def _determine_severity(self, drift_score: float) -> DriftSeverity:
        """Determine drift severity based on score"""
        thresholds = self.config.severity_thresholds
        
        if drift_score >= thresholds['critical']:
            return DriftSeverity.CRITICAL
        elif drift_score >= thresholds['high']:
            return DriftSeverity.HIGH
        elif drift_score >= thresholds['medium']:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW
    
    def _determine_drift_type(self, detections: List[Dict]) -> DriftType:
        """Determine type of drift based on detection patterns"""
        # Simple heuristic: if multiple features affected, likely gradual
        # If performance drops suddenly, likely sudden
        
        feature_detections = [d for d in detections if 'feature' in d]
        performance_detections = [d for d in detections if 'performance' in d['method'].lower()]
        
        if len(feature_detections) > 3:
            return DriftType.GRADUAL
        elif performance_detections and any(d['score'] > 0.2 for d in performance_detections):
            return DriftType.SUDDEN
        else:
            return DriftType.INCREMENTAL
    
    def _samples_since_last_drift(self) -> int:
        """Calculate samples since last drift detection"""
        if self.last_drift_time is None:
            return self.samples_seen
        
        # This is a simplified version - in practice, you'd track this more precisely
        return min(1000, self.samples_seen)  # Placeholder
    
    def _update_reference_window(self):
        """Update reference window after drift detection"""
        # Replace reference window with recent samples
        if len(self.current_window) >= self.config.reference_window_size // 2:
            recent_samples = list(self.current_window)[-self.config.reference_window_size // 2:]
            self.reference_window.clear()
            self.reference_window.extend(recent_samples)
            
            # Update feature statistics
            self.feature_statistics = self._calculate_feature_statistics()
            
            logger.info("Updated reference window after drift detection")
    
    def _update_feature_statistics(self, features: pd.Series):
        """Update running feature statistics"""
        if not self.feature_statistics:
            self.feature_statistics = {}
        
        for feature, value in features.items():
            if pd.notna(value) and isinstance(value, (int, float)):
                if feature not in self.feature_statistics:
                    self.feature_statistics[feature] = {
                        'sum': 0.0,
                        'sum_sq': 0.0,
                        'count': 0,
                        'mean': 0.0,
                        'std': 0.0
                    }
                
                stats = self.feature_statistics[feature]
                stats['count'] += 1
                stats['sum'] += value
                stats['sum_sq'] += value * value
                
                # Update mean and std
                stats['mean'] = stats['sum'] / stats['count']
                if stats['count'] > 1:
                    variance = (stats['sum_sq'] - stats['sum'] * stats['mean']) / (stats['count'] - 1)
                    stats['std'] = np.sqrt(max(0, variance))
    
    def _calculate_feature_statistics(self) -> Dict:
        """Calculate current feature statistics from current window"""
        if not self.current_window:
            return {}
        
        df = pd.DataFrame(list(self.current_window))
        stats = {}
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    stats[col] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'count': len(col_data)
                    }
        
        return stats
    
    def get_drift_history(self) -> List[DriftDetection]:
        """Get history of detected drifts"""
        return self.drift_history.copy()
    
    def get_statistics(self) -> Dict:
        """Get detector statistics"""
        return {
            'samples_seen': self.samples_seen,
            'warmup_complete': self.warmup_complete,
            'reference_window_size': len(self.reference_window),
            'current_window_size': len(self.current_window),
            'performance_history_size': len(self.performance_history),
            'total_drifts_detected': len(self.drift_history),
            'last_drift_time': self.last_drift_time.isoformat() if self.last_drift_time else None,
            'feature_statistics_count': len(self.feature_statistics),
            'recent_performance': list(self.performance_history)[-10:] if self.performance_history else []
        }
    
    def reset(self):
        """Reset detector to initial state"""
        self.reference_window.clear()
        self.current_window.clear()
        self.performance_history.clear()
        self.feature_statistics.clear()
        self.drift_history.clear()
        
        self.samples_seen = 0
        self.warmup_complete = False
        self.last_drift_time = None
        
        # Reset internal detectors
        self.adwin_detector = ADWINDetector(self.config.delta)
        self.ph_detector = PageHinkleyDetector(
            self.config.min_instances,
            self.config.threshold,
            self.config.alpha
        )
        
        logger.info("Reset concept drift detector")


# Factory function
def create_drift_detector(detector_type: str = "comprehensive", **kwargs) -> ConceptDriftDetector:
    """
    Factory function to create drift detectors.
    
    Args:
        detector_type: Type of detector (only 'comprehensive' supported currently)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured drift detector
    """
    config = DriftDetectorConfig(**kwargs)
    return ConceptDriftDetector(config)


# Predefined configurations
SENSITIVE_DETECTOR = DriftDetectorConfig(
    delta=0.001,
    threshold=30.0,
    performance_threshold=0.02,
    p_value_threshold=0.01
)

ROBUST_DETECTOR = DriftDetectorConfig(
    delta=0.01,
    threshold=100.0,
    performance_threshold=0.1,
    p_value_threshold=0.001
)