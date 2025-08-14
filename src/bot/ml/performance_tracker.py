"""
Model Performance Tracking System
Phase 2.5 - Day 5

Tracks ML model performance over time with drift detection and automated retraining triggers.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import sqlite3
from enum import Enum

from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model status states"""
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    RETRAINING = "retraining"
    RETIRED = "retired"


class DriftType(Enum):
    """Types of drift detected"""
    NONE = "none"
    CONCEPT = "concept"  # Target distribution change
    FEATURE = "feature"  # Feature distribution change
    PERFORMANCE = "performance"  # Model performance degradation


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: datetime
    model_id: str
    model_version: str
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    
    # Trading metrics
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    returns: Optional[float] = None
    
    # Data characteristics
    n_predictions: int = 0
    n_features: int = 0
    
    # Drift indicators
    feature_drift_score: Optional[float] = None
    prediction_drift_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class ModelHealth:
    """Model health status"""
    model_id: str
    status: ModelStatus
    last_update: datetime
    
    # Performance tracking
    current_accuracy: float
    baseline_accuracy: float
    performance_trend: str  # 'improving', 'stable', 'degrading'
    
    # Drift detection
    drift_detected: bool
    drift_type: DriftType
    drift_score: float
    
    # Recommendations
    needs_retraining: bool
    confidence_score: float  # 0-1, confidence in model predictions
    
    # History
    performance_history: List[float] = field(default_factory=list)
    drift_history: List[float] = field(default_factory=list)


class PerformanceTracker:
    """
    Tracks and monitors ML model performance over time.
    
    Features:
    - Performance metric tracking
    - Drift detection (concept and feature drift)
    - Automated retraining triggers
    - Performance visualization
    - Model comparison
    """
    
    def __init__(self, db_path: str = "model_performance.db"):
        self.db_path = db_path
        self.models: Dict[str, ModelHealth] = {}
        
        # Performance thresholds
        self.min_accuracy = 0.55
        self.degradation_threshold = 0.05  # 5% performance drop
        self.drift_threshold = 0.1  # Statistical significance for drift
        
        # Tracking windows
        self.performance_window = 100  # Last 100 predictions
        self.drift_window = 500  # Last 500 predictions for drift detection
        
        # Initialize database
        self._init_database()
        
        logger.info(f"PerformanceTracker initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize performance tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                model_id TEXT,
                model_version TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                roc_auc REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                returns REAL,
                n_predictions INTEGER,
                n_features INTEGER,
                feature_drift_score REAL,
                prediction_drift_score REAL
            )
        """)
        
        # Create model health table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_health (
                model_id TEXT PRIMARY KEY,
                status TEXT,
                last_update DATETIME,
                current_accuracy REAL,
                baseline_accuracy REAL,
                performance_trend TEXT,
                drift_detected BOOLEAN,
                drift_type TEXT,
                drift_score REAL,
                needs_retraining BOOLEAN,
                confidence_score REAL
            )
        """)
        
        # Create predictions log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                model_id TEXT,
                features TEXT,  -- JSON
                prediction REAL,
                actual REAL,
                confidence REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def track_prediction(self,
                        model_id: str,
                        features: Dict[str, float],
                        prediction: float,
                        actual: Optional[float] = None,
                        confidence: Optional[float] = None):
        """
        Track individual prediction.
        
        Args:
            model_id: Model identifier
            features: Feature values used
            prediction: Model prediction
            actual: Actual outcome (if available)
            confidence: Prediction confidence
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions_log 
            (timestamp, model_id, features, prediction, actual, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            model_id,
            json.dumps(features),
            prediction,
            actual,
            confidence
        ))
        
        conn.commit()
        conn.close()
        
        # Update performance if actual is available
        if actual is not None:
            self._update_running_performance(model_id)
    
    def update_performance(self,
                          model_id: str,
                          metrics: PerformanceMetrics):
        """
        Update model performance metrics.
        
        Args:
            model_id: Model identifier
            metrics: Performance metrics
        """
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO performance_metrics
            (timestamp, model_id, model_version, accuracy, precision, recall,
             f1_score, roc_auc, sharpe_ratio, max_drawdown, win_rate, returns,
             n_predictions, n_features, feature_drift_score, prediction_drift_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.timestamp,
            metrics.model_id,
            metrics.model_version,
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1_score,
            metrics.roc_auc,
            metrics.sharpe_ratio,
            metrics.max_drawdown,
            metrics.win_rate,
            metrics.returns,
            metrics.n_predictions,
            metrics.n_features,
            metrics.feature_drift_score,
            metrics.prediction_drift_score
        ))
        
        conn.commit()
        conn.close()
        
        # Update model health
        self._update_model_health(model_id, metrics)
        
        logger.info(f"Updated performance for {model_id}: accuracy={metrics.accuracy:.3f}")
    
    def detect_drift(self,
                    model_id: str,
                    current_features: pd.DataFrame,
                    reference_features: pd.DataFrame) -> Tuple[bool, DriftType, float]:
        """
        Detect feature and concept drift.
        
        Args:
            model_id: Model identifier
            current_features: Recent feature data
            reference_features: Reference/training feature data
            
        Returns:
            Tuple of (drift_detected, drift_type, drift_score)
        """
        drift_scores = []
        
        # Feature drift detection (Kolmogorov-Smirnov test)
        feature_drift_detected = False
        for column in current_features.columns:
            if column in reference_features.columns:
                statistic, p_value = stats.ks_2samp(
                    reference_features[column].dropna(),
                    current_features[column].dropna()
                )
                
                if p_value < self.drift_threshold:
                    feature_drift_detected = True
                    drift_scores.append(1 - p_value)
        
        # Calculate overall drift score
        drift_score = np.mean(drift_scores) if drift_scores else 0
        
        # Determine drift type
        if feature_drift_detected:
            drift_type = DriftType.FEATURE
        else:
            # Check performance drift
            if self._check_performance_drift(model_id):
                drift_type = DriftType.PERFORMANCE
                drift_score = max(drift_score, 0.5)  # Ensure significant score
            else:
                drift_type = DriftType.NONE
        
        drift_detected = drift_type != DriftType.NONE
        
        if drift_detected:
            logger.warning(f"Drift detected for {model_id}: {drift_type.value} (score={drift_score:.3f})")
        
        return drift_detected, drift_type, drift_score
    
    def check_retraining_needed(self, model_id: str) -> bool:
        """
        Check if model needs retraining.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if retraining is needed
        """
        if model_id not in self.models:
            return False
        
        health = self.models[model_id]
        
        # Check various conditions
        conditions = [
            health.status == ModelStatus.FAILED,
            health.drift_detected and health.drift_score > 0.2,
            health.current_accuracy < self.min_accuracy,
            health.current_accuracy < health.baseline_accuracy - self.degradation_threshold,
            health.performance_trend == 'degrading' and len(health.performance_history) > 10
        ]
        
        needs_retraining = any(conditions)
        
        if needs_retraining:
            logger.info(f"Model {model_id} needs retraining")
            health.needs_retraining = True
        
        return needs_retraining
    
    def get_model_health(self, model_id: str) -> Optional[ModelHealth]:
        """
        Get current model health status.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model health status
        """
        if model_id in self.models:
            return self.models[model_id]
        
        # Try to load from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM model_health WHERE model_id = ?
        """, (model_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            health = ModelHealth(
                model_id=row[0],
                status=ModelStatus(row[1]),
                last_update=datetime.fromisoformat(row[2]) if isinstance(row[2], str) else row[2],
                current_accuracy=row[3],
                baseline_accuracy=row[4],
                performance_trend=row[5],
                drift_detected=bool(row[6]),
                drift_type=DriftType(row[7]),
                drift_score=row[8],
                needs_retraining=bool(row[9]),
                confidence_score=row[10]
            )
            self.models[model_id] = health
            return health
        
        return None
    
    def get_performance_history(self,
                               model_id: str,
                               days: int = 30) -> pd.DataFrame:
        """
        Get performance history for a model.
        
        Args:
            model_id: Model identifier
            days: Number of days of history
            
        Returns:
            DataFrame with performance history
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM performance_metrics
            WHERE model_id = ?
            AND timestamp > datetime('now', '-{} days')
            ORDER BY timestamp
        """.format(days)
        
        df = pd.read_sql_query(query, conn, params=(model_id,))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """
        Compare performance of multiple models.
        
        Args:
            model_ids: List of model identifiers
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for model_id in model_ids:
            health = self.get_model_health(model_id)
            if health:
                # Get recent performance
                recent_perf = self.get_performance_history(model_id, days=7)
                
                if not recent_perf.empty:
                    comparison_data.append({
                        'Model': model_id,
                        'Status': health.status.value,
                        'Current Accuracy': health.current_accuracy,
                        'Trend': health.performance_trend,
                        'Drift Detected': health.drift_detected,
                        'Confidence': health.confidence_score,
                        'Avg F1 (7d)': recent_perf['f1_score'].mean(),
                        'Avg Sharpe (7d)': recent_perf['sharpe_ratio'].mean() if 'sharpe_ratio' in recent_perf else None,
                        'Needs Retraining': health.needs_retraining
                    })
        
        return pd.DataFrame(comparison_data)
    
    def _update_running_performance(self, model_id: str):
        """Update running performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent predictions
        cursor.execute("""
            SELECT prediction, actual FROM predictions_log
            WHERE model_id = ?
            AND actual IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT ?
        """, (model_id, self.performance_window))
        
        predictions = []
        actuals = []
        
        for row in cursor.fetchall():
            predictions.append(row[0])
            actuals.append(row[1])
        
        conn.close()
        
        if len(predictions) > 10:  # Need minimum samples
            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Convert to binary if needed
            if len(np.unique(actuals)) == 2:
                accuracy = accuracy_score(actuals, predictions > 0.5)
                f1 = f1_score(actuals, predictions > 0.5)
            else:
                accuracy = 1 - np.mean(np.abs(predictions - actuals))
                f1 = accuracy  # Simplified for regression
            
            # Update model health
            if model_id in self.models:
                health = self.models[model_id]
                health.current_accuracy = accuracy
                health.performance_history.append(accuracy)
                
                # Keep only recent history
                if len(health.performance_history) > 100:
                    health.performance_history = health.performance_history[-100:]
    
    def _update_model_health(self, model_id: str, metrics: PerformanceMetrics):
        """Update model health status"""
        if model_id not in self.models:
            # Initialize new model health
            self.models[model_id] = ModelHealth(
                model_id=model_id,
                status=ModelStatus.ACTIVE,
                last_update=datetime.now(),
                current_accuracy=metrics.accuracy,
                baseline_accuracy=metrics.accuracy,
                performance_trend='stable',
                drift_detected=False,
                drift_type=DriftType.NONE,
                drift_score=0.0,
                needs_retraining=False,
                confidence_score=metrics.accuracy
            )
        else:
            health = self.models[model_id]
            health.last_update = datetime.now()
            health.current_accuracy = metrics.accuracy
            
            # Update performance history
            health.performance_history.append(metrics.accuracy)
            if len(health.performance_history) > 100:
                health.performance_history = health.performance_history[-100:]
            
            # Determine trend
            if len(health.performance_history) >= 10:
                recent = np.mean(health.performance_history[-5:])
                older = np.mean(health.performance_history[-10:-5])
                
                if recent > older + 0.02:
                    health.performance_trend = 'improving'
                elif recent < older - 0.02:
                    health.performance_trend = 'degrading'
                else:
                    health.performance_trend = 'stable'
            
            # Update status
            if metrics.accuracy < self.min_accuracy:
                health.status = ModelStatus.FAILED
            elif metrics.accuracy < health.baseline_accuracy - self.degradation_threshold:
                health.status = ModelStatus.DEGRADED
            else:
                health.status = ModelStatus.ACTIVE
            
            # Update drift if available
            if metrics.feature_drift_score:
                health.drift_score = metrics.feature_drift_score
                health.drift_detected = metrics.feature_drift_score > self.drift_threshold
            
            # Update confidence
            health.confidence_score = min(1.0, metrics.accuracy / health.baseline_accuracy)
        
        # Save to database
        self._save_model_health(health)
    
    def _save_model_health(self, health: ModelHealth):
        """Save model health to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO model_health
            (model_id, status, last_update, current_accuracy, baseline_accuracy,
             performance_trend, drift_detected, drift_type, drift_score,
             needs_retraining, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            health.model_id,
            health.status.value,
            health.last_update,
            health.current_accuracy,
            health.baseline_accuracy,
            health.performance_trend,
            health.drift_detected,
            health.drift_type.value,
            health.drift_score,
            health.needs_retraining,
            health.confidence_score
        ))
        
        conn.commit()
        conn.close()
    
    def _check_performance_drift(self, model_id: str) -> bool:
        """Check for performance drift"""
        if model_id not in self.models:
            return False
        
        health = self.models[model_id]
        
        # Need sufficient history
        if len(health.performance_history) < 20:
            return False
        
        # Compare recent vs older performance
        recent = health.performance_history[-10:]
        older = health.performance_history[-20:-10]
        
        # Statistical test for difference
        statistic, p_value = stats.ttest_ind(older, recent)
        
        # Check if significant degradation
        return p_value < 0.05 and np.mean(recent) < np.mean(older)
    
    def generate_report(self, model_id: str) -> Dict[str, Any]:
        """Generate performance report for a model"""
        health = self.get_model_health(model_id)
        if not health:
            return {}
        
        # Get performance history
        history = self.get_performance_history(model_id, days=30)
        
        report = {
            'model_id': model_id,
            'status': health.status.value,
            'last_update': health.last_update.isoformat(),
            'current_performance': {
                'accuracy': health.current_accuracy,
                'trend': health.performance_trend,
                'confidence': health.confidence_score
            },
            'drift': {
                'detected': health.drift_detected,
                'type': health.drift_type.value,
                'score': health.drift_score
            },
            'recommendations': {
                'needs_retraining': health.needs_retraining,
                'action': self._get_recommendation(health)
            },
            'history_summary': {
                'avg_accuracy_30d': history['accuracy'].mean() if not history.empty else 0,
                'avg_f1_30d': history['f1_score'].mean() if not history.empty and 'f1_score' in history else 0,
                'total_predictions_30d': history['n_predictions'].sum() if not history.empty and 'n_predictions' in history else 0
            }
        }
        
        return report
    
    def _get_recommendation(self, health: ModelHealth) -> str:
        """Get recommendation based on model health"""
        if health.status == ModelStatus.FAILED:
            return "URGENT: Model has failed. Immediate retraining required."
        elif health.status == ModelStatus.DEGRADED:
            return "WARNING: Model performance degraded. Schedule retraining."
        elif health.drift_detected:
            return "ATTENTION: Drift detected. Monitor closely and consider retraining."
        elif health.performance_trend == 'degrading':
            return "MONITOR: Performance trending down. Prepare for retraining."
        else:
            return "OK: Model performing within acceptable parameters."


def create_tracker(db_path: str = "model_performance.db") -> PerformanceTracker:
    """Create performance tracker instance"""
    return PerformanceTracker(db_path)


if __name__ == "__main__":
    # Example usage
    tracker = create_tracker()
    
    # Track some predictions
    for i in range(100):
        tracker.track_prediction(
            model_id="xgboost_v1",
            features={'feature_1': np.random.randn(), 'feature_2': np.random.randn()},
            prediction=np.random.random(),
            actual=np.random.randint(0, 2),
            confidence=np.random.random()
        )
    
    # Update performance metrics
    metrics = PerformanceMetrics(
        timestamp=datetime.now(),
        model_id="xgboost_v1",
        model_version="1.0",
        accuracy=0.75,
        precision=0.72,
        recall=0.78,
        f1_score=0.75,
        roc_auc=0.82,
        sharpe_ratio=1.2,
        max_drawdown=0.15,
        n_predictions=100
    )
    
    tracker.update_performance("xgboost_v1", metrics)
    
    # Check model health
    health = tracker.get_model_health("xgboost_v1")
    if health:
        print(f"Model Status: {health.status.value}")
        print(f"Current Accuracy: {health.current_accuracy:.3f}")
        print(f"Needs Retraining: {health.needs_retraining}")
    
    # Generate report
    report = tracker.generate_report("xgboost_v1")
    print("\nModel Report:")
    print(json.dumps(report, indent=2, default=str))