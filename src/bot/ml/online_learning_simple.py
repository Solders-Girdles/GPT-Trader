"""
Simplified Online Learning Pipeline for GPT-Trader
Phase 3 - ADAPT-001 through ADAPT-008: Complete online learning system

This is a simplified version that doesn't depend on complex base classes,
making it easier to test and integrate.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import pickle
import json
import warnings
from pathlib import Path
import threading
import time

# Machine learning imports
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Local imports - use absolute imports to avoid circular dependencies
from .learning_scheduler import LearningRateScheduler, SchedulerConfig, SchedulerType, create_scheduler
from .drift_detector import ConceptDriftDetector, DriftDetectorConfig, DriftDetection, create_drift_detector

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Online learning modes"""
    INCREMENTAL = "incremental"
    MINI_BATCH = "mini_batch"
    STREAM = "stream"
    ADAPTIVE = "adaptive"


class UpdateStrategy(Enum):
    """Model update strategies"""
    IMMEDIATE = "immediate"
    BATCH = "batch"
    DRIFT_TRIGGERED = "drift_triggered"
    SCHEDULED = "scheduled"


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning pipeline"""
    # Learning parameters
    learning_mode: LearningMode = LearningMode.ADAPTIVE
    update_strategy: UpdateStrategy = UpdateStrategy.DRIFT_TRIGGERED
    batch_size: int = 32
    min_batch_size: int = 10
    max_batch_size: int = 1000
    
    # Memory management
    memory_buffer_size: int = 10000
    priority_replay: bool = True
    importance_decay: float = 0.99
    
    # Model parameters
    base_estimator: str = "sgd"  # sgd, random_forest
    n_estimators_rf: int = 10  # For RandomForest incremental
    warm_start: bool = True
    
    # Feature engineering
    incremental_scaling: bool = True
    feature_update_frequency: int = 100
    correlation_update_frequency: int = 500
    
    # Performance monitoring
    performance_window: int = 1000
    convergence_patience: int = 100
    convergence_threshold: float = 1e-4
    min_samples_for_update: int = 10
    
    # Safety parameters
    max_learning_rate: float = 0.1
    min_learning_rate: float = 1e-6
    gradient_clip_value: float = 1.0
    
    # Scheduling
    scheduler_config: Optional[SchedulerConfig] = None
    drift_detector_config: Optional[DriftDetectorConfig] = None
    
    # Warm starting
    initial_training_size: int = 1000
    warmup_epochs: int = 5
    
    # Advanced features
    ensemble_learning: bool = False
    model_averaging: bool = True
    uncertainty_estimation: bool = True


@dataclass
class MemorySample:
    """Sample stored in memory buffer"""
    features: np.ndarray
    target: float
    timestamp: datetime
    importance: float = 1.0
    prediction: Optional[float] = None
    loss: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningCurve:
    """Learning curve tracking"""
    timestamps: List[datetime] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    sample_counts: List[int] = field(default_factory=list)
    drift_detections: List[datetime] = field(default_factory=list)


class SimpleOnlineLearningPipeline:
    """
    Simplified online learning pipeline with adaptive capabilities.
    
    Features:
    - SGD-based incremental learning
    - Adaptive learning rate scheduling
    - Concept drift detection and adaptation
    - Memory buffer with priority replay
    - Incremental feature engineering
    - Convergence monitoring
    - Warm starting and transfer learning
    """
    
    def __init__(self, config: OnlineLearningConfig):
        """Initialize online learning pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Initialize models
        self.primary_model = None
        self.backup_model = None
        self.model_history = deque(maxlen=5)
        
        # Initialize components
        self.scheduler = create_scheduler(
            config.scheduler_config.scheduler_type if config.scheduler_config else SchedulerType.ADAPTIVE,
            **(config.scheduler_config.__dict__ if config.scheduler_config else {})
        )
        
        self.drift_detector = create_drift_detector(
            **(config.drift_detector_config.__dict__ if config.drift_detector_config else {})
        )
        
        # Memory management
        self.memory_buffer = deque(maxlen=config.memory_buffer_size)
        self.validation_buffer = deque(maxlen=config.memory_buffer_size // 10)
        
        # Feature engineering
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_stats = {}
        self.feature_correlations = {}
        self.feature_names = []
        
        # Performance tracking
        self.learning_curve = LearningCurve()
        self.performance_history = deque(maxlen=config.performance_window)
        self.sample_count = 0
        self.last_update_time = datetime.now()
        self.convergence_counter = 0
        
        # State management
        self.is_initialized = False
        self.is_converged = False
        self.last_drift_adaptation = None
        self.adaptation_count = 0
        
        # Threading for async operations
        self.update_lock = threading.Lock()
        self.background_thread = None
        self.should_stop = False
        
        logger.info(f"Initialized simple online learning pipeline with {config.learning_mode.value} mode")
    
    def initialize(self, 
                   initial_data: pd.DataFrame, 
                   initial_targets: pd.Series,
                   feature_names: Optional[List[str]] = None):
        """
        Initialize pipeline with initial training data.
        
        Args:
            initial_data: Initial training features
            initial_targets: Initial training targets
            feature_names: Names of features
        """
        logger.info("Initializing online learning pipeline with initial data")
        
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = list(initial_data.columns) if hasattr(initial_data, 'columns') else [f"feature_{i}" for i in range(initial_data.shape[1])]
        
        # Initialize feature scaling
        if self.config.incremental_scaling:
            self.scaler.fit(initial_data)
            scaled_data = self.scaler.transform(initial_data)
        else:
            scaled_data = initial_data.values if hasattr(initial_data, 'values') else initial_data
        
        # Initialize label encoding for classification
        if self._is_classification_task(initial_targets):
            unique_labels = initial_targets.unique()
            if len(unique_labels) > 2 or not all(label in [0, 1] for label in unique_labels):
                self.label_encoder.fit(initial_targets)
                encoded_targets = self.label_encoder.transform(initial_targets)
            else:
                encoded_targets = initial_targets.values
        else:
            encoded_targets = initial_targets.values
        
        # Create initial model
        self.primary_model = self._create_model()
        
        # Initial training
        if self.config.base_estimator == "sgd":
            # Warm-up training for SGD
            for epoch in range(self.config.warmup_epochs):
                self.primary_model.partial_fit(scaled_data, encoded_targets)
        else:
            # Full fit for other models
            self.primary_model.fit(scaled_data, encoded_targets)
        
        # Initialize feature statistics
        self._update_feature_statistics(initial_data)
        
        # Store initial samples in memory buffer
        for i in range(len(initial_data)):
            sample = MemorySample(
                features=scaled_data[i],
                target=encoded_targets[i],
                timestamp=datetime.now(),
                importance=1.0
            )
            self.memory_buffer.append(sample)
        
        # Create backup model
        self.backup_model = self._create_model()
        if self.config.base_estimator == "sgd":
            self.backup_model.partial_fit(scaled_data, encoded_targets)
        else:
            self.backup_model.fit(scaled_data, encoded_targets)
        
        self.sample_count = len(initial_data)
        self.is_initialized = True
        
        logger.info(f"Pipeline initialized with {len(initial_data)} samples")
    
    def update(self, 
               features: Union[np.ndarray, pd.DataFrame, pd.Series],
               target: Union[float, int],
               force_update: bool = False) -> Dict[str, Any]:
        """
        Update model with new sample.
        
        Args:
            features: New feature vector
            target: True target value
            force_update: Force immediate model update
            
        Returns:
            Update result with metrics
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        with self.update_lock:
            return self._internal_update(features, target, force_update)
    
    def _internal_update(self, features, target, force_update):
        """Internal update method (thread-safe)"""
        start_time = time.time()
        
        # Convert inputs to arrays
        if isinstance(features, pd.DataFrame):
            feature_array = features.values[0] if len(features) == 1 else features.values
        elif isinstance(features, pd.Series):
            feature_array = features.values
        else:
            feature_array = np.array(features).reshape(1, -1) if np.array(features).ndim == 1 else features
        
        # Ensure 1D array for single sample
        if feature_array.ndim == 2 and feature_array.shape[0] == 1:
            feature_array = feature_array[0]
        
        # Preprocess features
        if self.config.incremental_scaling:
            try:
                scaled_features = self.scaler.transform(feature_array.reshape(1, -1))[0]
            except Exception as e:
                logger.warning(f"Scaling failed, using raw features: {e}")
                scaled_features = feature_array
        else:
            scaled_features = feature_array
        
        # Encode target if necessary
        if hasattr(self.label_encoder, 'classes_'):
            try:
                encoded_target = self.label_encoder.transform([target])[0]
            except ValueError:
                # New class, expand encoder
                self.label_encoder.classes_ = np.append(self.label_encoder.classes_, target)
                encoded_target = self.label_encoder.transform([target])[0]
        else:
            encoded_target = target
        
        # Make prediction for drift detection and importance calculation
        prediction = self.predict(feature_array.reshape(1, -1))[0] if feature_array.ndim == 1 else self.predict(feature_array)[0]
        
        # Calculate loss for importance weighting
        if self._is_classification_task([encoded_target]):
            try:
                loss = log_loss([encoded_target], [prediction]) if 0 < prediction < 1 else abs(prediction - encoded_target)
            except:
                loss = abs(prediction - encoded_target)
        else:
            loss = (prediction - encoded_target) ** 2
        
        # Create memory sample
        sample = MemorySample(
            features=scaled_features,
            target=encoded_target,
            timestamp=datetime.now(),
            importance=min(2.0, 1.0 + loss),  # Higher importance for harder samples
            prediction=prediction,
            loss=loss
        )
        
        # Add to memory buffer
        self.memory_buffer.append(sample)
        
        # Update feature statistics
        if self.sample_count % self.config.feature_update_frequency == 0:
            self._update_feature_statistics_incremental(feature_array)
        
        # Drift detection
        drift_detection = self.drift_detector.add_sample(
            features=feature_array.reshape(1, -1) if feature_array.ndim == 1 else feature_array,
            target=encoded_target,
            prediction=prediction
        )
        
        # Determine if update is needed
        should_update = (
            force_update or
            len(self.memory_buffer) >= self.config.batch_size or
            (drift_detection is not None) or
            (self.config.update_strategy == UpdateStrategy.IMMEDIATE)
        )
        
        update_result = {
            'sample_added': True,
            'model_updated': False,
            'drift_detected': drift_detection is not None,
            'prediction': prediction,
            'loss': loss,
            'processing_time': time.time() - start_time
        }
        
        # Perform model update if needed
        if should_update:
            update_metrics = self._perform_model_update(drift_detection)
            update_result.update(update_metrics)
        
        self.sample_count += 1
        return update_result
    
    def _perform_model_update(self, drift_detection: Optional[DriftDetection]) -> Dict[str, Any]:
        """Perform actual model update"""
        update_start = time.time()
        
        # Prepare batch for training
        if self.config.priority_replay and len(self.memory_buffer) > self.config.min_batch_size:
            batch_samples = self._sample_priority_batch()
        else:
            batch_size = min(self.config.batch_size, len(self.memory_buffer))
            batch_samples = list(self.memory_buffer)[-batch_size:]
        
        if len(batch_samples) < self.config.min_batch_size:
            return {'model_updated': False, 'reason': 'insufficient_samples'}
        
        # Prepare batch data
        X_batch = np.array([sample.features for sample in batch_samples])
        y_batch = np.array([sample.target for sample in batch_samples])
        
        # Handle drift adaptation
        if drift_detection is not None:
            adaptation_result = self._adapt_to_drift(drift_detection, X_batch, y_batch)
            self.last_drift_adaptation = datetime.now()
            self.adaptation_count += 1
        else:
            adaptation_result = {}
        
        # Update learning rate
        recent_performance = np.mean(self.performance_history) if self.performance_history else 0.5
        current_lr = self.scheduler.step(recent_performance)
        
        # Update model
        try:
            if self.config.base_estimator == "sgd" and hasattr(self.primary_model, 'partial_fit'):
                # Set learning rate if supported
                if hasattr(self.primary_model, 'set_params'):
                    self.primary_model.set_params(learning_rate='constant', eta0=current_lr)
                
                self.primary_model.partial_fit(X_batch, y_batch)
            
            else:
                # For models that don't support partial_fit, retrain on recent data
                recent_size = min(1000, len(self.memory_buffer))
                recent_samples = list(self.memory_buffer)[-recent_size:]
                X_recent = np.array([sample.features for sample in recent_samples])
                y_recent = np.array([sample.target for sample in recent_samples])
                
                self.primary_model.fit(X_recent, y_recent)
            
            # Update backup model occasionally
            if self.sample_count % 100 == 0:
                self._update_backup_model(X_batch, y_batch)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(batch_samples)
            
            # Update learning curve
            self._update_learning_curve(performance_metrics, current_lr)
            
            # Check convergence
            convergence_status = self._check_convergence(performance_metrics)
            
            # Decay importance values
            self._decay_importance_values()
            
            self.last_update_time = datetime.now()
            
            return {
                'model_updated': True,
                'batch_size': len(batch_samples),
                'learning_rate': current_lr,
                'performance_metrics': performance_metrics,
                'convergence_status': convergence_status,
                'adaptation_result': adaptation_result,
                'update_time': time.time() - update_start
            }
        
        except Exception as e:
            logger.error(f"Model update failed: {e}")
            return {
                'model_updated': False,
                'error': str(e),
                'update_time': time.time() - update_start
            }
    
    def _sample_priority_batch(self) -> List[MemorySample]:
        """Sample batch using priority replay"""
        if len(self.memory_buffer) <= self.config.batch_size:
            return list(self.memory_buffer)
        
        # Calculate sampling probabilities based on importance
        importances = np.array([sample.importance for sample in self.memory_buffer])
        probabilities = importances / np.sum(importances)
        
        # Sample indices
        indices = np.random.choice(
            len(self.memory_buffer),
            size=min(self.config.batch_size, len(self.memory_buffer)),
            replace=False,
            p=probabilities
        )
        
        return [self.memory_buffer[i] for i in indices]
    
    def _adapt_to_drift(self, drift_detection: DriftDetection, X_batch: np.ndarray, y_batch: np.ndarray) -> Dict[str, Any]:
        """Adapt model to detected concept drift"""
        logger.info(f"Adapting to {drift_detection.drift_severity.value} concept drift")
        
        adaptation_result = {
            'drift_type': drift_detection.drift_type.value if drift_detection.drift_type else 'unknown',
            'drift_severity': drift_detection.drift_severity.value,
            'adaptation_strategy': None
        }
        
        if drift_detection.drift_severity.value in ['high', 'critical']:
            # Severe drift: reset model or use backup
            if self.backup_model is not None:
                logger.info("Switching to backup model due to severe drift")
                self.primary_model = self.backup_model
                self.backup_model = self._create_model()
                adaptation_result['adaptation_strategy'] = 'model_switch'
            else:
                logger.info("Resetting model due to severe drift")
                self.primary_model = self._create_model()
                # Warm-up with recent data
                recent_samples = list(self.memory_buffer)[-self.config.initial_training_size:]
                if len(recent_samples) >= 10:
                    X_recent = np.array([s.features for s in recent_samples])
                    y_recent = np.array([s.target for s in recent_samples])
                    if self.config.base_estimator == "sgd":
                        self.primary_model.partial_fit(X_recent, y_recent)
                    else:
                        self.primary_model.fit(X_recent, y_recent)
                adaptation_result['adaptation_strategy'] = 'model_reset'
        
        elif drift_detection.drift_severity.value == 'medium':
            # Moderate drift: increase learning rate and focus on recent data
            self.scheduler.current_lr *= 1.5  # Temporary increase
            adaptation_result['adaptation_strategy'] = 'increased_learning_rate'
        
        else:
            # Low drift: just increase model update frequency
            adaptation_result['adaptation_strategy'] = 'increased_updates'
        
        return adaptation_result
    
    def _update_backup_model(self, X_batch: np.ndarray, y_batch: np.ndarray):
        """Update backup model"""
        if self.backup_model is None:
            self.backup_model = self._create_model()
        
        try:
            if self.config.base_estimator == "sgd" and hasattr(self.backup_model, 'partial_fit'):
                self.backup_model.partial_fit(X_batch, y_batch)
            else:
                # Use larger sample for backup model
                recent_size = min(2000, len(self.memory_buffer))
                recent_samples = list(self.memory_buffer)[-recent_size:]
                X_recent = np.array([sample.features for sample in recent_samples])
                y_recent = np.array([sample.target for sample in recent_samples])
                self.backup_model.fit(X_recent, y_recent)
        except Exception as e:
            logger.warning(f"Backup model update failed: {e}")
    
    def _calculate_performance_metrics(self, batch_samples: List[MemorySample]) -> Dict[str, float]:
        """Calculate performance metrics on batch"""
        if len(batch_samples) < 2:
            return {}
        
        X = np.array([sample.features for sample in batch_samples])
        y_true = np.array([sample.target for sample in batch_samples])
        
        try:
            y_pred = self.primary_model.predict(X)
            
            metrics = {}
            
            if self._is_classification_task(y_true):
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                if hasattr(self.primary_model, 'predict_proba'):
                    try:
                        y_proba = self.primary_model.predict_proba(X)
                        if y_proba.shape[1] > 1:  # Multi-class
                            metrics['log_loss'] = log_loss(y_true, y_proba)
                    except:
                        pass
            else:
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
            
            # Calculate loss for learning curve
            if 'accuracy' in metrics:
                metrics['loss'] = 1.0 - metrics['accuracy']
            elif 'mse' in metrics:
                metrics['loss'] = metrics['mse']
            
            # Update performance history
            main_metric = metrics.get('accuracy', 1.0 - metrics.get('mse', 1.0))
            self.performance_history.append(main_metric)
            
            return metrics
        
        except Exception as e:
            logger.warning(f"Performance calculation failed: {e}")
            return {}
    
    def _update_learning_curve(self, performance_metrics: Dict[str, float], learning_rate: float):
        """Update learning curve tracking"""
        self.learning_curve.timestamps.append(datetime.now())
        self.learning_curve.learning_rates.append(learning_rate)
        self.learning_curve.sample_counts.append(self.sample_count)
        
        if 'loss' in performance_metrics:
            self.learning_curve.train_losses.append(performance_metrics['loss'])
        
        # Calculate validation loss if we have validation data
        if len(self.validation_buffer) > 10:
            val_samples = list(self.validation_buffer)[-50:]
            X_val = np.array([sample.features for sample in val_samples])
            y_val = np.array([sample.target for sample in val_samples])
            
            try:
                y_pred = self.primary_model.predict(X_val)
                if self._is_classification_task(y_val):
                    val_loss = 1.0 - accuracy_score(y_val, y_pred)
                else:
                    val_loss = mean_squared_error(y_val, y_pred)
                
                self.learning_curve.val_losses.append(val_loss)
            except:
                self.learning_curve.val_losses.append(0.0)
        else:
            self.learning_curve.val_losses.append(0.0)
        
        # Keep only recent history
        max_history = 1000
        if len(self.learning_curve.timestamps) > max_history:
            self.learning_curve.timestamps = self.learning_curve.timestamps[-max_history:]
            self.learning_curve.train_losses = self.learning_curve.train_losses[-max_history:]
            self.learning_curve.val_losses = self.learning_curve.val_losses[-max_history:]
            self.learning_curve.learning_rates = self.learning_curve.learning_rates[-max_history:]
            self.learning_curve.sample_counts = self.learning_curve.sample_counts[-max_history:]
    
    def _check_convergence(self, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check if model has converged"""
        convergence_status = {
            'is_converged': False,
            'patience_counter': self.convergence_counter,
            'reason': None
        }
        
        if len(self.performance_history) < self.config.convergence_patience:
            convergence_status['reason'] = 'insufficient_history'
            return convergence_status
        
        # Check if performance has stabilized
        recent_performance = list(self.performance_history)[-self.config.convergence_patience:]
        performance_std = np.std(recent_performance)
        performance_mean = np.mean(recent_performance)
        
        cv = performance_std / performance_mean if performance_mean > 0 else float('inf')
        
        if cv < self.config.convergence_threshold:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0
        
        # Check learning rate convergence
        lr_converged = self.scheduler.is_converged()
        
        if self.convergence_counter >= self.config.convergence_patience and lr_converged:
            self.is_converged = True
            convergence_status['is_converged'] = True
            convergence_status['reason'] = 'performance_stabilized'
            logger.info("Model convergence detected")
        
        convergence_status['performance_cv'] = cv
        convergence_status['lr_converged'] = lr_converged
        
        return convergence_status
    
    def _decay_importance_values(self):
        """Decay importance values in memory buffer"""
        for sample in self.memory_buffer:
            sample.importance *= self.config.importance_decay
    
    def predict(self, features: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the current model.
        
        Args:
            features: Feature matrix
            
        Returns:
            Predictions array
        """
        if not self.is_initialized or self.primary_model is None:
            raise RuntimeError("Model not initialized")
        
        # Preprocess features
        if isinstance(features, pd.DataFrame):
            feature_array = features.values
        else:
            feature_array = np.array(features)
        
        if self.config.incremental_scaling:
            try:
                scaled_features = self.scaler.transform(feature_array)
            except Exception as e:
                logger.warning(f"Scaling failed during prediction: {e}")
                scaled_features = feature_array
        else:
            scaled_features = feature_array
        
        # Make prediction
        predictions = self.primary_model.predict(scaled_features)
        
        # Decode predictions if necessary
        if hasattr(self.label_encoder, 'classes_'):
            try:
                predictions = self.label_encoder.inverse_transform(predictions.astype(int))
            except:
                pass  # Keep original predictions if decoding fails
        
        return predictions
    
    def predict_proba(self, features: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get prediction probabilities (for classification models).
        
        Args:
            features: Feature matrix
            
        Returns:
            Probability array
        """
        if not self.is_initialized or self.primary_model is None:
            raise RuntimeError("Model not initialized")
        
        if not hasattr(self.primary_model, 'predict_proba'):
            raise NotImplementedError("Model does not support probability predictions")
        
        # Preprocess features
        if isinstance(features, pd.DataFrame):
            feature_array = features.values
        else:
            feature_array = np.array(features)
        
        if self.config.incremental_scaling:
            try:
                scaled_features = self.scaler.transform(feature_array)
            except Exception as e:
                logger.warning(f"Scaling failed during prediction: {e}")
                scaled_features = feature_array
        else:
            scaled_features = feature_array
        
        return self.primary_model.predict_proba(scaled_features)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'is_initialized': self.is_initialized,
            'is_converged': self.is_converged,
            'sample_count': self.sample_count,
            'memory_buffer_size': len(self.memory_buffer),
            'adaptation_count': self.adaptation_count,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'last_drift_adaptation': self.last_drift_adaptation.isoformat() if self.last_drift_adaptation else None,
            'current_learning_rate': self.scheduler.get_current_lr(),
            'performance_history_size': len(self.performance_history),
            'recent_performance': list(self.performance_history)[-10:] if self.performance_history else [],
            'scheduler_stats': self.scheduler.get_statistics(),
            'drift_detector_stats': self.drift_detector.get_statistics(),
            'feature_names': self.feature_names,
            'model_type': self.config.base_estimator
        }
    
    def get_learning_curve(self) -> LearningCurve:
        """Get learning curve data"""
        return self.learning_curve
    
    def _create_model(self):
        """Create a new model instance"""
        if self.config.base_estimator == "sgd":
            # Use SGDClassifier by default
            return SGDClassifier(
                loss='log_loss',
                learning_rate='constant',
                eta0=self.config.max_learning_rate / 10,
                random_state=42,
                warm_start=self.config.warm_start
            )
        
        elif self.config.base_estimator == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators_rf,
                warm_start=self.config.warm_start,
                random_state=42,
                n_jobs=1  # Avoid threading issues
            )
        
        else:
            raise ValueError(f"Unsupported base estimator: {self.config.base_estimator}")
    
    def _is_classification_task(self, targets) -> bool:
        """Determine if this is a classification task"""
        unique_targets = np.unique(targets)
        return len(unique_targets) <= 10 and all(isinstance(t, (int, np.integer)) or t in [0, 1] for t in unique_targets)
    
    def _update_feature_statistics(self, features: pd.DataFrame):
        """Update feature statistics (initial)"""
        for col in features.columns:
            if pd.api.types.is_numeric_dtype(features[col]):
                self.feature_stats[col] = {
                    'mean': float(features[col].mean()),
                    'std': float(features[col].std()),
                    'min': float(features[col].min()),
                    'max': float(features[col].max())
                }
    
    def _update_feature_statistics_incremental(self, features: np.ndarray):
        """Update feature statistics incrementally"""
        # Simple incremental update - could be made more sophisticated
        for i, feature_name in enumerate(self.feature_names[:len(features)]):
            if feature_name in self.feature_stats:
                old_stats = self.feature_stats[feature_name]
                new_value = features[i]
                
                # Update with exponential moving average
                alpha = 0.01  # Learning rate for statistics
                old_stats['mean'] = (1 - alpha) * old_stats['mean'] + alpha * new_value
                
                # Update other statistics similarly
                if new_value < old_stats['min']:
                    old_stats['min'] = new_value
                if new_value > old_stats['max']:
                    old_stats['max'] = new_value


# Factory function
def create_simple_online_learning_pipeline(config_dict: Optional[Dict] = None) -> SimpleOnlineLearningPipeline:
    """
    Factory function to create simple online learning pipeline.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configured online learning pipeline
    """
    if config_dict is None:
        config_dict = {}
    
    config = OnlineLearningConfig(**config_dict)
    return SimpleOnlineLearningPipeline(config)


# Predefined configurations
CONSERVATIVE_CONFIG = OnlineLearningConfig(
    learning_mode=LearningMode.MINI_BATCH,
    update_strategy=UpdateStrategy.BATCH,
    batch_size=64,
    memory_buffer_size=5000,
    convergence_patience=200
)

AGGRESSIVE_CONFIG = OnlineLearningConfig(
    learning_mode=LearningMode.STREAM,
    update_strategy=UpdateStrategy.IMMEDIATE,
    batch_size=16,
    memory_buffer_size=20000,
    convergence_patience=50
)

DRIFT_ADAPTIVE_CONFIG = OnlineLearningConfig(
    learning_mode=LearningMode.ADAPTIVE,
    update_strategy=UpdateStrategy.DRIFT_TRIGGERED,
    batch_size=32,
    memory_buffer_size=10000,
    priority_replay=True,
    convergence_patience=100
)