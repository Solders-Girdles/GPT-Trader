"""
Integrated ML Pipeline
Phase 2.5 - Day 6

Integrates feature engineering, selection, validation, and tracking.
Provides a unified interface for the ML components.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import json

# Import our ML components
from .feature_engineering_v2 import OptimizedFeatureEngineer, FeatureConfig
from .feature_selector import AdvancedFeatureSelector, FeatureSelectionConfig, SelectionMethod
from .model_validation import ModelValidator, ValidationConfig, ModelPerformance
from .performance_tracker import PerformanceTracker, PerformanceMetrics

logger = logging.getLogger(__name__)


class IntegratedMLPipeline:
    """
    Unified ML pipeline integrating all components.
    
    Components:
    - Feature engineering (feature_engineering_v2.py)
    - Feature selection (feature_selector.py)
    - Model validation (model_validation.py)
    - Performance tracking (performance_tracker.py)
    
    This ensures seamless integration with existing codebase.
    """
    
    def __init__(self, 
                 feature_config: Optional[FeatureConfig] = None,
                 selection_config: Optional[FeatureSelectionConfig] = None,
                 validation_config: Optional[ValidationConfig] = None,
                 tracking_db_path: str = "model_performance.db"):
        
        # Initialize components
        self.feature_engineer = OptimizedFeatureEngineer(feature_config)
        self.feature_selector = AdvancedFeatureSelector(selection_config)
        self.model_validator = ModelValidator(validation_config)
        self.performance_tracker = PerformanceTracker(tracking_db_path)
        
        # Pipeline state
        self.current_features: Optional[pd.DataFrame] = None
        self.selected_feature_names: Optional[List[str]] = None
        self.current_model = None
        self.pipeline_config = {}
        
        logger.info("IntegratedMLPipeline initialized with all components")
    
    def prepare_features(self, 
                        raw_data: pd.DataFrame,
                        target: Optional[pd.Series] = None,
                        use_selection: bool = True) -> pd.DataFrame:
        """
        Prepare features from raw OHLCV data.
        
        Args:
            raw_data: DataFrame with OHLCV columns
            target: Optional target variable for supervised selection
            use_selection: Whether to apply feature selection
            
        Returns:
            Processed feature DataFrame
        """
        logger.info("Starting feature preparation pipeline")
        
        # Step 1: Engineer features
        logger.info("Step 1: Engineering features...")
        all_features = self.feature_engineer.engineer_features(raw_data)
        logger.info(f"Engineered {len(all_features.columns)} features")
        
        # Step 2: Apply feature selection if requested
        if use_selection and target is not None:
            logger.info("Step 2: Applying feature selection...")
            
            # Align features and target
            common_index = all_features.index.intersection(target.index)
            features_aligned = all_features.loc[common_index]
            target_aligned = target.loc[common_index]
            
            # Select features using ensemble method
            selected_features, scores = self.feature_selector.select_features(
                features_aligned, 
                target_aligned,
                method=SelectionMethod.ENSEMBLE
            )
            
            # Keep only selected features
            self.selected_feature_names = selected_features
            self.current_features = all_features[selected_features]
            
            logger.info(f"Selected {len(selected_features)} features")
            
            # Log feature importance
            self._log_feature_importance(scores)
            
        else:
            logger.info("Step 2: Skipping feature selection")
            self.current_features = all_features
            self.selected_feature_names = list(all_features.columns)
        
        return self.current_features
    
    def train_and_validate(self,
                          model_class: Any,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_test: pd.DataFrame,
                          y_test: pd.Series,
                          model_params: Optional[Dict] = None,
                          model_name: str = "model") -> ModelPerformance:
        """
        Train model with validation and performance tracking.
        
        Args:
            model_class: Model class to instantiate
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            model_params: Model hyperparameters
            model_name: Name for tracking
            
        Returns:
            Model performance metrics
        """
        if model_params is None:
            model_params = {}
        
        logger.info(f"Training {model_name} with validation")
        
        # Instantiate and train model
        model = model_class(**model_params)
        
        # Train with timing
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Validate model
        performance = self.model_validator.validate_model(
            model, X_train, y_train, X_test, y_test,
            model_name=model_name
        )
        performance.training_time = training_time
        
        # Track performance
        self._track_model_performance(model_name, performance)
        
        # Store current model
        self.current_model = model
        
        return performance
    
    def walk_forward_validation(self,
                              model_class: Any,
                              X: pd.DataFrame,
                              y: pd.Series,
                              model_params: Optional[Dict] = None,
                              model_name: str = "model") -> List[ModelPerformance]:
        """
        Perform walk-forward validation.
        
        Args:
            model_class: Model class to use
            X: Feature matrix
            y: Target variable
            model_params: Model parameters
            model_name: Model name for tracking
            
        Returns:
            List of performance results for each period
        """
        logger.info(f"Starting walk-forward validation for {model_name}")
        
        results = self.model_validator.walk_forward_analysis(
            model_class, X, y, model_params
        )
        
        # Track each period's performance
        for i, performance in enumerate(results):
            performance.model_name = f"{model_name}_fold_{i}"
            self._track_model_performance(performance.model_name, performance)
        
        # Log summary statistics
        accuracies = [p.accuracy for p in results]
        logger.info(f"Walk-forward results: Mean accuracy={np.mean(accuracies):.3f}, "
                   f"Std={np.std(accuracies):.3f}")
        
        return results
    
    def check_model_health(self, model_id: str) -> Dict[str, Any]:
        """
        Check model health and drift status.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Health status dictionary
        """
        health = self.performance_tracker.get_model_health(model_id)
        
        if health:
            # Check if retraining needed
            needs_retraining = self.performance_tracker.check_retraining_needed(model_id)
            
            return {
                'model_id': model_id,
                'status': health.status.value,
                'current_accuracy': health.current_accuracy,
                'baseline_accuracy': health.baseline_accuracy,
                'trend': health.performance_trend,
                'drift_detected': health.drift_detected,
                'needs_retraining': needs_retraining,
                'confidence': health.confidence_score,
                'last_update': health.last_update.isoformat()
            }
        
        return {'model_id': model_id, 'status': 'not_found'}
    
    def detect_feature_drift(self,
                            current_data: pd.DataFrame,
                            reference_data: pd.DataFrame,
                            model_id: str) -> Tuple[bool, float]:
        """
        Detect feature drift between current and reference data.
        
        Args:
            current_data: Recent feature data
            reference_data: Reference/training feature data
            model_id: Model identifier
            
        Returns:
            Tuple of (drift_detected, drift_score)
        """
        # Prepare features for both datasets
        current_features = self.feature_engineer.engineer_features(current_data)
        reference_features = self.feature_engineer.engineer_features(reference_data)
        
        # Select same features
        if self.selected_feature_names:
            current_features = current_features[self.selected_feature_names]
            reference_features = reference_features[self.selected_feature_names]
        
        # Detect drift
        drift_detected, drift_type, drift_score = self.performance_tracker.detect_drift(
            model_id, current_features, reference_features
        )
        
        if drift_detected:
            logger.warning(f"Feature drift detected for {model_id}: "
                         f"type={drift_type.value}, score={drift_score:.3f}")
        
        return drift_detected, drift_score
    
    def save_pipeline(self, directory: str):
        """
        Save entire pipeline configuration and models.
        
        Args:
            directory: Directory to save pipeline
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save feature engineering config
        feature_config = {
            'selected_features': self.feature_engineer.selected_features,
            'feature_importance': self.feature_engineer.feature_importance,
            'config': {
                'max_features': self.feature_engineer.config.max_features,
                'correlation_threshold': self.feature_engineer.config.correlation_threshold
            }
        }
        with open(dir_path / 'feature_config.json', 'w') as f:
            json.dump(feature_config, f, indent=2)
        
        # Save feature selection config
        self.feature_selector.save_selection_config(
            str(dir_path / 'selection_config.json')
        )
        
        # Save scaler if exists
        if self.feature_engineer.scaler:
            joblib.dump(
                self.feature_engineer.scaler,
                dir_path / 'feature_scaler.joblib'
            )
        
        # Save current model if exists
        if self.current_model:
            joblib.dump(
                self.current_model,
                dir_path / 'model.joblib'
            )
        
        # Save pipeline config
        self.pipeline_config = {
            'n_features': len(self.selected_feature_names) if self.selected_feature_names else 0,
            'feature_names': self.selected_feature_names,
            'timestamp': datetime.now().isoformat()
        }
        with open(dir_path / 'pipeline_config.json', 'w') as f:
            json.dump(self.pipeline_config, f, indent=2)
        
        logger.info(f"Pipeline saved to {directory}")
    
    def load_pipeline(self, directory: str):
        """
        Load pipeline configuration and models.
        
        Args:
            directory: Directory containing saved pipeline
        """
        dir_path = Path(directory)
        
        # Load feature config
        with open(dir_path / 'feature_config.json', 'r') as f:
            feature_config = json.load(f)
            self.feature_engineer.selected_features = feature_config['selected_features']
            self.feature_engineer.feature_importance = feature_config['feature_importance']
        
        # Load selection config
        self.feature_selector.load_selection_config(
            str(dir_path / 'selection_config.json')
        )
        
        # Load scaler
        scaler_path = dir_path / 'feature_scaler.joblib'
        if scaler_path.exists():
            self.feature_engineer.scaler = joblib.load(scaler_path)
        
        # Load model
        model_path = dir_path / 'model.joblib'
        if model_path.exists():
            self.current_model = joblib.load(model_path)
        
        # Load pipeline config
        with open(dir_path / 'pipeline_config.json', 'r') as f:
            self.pipeline_config = json.load(f)
            self.selected_feature_names = self.pipeline_config.get('feature_names')
        
        logger.info(f"Pipeline loaded from {directory}")
    
    def _track_model_performance(self, model_id: str, performance: ModelPerformance):
        """Track model performance in database"""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            model_id=model_id,
            model_version="1.0",
            accuracy=performance.accuracy,
            precision=performance.precision,
            recall=performance.recall,
            f1_score=performance.f1_score,
            roc_auc=performance.roc_auc,
            n_predictions=performance.n_samples_test or 0,
            n_features=len(self.selected_feature_names) if self.selected_feature_names else 0
        )
        
        self.performance_tracker.update_performance(model_id, metrics)
    
    def _log_feature_importance(self, scores: pd.DataFrame):
        """Log feature importance scores"""
        if scores is not None and not scores.empty:
            # Get top 10 features
            top_features = scores.head(10)
            
            logger.info("Top 10 features by importance:")
            for i, row in top_features.iterrows():
                if isinstance(row, pd.Series):
                    feature_name = row.get('feature', i)
                    score = row.drop('feature').max() if 'feature' in row else row.max()
                    logger.info(f"  {feature_name}: {score:.4f}")
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline configuration and performance"""
        summary = {
            'feature_engineering': {
                'total_features_engineered': len(self.feature_engineer.selected_features) 
                    if hasattr(self.feature_engineer, 'selected_features') else 0,
                'features_after_selection': len(self.selected_feature_names) 
                    if self.selected_feature_names else 0,
                'correlation_threshold': self.feature_selector.config.correlation_threshold,
                'selection_method': self.feature_selector.config.primary_method.value
            },
            'validation': {
                'cv_folds': self.model_validator.config.n_splits,
                'test_size': self.model_validator.config.test_size,
                'primary_metric': self.model_validator.config.primary_metric
            },
            'performance_tracking': {
                'models_tracked': len(self.performance_tracker.models),
                'database_path': self.performance_tracker.db_path
            },
            'current_model': {
                'loaded': self.current_model is not None,
                'type': type(self.current_model).__name__ if self.current_model else None
            }
        }
        
        return summary


def create_integrated_pipeline(**kwargs) -> IntegratedMLPipeline:
    """
    Create integrated ML pipeline with custom configuration.
    
    Args:
        feature_config: Feature engineering configuration
        selection_config: Feature selection configuration
        validation_config: Validation configuration
        tracking_db_path: Path to performance tracking database
        
    Returns:
        Configured pipeline instance
    """
    return IntegratedMLPipeline(**kwargs)


if __name__ == "__main__":
    # Example usage showing integration
    import yfinance as yf
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    
    # Get sample data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="2y")
    data.columns = [c.lower() for c in data.columns]
    
    # Create target (next day return direction)
    target = (data['close'].pct_change().shift(-1) > 0).astype(int)
    
    # Create pipeline with aggressive feature selection
    pipeline = create_integrated_pipeline(
        selection_config=FeatureSelectionConfig(
            n_features_target=50,
            correlation_threshold=0.7,  # Aggressive
            primary_method=SelectionMethod.ENSEMBLE
        ),
        validation_config=ValidationConfig(
            n_splits=5,
            primary_metric='f1'
        )
    )
    
    # Prepare features
    features = pipeline.prepare_features(data, target, use_selection=True)
    print(f"Prepared {features.shape[1]} features from {len(data)} samples")
    
    # Split data
    split_idx = int(len(features) * 0.8)
    X_train = features.iloc[:split_idx]
    y_train = target.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_test = target.iloc[split_idx:]
    
    # Train and validate models
    models = {
        'RandomForest': (RandomForestClassifier, {'n_estimators': 100, 'random_state': 42}),
        'XGBoost': (XGBClassifier, {'n_estimators': 100, 'random_state': 42, 'eval_metric': 'logloss'})
    }
    
    for model_name, (model_class, params) in models.items():
        print(f"\nTraining {model_name}...")
        performance = pipeline.train_and_validate(
            model_class, X_train, y_train, X_test, y_test,
            model_params=params,
            model_name=model_name
        )
        print(f"{model_name} - Accuracy: {performance.accuracy:.3f}, F1: {performance.f1_score:.3f}")
        
        # Check model health
        health = pipeline.check_model_health(model_name)
        print(f"Model health: {health['status']}, Confidence: {health.get('confidence', 0):.3f}")
    
    # Save pipeline
    pipeline.save_pipeline('ml_pipeline_v2')
    
    # Get summary
    summary = pipeline.get_pipeline_summary()
    print("\nPipeline Summary:")
    print(json.dumps(summary, indent=2))