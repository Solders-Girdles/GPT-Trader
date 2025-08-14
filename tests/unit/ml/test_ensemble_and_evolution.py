"""
Tests for Week 6 Ensemble Management and Feature Evolution
Phase 3, Week 6: ADAPT-017 to ADAPT-032
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


class TestEnsembleManager:
    """Test ensemble management system"""
    
    def test_ensemble_import(self):
        """Test ensemble manager import"""
        from src.bot.ml.ensemble_manager import (
            EnsembleManager,
            EnsembleMethod,
            WeightingStrategy,
            DynamicWeightOptimizer,
            BayesianModelAveraging
        )
        assert EnsembleManager is not None
        assert DynamicWeightOptimizer is not None
        assert BayesianModelAveraging is not None
    
    def test_ensemble_creation(self):
        """Test ensemble manager creation"""
        from src.bot.ml.ensemble_manager import EnsembleManager, EnsembleMethod
        
        ensemble = EnsembleManager(method=EnsembleMethod.DYNAMIC)
        assert ensemble is not None
        assert ensemble.method == EnsembleMethod.DYNAMIC
        assert ensemble.max_models == 10
    
    def test_add_models(self):
        """Test adding models to ensemble"""
        from src.bot.ml.ensemble_manager import EnsembleManager
        
        ensemble = EnsembleManager()
        
        # Add models
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
        
        assert ensemble.add_model("rf_model", rf)
        assert ensemble.add_model("gb_model", gb)
        assert len(ensemble.models) == 2
    
    def test_dynamic_weighting(self):
        """Test dynamic weight optimization"""
        from src.bot.ml.ensemble_manager import DynamicWeightOptimizer
        
        optimizer = DynamicWeightOptimizer()
        
        # Create sample predictions
        np.random.seed(42)
        predictions = np.random.randn(100, 3)  # 100 samples, 3 models
        targets = np.random.randn(100)
        
        # Optimize weights
        weights = optimizer.optimize_weights(predictions, targets)
        
        assert len(weights) == 3
        assert np.abs(np.sum(weights) - 1.0) < 0.01  # Sum to 1
        assert all(w >= 0 for w in weights)  # Non-negative
    
    def test_bayesian_averaging(self):
        """Test Bayesian model averaging"""
        from src.bot.ml.ensemble_manager import BayesianModelAveraging
        
        bma = BayesianModelAveraging()
        
        # Test weight calculation
        likelihoods = {
            'model1': 0.8,
            'model2': 0.6,
            'model3': 0.7
        }
        
        weights = bma.calculate_bma_weights(likelihoods)
        
        assert len(weights) == 3
        assert np.abs(sum(weights.values()) - 1.0) < 0.01
        assert weights['model1'] > weights['model2']  # Higher likelihood = higher weight
    
    def test_diversity_analysis(self):
        """Test ensemble diversity analysis"""
        from src.bot.ml.ensemble_manager import DiversityAnalyzer
        
        analyzer = DiversityAnalyzer()
        
        # Create diverse predictions
        np.random.seed(42)
        pred1 = np.random.randint(0, 2, 100)
        pred2 = np.random.randint(0, 2, 100)
        pred3 = 1 - pred1  # Opposite of pred1
        
        predictions = [pred1, pred2, pred3]
        
        # Calculate diversity
        disagreement = analyzer.calculate_disagreement(predictions)
        correlation_div = analyzer.calculate_correlation_diversity(predictions)
        
        assert 0 <= disagreement <= 1
        assert 0 <= correlation_div <= 1
        assert correlation_div > 0.3  # Should have some diversity
    
    def test_stacking_meta_learner(self):
        """Test stacking meta-learner"""
        from src.bot.ml.ensemble_manager import StackingMetaLearner
        
        meta_learner = StackingMetaLearner()
        
        # Create base predictions
        np.random.seed(42)
        base_preds = [
            np.random.randn(100),
            np.random.randn(100),
            np.random.randn(100)
        ]
        targets = np.random.randint(0, 2, 100)
        
        # Fit meta-learner
        meta_learner.fit(base_preds, targets)
        assert meta_learner.is_fitted
        
        # Make predictions
        test_preds = [
            np.random.randn(20),
            np.random.randn(20),
            np.random.randn(20)
        ]
        meta_predictions = meta_learner.predict(test_preds)
        assert len(meta_predictions) == 20


class TestFeatureEvolution:
    """Test feature evolution tracking"""
    
    def test_feature_evolution_import(self):
        """Test feature evolution import"""
        from src.bot.ml.feature_evolution import (
            FeatureEvolutionSystem,
            FeatureImportanceTracker,
            FeatureObsolescenceDetector,
            AdaptiveFeatureEngineer
        )
        assert FeatureEvolutionSystem is not None
        assert FeatureImportanceTracker is not None
        assert FeatureObsolescenceDetector is not None
        assert AdaptiveFeatureEngineer is not None
    
    def test_feature_registration(self):
        """Test feature registration"""
        from src.bot.ml.feature_evolution import FeatureEvolutionSystem, FeatureType
        
        system = FeatureEvolutionSystem()
        
        # Register features
        system.register_feature("feature_1", FeatureType.NUMERICAL)
        system.register_feature("feature_2", FeatureType.CATEGORICAL)
        
        assert len(system.features_metadata) == 2
        assert "feature_1" in system.features_metadata
    
    def test_importance_tracking(self):
        """Test feature importance tracking"""
        from src.bot.ml.feature_evolution import FeatureImportanceTracker
        
        tracker = FeatureImportanceTracker(window_size=10)
        
        # Update importance over time
        for i in range(5):
            importance = {
                'feature_1': 0.5 + i * 0.1,  # Increasing
                'feature_2': 0.5 - i * 0.1,  # Decreasing
                'feature_3': 0.5  # Stable
            }
            tracker.update_importance(importance)
        
        # Check trends
        trend_1 = tracker.calculate_importance_trend('feature_1')
        trend_2 = tracker.calculate_importance_trend('feature_2')
        trend_3 = tracker.calculate_importance_trend('feature_3')
        
        assert trend_1 > 0  # Increasing
        assert trend_2 < 0  # Decreasing
        assert abs(trend_3) < 0.1  # Stable
    
    def test_obsolescence_detection(self):
        """Test obsolescence detection"""
        from src.bot.ml.feature_evolution import (
            FeatureObsolescenceDetector,
            FeatureMetadata,
            FeatureType
        )
        
        detector = FeatureObsolescenceDetector(obsolescence_threshold=0.7)
        
        # Create feature with declining importance
        metadata = FeatureMetadata(
            name="old_feature",
            feature_type=FeatureType.NUMERICAL,
            created_at=datetime.now() - timedelta(days=60)
        )
        
        # Simulate declining importance
        for i in range(20):
            importance = 0.8 - i * 0.03
            metadata.importance_history.append(max(0, importance))
        
        metadata.last_used = datetime.now() - timedelta(days=45)
        
        # Calculate obsolescence
        score = detector.calculate_obsolescence_score(metadata)
        assert 0 <= score <= 1
        assert score > 0.5  # Should be somewhat obsolete
    
    def test_adaptive_feature_engineering(self):
        """Test adaptive feature engineering"""
        from src.bot.ml.feature_evolution import AdaptiveFeatureEngineer
        
        engineer = AdaptiveFeatureEngineer()
        
        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100)
        })
        target = pd.Series(np.random.randn(100))
        
        # Suggest new features
        suggestions = engineer.suggest_new_features(
            data, target, list(data.columns)
        )
        
        assert len(suggestions) > 0
        assert all('name' in s for s in suggestions)
        assert all('type' in s for s in suggestions)
        assert all('formula' in s for s in suggestions)
    
    def test_feature_evolution_analysis(self):
        """Test feature evolution analysis"""
        from src.bot.ml.feature_evolution import FeatureEvolutionSystem, FeatureType
        
        system = FeatureEvolutionSystem()
        
        # Register and update features
        for i in range(10):
            system.register_feature(f"feature_{i}", FeatureType.NUMERICAL)
        
        # Simulate importance updates
        for _ in range(5):
            importance = {f"feature_{i}": np.random.random() for i in range(10)}
            system.update_feature_importance(importance, "model_1")
        
        # Analyze evolution
        report = system.analyze_feature_evolution()
        
        assert report is not None
        assert report.total_features == 10
        assert len(report.top_important) > 0
        assert 0 <= report.feature_stability <= 1
    
    def test_feature_pruning(self):
        """Test feature pruning"""
        from src.bot.ml.feature_evolution import (
            FeatureEvolutionSystem,
            FeatureType,
            FeatureStatus
        )
        
        system = FeatureEvolutionSystem()
        
        # Register features
        for i in range(5):
            system.register_feature(f"feature_{i}", FeatureType.NUMERICAL)
        
        # Mark some as obsolete
        system.features_metadata['feature_0'].status = FeatureStatus.OBSOLETE
        system.features_metadata['feature_0'].obsolescence_score = 0.8
        system.features_metadata['feature_1'].status = FeatureStatus.OBSOLETE
        system.features_metadata['feature_1'].obsolescence_score = 0.9
        
        # Prune
        pruned = system.prune_features()
        
        assert len(pruned) == 2
        assert 'feature_0' in pruned
        assert 'feature_1' in pruned
        assert len(system.features_metadata) == 3


class TestWeek6Integration:
    """Integration tests for Week 6 components"""
    
    def test_ensemble_with_evolution(self):
        """Test integration of ensemble and feature evolution"""
        from src.bot.ml.ensemble_manager import EnsembleManager
        from src.bot.ml.feature_evolution import FeatureEvolutionSystem
        
        # Create systems
        ensemble = EnsembleManager()
        evolution = FeatureEvolutionSystem()
        
        # Both should initialize without errors
        assert ensemble is not None
        assert evolution is not None
    
    def test_week6_components_available(self):
        """Test all Week 6 components are available"""
        components = []
        
        try:
            from src.bot.ml.ensemble_manager import EnsembleManager
            components.append("✅ Ensemble Manager")
        except:
            components.append("❌ Ensemble Manager")
        
        try:
            from src.bot.ml.ensemble_manager import DynamicWeightOptimizer
            components.append("✅ Dynamic Weight Optimizer")
        except:
            components.append("❌ Dynamic Weight Optimizer")
        
        try:
            from src.bot.ml.ensemble_manager import BayesianModelAveraging
            components.append("✅ Bayesian Model Averaging")
        except:
            components.append("❌ Bayesian Model Averaging")
        
        try:
            from src.bot.ml.feature_evolution import FeatureEvolutionSystem
            components.append("✅ Feature Evolution System")
        except:
            components.append("❌ Feature Evolution System")
        
        try:
            from src.bot.ml.feature_evolution import AdaptiveFeatureEngineer
            components.append("✅ Adaptive Feature Engineer")
        except:
            components.append("❌ Adaptive Feature Engineer")
        
        print("\nWeek 6 Components Status:")
        for component in components:
            print(f"  {component}")
        
        # All should be available
        working = len([c for c in components if "✅" in c])
        assert working == 5, f"Only {working}/5 components available"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])