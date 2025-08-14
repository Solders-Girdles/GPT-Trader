"""
Model Versioning and Lifecycle Management
Phase 3, Week 5-6: ADAPT-014

Comprehensive model versioning system with automatic version management,
rollback capabilities, and A/B testing support for GPT-Trader.
"""

import logging
import shutil
import hashlib
import json
import pickle
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import semver
import git
from abc import ABC, abstractmethod

# Database and serialization
import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    SHADOW = "shadow"
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    ROLLBACK_READY = "rollback_ready"


class VersionType(Enum):
    """Semantic version types"""
    MAJOR = "major"      # Breaking changes
    MINOR = "minor"      # New features
    PATCH = "patch"      # Bug fixes
    HOTFIX = "hotfix"    # Emergency fixes


class ModelFormat(Enum):
    """Supported model formats"""
    JOBLIB = "joblib"
    PICKLE = "pickle"
    ONNX = "onnx"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    CUSTOM = "custom"


@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    model_id: str
    version: str
    stage: ModelStage
    
    # Model information
    model_type: str
    model_format: ModelFormat
    model_size_bytes: int
    model_hash: str  # SHA256
    
    # Training information
    training_dataset_hash: str
    training_start_time: datetime
    training_end_time: datetime
    training_duration_seconds: float
    training_config: Dict[str, Any]
    
    # Performance metrics
    validation_accuracy: float
    validation_precision: float
    validation_recall: float
    validation_f1: float
    validation_roc_auc: float
    
    # Trading metrics
    backtest_sharpe_ratio: Optional[float] = None
    backtest_max_drawdown: Optional[float] = None
    backtest_total_return: Optional[float] = None
    
    # Dependencies
    feature_set_version: str
    dependencies: Dict[str, str] = field(default_factory=dict)
    
    # Lifecycle tracking
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    promoted_by: Optional[str] = None
    promoted_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    
    # Deployment information
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    
    # Tags and annotations
    tags: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # Rollback information
    parent_version: Optional[str] = None
    rollback_safe: bool = True
    rollback_tested: bool = False


@dataclass
class VersionComparisonResult:
    """Result of version comparison"""
    old_version: str
    new_version: str
    comparison_type: str
    
    # Performance comparison
    accuracy_improvement: float
    precision_improvement: float
    recall_improvement: float
    f1_improvement: float
    
    # Trading performance
    sharpe_improvement: Optional[float] = None
    drawdown_improvement: Optional[float] = None
    return_improvement: Optional[float] = None
    
    # Statistical significance
    is_statistically_significant: bool = False
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
    # Recommendation
    recommended_action: str = "hold"  # promote, hold, rollback
    recommendation_confidence: float = 0.5


class ModelVersioning:
    """
    Comprehensive model versioning system.
    
    Features:
    - Semantic versioning
    - Automatic version tagging
    - Model lifecycle management
    - Rollback capabilities
    - A/B testing support
    - Model comparison and validation
    - Automated cleanup
    """
    
    def __init__(self, 
                 base_path: Union[str, Path] = "models",
                 enable_git_tracking: bool = True,
                 max_versions_per_model: int = 10,
                 retention_days: int = 90):
        """Initialize model versioning system
        
        Args:
            base_path: Base directory for model storage
            enable_git_tracking: Enable Git-based versioning
            max_versions_per_model: Maximum versions to keep per model
            retention_days: Days to retain old versions
        """
        self.base_path = Path(base_path)
        self.enable_git_tracking = enable_git_tracking
        self.max_versions_per_model = max_versions_per_model
        self.retention_days = retention_days
        
        # Create directory structure
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "metadata").mkdir(exist_ok=True)
        (self.base_path / "models").mkdir(exist_ok=True)
        (self.base_path / "archive").mkdir(exist_ok=True)
        
        # Model registry
        self.models: Dict[str, List[ModelMetadata]] = {}
        self.current_versions: Dict[str, str] = {}  # model_id -> current version
        self.production_versions: Dict[str, str] = {}  # model_id -> production version
        
        # Git repository (if enabled)
        self.git_repo = None
        if enable_git_tracking:
            self._init_git_repo()
        
        # Load existing models
        self._load_existing_models()
        
        logger.info(f"Initialized model versioning system at {self.base_path}")
    
    def _init_git_repo(self):
        """Initialize Git repository for version tracking"""
        try:
            if (self.base_path / ".git").exists():
                self.git_repo = git.Repo(self.base_path)
            else:
                self.git_repo = git.Repo.init(self.base_path)
                
                # Create .gitignore
                gitignore_path = self.base_path / ".gitignore"
                if not gitignore_path.exists():
                    with open(gitignore_path, 'w') as f:
                        f.write("*.pyc\n__pycache__/\n.DS_Store\n*.log\n")
                
                # Initial commit
                self.git_repo.index.add([".gitignore"])
                self.git_repo.index.commit("Initial commit")
            
            logger.info("Git repository initialized for model versioning")
        
        except Exception as e:
            logger.warning(f"Failed to initialize Git repository: {e}")
            self.enable_git_tracking = False
    
    def _load_existing_models(self):
        """Load existing models from disk"""
        metadata_dir = self.base_path / "metadata"
        
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                metadata = ModelMetadata(**metadata_dict)
                
                if metadata.model_id not in self.models:
                    self.models[metadata.model_id] = []
                
                self.models[metadata.model_id].append(metadata)
                
                # Update current version
                if metadata.stage == ModelStage.PRODUCTION:
                    self.production_versions[metadata.model_id] = metadata.version
                
                if (metadata.model_id not in self.current_versions or
                    self._is_newer_version(metadata.version, 
                                          self.current_versions[metadata.model_id])):
                    self.current_versions[metadata.model_id] = metadata.version
            
            except Exception as e:
                logger.error(f"Failed to load metadata from {metadata_file}: {e}")
        
        logger.info(f"Loaded {len(self.models)} model families with "
                   f"{sum(len(versions) for versions in self.models.values())} total versions")
    
    def create_version(self,
                      model_id: str,
                      model_object: Any,
                      metadata: ModelMetadata,
                      version_type: VersionType = VersionType.MINOR,
                      custom_version: Optional[str] = None) -> str:
        """Create a new model version
        
        Args:
            model_id: Model identifier
            model_object: Trained model object
            metadata: Model metadata
            version_type: Type of version increment
            custom_version: Custom version string (overrides auto-generation)
            
        Returns:
            Version string
        """
        try:
            # Generate version number
            if custom_version:
                version = custom_version
            else:
                version = self._generate_version_number(model_id, version_type)
            
            # Update metadata
            metadata.model_id = model_id
            metadata.version = version
            metadata.created_at = datetime.now()
            
            # Calculate model hash
            model_path = self._get_model_path(model_id, version)
            self._save_model(model_object, model_path, metadata.model_format)
            metadata.model_hash = self._calculate_model_hash(model_path)
            metadata.model_size_bytes = model_path.stat().st_size
            
            # Save metadata
            self._save_metadata(metadata)
            
            # Update registry
            if model_id not in self.models:
                self.models[model_id] = []
            
            self.models[model_id].append(metadata)
            self.current_versions[model_id] = version
            
            # Git commit (if enabled)
            if self.enable_git_tracking:
                self._git_commit_version(model_id, version, "Create new model version")
            
            # Cleanup old versions
            self._cleanup_old_versions(model_id)
            
            logger.info(f"Created model version {model_id}:{version}")
            return version
        
        except Exception as e:
            logger.error(f"Failed to create model version: {e}")
            raise
    
    def promote_version(self,
                       model_id: str,
                       version: str,
                       target_stage: ModelStage,
                       promoted_by: str,
                       validation_results: Optional[Dict[str, Any]] = None) -> bool:
        """Promote a model version to a higher stage
        
        Args:
            model_id: Model identifier
            version: Version to promote
            target_stage: Target stage
            promoted_by: Who promoted the model
            validation_results: Validation results
            
        Returns:
            True if promotion successful
        """
        try:
            metadata = self.get_version_metadata(model_id, version)
            if not metadata:
                logger.error(f"Version {model_id}:{version} not found")
                return False
            
            # Validate promotion
            if not self._validate_promotion(metadata, target_stage):
                logger.error(f"Promotion validation failed for {model_id}:{version}")
                return False
            
            # Update metadata
            old_stage = metadata.stage
            metadata.stage = target_stage
            metadata.promoted_by = promoted_by
            metadata.promoted_at = datetime.now()
            
            if validation_results:
                metadata.annotations.update(validation_results)
            
            # Special handling for production promotion
            if target_stage == ModelStage.PRODUCTION:
                # Demote current production version
                self._demote_current_production(model_id)
                self.production_versions[model_id] = version
                
                # Mark as rollback ready
                metadata.rollback_safe = True
                metadata.rollback_tested = True
            
            # Save updated metadata
            self._save_metadata(metadata)
            
            # Git commit
            if self.enable_git_tracking:
                self._git_commit_version(
                    model_id, version, 
                    f"Promote {model_id}:{version} from {old_stage.value} to {target_stage.value}"
                )
            
            logger.info(f"Promoted {model_id}:{version} from {old_stage.value} to {target_stage.value}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to promote version: {e}")
            return False
    
    def rollback_to_version(self,
                           model_id: str,
                           target_version: str,
                           rollback_by: str,
                           reason: str) -> bool:
        """Rollback to a previous model version
        
        Args:
            model_id: Model identifier
            target_version: Version to rollback to
            rollback_by: Who initiated the rollback
            reason: Reason for rollback
            
        Returns:
            True if rollback successful
        """
        try:
            target_metadata = self.get_version_metadata(model_id, target_version)
            if not target_metadata:
                logger.error(f"Target version {model_id}:{target_version} not found")
                return False
            
            # Validate rollback
            if not target_metadata.rollback_safe:
                logger.error(f"Version {model_id}:{target_version} is not marked as rollback safe")
                return False
            
            # Get current production version
            current_production = self.production_versions.get(model_id)
            if current_production:
                current_metadata = self.get_version_metadata(model_id, current_production)
                if current_metadata:
                    # Demote current production
                    current_metadata.stage = ModelStage.ROLLBACK_READY
                    current_metadata.annotations["rollback_reason"] = reason
                    current_metadata.annotations["rollback_at"] = datetime.now().isoformat()
                    current_metadata.annotations["rollback_by"] = rollback_by
                    self._save_metadata(current_metadata)
            
            # Promote target version to production
            target_metadata.stage = ModelStage.PRODUCTION
            target_metadata.promoted_by = rollback_by
            target_metadata.promoted_at = datetime.now()
            target_metadata.annotations["rollback_from"] = current_production or "unknown"
            target_metadata.annotations["rollback_reason"] = reason
            
            self._save_metadata(target_metadata)
            self.production_versions[model_id] = target_version
            
            # Git commit
            if self.enable_git_tracking:
                self._git_commit_version(
                    model_id, target_version,
                    f"Rollback {model_id} to version {target_version}: {reason}"
                )
            
            logger.info(f"Rolled back {model_id} to version {target_version}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to rollback version: {e}")
            return False
    
    def compare_versions(self,
                        model_id: str,
                        version1: str,
                        version2: str,
                        test_data: Optional[pd.DataFrame] = None) -> VersionComparisonResult:
        """Compare two model versions
        
        Args:
            model_id: Model identifier
            version1: First version (typically older)
            version2: Second version (typically newer)
            test_data: Test data for comparison
            
        Returns:
            Comparison results
        """
        try:
            metadata1 = self.get_version_metadata(model_id, version1)
            metadata2 = self.get_version_metadata(model_id, version2)
            
            if not metadata1 or not metadata2:
                raise ValueError("One or both versions not found")
            
            # Performance comparison
            accuracy_improvement = metadata2.validation_accuracy - metadata1.validation_accuracy
            precision_improvement = metadata2.validation_precision - metadata1.validation_precision
            recall_improvement = metadata2.validation_recall - metadata1.validation_recall
            f1_improvement = metadata2.validation_f1 - metadata1.validation_f1
            
            # Trading performance comparison
            sharpe_improvement = None
            drawdown_improvement = None
            return_improvement = None
            
            if (metadata1.backtest_sharpe_ratio and metadata2.backtest_sharpe_ratio):
                sharpe_improvement = metadata2.backtest_sharpe_ratio - metadata1.backtest_sharpe_ratio
            
            if (metadata1.backtest_max_drawdown and metadata2.backtest_max_drawdown):
                drawdown_improvement = metadata1.backtest_max_drawdown - metadata2.backtest_max_drawdown
            
            if (metadata1.backtest_total_return and metadata2.backtest_total_return):
                return_improvement = metadata2.backtest_total_return - metadata1.backtest_total_return
            
            # Statistical significance (if test data provided)
            is_significant = False
            p_value = None
            confidence_interval = None
            
            if test_data is not None:
                is_significant, p_value, confidence_interval = self._test_statistical_significance(
                    model_id, version1, version2, test_data
                )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                accuracy_improvement, f1_improvement, sharpe_improvement,
                is_significant
            )
            
            return VersionComparisonResult(
                old_version=version1,
                new_version=version2,
                comparison_type="performance",
                accuracy_improvement=accuracy_improvement,
                precision_improvement=precision_improvement,
                recall_improvement=recall_improvement,
                f1_improvement=f1_improvement,
                sharpe_improvement=sharpe_improvement,
                drawdown_improvement=drawdown_improvement,
                return_improvement=return_improvement,
                is_statistically_significant=is_significant,
                p_value=p_value,
                confidence_interval=confidence_interval,
                recommended_action=recommendation["action"],
                recommendation_confidence=recommendation["confidence"]
            )
        
        except Exception as e:
            logger.error(f"Failed to compare versions: {e}")
            raise
    
    def archive_version(self,
                       model_id: str,
                       version: str,
                       archive_reason: str = "automated_cleanup") -> bool:
        """Archive a model version
        
        Args:
            model_id: Model identifier
            version: Version to archive
            archive_reason: Reason for archiving
            
        Returns:
            True if archival successful
        """
        try:
            metadata = self.get_version_metadata(model_id, version)
            if not metadata:
                logger.error(f"Version {model_id}:{version} not found")
                return False
            
            # Cannot archive production models
            if metadata.stage == ModelStage.PRODUCTION:
                logger.error(f"Cannot archive production version {model_id}:{version}")
                return False
            
            # Move model files to archive
            model_path = self._get_model_path(model_id, version)
            archive_path = self.base_path / "archive" / model_id / version
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            if model_path.exists():
                shutil.move(str(model_path), str(archive_path))
            
            # Update metadata
            metadata.stage = ModelStage.ARCHIVED
            metadata.archived_at = datetime.now()
            metadata.annotations["archive_reason"] = archive_reason
            
            self._save_metadata(metadata)
            
            # Git commit
            if self.enable_git_tracking:
                self._git_commit_version(
                    model_id, version,
                    f"Archive {model_id}:{version}: {archive_reason}"
                )
            
            logger.info(f"Archived {model_id}:{version}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to archive version: {e}")
            return False
    
    def delete_version(self,
                      model_id: str,
                      version: str,
                      force: bool = False) -> bool:
        """Permanently delete a model version
        
        Args:
            model_id: Model identifier
            version: Version to delete
            force: Force deletion even if not archived
            
        Returns:
            True if deletion successful
        """
        try:
            metadata = self.get_version_metadata(model_id, version)
            if not metadata:
                logger.error(f"Version {model_id}:{version} not found")
                return False
            
            # Safety checks
            if metadata.stage == ModelStage.PRODUCTION and not force:
                logger.error(f"Cannot delete production version {model_id}:{version}")
                return False
            
            if metadata.stage \!= ModelStage.ARCHIVED and not force:
                logger.error(f"Version {model_id}:{version} must be archived before deletion")
                return False
            
            # Delete model files
            model_path = self._get_model_path(model_id, version)
            archive_path = self.base_path / "archive" / model_id / version
            
            if model_path.exists():
                if model_path.is_file():
                    model_path.unlink()
                else:
                    shutil.rmtree(model_path)
            
            if archive_path.exists():
                if archive_path.is_file():
                    archive_path.unlink()
                else:
                    shutil.rmtree(archive_path)
            
            # Delete metadata
            metadata_path = self.base_path / "metadata" / f"{model_id}_{version}.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Remove from registry
            if model_id in self.models:
                self.models[model_id] = [
                    m for m in self.models[model_id] if m.version \!= version
                ]
                
                if not self.models[model_id]:
                    del self.models[model_id]
            
            # Update current version if necessary
            if self.current_versions.get(model_id) == version:
                remaining_versions = self.list_versions(model_id)
                if remaining_versions:
                    self.current_versions[model_id] = max(remaining_versions)
                else:
                    del self.current_versions[model_id]
            
            logger.info(f"Deleted {model_id}:{version}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete version: {e}")
            return False
    
    def load_model(self, 
                   model_id: str, 
                   version: Optional[str] = None) -> Optional[Any]:
        """Load a model from storage
        
        Args:
            model_id: Model identifier
            version: Specific version (uses current if not specified)
            
        Returns:
            Loaded model object
        """
        try:
            if version is None:
                version = self.current_versions.get(model_id)
                if not version:
                    logger.error(f"No current version found for model {model_id}")
                    return None
            
            metadata = self.get_version_metadata(model_id, version)
            if not metadata:
                logger.error(f"Metadata not found for {model_id}:{version}")
                return None
            
            model_path = self._get_model_path(model_id, version)
            if not model_path.exists():
                # Check archive
                archive_path = self.base_path / "archive" / model_id / version
                if archive_path.exists():
                    model_path = archive_path
                else:
                    logger.error(f"Model file not found for {model_id}:{version}")
                    return None
            
            return self._load_model(model_path, metadata.model_format)
        
        except Exception as e:
            logger.error(f"Failed to load model {model_id}:{version}: {e}")
            return None
    
    def get_version_metadata(self, 
                           model_id: str, 
                           version: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific version"""
        if model_id in self.models:
            for metadata in self.models[model_id]:
                if metadata.version == version:
                    return metadata
        return None
    
    def list_models(self) -> List[str]:
        """List all model IDs"""
        return list(self.models.keys())
    
    def list_versions(self, model_id: str) -> List[str]:
        """List all versions for a model"""
        if model_id in self.models:
            return [metadata.version for metadata in self.models[model_id]]
        return []
    
    def get_production_version(self, model_id: str) -> Optional[str]:
        """Get current production version for a model"""
        return self.production_versions.get(model_id)
    
    def get_latest_version(self, model_id: str) -> Optional[str]:
        """Get latest version for a model"""
        return self.current_versions.get(model_id)
    
    def cleanup_old_versions(self, 
                           model_id: Optional[str] = None,
                           dry_run: bool = False) -> Dict[str, int]:
        """Clean up old model versions
        
        Args:
            model_id: Specific model to clean (all models if None)
            dry_run: Only report what would be cleaned
            
        Returns:
            Dictionary of cleanup counts
        """
        cleanup_stats = {"archived": 0, "deleted": 0}
        
        models_to_clean = [model_id] if model_id else self.list_models()
        
        for mid in models_to_clean:
            if mid in self.models:
                cleanup_stats.update(self._cleanup_old_versions(mid, dry_run))
        
        return cleanup_stats
    
    # Private helper methods
    
    def _generate_version_number(self, 
                                model_id: str, 
                                version_type: VersionType) -> str:
        """Generate semantic version number"""
        current_version = self.current_versions.get(model_id, "0.0.0")
        
        try:
            if version_type == VersionType.MAJOR:
                return semver.bump_major(current_version)
            elif version_type == VersionType.MINOR:
                return semver.bump_minor(current_version)
            elif version_type == VersionType.PATCH:
                return semver.bump_patch(current_version)
            elif version_type == VersionType.HOTFIX:
                # Hotfix is treated as patch with special suffix
                base_version = semver.bump_patch(current_version)
                return f"{base_version}-hotfix"
            else:
                return semver.bump_minor(current_version)
        
        except ValueError:
            # If current version is not semantic, start fresh
            if version_type == VersionType.MAJOR:
                return "1.0.0"
            else:
                return "0.1.0"
    
    def _is_newer_version(self, version1: str, version2: str) -> bool:
        """Check if version1 is newer than version2"""
        try:
            return semver.compare(version1, version2) > 0
        except ValueError:
            # Fallback to string comparison if not semantic
            return version1 > version2
    
    def _get_model_path(self, model_id: str, version: str) -> Path:
        """Get path for model storage"""
        return self.base_path / "models" / model_id / version
    
    def _save_model(self, 
                   model_object: Any, 
                   model_path: Path, 
                   model_format: ModelFormat):
        """Save model to disk"""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if model_format == ModelFormat.JOBLIB:
            joblib.dump(model_object, model_path / "model.joblib")
        elif model_format == ModelFormat.PICKLE:
            with open(model_path / "model.pkl", 'wb') as f:
                pickle.dump(model_object, f)
        else:
            raise ValueError(f"Unsupported model format: {model_format}")
    
    def _load_model(self, model_path: Path, model_format: ModelFormat) -> Any:
        """Load model from disk"""
        if model_format == ModelFormat.JOBLIB:
            return joblib.load(model_path / "model.joblib")
        elif model_format == ModelFormat.PICKLE:
            with open(model_path / "model.pkl", 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported model format: {model_format}")
    
    def _save_metadata(self, metadata: ModelMetadata):
        """Save metadata to disk"""
        metadata_path = (self.base_path / "metadata" / 
                        f"{metadata.model_id}_{metadata.version}.json")
        
        # Convert to serializable format
        metadata_dict = {
            k: v.isoformat() if isinstance(v, datetime) else
               v.value if isinstance(v, Enum) else v
            for k, v in metadata.__dict__.items()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
    
    def _calculate_model_hash(self, model_path: Path) -> str:
        """Calculate SHA256 hash of model files"""
        hasher = hashlib.sha256()
        
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _validate_promotion(self, 
                           metadata: ModelMetadata, 
                           target_stage: ModelStage) -> bool:
        """Validate if promotion is allowed"""
        current_stage = metadata.stage
        
        # Define allowed promotion paths
        allowed_promotions = {
            ModelStage.DEVELOPMENT: [ModelStage.TESTING, ModelStage.STAGING],
            ModelStage.TESTING: [ModelStage.STAGING, ModelStage.SHADOW],
            ModelStage.STAGING: [ModelStage.SHADOW, ModelStage.CANDIDATE],
            ModelStage.SHADOW: [ModelStage.CANDIDATE, ModelStage.PRODUCTION],
            ModelStage.CANDIDATE: [ModelStage.PRODUCTION, ModelStage.STAGING],
            ModelStage.PRODUCTION: [ModelStage.DEPRECATED, ModelStage.ROLLBACK_READY]
        }
        
        return target_stage in allowed_promotions.get(current_stage, [])
    
    def _demote_current_production(self, model_id: str):
        """Demote current production version"""
        current_production = self.production_versions.get(model_id)
        if current_production:
            metadata = self.get_version_metadata(model_id, current_production)
            if metadata:
                metadata.stage = ModelStage.ROLLBACK_READY
                self._save_metadata(metadata)
    
    def _git_commit_version(self, model_id: str, version: str, message: str):
        """Commit version to Git"""
        if not self.git_repo:
            return
        
        try:
            # Add all files
            self.git_repo.git.add(".")
            
            # Create commit
            commit_message = f"{message}\n\nModel: {model_id}\nVersion: {version}"
            self.git_repo.index.commit(commit_message)
            
            # Create tag
            tag_name = f"{model_id}-{version}"
            self.git_repo.create_tag(tag_name, message=f"Version {version} of {model_id}")
            
        except Exception as e:
            logger.warning(f"Git commit failed: {e}")
    
    def _cleanup_old_versions(self, model_id: str, dry_run: bool = False) -> Dict[str, int]:
        """Clean up old versions for a specific model"""
        stats = {"archived": 0, "deleted": 0}
        
        if model_id not in self.models:
            return stats
        
        versions = self.models[model_id]
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        # Sort versions by creation date
        versions_by_date = sorted(versions, key=lambda x: x.created_at, reverse=True)
        
        # Keep production and recent versions
        to_archive = []
        to_delete = []
        
        for i, metadata in enumerate(versions_by_date):
            # Never clean production versions
            if metadata.stage == ModelStage.PRODUCTION:
                continue
            
            # Keep max_versions_per_model most recent
            if i >= self.max_versions_per_model:
                if metadata.stage == ModelStage.ARCHIVED:
                    if metadata.created_at < cutoff_date:
                        to_delete.append(metadata)
                else:
                    to_archive.append(metadata)
            
            # Archive old versions
            elif metadata.created_at < cutoff_date and metadata.stage \!= ModelStage.ARCHIVED:
                to_archive.append(metadata)
        
        # Perform cleanup
        if not dry_run:
            for metadata in to_archive:
                if self.archive_version(model_id, metadata.version, "automated_cleanup"):
                    stats["archived"] += 1
            
            for metadata in to_delete:
                if self.delete_version(model_id, metadata.version, force=True):
                    stats["deleted"] += 1
        else:
            stats["archived"] = len(to_archive)
            stats["deleted"] = len(to_delete)
        
        return stats
    
    def _test_statistical_significance(self,
                                     model_id: str,
                                     version1: str,
                                     version2: str,
                                     test_data: pd.DataFrame) -> Tuple[bool, float, Tuple[float, float]]:
        """Test statistical significance between model versions"""
        # This is a simplified implementation
        # In practice, you'd use proper statistical tests
        
        # Load models
        model1 = self.load_model(model_id, version1)
        model2 = self.load_model(model_id, version2)
        
        if not model1 or not model2:
            return False, 1.0, (0.0, 0.0)
        
        # Make predictions
        X = test_data.drop(['target'], axis=1, errors='ignore')
        y = test_data['target']
        
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)
        
        # Calculate accuracies
        acc1 = np.mean(pred1 == y)
        acc2 = np.mean(pred2 == y)
        
        # Simple t-test approximation
        diff = acc2 - acc1
        std_err = np.sqrt((acc1 * (1 - acc1) + acc2 * (1 - acc2)) / len(y))
        
        if std_err > 0:
            t_stat = diff / std_err
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            
            # 95% confidence interval
            ci_lower = diff - 1.96 * std_err
            ci_upper = diff + 1.96 * std_err
            
            return p_value < 0.05, p_value, (ci_lower, ci_upper)
        
        return False, 1.0, (0.0, 0.0)
    
    def _generate_recommendation(self,
                               accuracy_improvement: float,
                               f1_improvement: float,
                               sharpe_improvement: Optional[float],
                               is_significant: bool) -> Dict[str, Any]:
        """Generate deployment recommendation"""
        
        # Calculate overall improvement score
        improvements = [accuracy_improvement, f1_improvement]
        if sharpe_improvement is not None:
            improvements.append(sharpe_improvement / 10)  # Scale Sharpe ratio
        
        avg_improvement = np.mean(improvements)
        
        # Generate recommendation
        if avg_improvement > 0.05 and is_significant:
            return {"action": "promote", "confidence": 0.9}
        elif avg_improvement > 0.02:
            return {"action": "promote", "confidence": 0.7}
        elif avg_improvement > -0.02:
            return {"action": "hold", "confidence": 0.8}
        else:
            return {"action": "rollback", "confidence": 0.9}


# Factory functions
def create_model_versioning(base_path: str = "models",
                          enable_git: bool = True,
                          max_versions: int = 10) -> ModelVersioning:
    """Create model versioning system"""
    return ModelVersioning(
        base_path=base_path,
        enable_git_tracking=enable_git,
        max_versions_per_model=max_versions
    )
EOF < /dev/null