"""
Checkpoint Handler for Bot V2 Trading System

Provides atomic checkpoint operations with consistency guarantees,
version management, and rollback capabilities for system state.
"""

import os
import json
import gzip
import pickle
import hashlib
import asyncio
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class CheckpointStatus(Enum):
    """Checkpoint status states"""
    CREATING = "creating"
    VALID = "valid"
    INVALID = "invalid"
    CORRUPTED = "corrupted"
    DELETED = "deleted"


class CheckpointType(Enum):
    """Types of checkpoints"""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    EMERGENCY = "emergency"
    PRE_UPGRADE = "pre_upgrade"
    DAILY = "daily"


@dataclass
class Checkpoint:
    """Checkpoint data structure"""
    checkpoint_id: str
    timestamp: datetime
    state_snapshot: Dict[str, Any]
    version: int
    consistency_hash: str
    size_bytes: int
    status: CheckpointStatus = CheckpointStatus.VALID
    checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'checkpoint_id': self.checkpoint_id,
            'timestamp': self.timestamp.isoformat(),
            'state_snapshot': self.state_snapshot,
            'version': self.version,
            'consistency_hash': self.consistency_hash,
            'size_bytes': self.size_bytes,
            'status': self.status.value,
            'checkpoint_type': self.checkpoint_type.value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """Create from dictionary"""
        return cls(
            checkpoint_id=data['checkpoint_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            state_snapshot=data['state_snapshot'],
            version=data['version'],
            consistency_hash=data['consistency_hash'],
            size_bytes=data['size_bytes'],
            status=CheckpointStatus(data.get('status', 'valid')),
            checkpoint_type=CheckpointType(data.get('checkpoint_type', 'automatic')),
            metadata=data.get('metadata', {})
        )


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint operations"""
    checkpoint_dir: str = "/tmp/bot_v2/checkpoints"
    max_checkpoints: int = 50
    checkpoint_interval_minutes: int = 15
    compression_enabled: bool = True
    verification_enabled: bool = True
    pause_trading_during_checkpoint: bool = True
    checkpoint_timeout_seconds: int = 30
    retention_days: int = 7


class CheckpointHandler:
    """
    Manages atomic checkpoint operations with rollback capabilities.
    Ensures system consistency during checkpoint creation and restoration.
    """
    
    def __init__(self, state_manager: Any, config: Optional[CheckpointConfig] = None):
        self.state_manager = state_manager
        self.config = config or CheckpointConfig()
        self._checkpoint_lock = threading.Lock()
        self._checkpoint_in_progress = False
        self._checkpoint_history: List[Checkpoint] = []
        self._trading_paused = False
        
        # Create checkpoint directory
        self.checkpoint_path = Path(self.config.checkpoint_dir)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint history
        self._load_checkpoint_history()
        
        logger.info(f"CheckpointHandler initialized with {len(self._checkpoint_history)} checkpoints")
    
    async def create_checkpoint(self, 
                              checkpoint_id: Optional[str] = None,
                              checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC) -> Optional[Checkpoint]:
        """
        Create atomic checkpoint with consistency guarantees.
        
        Args:
            checkpoint_id: Optional checkpoint ID
            checkpoint_type: Type of checkpoint
            
        Returns:
            Created checkpoint or None if failed
        """
        if self._checkpoint_in_progress:
            logger.warning("Checkpoint already in progress")
            return None
        
        with self._checkpoint_lock:
            self._checkpoint_in_progress = True
            start_time = datetime.utcnow()
            
            try:
                # Generate checkpoint ID if not provided
                if not checkpoint_id:
                    checkpoint_id = self._generate_checkpoint_id(checkpoint_type)
                
                logger.info(f"Creating checkpoint {checkpoint_id} of type {checkpoint_type.value}")
                
                # Pause trading if configured
                if self.config.pause_trading_during_checkpoint:
                    await self._pause_trading_operations()
                
                # Capture system state
                state_snapshot = await self._capture_system_state()
                
                if not state_snapshot:
                    raise Exception("Failed to capture system state")
                
                # Calculate consistency hash
                consistency_hash = self._calculate_consistency_hash(state_snapshot)
                
                # Prepare checkpoint data
                checkpoint_data = json.dumps(state_snapshot, default=str)
                
                # Compress if enabled
                if self.config.compression_enabled:
                    checkpoint_data = gzip.compress(checkpoint_data.encode())
                    size_bytes = len(checkpoint_data)
                else:
                    size_bytes = len(checkpoint_data.encode())
                
                # Create checkpoint object
                checkpoint = Checkpoint(
                    checkpoint_id=checkpoint_id,
                    timestamp=datetime.utcnow(),
                    state_snapshot=state_snapshot,
                    version=self._get_next_version(),
                    consistency_hash=consistency_hash,
                    size_bytes=size_bytes,
                    status=CheckpointStatus.CREATING,
                    checkpoint_type=checkpoint_type,
                    metadata={
                        'creation_duration_ms': 0,
                        'positions_count': len(state_snapshot.get('positions', {})),
                        'orders_count': len(state_snapshot.get('orders', {}))
                    }
                )
                
                # Store checkpoint atomically
                if await self._store_checkpoint_atomic(checkpoint, checkpoint_data):
                    checkpoint.status = CheckpointStatus.VALID
                    
                    # Verify if enabled
                    if self.config.verification_enabled:
                        if not await self._verify_checkpoint(checkpoint):
                            checkpoint.status = CheckpointStatus.INVALID
                            logger.warning(f"Checkpoint {checkpoint_id} verification failed")
                    
                    # Update metadata
                    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    checkpoint.metadata['creation_duration_ms'] = duration_ms
                    
                    # Add to history
                    self._checkpoint_history.append(checkpoint)
                    self._cleanup_old_checkpoints()
                    
                    logger.info(f"Checkpoint {checkpoint_id} created successfully in {duration_ms:.2f}ms")
                    return checkpoint
                else:
                    raise Exception("Failed to store checkpoint")
                
            except asyncio.TimeoutError:
                logger.error(f"Checkpoint creation timed out after {self.config.checkpoint_timeout_seconds}s")
                return None
                
            except Exception as e:
                logger.error(f"Checkpoint creation failed: {e}")
                return None
                
            finally:
                # Resume trading
                if self.config.pause_trading_during_checkpoint:
                    await self._resume_trading_operations()
                
                self._checkpoint_in_progress = False
    
    async def restore_from_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """
        Restore system state from checkpoint.
        
        Args:
            checkpoint: Checkpoint to restore from
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Restoring from checkpoint {checkpoint.checkpoint_id}")
            
            # Verify checkpoint integrity
            if not await self._verify_checkpoint_integrity(checkpoint):
                logger.error("Checkpoint integrity check failed")
                return False
            
            # Pause system operations
            await self._pause_trading_operations()
            
            # Clear current state
            await self._clear_current_state()
            
            # Restore state from snapshot
            success = await self._restore_state_from_snapshot(checkpoint.state_snapshot)
            
            if success:
                # Verify restoration
                if await self._verify_restoration(checkpoint):
                    logger.info(f"Successfully restored from checkpoint {checkpoint.checkpoint_id}")
                else:
                    logger.warning("Restoration verification failed but state was restored")
                    success = False
            
            # Resume operations
            await self._resume_trading_operations()
            
            return success
            
        except Exception as e:
            logger.error(f"Checkpoint restoration failed: {e}")
            await self._resume_trading_operations()
            return False
    
    async def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Rollback system to specific checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to rollback to
            
        Returns:
            Success status
        """
        checkpoint = self.get_checkpoint(checkpoint_id)
        if not checkpoint:
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False
        
        # Create emergency checkpoint before rollback
        emergency_cp = await self.create_checkpoint(
            checkpoint_type=CheckpointType.EMERGENCY
        )
        
        if emergency_cp:
            logger.info(f"Created emergency checkpoint {emergency_cp.checkpoint_id} before rollback")
        
        # Perform rollback
        success = await self.restore_from_checkpoint(checkpoint)
        
        if success:
            logger.info(f"Successfully rolled back to checkpoint {checkpoint_id}")
        else:
            logger.error(f"Rollback to checkpoint {checkpoint_id} failed")
            
            # Try to restore emergency checkpoint
            if emergency_cp:
                logger.info("Attempting to restore emergency checkpoint")
                await self.restore_from_checkpoint(emergency_cp)
        
        return success
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get checkpoint by ID"""
        for checkpoint in self._checkpoint_history:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        
        # Try loading from disk
        return self._load_checkpoint_from_disk(checkpoint_id)
    
    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get most recent valid checkpoint"""
        valid_checkpoints = [
            cp for cp in self._checkpoint_history
            if cp.status == CheckpointStatus.VALID
        ]
        
        if valid_checkpoints:
            return max(valid_checkpoints, key=lambda cp: cp.timestamp)
        
        return None
    
    async def find_valid_checkpoint(self, 
                                   before: Optional[datetime] = None) -> Optional[Checkpoint]:
        """
        Find last valid checkpoint before given time.
        
        Args:
            before: Time constraint
            
        Returns:
            Valid checkpoint or None
        """
        valid_checkpoints = [
            cp for cp in self._checkpoint_history
            if cp.status == CheckpointStatus.VALID
        ]
        
        if before:
            valid_checkpoints = [
                cp for cp in valid_checkpoints
                if cp.timestamp < before
            ]
        
        if not valid_checkpoints:
            return None
        
        # Verify integrity of most recent checkpoint
        for checkpoint in sorted(valid_checkpoints, key=lambda cp: cp.timestamp, reverse=True):
            if await self._verify_checkpoint_integrity(checkpoint):
                return checkpoint
        
        return None
    
    async def _capture_system_state(self) -> Dict[str, Any]:
        """Capture complete system state"""
        state = {
            'timestamp': datetime.utcnow().isoformat(),
            'positions': {},
            'orders': {},
            'portfolio': {},
            'ml_models': {},
            'configuration': {},
            'performance_metrics': {},
            'market_data_cache': {}
        }
        
        try:
            # Capture trading positions
            position_keys = await self.state_manager.get_keys_by_pattern("position:*")
            for key in position_keys:
                value = await self.state_manager.get_state(key)
                if value:
                    state['positions'][key] = value
            
            # Capture open orders
            order_keys = await self.state_manager.get_keys_by_pattern("order:*")
            for key in order_keys:
                value = await self.state_manager.get_state(key)
                if value and value.get('status') != 'filled':
                    state['orders'][key] = value
            
            # Capture portfolio state
            portfolio_data = await self.state_manager.get_state("portfolio_current")
            if portfolio_data:
                state['portfolio'] = portfolio_data
            
            # Capture ML model states
            ml_keys = await self.state_manager.get_keys_by_pattern("ml_model:*")
            for key in ml_keys:
                value = await self.state_manager.get_state(key)
                if value:
                    state['ml_models'][key] = value
            
            # Capture configuration
            config_keys = await self.state_manager.get_keys_by_pattern("config:*")
            for key in config_keys:
                value = await self.state_manager.get_state(key)
                if value:
                    state['configuration'][key] = value
            
            # Capture performance metrics
            metrics_data = await self.state_manager.get_state("performance_metrics")
            if metrics_data:
                state['performance_metrics'] = metrics_data
            
            logger.debug(f"Captured state with {len(state['positions'])} positions, "
                        f"{len(state['orders'])} orders")
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to capture system state: {e}")
            return {}
    
    async def _store_checkpoint_atomic(self, checkpoint: Checkpoint, data: bytes) -> bool:
        """Store checkpoint atomically"""
        try:
            # Prepare file paths
            checkpoint_file = self.checkpoint_path / f"{checkpoint.checkpoint_id}.checkpoint"
            temp_file = self.checkpoint_path / f"{checkpoint.checkpoint_id}.tmp"
            metadata_file = self.checkpoint_path / f"{checkpoint.checkpoint_id}.meta"
            
            # Write to temporary file first
            with open(temp_file, 'wb') as f:
                f.write(data)
            
            # Write metadata
            metadata = checkpoint.to_dict()
            del metadata['state_snapshot']  # Don't duplicate large data
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Atomic rename
            temp_file.rename(checkpoint_file)
            
            logger.debug(f"Checkpoint {checkpoint.checkpoint_id} stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store checkpoint: {e}")
            
            # Cleanup temporary files
            for suffix in ['.tmp', '.checkpoint', '.meta']:
                file_path = self.checkpoint_path / f"{checkpoint.checkpoint_id}{suffix}"
                if file_path.exists():
                    file_path.unlink()
            
            return False
    
    def _calculate_consistency_hash(self, state_snapshot: Dict[str, Any]) -> str:
        """Calculate hash for consistency verification"""
        # Create deterministic string representation
        state_str = json.dumps(state_snapshot, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    async def _verify_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Verify checkpoint validity"""
        try:
            # Recalculate hash
            calculated_hash = self._calculate_consistency_hash(checkpoint.state_snapshot)
            
            if calculated_hash != checkpoint.consistency_hash:
                logger.error(f"Checkpoint {checkpoint.checkpoint_id} hash mismatch")
                return False
            
            # Verify critical data presence
            if not checkpoint.state_snapshot.get('timestamp'):
                logger.error(f"Checkpoint {checkpoint.checkpoint_id} missing timestamp")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint verification failed: {e}")
            return False
    
    async def _verify_checkpoint_integrity(self, checkpoint: Checkpoint) -> bool:
        """Verify checkpoint file integrity"""
        try:
            checkpoint_file = self.checkpoint_path / f"{checkpoint.checkpoint_id}.checkpoint"
            
            if not checkpoint_file.exists():
                logger.error(f"Checkpoint file {checkpoint_file} not found")
                return False
            
            # Load and verify data
            with open(checkpoint_file, 'rb') as f:
                data = f.read()
            
            # Decompress if needed
            if self.config.compression_enabled:
                data = gzip.decompress(data)
            
            # Parse and verify
            state_snapshot = json.loads(data)
            calculated_hash = self._calculate_consistency_hash(state_snapshot)
            
            return calculated_hash == checkpoint.consistency_hash
            
        except Exception as e:
            logger.error(f"Integrity check failed: {e}")
            return False
    
    async def _clear_current_state(self):
        """Clear current system state before restoration"""
        try:
            # Clear positions
            position_keys = await self.state_manager.get_keys_by_pattern("position:*")
            for key in position_keys:
                await self.state_manager.delete_state(key)
            
            # Clear orders
            order_keys = await self.state_manager.get_keys_by_pattern("order:*")
            for key in order_keys:
                await self.state_manager.delete_state(key)
            
            logger.debug("Current state cleared for restoration")
            
        except Exception as e:
            logger.error(f"Failed to clear current state: {e}")
    
    async def _restore_state_from_snapshot(self, state_snapshot: Dict[str, Any]) -> bool:
        """Restore state from snapshot"""
        try:
            from .state_manager import StateCategory
            
            # Restore positions
            for key, value in state_snapshot.get('positions', {}).items():
                await self.state_manager.set_state(key, value, StateCategory.HOT)
            
            # Restore orders
            for key, value in state_snapshot.get('orders', {}).items():
                await self.state_manager.set_state(key, value, StateCategory.HOT)
            
            # Restore portfolio
            if 'portfolio' in state_snapshot:
                await self.state_manager.set_state(
                    "portfolio_current", 
                    state_snapshot['portfolio'],
                    StateCategory.HOT
                )
            
            # Restore ML models
            for key, value in state_snapshot.get('ml_models', {}).items():
                await self.state_manager.set_state(key, value, StateCategory.WARM)
            
            # Restore configuration
            for key, value in state_snapshot.get('configuration', {}).items():
                await self.state_manager.set_state(key, value, StateCategory.WARM)
            
            # Restore performance metrics
            if 'performance_metrics' in state_snapshot:
                await self.state_manager.set_state(
                    "performance_metrics",
                    state_snapshot['performance_metrics'],
                    StateCategory.WARM
                )
            
            logger.info(f"Restored {len(state_snapshot.get('positions', {}))} positions, "
                       f"{len(state_snapshot.get('orders', {}))} orders")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore state: {e}")
            return False
    
    async def _verify_restoration(self, checkpoint: Checkpoint) -> bool:
        """Verify successful restoration"""
        try:
            # Capture current state
            current_state = await self._capture_system_state()
            
            # Compare key metrics
            original_positions = len(checkpoint.state_snapshot.get('positions', {}))
            current_positions = len(current_state.get('positions', {}))
            
            if original_positions != current_positions:
                logger.warning(f"Position count mismatch: {original_positions} vs {current_positions}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Restoration verification failed: {e}")
            return False
    
    async def _pause_trading_operations(self):
        """Pause trading operations for consistency"""
        self._trading_paused = True
        logger.debug("Trading operations paused for checkpoint")
        
        # Signal to trading systems
        await self.state_manager.set_state("system:trading_paused", True)
        
        # Wait for operations to complete
        await asyncio.sleep(0.1)
    
    async def _resume_trading_operations(self):
        """Resume trading operations"""
        self._trading_paused = False
        await self.state_manager.set_state("system:trading_paused", False)
        logger.debug("Trading operations resumed")
    
    def _generate_checkpoint_id(self, checkpoint_type: CheckpointType) -> str:
        """Generate unique checkpoint ID"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        type_prefix = checkpoint_type.value[:3].upper()
        return f"{type_prefix}_{timestamp}"
    
    def _get_next_version(self) -> int:
        """Get next checkpoint version number"""
        if not self._checkpoint_history:
            return 1
        return max(cp.version for cp in self._checkpoint_history) + 1
    
    def _load_checkpoint_history(self):
        """Load checkpoint history from disk"""
        try:
            for meta_file in self.checkpoint_path.glob("*.meta"):
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                
                # Create checkpoint without loading full state
                checkpoint = Checkpoint(
                    checkpoint_id=metadata['checkpoint_id'],
                    timestamp=datetime.fromisoformat(metadata['timestamp']),
                    state_snapshot={},  # Don't load full state into memory
                    version=metadata['version'],
                    consistency_hash=metadata['consistency_hash'],
                    size_bytes=metadata['size_bytes'],
                    status=CheckpointStatus(metadata.get('status', 'valid')),
                    checkpoint_type=CheckpointType(metadata.get('checkpoint_type', 'automatic')),
                    metadata=metadata.get('metadata', {})
                )
                
                self._checkpoint_history.append(checkpoint)
            
            # Sort by timestamp
            self._checkpoint_history.sort(key=lambda cp: cp.timestamp)
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint history: {e}")
    
    def _load_checkpoint_from_disk(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load specific checkpoint from disk"""
        try:
            checkpoint_file = self.checkpoint_path / f"{checkpoint_id}.checkpoint"
            metadata_file = self.checkpoint_path / f"{checkpoint_id}.meta"
            
            if not checkpoint_file.exists() or not metadata_file.exists():
                return None
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load checkpoint data
            with open(checkpoint_file, 'rb') as f:
                data = f.read()
            
            # Decompress if needed
            if self.config.compression_enabled:
                data = gzip.decompress(data)
            
            # Parse state snapshot
            state_snapshot = json.loads(data)
            
            # Create checkpoint object
            checkpoint = Checkpoint.from_dict(metadata)
            checkpoint.state_snapshot = state_snapshot
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints based on retention policy"""
        try:
            # Keep only max_checkpoints
            if len(self._checkpoint_history) > self.config.max_checkpoints:
                checkpoints_to_remove = len(self._checkpoint_history) - self.config.max_checkpoints
                
                for checkpoint in self._checkpoint_history[:checkpoints_to_remove]:
                    # Delete files
                    for suffix in ['.checkpoint', '.meta']:
                        file_path = self.checkpoint_path / f"{checkpoint.checkpoint_id}{suffix}"
                        if file_path.exists():
                            file_path.unlink()
                    
                    checkpoint.status = CheckpointStatus.DELETED
                
                # Remove from history
                self._checkpoint_history = self._checkpoint_history[checkpoints_to_remove:]
            
            # Remove checkpoints older than retention period
            cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days)
            
            old_checkpoints = [
                cp for cp in self._checkpoint_history
                if cp.timestamp < cutoff_date and cp.checkpoint_type != CheckpointType.PRE_UPGRADE
            ]
            
            for checkpoint in old_checkpoints:
                # Delete files
                for suffix in ['.checkpoint', '.meta']:
                    file_path = self.checkpoint_path / f"{checkpoint.checkpoint_id}{suffix}"
                    if file_path.exists():
                        file_path.unlink()
                
                checkpoint.status = CheckpointStatus.DELETED
                self._checkpoint_history.remove(checkpoint)
            
            if old_checkpoints:
                logger.info(f"Cleaned up {len(old_checkpoints)} old checkpoints")
                
        except Exception as e:
            logger.error(f"Checkpoint cleanup failed: {e}")
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics"""
        valid_checkpoints = [
            cp for cp in self._checkpoint_history
            if cp.status == CheckpointStatus.VALID
        ]
        
        return {
            'total_checkpoints': len(self._checkpoint_history),
            'valid_checkpoints': len(valid_checkpoints),
            'invalid_checkpoints': len([
                cp for cp in self._checkpoint_history
                if cp.status == CheckpointStatus.INVALID
            ]),
            'total_size_bytes': sum(cp.size_bytes for cp in self._checkpoint_history),
            'oldest_checkpoint': min(self._checkpoint_history, key=lambda cp: cp.timestamp).timestamp
                if self._checkpoint_history else None,
            'newest_checkpoint': max(self._checkpoint_history, key=lambda cp: cp.timestamp).timestamp
                if self._checkpoint_history else None,
            'average_size_bytes': sum(cp.size_bytes for cp in valid_checkpoints) / len(valid_checkpoints)
                if valid_checkpoints else 0
        }


# Convenience functions
async def create_checkpoint(state_manager: Any, 
                          checkpoint_type: CheckpointType = CheckpointType.MANUAL) -> Optional[Checkpoint]:
    """Create a checkpoint"""
    handler = CheckpointHandler(state_manager)
    return await handler.create_checkpoint(checkpoint_type=checkpoint_type)


async def restore_latest_checkpoint(state_manager: Any) -> bool:
    """Restore from latest checkpoint"""
    handler = CheckpointHandler(state_manager)
    latest = handler.get_latest_checkpoint()
    
    if latest:
        return await handler.restore_from_checkpoint(latest)
    
    return False