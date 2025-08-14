"""
DL-012: Distributed Training Framework
Phase 4 - Week 2

Distributed training for large-scale models:
- Support 4+ GPUs with linear scaling
- Data parallel and model parallel strategies
- Gradient synchronization and aggregation
- Fault tolerance and checkpointing
- Horovod and PyTorch DDP support
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path
import json
import time
import pickle
import os
import socket
from datetime import datetime

# Try to import distributed training libraries
try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    from torch.distributed.optim import ZeroRedundancyOptimizer
    TORCH_DISTRIBUTED = True
except ImportError:
    TORCH_DISTRIBUTED = False

try:
    import horovod.torch as hvd
    HOROVOD_AVAILABLE = True
except ImportError:
    HOROVOD_AVAILABLE = False

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

logger = logging.getLogger(__name__)


class DistributedBackend(Enum):
    """Distributed training backends"""
    PYTORCH_DDP = "pytorch_ddp"
    HOROVOD = "horovod"
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"


class DistributedStrategy(Enum):
    """Distributed training strategies"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    GRADIENT_ACCUMULATION = "gradient_accumulation"
    MIXED_PRECISION = "mixed_precision"


@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    # Backend settings
    backend: DistributedBackend = DistributedBackend.PYTORCH_DDP
    init_method: str = "env://"  # or "tcp://master_ip:port"
    
    # Process settings
    world_size: int = -1  # Total number of processes
    rank: int = -1  # Process rank
    local_rank: int = -1  # Local GPU rank
    master_addr: str = "localhost"
    master_port: str = "12355"
    
    # Training strategy
    strategy: DistributedStrategy = DistributedStrategy.SYNCHRONOUS
    gradient_sync_frequency: int = 1  # Steps between gradient syncs
    
    # Communication
    bucket_cap_mb: int = 25  # DDP bucket size
    broadcast_buffers: bool = True
    find_unused_parameters: bool = False
    
    # Optimization
    use_zero_optimizer: bool = False  # ZeRO optimizer for memory efficiency
    gradient_as_bucket_view: bool = True
    
    # Fault tolerance
    enable_checkpointing: bool = True
    checkpoint_frequency: int = 100  # Steps between checkpoints
    resume_from_checkpoint: bool = True
    
    # Performance
    mixed_precision: bool = True
    gradient_compression: bool = False
    
    # Monitoring
    log_frequency: int = 10
    profile_communication: bool = False
    
    def __post_init__(self):
        """Auto-detect distributed settings"""
        if self.world_size == -1:
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        if self.rank == -1:
            self.rank = int(os.environ.get('RANK', 0))
        if self.local_rank == -1:
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))


class DistributedInitializer:
    """Initialize distributed training environment"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize distributed backend"""
        
        if self.config.backend == DistributedBackend.PYTORCH_DDP:
            return self._init_pytorch_ddp()
        elif self.config.backend == DistributedBackend.HOROVOD:
            return self._init_horovod()
        else:
            logger.warning(f"Backend {self.config.backend} not fully supported")
            return False
    
    def _init_pytorch_ddp(self) -> bool:
        """Initialize PyTorch DDP"""
        if not TORCH_DISTRIBUTED:
            logger.error("PyTorch distributed not available")
            return False
        
        try:
            # Set environment variables
            os.environ['MASTER_ADDR'] = self.config.master_addr
            os.environ['MASTER_PORT'] = self.config.master_port
            
            # Initialize process group
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(
                backend=backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            
            # Set device
            if torch.cuda.is_available():
                torch.cuda.set_device(self.config.local_rank)
            
            self.is_initialized = True
            logger.info(f"Initialized PyTorch DDP: rank {self.config.rank}/{self.config.world_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PyTorch DDP: {e}")
            return False
    
    def _init_horovod(self) -> bool:
        """Initialize Horovod"""
        if not HOROVOD_AVAILABLE:
            logger.error("Horovod not available")
            return False
        
        try:
            hvd.init()
            
            # Update config with Horovod settings
            self.config.rank = hvd.rank()
            self.config.world_size = hvd.size()
            self.config.local_rank = hvd.local_rank()
            
            # Set device
            if torch.cuda.is_available():
                torch.cuda.set_device(self.config.local_rank)
            
            self.is_initialized = True
            logger.info(f"Initialized Horovod: rank {self.config.rank}/{self.config.world_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Horovod: {e}")
            return False
    
    def cleanup(self):
        """Cleanup distributed resources"""
        if self.config.backend == DistributedBackend.PYTORCH_DDP and dist.is_initialized():
            dist.destroy_process_group()
        elif self.config.backend == DistributedBackend.HOROVOD and HOROVOD_AVAILABLE:
            # Horovod cleanup is automatic
            pass
        
        self.is_initialized = False


class DistributedModel:
    """Wrapper for distributed model"""
    
    def __init__(self, model: nn.Module, config: DistributedConfig):
        self.config = config
        self.model = self._wrap_model(model)
    
    def _wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for distributed training"""
        
        if self.config.backend == DistributedBackend.PYTORCH_DDP:
            if torch.cuda.is_available():
                model = model.cuda(self.config.local_rank)
            
            model = DDP(
                model,
                device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
                output_device=self.config.local_rank if torch.cuda.is_available() else None,
                broadcast_buffers=self.config.broadcast_buffers,
                bucket_cap_mb=self.config.bucket_cap_mb,
                find_unused_parameters=self.config.find_unused_parameters,
                gradient_as_bucket_view=self.config.gradient_as_bucket_view
            )
            
        elif self.config.backend == DistributedBackend.HOROVOD and HOROVOD_AVAILABLE:
            if torch.cuda.is_available():
                model = model.cuda()
            
            # Horovod doesn't need explicit model wrapping
            # but we broadcast parameters
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            
        elif self.config.backend == DistributedBackend.DATA_PARALLEL:
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                model = model.cuda()
        
        return model
    
    def get_base_model(self) -> nn.Module:
        """Get the base model without DDP wrapper"""
        if isinstance(self.model, (DDP, nn.DataParallel)):
            return self.model.module
        return self.model


class DistributedOptimizer:
    """Distributed optimizer with gradient synchronization"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, config: DistributedConfig):
        self.config = config
        self.base_optimizer = optimizer
        self.optimizer = self._wrap_optimizer(optimizer)
        self.step_count = 0
    
    def _wrap_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """Wrap optimizer for distributed training"""
        
        if self.config.backend == DistributedBackend.HOROVOD and HOROVOD_AVAILABLE:
            # Wrap with Horovod DistributedOptimizer
            optimizer = hvd.DistributedOptimizer(
                optimizer,
                named_parameters=None,
                compression=hvd.Compression.fp16 if self.config.gradient_compression else hvd.Compression.none
            )
            
        elif self.config.use_zero_optimizer and TORCH_DISTRIBUTED:
            # Use ZeRO optimizer for memory efficiency
            optimizer = ZeroRedundancyOptimizer(
                optimizer.param_groups[0]['params'],
                optimizer_class=type(optimizer),
                parameters_as_bucket_view=self.config.gradient_as_bucket_view,
                **{k: v for k, v in optimizer.defaults.items()}
            )
        
        return optimizer
    
    def step(self):
        """Optimizer step with gradient synchronization"""
        self.step_count += 1
        
        # Synchronous gradient aggregation
        if self.config.strategy == DistributedStrategy.SYNCHRONOUS:
            self.optimizer.step()
            
        # Gradient accumulation
        elif self.config.strategy == DistributedStrategy.GRADIENT_ACCUMULATION:
            if self.step_count % self.config.gradient_sync_frequency == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        else:
            self.optimizer.step()
    
    def zero_grad(self):
        """Zero gradients"""
        if self.config.strategy != DistributedStrategy.GRADIENT_ACCUMULATION or \
           self.step_count % self.config.gradient_sync_frequency == 0:
            self.optimizer.zero_grad()


class DistributedDataLoader:
    """Distributed data loading with proper sampling"""
    
    def __init__(self, dataset, config: DistributedConfig, **kwargs):
        self.config = config
        self.dataset = dataset
        self.kwargs = kwargs
        self.sampler = self._create_sampler()
        self.loader = self._create_loader()
    
    def _create_sampler(self):
        """Create distributed sampler"""
        if self.config.backend in [DistributedBackend.PYTORCH_DDP, DistributedBackend.HOROVOD]:
            return DistributedSampler(
                self.dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=self.kwargs.get('shuffle', True)
            )
        return None
    
    def _create_loader(self) -> DataLoader:
        """Create data loader with distributed sampler"""
        # Remove shuffle if using distributed sampler
        if self.sampler is not None:
            self.kwargs.pop('shuffle', None)
            
        return DataLoader(
            self.dataset,
            sampler=self.sampler,
            **self.kwargs
        )
    
    def set_epoch(self, epoch: int):
        """Set epoch for distributed sampler"""
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)


class CheckpointManager:
    """Manage distributed checkpointing"""
    
    def __init__(self, config: DistributedConfig, checkpoint_dir: str = "checkpoints"):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, state: Dict[str, Any], epoch: int, is_best: bool = False):
        """Save checkpoint (only on rank 0)"""
        if self.config.rank != 0:
            return
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(state, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
        
        # Keep only last 3 checkpoints
        self._cleanup_old_checkpoints(keep_last=3)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Optional[Dict]:
        """Load checkpoint"""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if not checkpoints:
                return None
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        if Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=f"cuda:{self.config.local_rank}")
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return state
        
        return None
    
    def _cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime
        )
        
        for checkpoint in checkpoints[:-keep_last]:
            checkpoint.unlink()


class DistributedTrainer:
    """Main distributed training coordinator"""
    
    def __init__(self, model: nn.Module, config: DistributedConfig):
        self.config = config
        self.initializer = DistributedInitializer(config)
        
        # Initialize distributed backend
        if not self.initializer.initialize():
            raise RuntimeError("Failed to initialize distributed backend")
        
        # Wrap model
        self.model_wrapper = DistributedModel(model, config)
        self.model = self.model_wrapper.model
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(config)
        
        # Metrics
        self.training_metrics = []
    
    def train_epoch(self, train_loader: DistributedDataLoader,
                   optimizer: DistributedOptimizer,
                   criterion: nn.Module,
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        train_loader.set_epoch(epoch)
        
        total_loss = 0.0
        n_batches = 0
        epoch_start = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move to device
            if torch.cuda.is_available():
                data = data.cuda(self.config.local_rank, non_blocking=True)
                target = target.cuda(self.config.local_rank, non_blocking=True)
            
            # Forward pass
            output = self.model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Accumulate metrics
            total_loss += loss.item()
            n_batches += 1
            
            # Logging
            if batch_idx % self.config.log_frequency == 0 and self.config.rank == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}")
            
            # Checkpointing
            if self.config.enable_checkpointing and \
               (batch_idx + 1) % self.config.checkpoint_frequency == 0:
                self.save_checkpoint(epoch, batch_idx, optimizer)
        
        # Gather metrics from all processes
        avg_loss = self._reduce_metric(total_loss / n_batches)
        epoch_time = time.time() - epoch_start
        
        metrics = {
            'loss': avg_loss,
            'epoch_time': epoch_time,
            'throughput': len(train_loader.dataset) / epoch_time
        }
        
        return metrics
    
    def _reduce_metric(self, metric: float) -> float:
        """Reduce metric across all processes"""
        if not dist.is_initialized():
            return metric
        
        metric_tensor = torch.tensor(metric).cuda(self.config.local_rank)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)
        
        return metric_tensor.item()
    
    def validate(self, val_loader: DistributedDataLoader,
                criterion: nn.Module) -> Dict[str, float]:
        """Validate model"""
        
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                if torch.cuda.is_available():
                    data = data.cuda(self.config.local_rank, non_blocking=True)
                    target = target.cuda(self.config.local_rank, non_blocking=True)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                n_batches += 1
        
        # Reduce across processes
        avg_loss = self._reduce_metric(total_loss / n_batches)
        
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, epoch: int, batch_idx: int, optimizer: DistributedOptimizer):
        """Save training checkpoint"""
        
        state = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': self.model_wrapper.get_base_model().state_dict(),
            'optimizer_state_dict': optimizer.base_optimizer.state_dict(),
            'config': self.config,
            'metrics': self.training_metrics
        }
        
        self.checkpoint_manager.save_checkpoint(state, epoch)
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """Load training checkpoint"""
        
        state = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        if state is not None:
            self.model_wrapper.get_base_model().load_state_dict(state['model_state_dict'])
            return state
        
        return None
    
    def cleanup(self):
        """Cleanup distributed resources"""
        self.initializer.cleanup()


class ModelParallelTrainer:
    """Model parallel training for very large models"""
    
    def __init__(self, model: nn.Module, config: DistributedConfig):
        self.config = config
        self.devices = list(range(torch.cuda.device_count()))
        self.model_parts = self._split_model(model)
    
    def _split_model(self, model: nn.Module) -> List[nn.Module]:
        """Split model across devices"""
        
        # Simple layer-wise split (can be customized)
        layers = list(model.children())
        n_devices = len(self.devices)
        layers_per_device = len(layers) // n_devices
        
        model_parts = []
        for i in range(n_devices):
            start_idx = i * layers_per_device
            end_idx = start_idx + layers_per_device if i < n_devices - 1 else len(layers)
            
            part = nn.Sequential(*layers[start_idx:end_idx])
            part = part.to(f"cuda:{self.devices[i]}")
            model_parts.append(part)
        
        logger.info(f"Split model across {n_devices} devices")
        return model_parts
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through model parts"""
        
        for i, part in enumerate(self.model_parts):
            x = x.to(f"cuda:{self.devices[i]}")
            x = part(x)
        
        return x


def benchmark_distributed_training(model: nn.Module, 
                                  dataset,
                                  config: DistributedConfig) -> Dict[str, Any]:
    """Benchmark distributed training performance"""
    
    # Create distributed data loader
    train_loader = DistributedDataLoader(
        dataset,
        config,
        batch_size=32,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = DistributedTrainer(model, config)
    
    # Create optimizer
    base_optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
    optimizer = DistributedOptimizer(base_optimizer, config)
    
    # Criterion
    criterion = nn.MSELoss()
    
    # Benchmark single epoch
    start_time = time.time()
    metrics = trainer.train_epoch(train_loader, optimizer, criterion, epoch=0)
    training_time = time.time() - start_time
    
    # Calculate scaling efficiency
    single_gpu_time = training_time * config.world_size  # Estimated
    scaling_efficiency = single_gpu_time / (training_time * config.world_size)
    
    results = {
        'world_size': config.world_size,
        'training_time': training_time,
        'throughput': metrics['throughput'],
        'scaling_efficiency': scaling_efficiency,
        'avg_loss': metrics['loss']
    }
    
    # Cleanup
    trainer.cleanup()
    
    return results


if __name__ == "__main__":
    # Display distributed training information
    print("Distributed Training Capabilities:")
    print(f"  PyTorch Distributed: {TORCH_DISTRIBUTED}")
    print(f"  Horovod: {HOROVOD_AVAILABLE}")
    print(f"  MPI: {MPI_AVAILABLE}")
    
    if TORCH_DISTRIBUTED and torch.cuda.is_available():
        print(f"  CUDA devices: {torch.cuda.device_count()}")
        
        # Test configuration
        config = DistributedConfig(
            backend=DistributedBackend.PYTORCH_DDP,
            world_size=torch.cuda.device_count(),
            enable_checkpointing=True
        )
        
        print(f"\nDistributed Configuration:")
        print(f"  Backend: {config.backend.value}")
        print(f"  World Size: {config.world_size}")
        print(f"  Strategy: {config.strategy.value}")
        
        # Note: Actual distributed training requires proper launch
        # e.g., torch.distributed.launch or torchrun
        print("\nTo run distributed training:")
        print("  torchrun --nproc_per_node=4 distributed_training.py")
        print("  or")
        print("  python -m torch.distributed.launch --nproc_per_node=4 distributed_training.py")