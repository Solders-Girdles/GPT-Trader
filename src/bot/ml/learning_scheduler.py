"""
Learning Rate Scheduler for Online Learning Pipeline
Phase 3 - ADAPT-002: Adaptive learning rate scheduling

Implements multiple learning rate scheduling strategies with auto-adjustment
based on convergence metrics and performance tracking.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class SchedulerType(Enum):
    """Available learning rate scheduler types"""
    EXPONENTIAL = "exponential"
    STEP = "step"
    COSINE = "cosine"
    PLATEAU = "plateau"
    ADAPTIVE = "adaptive"
    CYCLICAL = "cyclical"


@dataclass
class SchedulerConfig:
    """Configuration for learning rate schedulers"""
    scheduler_type: SchedulerType = SchedulerType.ADAPTIVE
    initial_lr: float = 0.001
    min_lr: float = 1e-6
    max_lr: float = 0.1
    
    # Exponential decay
    decay_rate: float = 0.95
    decay_steps: int = 1000
    
    # Step decay
    step_size: int = 500
    gamma: float = 0.1
    
    # Cosine annealing
    T_max: int = 1000
    eta_min: float = 1e-6
    
    # Plateau reduction
    patience: int = 50
    factor: float = 0.5
    threshold: float = 1e-4
    cooldown: int = 10
    
    # Adaptive parameters
    increase_factor: float = 1.05
    decrease_factor: float = 0.95
    performance_window: int = 100
    min_improvement: float = 1e-4
    
    # Cyclical parameters
    base_lr: float = 1e-4
    max_lr_cycle: float = 1e-2
    step_size_up: int = 200
    step_size_down: int = 200
    mode: str = "triangular"


class LearningRateScheduler:
    """
    Adaptive learning rate scheduler with multiple strategies.
    
    Features:
    - Multiple scheduling algorithms
    - Auto-adjustment based on performance
    - Learning rate history tracking
    - Convergence detection
    """
    
    def __init__(self, config: SchedulerConfig):
        """Initialize learning rate scheduler
        
        Args:
            config: Scheduler configuration
        """
        self.config = config
        self.current_lr = config.initial_lr
        self.step_count = 0
        self.performance_history: List[float] = []
        self.lr_history: List[float] = []
        self.best_performance = -float('inf')
        self.patience_counter = 0
        self.cooldown_counter = 0
        self.cycle_step = 0
        
        logger.info(f"Initialized {config.scheduler_type.value} learning rate scheduler")
    
    def step(self, performance_metric: Optional[float] = None) -> float:
        """
        Update learning rate based on step count and performance.
        
        Args:
            performance_metric: Current performance metric (higher is better)
            
        Returns:
            Updated learning rate
        """
        self.step_count += 1
        
        if performance_metric is not None:
            self.performance_history.append(performance_metric)
            
            # Keep only recent history
            if len(self.performance_history) > self.config.performance_window:
                self.performance_history.pop(0)
        
        # Update learning rate based on scheduler type
        if self.config.scheduler_type == SchedulerType.EXPONENTIAL:
            self._exponential_decay()
        elif self.config.scheduler_type == SchedulerType.STEP:
            self._step_decay()
        elif self.config.scheduler_type == SchedulerType.COSINE:
            self._cosine_annealing()
        elif self.config.scheduler_type == SchedulerType.PLATEAU:
            self._plateau_reduction(performance_metric)
        elif self.config.scheduler_type == SchedulerType.ADAPTIVE:
            self._adaptive_adjustment(performance_metric)
        elif self.config.scheduler_type == SchedulerType.CYCLICAL:
            self._cyclical_lr()
        
        # Ensure learning rate is within bounds
        self.current_lr = np.clip(
            self.current_lr, 
            self.config.min_lr, 
            self.config.max_lr
        )
        
        self.lr_history.append(self.current_lr)
        
        # Keep only recent history
        if len(self.lr_history) > 10000:
            self.lr_history = self.lr_history[-5000:]
        
        return self.current_lr
    
    def _exponential_decay(self):
        """Exponential learning rate decay"""
        decay_factor = self.config.decay_rate ** (self.step_count // self.config.decay_steps)
        self.current_lr = self.config.initial_lr * decay_factor
    
    def _step_decay(self):
        """Step-wise learning rate decay"""
        if self.step_count % self.config.step_size == 0 and self.step_count > 0:
            self.current_lr *= self.config.gamma
    
    def _cosine_annealing(self):
        """Cosine annealing learning rate schedule"""
        self.current_lr = self.config.eta_min + 0.5 * (
            self.config.initial_lr - self.config.eta_min
        ) * (1 + np.cos(np.pi * self.step_count / self.config.T_max))
    
    def _plateau_reduction(self, performance_metric: Optional[float]):
        """Reduce learning rate when performance plateaus"""
        if performance_metric is None or len(self.performance_history) < 2:
            return
        
        # Check if we're in cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        
        # Check for improvement
        if performance_metric > self.best_performance + self.config.threshold:
            self.best_performance = performance_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Reduce learning rate if no improvement
        if self.patience_counter >= self.config.patience:
            self.current_lr *= self.config.factor
            self.patience_counter = 0
            self.cooldown_counter = self.config.cooldown
            logger.info(f"Reduced learning rate to {self.current_lr:.6f} due to plateau")
    
    def _adaptive_adjustment(self, performance_metric: Optional[float]):
        """Adaptive learning rate adjustment based on recent performance"""
        if performance_metric is None or len(self.performance_history) < 10:
            return
        
        # Calculate recent performance trend
        recent_window = min(10, len(self.performance_history))
        recent_performance = self.performance_history[-recent_window:]
        
        if len(recent_performance) < 5:
            return
        
        # Calculate improvement rate
        half_point = len(recent_performance) // 2
        first_half = np.mean(recent_performance[:half_point])
        second_half = np.mean(recent_performance[half_point:])
        
        improvement = (second_half - first_half) / abs(first_half) if first_half != 0 else 0
        
        # Adjust learning rate based on improvement
        if improvement > self.config.min_improvement:
            # Performance is improving, potentially increase LR
            self.current_lr *= self.config.increase_factor
        elif improvement < -self.config.min_improvement:
            # Performance is degrading, decrease LR
            self.current_lr *= self.config.decrease_factor
        
        # Additional check for variance (stability)
        performance_std = np.std(recent_performance)
        performance_mean = np.mean(recent_performance)
        cv = performance_std / abs(performance_mean) if performance_mean != 0 else 0
        
        # If performance is too volatile, reduce learning rate
        if cv > 0.1:  # High coefficient of variation
            self.current_lr *= 0.98
    
    def _cyclical_lr(self):
        """Cyclical learning rate (triangular pattern)"""
        cycle_length = self.config.step_size_up + self.config.step_size_down
        cycle_position = self.step_count % cycle_length
        
        if cycle_position <= self.config.step_size_up:
            # Increasing phase
            factor = cycle_position / self.config.step_size_up
        else:
            # Decreasing phase
            factor = 1 - (cycle_position - self.config.step_size_up) / self.config.step_size_down
        
        lr_range = self.config.max_lr_cycle - self.config.base_lr
        self.current_lr = self.config.base_lr + factor * lr_range
    
    def get_current_lr(self) -> float:
        """Get current learning rate"""
        return self.current_lr
    
    def get_lr_history(self) -> List[float]:
        """Get learning rate history"""
        return self.lr_history.copy()
    
    def get_performance_history(self) -> List[float]:
        """Get performance history"""
        return self.performance_history.copy()
    
    def reset(self):
        """Reset scheduler to initial state"""
        self.current_lr = self.config.initial_lr
        self.step_count = 0
        self.performance_history.clear()
        self.lr_history.clear()
        self.best_performance = -float('inf')
        self.patience_counter = 0
        self.cooldown_counter = 0
        self.cycle_step = 0
        
        logger.info("Reset learning rate scheduler")
    
    def is_converged(self, 
                     window_size: int = 50, 
                     tolerance: float = 1e-5) -> bool:
        """
        Check if learning rate has converged (stabilized).
        
        Args:
            window_size: Number of recent steps to check
            tolerance: Tolerance for considering LR stable
            
        Returns:
            True if learning rate has converged
        """
        if len(self.lr_history) < window_size:
            return False
        
        recent_lrs = self.lr_history[-window_size:]
        lr_std = np.std(recent_lrs)
        lr_mean = np.mean(recent_lrs)
        
        # Check if coefficient of variation is below tolerance
        cv = lr_std / lr_mean if lr_mean > 0 else float('inf')
        return cv < tolerance
    
    def suggest_optimal_lr(self) -> float:
        """
        Suggest optimal learning rate based on performance history.
        
        Returns:
            Suggested optimal learning rate
        """
        if len(self.performance_history) < 20 or len(self.lr_history) < 20:
            return self.config.initial_lr
        
        # Find LR that led to best performance improvements
        window_size = 10
        best_improvement = -float('inf')
        best_lr = self.config.initial_lr
        
        for i in range(window_size, len(self.performance_history)):
            # Calculate improvement over window
            before = np.mean(self.performance_history[i-window_size:i])
            after = np.mean(self.performance_history[i:i+window_size]) if i+window_size <= len(self.performance_history) else self.performance_history[i]
            
            improvement = (after - before) / abs(before) if before != 0 else 0
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_lr = self.lr_history[i]
        
        # Ensure suggested LR is within bounds
        return np.clip(best_lr, self.config.min_lr, self.config.max_lr)
    
    def get_statistics(self) -> Dict:
        """Get scheduler statistics"""
        stats = {
            'current_lr': self.current_lr,
            'step_count': self.step_count,
            'scheduler_type': self.config.scheduler_type.value,
            'lr_history_length': len(self.lr_history),
            'performance_history_length': len(self.performance_history),
            'is_converged': self.is_converged(),
            'suggested_optimal_lr': self.suggest_optimal_lr()
        }
        
        if self.lr_history:
            stats.update({
                'lr_min': min(self.lr_history),
                'lr_max': max(self.lr_history),
                'lr_mean': np.mean(self.lr_history),
                'lr_std': np.std(self.lr_history)
            })
        
        if self.performance_history:
            stats.update({
                'performance_min': min(self.performance_history),
                'performance_max': max(self.performance_history),
                'performance_mean': np.mean(self.performance_history),
                'performance_std': np.std(self.performance_history),
                'performance_trend': self._calculate_trend()
            })
        
        return stats
    
    def _calculate_trend(self) -> str:
        """Calculate performance trend"""
        if len(self.performance_history) < 10:
            return "insufficient_data"
        
        recent = self.performance_history[-10:]
        older = self.performance_history[-20:-10] if len(self.performance_history) >= 20 else self.performance_history[:-10]
        
        if not older:
            return "insufficient_data"
        
        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        
        improvement = (recent_mean - older_mean) / abs(older_mean) if older_mean != 0 else 0
        
        if improvement > 0.01:
            return "improving"
        elif improvement < -0.01:
            return "degrading"
        else:
            return "stable"
    
    def export_state(self) -> Dict:
        """Export scheduler state for persistence"""
        return {
            'config': {
                'scheduler_type': self.config.scheduler_type.value,
                'initial_lr': self.config.initial_lr,
                'min_lr': self.config.min_lr,
                'max_lr': self.config.max_lr,
                'decay_rate': self.config.decay_rate,
                'decay_steps': self.config.decay_steps,
                'step_size': self.config.step_size,
                'gamma': self.config.gamma,
                'T_max': self.config.T_max,
                'eta_min': self.config.eta_min,
                'patience': self.config.patience,
                'factor': self.config.factor,
                'threshold': self.config.threshold,
                'cooldown': self.config.cooldown,
                'increase_factor': self.config.increase_factor,
                'decrease_factor': self.config.decrease_factor,
                'performance_window': self.config.performance_window,
                'min_improvement': self.config.min_improvement,
                'base_lr': self.config.base_lr,
                'max_lr_cycle': self.config.max_lr_cycle,
                'step_size_up': self.config.step_size_up,
                'step_size_down': self.config.step_size_down,
                'mode': self.config.mode
            },
            'state': {
                'current_lr': self.current_lr,
                'step_count': self.step_count,
                'performance_history': self.performance_history,
                'lr_history': self.lr_history,
                'best_performance': self.best_performance,
                'patience_counter': self.patience_counter,
                'cooldown_counter': self.cooldown_counter,
                'cycle_step': self.cycle_step
            },
            'export_timestamp': datetime.now().isoformat()
        }
    
    def import_state(self, state_dict: Dict):
        """Import scheduler state from persistence"""
        try:
            # Import state
            state = state_dict['state']
            self.current_lr = state['current_lr']
            self.step_count = state['step_count']
            self.performance_history = state['performance_history']
            self.lr_history = state['lr_history']
            self.best_performance = state['best_performance']
            self.patience_counter = state['patience_counter']
            self.cooldown_counter = state['cooldown_counter']
            self.cycle_step = state['cycle_step']
            
            logger.info(f"Imported scheduler state from {state_dict.get('export_timestamp', 'unknown')}")
            
        except KeyError as e:
            logger.error(f"Failed to import scheduler state: missing key {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to import scheduler state: {e}")
            raise


# Factory function for creating schedulers
def create_scheduler(scheduler_type: Union[str, SchedulerType], **kwargs) -> LearningRateScheduler:
    """
    Factory function to create learning rate schedulers.
    
    Args:
        scheduler_type: Type of scheduler to create
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured learning rate scheduler
    """
    if isinstance(scheduler_type, str):
        scheduler_type = SchedulerType(scheduler_type)
    
    config = SchedulerConfig(scheduler_type=scheduler_type, **kwargs)
    return LearningRateScheduler(config)


# Predefined scheduler configurations
AGGRESSIVE_SCHEDULER = SchedulerConfig(
    scheduler_type=SchedulerType.ADAPTIVE,
    initial_lr=0.01,
    increase_factor=1.1,
    decrease_factor=0.9,
    min_improvement=0.001
)

CONSERVATIVE_SCHEDULER = SchedulerConfig(
    scheduler_type=SchedulerType.EXPONENTIAL,
    initial_lr=0.001,
    decay_rate=0.99,
    decay_steps=100
)

CYCLICAL_SCHEDULER = SchedulerConfig(
    scheduler_type=SchedulerType.CYCLICAL,
    base_lr=1e-4,
    max_lr_cycle=1e-2,
    step_size_up=100,
    step_size_down=100
)