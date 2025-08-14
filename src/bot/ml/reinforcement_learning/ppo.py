"""
RL-005: Proximal Policy Optimization (PPO)
Phase 4 - Week 3

PPO for stable and efficient trading:
- Clipped surrogate objective
- Adaptive KL penalty
- Mini-batch optimization
- Parallel environment collection
- Both discrete and continuous actions
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque
import time
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical, Normal
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

logger = logging.getLogger(__name__)


class PPOType(Enum):
    """Types of PPO variants"""
    CLIP = "clip"  # Clipped surrogate objective
    ADAPTIVE_KL = "adaptive_kl"  # Adaptive KL penalty
    HYBRID = "hybrid"  # Both clip and KL


@dataclass
class PPOConfig:
    """Configuration for PPO"""
    # Architecture
    state_dim: int = 50
    action_dim: int = 3
    hidden_dims: List[int] = None
    activation: str = "tanh"
    continuous_actions: bool = False
    
    # PPO parameters
    ppo_type: PPOType = PPOType.CLIP
    clip_ratio: float = 0.2  # Clip parameter epsilon
    target_kl: float = 0.01  # KL divergence threshold
    kl_coeff: float = 0.2  # Initial KL penalty coefficient
    
    # Training parameters
    learning_rate: float = 0.0003
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    
    # Optimization
    n_epochs: int = 10  # Number of optimization epochs per update
    batch_size: int = 64
    minibatch_size: int = 32
    
    # Coefficients
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    
    # Collection
    n_steps: int = 2048  # Steps to collect before update
    n_envs: int = 1  # Number of parallel environments
    
    # Scheduling
    lr_schedule: bool = True
    entropy_schedule: bool = True
    clip_schedule: bool = False
    
    # Early stopping
    early_stop_kl: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64]


if TORCH_AVAILABLE:
    
    class ActorCriticNetwork(nn.Module):
        """Combined Actor-Critic network for PPO"""
        
        def __init__(self, config: PPOConfig):
            super().__init__()
            self.config = config
            
            # Shared layers
            layers = []
            input_dim = config.state_dim
            
            for hidden_dim in config.hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                if config.activation == "tanh":
                    layers.append(nn.Tanh())
                elif config.activation == "relu":
                    layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(hidden_dim))
                input_dim = hidden_dim
            
            self.shared = nn.Sequential(*layers)
            
            # Actor head
            if config.continuous_actions:
                self.actor_mean = nn.Linear(input_dim, config.action_dim)
                self.actor_log_std = nn.Parameter(torch.zeros(1, config.action_dim))
            else:
                self.actor = nn.Linear(input_dim, config.action_dim)
            
            # Critic head
            self.critic = nn.Linear(input_dim, 1)
            
            # Initialize weights
            self.apply(self._init_weights)
        
        def _init_weights(self, module):
            """Initialize network weights"""
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        def forward(self, state: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
            """Forward pass returning action distribution and value"""
            features = self.shared(state)
            
            # Get action distribution
            if self.config.continuous_actions:
                mean = self.actor_mean(features)
                std = self.actor_log_std.exp().expand_as(mean)
                dist = Normal(mean, std)
            else:
                logits = self.actor(features)
                dist = Categorical(logits=logits)
            
            # Get value
            value = self.critic(features)
            
            return dist, value
        
        def get_action_and_value(self, state: torch.Tensor, action: Optional[torch.Tensor] = None):
            """Get action, log probability, entropy, and value"""
            dist, value = self.forward(state)
            
            if action is None:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            if self.config.continuous_actions:
                log_prob = log_prob.sum(dim=-1)
            
            entropy = dist.entropy()
            if self.config.continuous_actions:
                entropy = entropy.sum(dim=-1)
            
            return action, log_prob, entropy, value.squeeze(-1)
        
        def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
            """Evaluate actions for given states"""
            dist, values = self.forward(states)
            
            log_probs = dist.log_prob(actions)
            if self.config.continuous_actions:
                log_probs = log_probs.sum(dim=-1)
            
            entropy = dist.entropy()
            if self.config.continuous_actions:
                entropy = entropy.sum(dim=-1)
            
            return log_probs, entropy, values.squeeze(-1)
    
    
    class RolloutBuffer:
        """Buffer for storing rollout data"""
        
        def __init__(self, buffer_size: int, state_dim: int, action_dim: int, 
                    n_envs: int = 1, continuous: bool = False):
            self.buffer_size = buffer_size
            self.n_envs = n_envs
            self.continuous = continuous
            self.ptr = 0
            self.full = False
            
            # Storage
            self.states = np.zeros((buffer_size, n_envs, state_dim), dtype=np.float32)
            
            if continuous:
                self.actions = np.zeros((buffer_size, n_envs, action_dim), dtype=np.float32)
            else:
                self.actions = np.zeros((buffer_size, n_envs), dtype=np.int64)
            
            self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
            self.values = np.zeros((buffer_size, n_envs), dtype=np.float32)
            self.log_probs = np.zeros((buffer_size, n_envs), dtype=np.float32)
            self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
            self.advantages = np.zeros((buffer_size, n_envs), dtype=np.float32)
            self.returns = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        def add(self, state, action, reward, value, log_prob, done):
            """Add transition to buffer"""
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.values[self.ptr] = value
            self.log_probs[self.ptr] = log_prob
            self.dones[self.ptr] = done
            
            self.ptr += 1
            if self.ptr == self.buffer_size:
                self.full = True
                self.ptr = 0
        
        def compute_returns_and_advantages(self, last_value: np.ndarray, gamma: float, gae_lambda: float):
            """Compute returns and GAE advantages"""
            last_gae_lambda = 0
            
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - self.dones[step]
                    next_value = last_value
                else:
                    next_non_terminal = 1.0 - self.dones[step + 1]
                    next_value = self.values[step + 1]
                
                delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
                last_gae_lambda = delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
                self.advantages[step] = last_gae_lambda
            
            self.returns = self.advantages + self.values
        
        def get_samples(self, batch_size: int):
            """Get random samples for training"""
            indices = np.random.permutation(self.buffer_size * self.n_envs)
            
            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Flatten across environments
                states = self.states.reshape(-1, self.states.shape[-1])[batch_indices]
                
                if self.continuous:
                    actions = self.actions.reshape(-1, self.actions.shape[-1])[batch_indices]
                else:
                    actions = self.actions.reshape(-1)[batch_indices]
                
                log_probs = self.log_probs.reshape(-1)[batch_indices]
                advantages = self.advantages.reshape(-1)[batch_indices]
                returns = self.returns.reshape(-1)[batch_indices]
                
                yield states, actions, log_probs, advantages, returns
        
        def clear(self):
            """Clear buffer"""
            self.ptr = 0
            self.full = False
    
    
    class PPO:
        """Proximal Policy Optimization"""
        
        def __init__(self, config: PPOConfig):
            self.config = config
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Network
            self.policy = ActorCriticNetwork(config).to(self.device)
            
            # Optimizer
            self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate, eps=1e-5)
            
            # Rollout buffer
            self.rollout_buffer = RolloutBuffer(
                config.n_steps,
                config.state_dim,
                config.action_dim,
                config.n_envs,
                config.continuous_actions
            )
            
            # Adaptive KL
            if config.ppo_type in [PPOType.ADAPTIVE_KL, PPOType.HYBRID]:
                self.kl_coeff = config.kl_coeff
            
            # Training state
            self.n_updates = 0
            self.training_history = []
            
            # Learning rate scheduler
            if config.lr_schedule:
                self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=1000, eta_min=1e-5
                )
        
        def select_action(self, state: np.ndarray, deterministic: bool = False):
            """Select action from policy"""
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            with torch.no_grad():
                dist, value = self.policy(state_tensor)
                
                if deterministic:
                    if self.config.continuous_actions:
                        action = dist.mean
                    else:
                        action = dist.probs.argmax(dim=-1)
                else:
                    action = dist.sample()
                
                log_prob = dist.log_prob(action)
                if self.config.continuous_actions:
                    log_prob = log_prob.sum(dim=-1)
            
            return (
                action.cpu().numpy().squeeze(),
                value.cpu().numpy().squeeze(),
                log_prob.cpu().numpy().squeeze()
            )
        
        def collect_rollouts(self, env) -> bool:
            """Collect rollout data"""
            self.rollout_buffer.clear()
            
            state = env.reset()
            
            for step in range(self.config.n_steps):
                # Get action
                action, value, log_prob = self.select_action(state)
                
                # Step environment
                next_state, reward, done, info = env.step(action)
                
                # Store in buffer
                self.rollout_buffer.add(state, action, reward, value, log_prob, done)
                
                state = next_state
                
                if done:
                    state = env.reset()
            
            # Compute returns and advantages
            with torch.no_grad():
                _, last_value = self.policy(torch.FloatTensor(state).to(self.device))
                last_value = last_value.cpu().numpy().squeeze()
            
            self.rollout_buffer.compute_returns_and_advantages(
                last_value,
                self.config.discount_factor,
                self.config.gae_lambda
            )
            
            return True
        
        def update(self) -> Dict[str, float]:
            """Update policy using PPO"""
            
            # Training metrics
            policy_losses = []
            value_losses = []
            entropy_losses = []
            kl_divs = []
            clip_fractions = []
            
            # Get current entropy coefficient
            entropy_coeff = self.config.entropy_coeff
            if self.config.entropy_schedule:
                entropy_coeff *= max(0.1, 1.0 - self.n_updates / 1000)
            
            # Multiple epochs of optimization
            for epoch in range(self.config.n_epochs):
                approx_kl_divs = []
                
                # Mini-batch updates
                for states, actions, old_log_probs, advantages, returns in \
                        self.rollout_buffer.get_samples(self.config.minibatch_size):
                    
                    # Convert to tensors
                    states = torch.FloatTensor(states).to(self.device)
                    actions = torch.LongTensor(actions).to(self.device) if not self.config.continuous_actions \
                             else torch.FloatTensor(actions).to(self.device)
                    old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
                    advantages = torch.FloatTensor(advantages).to(self.device)
                    returns = torch.FloatTensor(returns).to(self.device)
                    
                    # Normalize advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    # Get current policy evaluation
                    log_probs, entropy, values = self.policy.evaluate_actions(states, actions)
                    
                    # Ratio for importance sampling
                    ratio = torch.exp(log_probs - old_log_probs)
                    
                    # Clipped surrogate loss
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
                    
                    if self.config.ppo_type == PPOType.CLIP:
                        policy_loss = -torch.min(surr1, surr2).mean()
                    elif self.config.ppo_type == PPOType.ADAPTIVE_KL:
                        kl_div = (old_log_probs - log_probs).mean()
                        policy_loss = -(ratio * advantages).mean() - self.kl_coeff * kl_div
                    else:  # HYBRID
                        kl_div = (old_log_probs - log_probs).mean()
                        policy_loss = -torch.min(surr1, surr2).mean() - self.kl_coeff * kl_div
                    
                    # Value loss
                    value_loss = F.mse_loss(values, returns)
                    
                    # Entropy loss
                    entropy_loss = -entropy.mean()
                    
                    # Total loss
                    loss = policy_loss + self.config.value_loss_coeff * value_loss + entropy_coeff * entropy_loss
                    
                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    
                    # Track metrics
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    
                    # KL divergence for early stopping
                    with torch.no_grad():
                        log_ratio = log_probs - old_log_probs
                        approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                        approx_kl_divs.append(approx_kl.item())
                        
                        # Clip fraction
                        clip_fraction = (torch.abs(ratio - 1) > self.config.clip_ratio).float().mean()
                        clip_fractions.append(clip_fraction.item())
                
                # Early stopping based on KL divergence
                if self.config.early_stop_kl:
                    if np.mean(approx_kl_divs) > 1.5 * self.config.target_kl:
                        logger.info(f"Early stopping at epoch {epoch} due to KL divergence")
                        break
                
                kl_divs.extend(approx_kl_divs)
            
            # Adaptive KL coefficient
            if self.config.ppo_type in [PPOType.ADAPTIVE_KL, PPOType.HYBRID]:
                mean_kl = np.mean(kl_divs)
                if mean_kl < self.config.target_kl / 1.5:
                    self.kl_coeff *= 0.5
                elif mean_kl > self.config.target_kl * 1.5:
                    self.kl_coeff *= 2
            
            # Update learning rate
            if self.config.lr_schedule:
                self.lr_scheduler.step()
            
            self.n_updates += 1
            
            return {
                'policy_loss': np.mean(policy_losses),
                'value_loss': np.mean(value_losses),
                'entropy': -np.mean(entropy_losses),
                'kl_divergence': np.mean(kl_divs),
                'clip_fraction': np.mean(clip_fractions),
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
        
        def train(self, env, total_timesteps: int) -> List[Dict[str, float]]:
            """Train PPO for specified timesteps"""
            
            timesteps = 0
            episode_rewards = []
            
            while timesteps < total_timesteps:
                # Collect rollouts
                self.collect_rollouts(env)
                timesteps += self.config.n_steps * self.config.n_envs
                
                # Update policy
                update_metrics = self.update()
                
                # Add timestep info
                update_metrics['timesteps'] = timesteps
                
                self.training_history.append(update_metrics)
                
                # Log progress
                if len(self.training_history) % 10 == 0:
                    recent_metrics = self.training_history[-10:]
                    avg_kl = np.mean([m['kl_divergence'] for m in recent_metrics])
                    avg_clip = np.mean([m['clip_fraction'] for m in recent_metrics])
                    
                    logger.info(f"Timesteps: {timesteps}, KL: {avg_kl:.4f}, Clip: {avg_clip:.3f}")
            
            return self.training_history
        
        def save(self, filepath: str):
            """Save PPO model"""
            save_dict = {
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'n_updates': self.n_updates,
                'training_history': self.training_history
            }
            
            if hasattr(self, 'kl_coeff'):
                save_dict['kl_coeff'] = self.kl_coeff
            
            torch.save(save_dict, filepath)
            logger.info(f"PPO model saved to {filepath}")
        
        def load(self, filepath: str):
            """Load PPO model"""
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.config = checkpoint['config']
            self.n_updates = checkpoint['n_updates']
            self.training_history = checkpoint['training_history']
            
            if 'kl_coeff' in checkpoint:
                self.kl_coeff = checkpoint['kl_coeff']
            
            logger.info(f"PPO model loaded from {filepath}")


def create_ppo_agent(state_dim: int, action_dim: int,
                    continuous: bool = False,
                    ppo_type: PPOType = PPOType.CLIP) -> PPO:
    """Factory function to create PPO agent"""
    
    config = PPOConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous_actions=continuous,
        ppo_type=ppo_type
    )
    
    return PPO(config)


if __name__ == "__main__":
    print("Proximal Policy Optimization (PPO) Implementation")
    
    if TORCH_AVAILABLE:
        # Test PPO with discrete actions
        print("\n=== Testing PPO (Discrete) ===")
        config = PPOConfig(
            state_dim=50,
            action_dim=3,
            continuous_actions=False,
            ppo_type=PPOType.CLIP
        )
        
        ppo_agent = PPO(config)
        
        # Test action selection
        test_state = np.random.randn(config.state_dim)
        action, value, log_prob = ppo_agent.select_action(test_state)
        print(f"Action: {action}, Value: {value:.3f}, Log Prob: {log_prob:.3f}")
        
        # Test PPO with continuous actions
        print("\n=== Testing PPO (Continuous) ===")
        continuous_config = PPOConfig(
            state_dim=50,
            action_dim=3,
            continuous_actions=True,
            ppo_type=PPOType.HYBRID
        )
        
        continuous_ppo = PPO(continuous_config)
        action, value, log_prob = continuous_ppo.select_action(test_state)
        print(f"Continuous Action Shape: {action.shape}")
        print(f"Value: {value:.3f}, Log Prob: {log_prob:.3f}")
        
        print(f"\nDevice: {ppo_agent.device}")
        print(f"Network parameters: {sum(p.numel() for p in ppo_agent.policy.parameters()):,}")
    else:
        print("PyTorch not available")