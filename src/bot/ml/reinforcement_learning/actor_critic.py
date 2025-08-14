"""
RL-004: Actor-Critic Architecture
Phase 4 - Week 3

Actor-Critic methods for trading:
- Advantage Actor-Critic (A2C)
- Asynchronous Advantage Actor-Critic (A3C)
- Soft Actor-Critic (SAC)
- TD3 (Twin Delayed DDPG)
- Continuous and discrete action spaces
"""

import logging
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import torch
    import torch.multiprocessing as torch_mp
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical, Normal

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

logger = logging.getLogger(__name__)


class ActorCriticType(Enum):
    """Types of Actor-Critic algorithms"""

    A2C = "a2c"  # Advantage Actor-Critic
    A3C = "a3c"  # Asynchronous Advantage Actor-Critic
    SAC = "sac"  # Soft Actor-Critic
    TD3 = "td3"  # Twin Delayed DDPG


@dataclass
class ActorCriticConfig:
    """Configuration for Actor-Critic methods"""

    # Architecture
    state_dim: int = 50
    action_dim: int = 3
    hidden_dims: list[int] = None
    activation: str = "relu"

    # Algorithm type
    algorithm: ActorCriticType = ActorCriticType.A2C
    continuous_actions: bool = False

    # Training parameters
    actor_lr: float = 0.0003
    critic_lr: float = 0.001
    discount_factor: float = 0.99
    tau: float = 0.005  # Soft update parameter

    # A2C/A3C specific
    n_steps: int = 5  # N-step returns
    gae_lambda: float = 0.95  # GAE parameter
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5

    # SAC specific
    alpha: float = 0.2  # Temperature parameter
    auto_alpha: bool = True  # Automatic temperature tuning
    target_entropy: float | None = None

    # TD3 specific
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2  # Delayed policy updates

    # A3C specific
    n_workers: int = 4  # Number of parallel workers

    # Buffer
    buffer_size: int = 100000
    batch_size: int = 256

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]

        if self.target_entropy is None and self.continuous_actions:
            self.target_entropy = -self.action_dim


if TORCH_AVAILABLE:

    class Actor(nn.Module):
        """Actor network for policy"""

        def __init__(self, config: ActorCriticConfig):
            super().__init__()
            self.config = config

            # Build layers
            layers = []
            input_dim = config.state_dim

            for hidden_dim in config.hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                if config.activation == "relu":
                    layers.append(nn.ReLU())
                elif config.activation == "tanh":
                    layers.append(nn.Tanh())
                input_dim = hidden_dim

            self.features = nn.Sequential(*layers)

            if config.continuous_actions:
                # Continuous action space
                self.mean = nn.Linear(input_dim, config.action_dim)
                self.log_std = nn.Linear(input_dim, config.action_dim)
            else:
                # Discrete action space
                self.logits = nn.Linear(input_dim, config.action_dim)

        def forward(self, state: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            """Forward pass"""
            features = self.features(state)

            if self.config.continuous_actions:
                mean = self.mean(features)
                log_std = self.log_std(features)
                log_std = torch.clamp(log_std, min=-20, max=2)
                return mean, log_std
            else:
                return self.logits(features)

        def get_action(self, state: torch.Tensor, deterministic: bool = False):
            """Sample action from policy"""
            if self.config.continuous_actions:
                mean, log_std = self.forward(state)
                std = log_std.exp()

                if deterministic:
                    action = mean
                else:
                    dist = Normal(mean, std)
                    action = dist.sample()

                # Compute log probability
                dist = Normal(mean, std)
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

                # Apply tanh squashing for bounded actions
                action = torch.tanh(action)

                # Correct log_prob for tanh
                log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(
                    dim=-1, keepdim=True
                )

                return action, log_prob
            else:
                logits = self.forward(state)
                dist = Categorical(logits=logits)

                if deterministic:
                    action = logits.argmax(dim=-1)
                else:
                    action = dist.sample()

                log_prob = dist.log_prob(action)

                return action, log_prob

    class Critic(nn.Module):
        """Critic network for value function"""

        def __init__(self, config: ActorCriticConfig, double_q: bool = False):
            super().__init__()
            self.config = config
            self.double_q = double_q

            # Build Q1 network
            self.q1 = self._build_network()

            # Build Q2 network for double Q-learning
            if double_q:
                self.q2 = self._build_network()

        def _build_network(self) -> nn.Sequential:
            """Build a Q-network"""
            layers = []

            if self.config.continuous_actions:
                input_dim = self.config.state_dim + self.config.action_dim
            else:
                input_dim = self.config.state_dim

            for hidden_dim in self.config.hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                if self.config.activation == "relu":
                    layers.append(nn.ReLU())
                elif self.config.activation == "tanh":
                    layers.append(nn.Tanh())
                input_dim = hidden_dim

            # Output layer
            if self.config.continuous_actions:
                layers.append(nn.Linear(input_dim, 1))
            else:
                layers.append(nn.Linear(input_dim, self.config.action_dim))

            return nn.Sequential(*layers)

        def forward(
            self, state: torch.Tensor, action: torch.Tensor | None = None
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            """Forward pass"""
            if self.config.continuous_actions and action is not None:
                x = torch.cat([state, action], dim=-1)
            else:
                x = state

            q1 = self.q1(x)

            if self.double_q:
                q2 = self.q2(x)
                return q1, q2

            return q1

    class A2C:
        """Advantage Actor-Critic (A2C)"""

        def __init__(self, config: ActorCriticConfig):
            self.config = config
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Networks
            self.actor = Actor(config).to(self.device)
            self.critic = Critic(config, double_q=False).to(self.device)

            # Optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

            # Storage for n-step returns
            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []
            self.log_probs = []
            self.dones = []

            # Training history
            self.training_history = []

        def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
            """Select action from policy"""
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob = self.actor.get_action(state_tensor, deterministic)
                value = self.critic(state_tensor)

            # Store for training
            if not deterministic:
                self.states.append(state)
                self.actions.append(action.cpu().numpy().squeeze())
                self.log_probs.append(log_prob)
                self.values.append(value)

            return action.cpu().numpy().squeeze()

        def compute_gae(self, next_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute Generalized Advantage Estimation"""
            rewards = torch.FloatTensor(self.rewards).to(self.device)
            dones = torch.FloatTensor(self.dones).to(self.device)
            values = torch.cat(self.values).squeeze()

            # Compute returns and advantages
            returns = []
            advantages = []

            gae = 0
            R = next_value

            for t in reversed(range(len(rewards))):
                R = rewards[t] + self.config.discount_factor * R * (1 - dones[t])
                returns.insert(0, R)

                if t == len(rewards) - 1:
                    next_val = next_value
                else:
                    next_val = values[t + 1]

                delta = (
                    rewards[t] + self.config.discount_factor * next_val * (1 - dones[t]) - values[t]
                )
                gae = delta + self.config.discount_factor * self.config.gae_lambda * gae * (
                    1 - dones[t]
                )
                advantages.insert(0, gae)

            returns = torch.stack(returns)
            advantages = torch.FloatTensor(advantages).to(self.device)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        def update(self, next_state: np.ndarray) -> dict[str, float]:
            """Update actor and critic networks"""

            if len(self.states) == 0:
                return {}

            # Get next value
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.critic(next_state_tensor)

            # Compute returns and advantages
            returns, advantages = self.compute_gae(next_value)

            # Convert to tensors
            states = torch.FloatTensor(self.states).to(self.device)
            actions = torch.FloatTensor(self.actions).to(self.device)
            log_probs = torch.cat(self.log_probs)

            # Actor loss
            if self.config.continuous_actions:
                new_actions, new_log_probs = self.actor.get_action(states)
            else:
                new_actions, new_log_probs = self.actor.get_action(states)

            ratio = (new_log_probs - log_probs).exp()
            actor_loss = -(ratio * advantages.detach()).mean()

            # Entropy regularization
            if self.config.continuous_actions:
                entropy = -new_log_probs.mean()
            else:
                logits = self.actor(states)
                dist = Categorical(logits=logits)
                entropy = dist.entropy().mean()

            actor_loss -= self.config.entropy_coeff * entropy

            # Critic loss
            values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(values, returns)

            # Update networks
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            self.critic_optimizer.step()

            # Clear storage
            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []
            self.log_probs = []
            self.dones = []

            return {
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "entropy": entropy.item(),
            }

        def store_transition(self, reward: float, done: bool):
            """Store transition data"""
            self.rewards.append(reward)
            self.dones.append(done)

    class SAC:
        """Soft Actor-Critic for continuous control"""

        def __init__(self, config: ActorCriticConfig):
            self.config = config
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Networks
            self.actor = Actor(config).to(self.device)
            self.critic = Critic(config, double_q=True).to(self.device)
            self.critic_target = Critic(config, double_q=True).to(self.device)

            # Copy target weights
            self.critic_target.load_state_dict(self.critic.state_dict())

            # Optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

            # Temperature parameter
            if config.auto_alpha:
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.actor_lr)
                self.target_entropy = config.target_entropy
            else:
                self.alpha = config.alpha

            # Replay buffer
            self.replay_buffer = deque(maxlen=config.buffer_size)

            # Training history
            self.training_history = []

        def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
            """Select action from policy"""
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, _ = self.actor.get_action(state_tensor, deterministic)

            return action.cpu().numpy().squeeze()

        def store_transition(self, state, action, reward, next_state, done):
            """Store transition in replay buffer"""
            self.replay_buffer.append((state, action, reward, next_state, done))

        def update(self) -> dict[str, float]:
            """Update SAC networks"""

            if len(self.replay_buffer) < self.config.batch_size:
                return {}

            # Sample batch
            batch = random.sample(self.replay_buffer, self.config.batch_size)
            states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
            actions = torch.FloatTensor([e[1] for e in batch]).to(self.device)
            rewards = torch.FloatTensor([e[2] for e in batch]).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
            dones = torch.FloatTensor([e[4] for e in batch]).unsqueeze(1).to(self.device)

            # Get current alpha
            if self.config.auto_alpha:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha

            # Update critic
            with torch.no_grad():
                next_actions, next_log_probs = self.actor.get_action(next_states)
                q1_next, q2_next = self.critic_target(next_states, next_actions)
                q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
                q_target = rewards + self.config.discount_factor * (1 - dones) * q_next

            q1, q2 = self.critic(states, actions)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update actor
            new_actions, log_probs = self.actor.get_action(states)
            q1_new, q2_new = self.critic(states, new_actions)
            q_new = torch.min(q1_new, q2_new)

            actor_loss = (alpha * log_probs - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update temperature
            if self.config.auto_alpha:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
            else:
                alpha_loss = torch.tensor(0.0)

            # Soft update target network
            for target_param, param in zip(
                self.critic_target.parameters(), self.critic.parameters(), strict=False
            ):
                target_param.data.copy_(
                    self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                )

            return {
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item(),
                "alpha_loss": alpha_loss.item(),
                "alpha": alpha.item() if isinstance(alpha, torch.Tensor) else alpha,
            }

    class TD3:
        """Twin Delayed DDPG for continuous control"""

        def __init__(self, config: ActorCriticConfig):
            self.config = config
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Networks
            self.actor = Actor(config).to(self.device)
            self.actor_target = Actor(config).to(self.device)
            self.critic = Critic(config, double_q=True).to(self.device)
            self.critic_target = Critic(config, double_q=True).to(self.device)

            # Copy target weights
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

            # Optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

            # Replay buffer
            self.replay_buffer = deque(maxlen=config.buffer_size)

            # Update counter for delayed policy updates
            self.update_counter = 0

            # Training history
            self.training_history = []

        def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
            """Select action with exploration noise"""
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, _ = self.actor.get_action(state_tensor, deterministic=True)
                action = action.cpu().numpy().squeeze()

            if add_noise:
                noise = np.random.normal(0, self.config.policy_noise, size=action.shape)
                action = np.clip(action + noise, -1, 1)

            return action

        def store_transition(self, state, action, reward, next_state, done):
            """Store transition in replay buffer"""
            self.replay_buffer.append((state, action, reward, next_state, done))

        def update(self) -> dict[str, float]:
            """Update TD3 networks"""

            if len(self.replay_buffer) < self.config.batch_size:
                return {}

            self.update_counter += 1

            # Sample batch
            batch = random.sample(self.replay_buffer, self.config.batch_size)
            states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
            actions = torch.FloatTensor([e[1] for e in batch]).to(self.device)
            rewards = torch.FloatTensor([e[2] for e in batch]).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
            dones = torch.FloatTensor([e[4] for e in batch]).unsqueeze(1).to(self.device)

            # Select next action with noise
            with torch.no_grad():
                next_actions, _ = self.actor_target.get_action(next_states, deterministic=True)

                # Add clipped noise
                noise = torch.randn_like(next_actions) * self.config.policy_noise
                noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
                next_actions = (next_actions + noise).clamp(-1, 1)

                # Compute target Q-values
                q1_next, q2_next = self.critic_target(next_states, next_actions)
                q_next = torch.min(q1_next, q2_next)
                q_target = rewards + self.config.discount_factor * (1 - dones) * q_next

            # Update critic
            q1, q2 = self.critic(states, actions)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = torch.tensor(0.0)

            # Delayed policy update
            if self.update_counter % self.config.policy_delay == 0:
                # Update actor
                new_actions, _ = self.actor.get_action(states, deterministic=True)
                q1_new, _ = self.critic(states, new_actions)
                actor_loss = -q1_new.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update target networks
                for target_param, param in zip(
                    self.actor_target.parameters(), self.actor.parameters(), strict=False
                ):
                    target_param.data.copy_(
                        self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                    )

                for target_param, param in zip(
                    self.critic_target.parameters(), self.critic.parameters(), strict=False
                ):
                    target_param.data.copy_(
                        self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                    )

            return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}


def create_actor_critic_agent(
    state_dim: int,
    action_dim: int,
    algorithm: ActorCriticType = ActorCriticType.A2C,
    continuous: bool = False,
) -> A2C | SAC | TD3:
    """Factory function to create actor-critic agent"""

    config = ActorCriticConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        algorithm=algorithm,
        continuous_actions=continuous,
    )

    if algorithm == ActorCriticType.A2C:
        return A2C(config)
    elif algorithm == ActorCriticType.SAC:
        config.continuous_actions = True  # SAC is for continuous control
        return SAC(config)
    elif algorithm == ActorCriticType.TD3:
        config.continuous_actions = True  # TD3 is for continuous control
        return TD3(config)
    else:
        return A2C(config)  # Default to A2C


if __name__ == "__main__":
    print("Actor-Critic Architecture Implementation")

    if TORCH_AVAILABLE:
        # Test A2C
        print("\n=== Testing A2C ===")
        config = ActorCriticConfig(
            state_dim=50, action_dim=3, algorithm=ActorCriticType.A2C, continuous_actions=False
        )

        a2c_agent = A2C(config)

        # Test action selection
        test_state = np.random.randn(config.state_dim)
        action = a2c_agent.select_action(test_state)
        print(f"A2C action: {action}")

        # Test SAC
        print("\n=== Testing SAC ===")
        sac_config = ActorCriticConfig(
            state_dim=50, action_dim=3, algorithm=ActorCriticType.SAC, continuous_actions=True
        )

        sac_agent = SAC(sac_config)
        action = sac_agent.select_action(test_state)
        print(f"SAC action shape: {action.shape}")

        # Test TD3
        print("\n=== Testing TD3 ===")
        td3_config = ActorCriticConfig(
            state_dim=50, action_dim=3, algorithm=ActorCriticType.TD3, continuous_actions=True
        )

        td3_agent = TD3(td3_config)
        action = td3_agent.select_action(test_state)
        print(f"TD3 action shape: {action.shape}")
    else:
        print("PyTorch not available")
