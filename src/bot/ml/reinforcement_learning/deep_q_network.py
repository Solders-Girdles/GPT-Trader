"""
RL-002: Deep Q-Network (DQN)
Phase 4 - Week 3

Deep Q-Network for complex trading strategies:
- Neural network function approximation
- Target network for stability
- Double DQN to reduce overestimation
- Dueling DQN for value/advantage separation
- Prioritized experience replay
"""

import logging
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, DQN functionality limited")

logger = logging.getLogger(__name__)


class DQNType(Enum):
    """Types of DQN architectures"""

    VANILLA = "vanilla"
    DOUBLE = "double"
    DUELING = "dueling"
    RAINBOW = "rainbow"  # Combination of improvements


@dataclass
class DQNConfig:
    """Configuration for Deep Q-Network"""

    # Network architecture
    state_dim: int = 50
    action_dim: int = 3
    hidden_layers: list[int] = None
    activation: str = "relu"

    # DQN variants
    dqn_type: DQNType = DQNType.DOUBLE
    use_dueling: bool = True
    use_noisy_nets: bool = False

    # Training parameters
    learning_rate: float = 0.0001
    discount_factor: float = 0.99
    tau: float = 0.001  # Soft update parameter

    # Experience replay
    buffer_size: int = 100000
    batch_size: int = 64
    prioritized_replay: bool = True
    alpha: float = 0.6  # Prioritization exponent
    beta_start: float = 0.4  # Importance sampling
    beta_end: float = 1.0

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Training schedule
    update_frequency: int = 4
    target_update_frequency: int = 1000

    # Performance
    gradient_clip: float = 10.0
    reward_scaling: float = 1.0

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 128, 64]


if TORCH_AVAILABLE:

    class NoisyLinear(nn.Module):
        """Noisy linear layer for exploration"""

        def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.std_init = std_init

            # Learnable parameters
            self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))

            # Factorized noise
            self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
            self.register_buffer("bias_epsilon", torch.empty(out_features))

            self.reset_parameters()
            self.reset_noise()

        def reset_parameters(self):
            """Initialize parameters"""
            mu_range = 1 / np.sqrt(self.in_features)
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

        def reset_noise(self):
            """Sample new noise"""
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)

        def _scale_noise(self, size: int) -> torch.Tensor:
            """Generate scaled noise"""
            x = torch.randn(size)
            return x.sign() * x.abs().sqrt()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with noisy weights"""
            if self.training:
                weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                weight = self.weight_mu
                bias = self.bias_mu

            return F.linear(x, weight, bias)

    class DuelingNetwork(nn.Module):
        """Dueling DQN architecture"""

        def __init__(self, config: DQNConfig):
            super().__init__()
            self.config = config

            # Shared feature layers
            layers = []
            input_dim = config.state_dim

            for hidden_dim in config.hidden_layers[:-1]:
                if config.use_noisy_nets:
                    layers.append(NoisyLinear(input_dim, hidden_dim))
                else:
                    layers.append(nn.Linear(input_dim, hidden_dim))

                if config.activation == "relu":
                    layers.append(nn.ReLU())
                elif config.activation == "tanh":
                    layers.append(nn.Tanh())

                input_dim = hidden_dim

            self.features = nn.Sequential(*layers)

            # Value stream
            last_hidden = config.hidden_layers[-1]
            if config.use_noisy_nets:
                self.value_stream = nn.Sequential(
                    NoisyLinear(input_dim, last_hidden), nn.ReLU(), NoisyLinear(last_hidden, 1)
                )
            else:
                self.value_stream = nn.Sequential(
                    nn.Linear(input_dim, last_hidden), nn.ReLU(), nn.Linear(last_hidden, 1)
                )

            # Advantage stream
            if config.use_noisy_nets:
                self.advantage_stream = nn.Sequential(
                    NoisyLinear(input_dim, last_hidden),
                    nn.ReLU(),
                    NoisyLinear(last_hidden, config.action_dim),
                )
            else:
                self.advantage_stream = nn.Sequential(
                    nn.Linear(input_dim, last_hidden),
                    nn.ReLU(),
                    nn.Linear(last_hidden, config.action_dim),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with dueling architecture"""
            features = self.features(x)

            value = self.value_stream(features)
            advantage = self.advantage_stream(features)

            # Combine value and advantage
            q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))

            return q_values

        def reset_noise(self):
            """Reset noise in noisy layers"""
            if self.config.use_noisy_nets:
                for module in self.modules():
                    if isinstance(module, NoisyLinear):
                        module.reset_noise()

    class VanillaDQN(nn.Module):
        """Standard DQN architecture"""

        def __init__(self, config: DQNConfig):
            super().__init__()
            self.config = config

            layers = []
            input_dim = config.state_dim

            for hidden_dim in config.hidden_layers:
                layers.append(nn.Linear(input_dim, hidden_dim))

                if config.activation == "relu":
                    layers.append(nn.ReLU())
                elif config.activation == "tanh":
                    layers.append(nn.Tanh())

                layers.append(nn.Dropout(0.1))
                input_dim = hidden_dim

            layers.append(nn.Linear(input_dim, config.action_dim))

            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass"""
            return self.network(x)

    class PrioritizedReplayBuffer:
        """Prioritized experience replay buffer"""

        def __init__(self, capacity: int, alpha: float = 0.6):
            self.capacity = capacity
            self.alpha = alpha
            self.buffer = []
            self.priorities = np.zeros(capacity, dtype=np.float32)
            self.position = 0
            self.max_priority = 1.0

        def push(self, state, action, reward, next_state, done):
            """Add experience with maximum priority"""
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)

            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = self.max_priority

            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size: int, beta: float = 0.4) -> tuple:
            """Sample batch with importance sampling weights"""
            if len(self.buffer) < batch_size:
                batch_size = len(self.buffer)

            # Calculate sampling probabilities
            priorities = self.priorities[: len(self.buffer)]
            probabilities = priorities**self.alpha
            probabilities /= probabilities.sum()

            # Sample indices
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

            # Calculate importance sampling weights
            total = len(self.buffer)
            weights = (total * probabilities[indices]) ** (-beta)
            weights /= weights.max()

            # Get samples
            batch = [self.buffer[idx] for idx in indices]
            states = torch.FloatTensor([e[0] for e in batch])
            actions = torch.LongTensor([e[1] for e in batch])
            rewards = torch.FloatTensor([e[2] for e in batch])
            next_states = torch.FloatTensor([e[3] for e in batch])
            dones = torch.FloatTensor([e[4] for e in batch])
            weights = torch.FloatTensor(weights)

            return states, actions, rewards, next_states, dones, indices, weights

        def update_priorities(self, indices: list[int], priorities: np.ndarray):
            """Update priorities for sampled experiences"""
            for idx, priority in zip(indices, priorities, strict=False):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)

        def __len__(self):
            return len(self.buffer)

    class DQNAgent:
        """Deep Q-Network agent"""

        def __init__(self, config: DQNConfig):
            self.config = config
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize networks
            if config.use_dueling:
                self.q_network = DuelingNetwork(config).to(self.device)
                self.target_network = DuelingNetwork(config).to(self.device)
            else:
                self.q_network = VanillaDQN(config).to(self.device)
                self.target_network = VanillaDQN(config).to(self.device)

            # Copy weights to target network
            self.target_network.load_state_dict(self.q_network.state_dict())

            # Optimizer
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

            # Replay buffer
            if config.prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(config.buffer_size, config.alpha)
            else:
                self.replay_buffer = deque(maxlen=config.buffer_size)

            # Training state
            self.epsilon = config.epsilon_start
            self.beta = config.beta_start
            self.steps = 0
            self.episodes = 0
            self.training_history = []

        def select_action(self, state: np.ndarray, training: bool = True) -> int:
            """Select action using epsilon-greedy or noisy nets"""

            if training and not self.config.use_noisy_nets:
                # Epsilon-greedy exploration
                if random.random() < self.epsilon:
                    return random.randint(0, self.config.action_dim - 1)

            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Get Q-values
            with torch.no_grad():
                q_values = self.q_network(state_tensor)

            # Select best action
            return q_values.argmax(dim=-1).item()

        def store_transition(self, state, action, reward, next_state, done):
            """Store transition in replay buffer"""

            # Scale reward
            reward = reward * self.config.reward_scaling

            if self.config.prioritized_replay:
                self.replay_buffer.push(state, action, reward, next_state, done)
            else:
                self.replay_buffer.append((state, action, reward, next_state, done))

        def train_step(self) -> float:
            """Single training step"""

            if self.config.prioritized_replay:
                if len(self.replay_buffer) < self.config.batch_size:
                    return 0.0

                # Sample with priorities
                states, actions, rewards, next_states, dones, indices, weights = (
                    self.replay_buffer.sample(self.config.batch_size, self.beta)
                )

                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)
                weights = weights.to(self.device)

            else:
                if len(self.replay_buffer) < self.config.batch_size:
                    return 0.0

                # Random sampling
                batch = random.sample(self.replay_buffer, self.config.batch_size)
                states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
                actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
                rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
                next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
                dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
                weights = torch.ones_like(rewards)

            # Current Q-values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Next Q-values
            with torch.no_grad():
                if self.config.dqn_type == DQNType.DOUBLE:
                    # Double DQN: use online network to select action, target network to evaluate
                    next_actions = self.q_network(next_states).argmax(dim=-1)
                    next_q_values = (
                        self.target_network(next_states)
                        .gather(1, next_actions.unsqueeze(1))
                        .squeeze(1)
                    )
                else:
                    # Vanilla DQN
                    next_q_values = self.target_network(next_states).max(dim=-1)[0]

                target_q_values = rewards + self.config.discount_factor * next_q_values * (
                    1 - dones
                )

            # Compute loss
            td_errors = target_q_values - current_q_values

            if self.config.prioritized_replay:
                # Weighted loss for prioritized replay
                loss = (weights * td_errors.pow(2)).mean()

                # Update priorities
                priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
                self.replay_buffer.update_priorities(indices, priorities)
            else:
                loss = td_errors.pow(2).mean()

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip)

            self.optimizer.step()

            # Reset noise if using noisy nets
            if self.config.use_noisy_nets and hasattr(self.q_network, "reset_noise"):
                self.q_network.reset_noise()

            self.steps += 1

            # Update target network
            if self.steps % self.config.target_update_frequency == 0:
                self.update_target_network()

            return loss.item()

        def update_target_network(self):
            """Update target network weights"""

            if self.config.tau < 1.0:
                # Soft update
                for target_param, param in zip(
                    self.target_network.parameters(), self.q_network.parameters(), strict=False
                ):
                    target_param.data.copy_(
                        self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                    )
            else:
                # Hard update
                self.target_network.load_state_dict(self.q_network.state_dict())

        def update_exploration(self):
            """Update exploration parameters"""

            # Decay epsilon
            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

            # Anneal beta for prioritized replay
            if self.config.prioritized_replay:
                progress = min(1.0, self.episodes / 1000)
                self.beta = self.config.beta_start + progress * (
                    self.config.beta_end - self.config.beta_start
                )

        def train_episode(self, env) -> dict[str, float]:
            """Train for one episode"""

            state = env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0

            done = False
            while not done:
                # Select action
                action = self.select_action(state, training=True)

                # Step environment
                next_state, reward, done, _ = env.step(action)

                # Store transition
                self.store_transition(state, action, reward, next_state, done)

                # Train
                if self.steps % self.config.update_frequency == 0:
                    loss = self.train_step()
                    episode_loss += loss

                episode_reward += reward
                state = next_state
                steps += 1

            # Update exploration
            self.update_exploration()
            self.episodes += 1

            metrics = {
                "episode": self.episodes,
                "reward": episode_reward,
                "loss": episode_loss / max(1, steps // self.config.update_frequency),
                "epsilon": self.epsilon,
                "beta": self.beta if self.config.prioritized_replay else 0,
                "steps": steps,
            }

            self.training_history.append(metrics)

            return metrics

        def save(self, filepath: str):
            """Save agent"""
            save_dict = {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
                "epsilon": self.epsilon,
                "beta": self.beta,
                "steps": self.steps,
                "episodes": self.episodes,
                "training_history": self.training_history,
            }

            torch.save(save_dict, filepath)
            logger.info(f"DQN agent saved to {filepath}")

        def load(self, filepath: str):
            """Load agent"""
            checkpoint = torch.load(filepath, map_location=self.device)

            self.q_network.load_state_dict(checkpoint["q_network"])
            self.target_network.load_state_dict(checkpoint["target_network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.config = checkpoint["config"]
            self.epsilon = checkpoint["epsilon"]
            self.beta = checkpoint.get("beta", self.config.beta_start)
            self.steps = checkpoint["steps"]
            self.episodes = checkpoint["episodes"]
            self.training_history = checkpoint["training_history"]

            logger.info(f"DQN agent loaded from {filepath}")


def create_dqn_agent(
    state_dim: int, action_dim: int, dqn_type: DQNType = DQNType.DOUBLE
) -> DQNAgent:
    """Factory function to create DQN agent"""

    config = DQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        dqn_type=dqn_type,
        use_dueling=True,
        prioritized_replay=True,
    )

    return DQNAgent(config)


if __name__ == "__main__":
    print("Deep Q-Network (DQN) Implementation")

    if TORCH_AVAILABLE:
        # Test DQN
        config = DQNConfig(
            state_dim=50,
            action_dim=3,
            dqn_type=DQNType.DOUBLE,
            use_dueling=True,
            prioritized_replay=True,
        )

        agent = DQNAgent(config)

        print("\nDQN Configuration:")
        print(f"  Type: {config.dqn_type.value}")
        print(f"  Dueling: {config.use_dueling}")
        print(f"  Prioritized Replay: {config.prioritized_replay}")
        print(f"  Device: {agent.device}")

        # Test forward pass
        test_state = np.random.randn(config.state_dim)
        action = agent.select_action(test_state, training=False)
        print(f"\nTest action selection: {action}")

        # Test training step
        for i in range(100):
            state = np.random.randn(config.state_dim)
            action = np.random.randint(0, config.action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(config.state_dim)
            done = i == 99

            agent.store_transition(state, action, reward, next_state, done)

        loss = agent.train_step()
        print(f"Training loss: {loss:.4f}")
    else:
        print("PyTorch not available - DQN requires PyTorch")
