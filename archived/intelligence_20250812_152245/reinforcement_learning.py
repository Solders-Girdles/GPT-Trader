"""
Reinforcement Learning Framework for GPT-Trader Phase 2.

This module provides sophisticated RL capabilities for trading:
- Deep Q-Networks (DQN) and variants for action selection
- Actor-Critic methods for continuous action spaces
- Policy gradient methods with experience replay
- Multi-agent RL for portfolio management
- Hierarchical RL for strategic decision making
- Custom trading environments with realistic constraints

Enables learning of optimal trading policies through market interaction.
"""

from __future__ import annotations

import logging
import random
import warnings
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Core RL components (can work without external RL libraries)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available. Install with: pip install torch")

# Optional RL libraries
try:
    import gym
    from gym import spaces

    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    warnings.warn("OpenAI Gym not available. Install with: pip install gymnasium")

try:
    from stable_baselines3 import A2C, PPO, SAC, TD3
    from stable_baselines3.common.env_checker import check_env

    HAS_STABLE_BASELINES = True
except ImportError:
    HAS_STABLE_BASELINES = False
    warnings.warn("Stable-Baselines3 not available. Install with: pip install stable-baselines3")

from bot.utils.base import BaseConfig

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


@dataclass
class RLConfig(BaseConfig):
    """Configuration for reinforcement learning framework."""

    # Environment parameters
    env_type: str = "trading"  # trading, portfolio, multi_asset
    max_episode_steps: int = 1000
    observation_window: int = 50

    # Action space parameters
    action_type: str = "discrete"  # discrete, continuous
    n_discrete_actions: int = 3  # buy, sell, hold
    action_bounds: tuple[float, float] = (-1.0, 1.0)  # For continuous actions

    # Reward function parameters
    reward_function: str = "profit"  # profit, sharpe, sortino, custom
    risk_penalty: float = 0.1
    transaction_cost: float = 0.001
    position_penalty: float = 0.05

    # Algorithm parameters
    algorithm: str = "dqn"  # dqn, ddqn, a2c, ppo, sac, td3
    learning_rate: float = 0.0003
    batch_size: int = 64
    buffer_size: int = 100000

    # Training parameters
    episodes: int = 1000
    max_steps_per_episode: int = 1000
    target_update_frequency: int = 1000

    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Network architecture
    hidden_layers: list[int] = field(default_factory=lambda: [256, 128])
    activation: str = "relu"
    dropout: float = 0.1

    # Training stability
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update coefficient
    gradient_clip: float = 1.0

    # Evaluation parameters
    eval_frequency: int = 100
    eval_episodes: int = 10

    # Random state
    random_state: int = 42


@dataclass
class TrainingResult:
    """Results from RL training session."""

    episode: int
    total_reward: float
    episode_length: int
    average_reward: float
    epsilon: float
    loss: float = 0.0
    portfolio_value: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0


class ReplayBuffer:
    """Experience replay buffer for off-policy RL algorithms."""

    def __init__(self, capacity: int, random_state: int = 42) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.random_state = random_state
        random.seed(random_state)

    def push(self, experience: Experience) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample batch of experiences."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class BaseRLEnvironment(ABC):
    """Base class for trading RL environments."""

    def __init__(self, data: pd.DataFrame, config: RLConfig) -> None:
        self.data = data
        self.config = config
        self.current_step = 0
        self.done = False

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        pass

    @abstractmethod
    def step(self, action: int | float | np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Execute action and return next state, reward, done, info."""
        pass

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        pass

    @abstractmethod
    def calculate_reward(self, action: int | float, portfolio_return: float) -> float:
        """Calculate reward for given action and return."""
        pass


class TradingEnvironment(BaseRLEnvironment):
    """
    Single-asset trading environment for RL agents.

    State: Price history, technical indicators, portfolio state
    Actions: Buy, Sell, Hold (discrete) or position size (continuous)
    Rewards: Profit, risk-adjusted returns, or custom metrics
    """

    def __init__(
        self, data: pd.DataFrame, config: RLConfig, initial_balance: float = 10000.0
    ) -> None:
        super().__init__(data, config)

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0.0  # Current position size
        self.portfolio_value = initial_balance

        # Performance tracking
        self.portfolio_history = []
        self.returns_history = []
        self.trades = []

        # Prepare features
        self.features = self._prepare_features(data)

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for RL environment."""
        features = pd.DataFrame(index=data.index)

        # Price features
        if "Close" in data.columns:
            features["close"] = data["Close"]
            features["returns"] = data["Close"].pct_change().fillna(0)
            features["log_returns"] = np.log(data["Close"] / data["Close"].shift(1)).fillna(0)

            # Moving averages
            for window in [5, 10, 20, 50]:
                features[f"ma_{window}"] = data["Close"].rolling(window).mean()
                features[f"ma_ratio_{window}"] = data["Close"] / features[f"ma_{window}"] - 1

            # Volatility
            features["volatility_10"] = features["returns"].rolling(10).std()
            features["volatility_20"] = features["returns"].rolling(20).std()

            # Momentum indicators
            features["momentum_5"] = data["Close"] / data["Close"].shift(5) - 1
            features["momentum_10"] = data["Close"] / data["Close"].shift(10) - 1

            # RSI
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-8)
            features["rsi"] = 100 - (100 / (1 + rs))

        # Volume features
        if "Volume" in data.columns:
            features["volume"] = data["Volume"]
            features["volume_ma"] = data["Volume"].rolling(20).mean()
            features["volume_ratio"] = data["Volume"] / (features["volume_ma"] + 1e-8)

        # Market state features
        if "Close" in data.columns:
            # Bollinger Bands
            bb_window = 20
            bb_ma = data["Close"].rolling(bb_window).mean()
            bb_std = data["Close"].rolling(bb_window).std()
            features["bb_upper"] = bb_ma + 2 * bb_std
            features["bb_lower"] = bb_ma - 2 * bb_std
            features["bb_position"] = (data["Close"] - bb_ma) / (2 * bb_std)

        # Fill missing values
        features = features.fillna(method="ffill").fillna(0)

        return features

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = self.config.observation_window
        self.done = False
        self.balance = self.initial_balance
        self.position = 0.0
        self.portfolio_value = self.initial_balance

        self.portfolio_history = [self.initial_balance]
        self.returns_history = []
        self.trades = []

        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        """Get current observation state."""
        if self.current_step < self.config.observation_window:
            # Pad with zeros if not enough history
            start_idx = 0
            padding = self.config.observation_window - self.current_step
        else:
            start_idx = self.current_step - self.config.observation_window
            padding = 0

        end_idx = self.current_step

        # Get feature window
        feature_window = self.features.iloc[start_idx:end_idx]

        # Normalize features
        feature_values = []
        for col in self.features.columns:
            if col == "close":
                # Use returns for price normalization
                if len(feature_window) > 1:
                    normalized = feature_window[col].pct_change().fillna(0).values
                else:
                    normalized = np.array([0.0])
            else:
                # Standardize other features
                values = feature_window[col].values
                if len(values) > 1 and np.std(values) > 0:
                    normalized = (values - np.mean(values)) / (np.std(values) + 1e-8)
                else:
                    normalized = values

            feature_values.extend(normalized)

        # Add padding if needed
        if padding > 0:
            feature_values = [0.0] * (padding * len(self.features.columns)) + feature_values

        # Add portfolio state
        portfolio_features = [
            self.position,  # Current position
            self.portfolio_value / self.initial_balance - 1,  # Portfolio return
            len(self.returns_history) / 252 if self.returns_history else 0,  # Time factor
        ]

        feature_values.extend(portfolio_features)

        return np.array(feature_values, dtype=np.float32)

    def step(self, action: int | float | np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Execute trading action."""
        if self.done or self.current_step >= len(self.features) - 1:
            return self.get_observation(), 0.0, True, {}

        # Get current price
        current_price = self.features["close"].iloc[self.current_step]

        # Execute action
        if self.config.action_type == "discrete":
            action = int(action)
            self._execute_discrete_action(action, current_price)
        else:
            self._execute_continuous_action(float(action), current_price)

        # Move to next step
        self.current_step += 1
        next_price = (
            self.features["close"].iloc[self.current_step]
            if self.current_step < len(self.features)
            else current_price
        )

        # Calculate portfolio return
        price_return = (next_price - current_price) / current_price
        portfolio_return = self.position * price_return

        # Update portfolio value
        self.portfolio_value *= 1 + portfolio_return
        self.returns_history.append(portfolio_return)
        self.portfolio_history.append(self.portfolio_value)

        # Calculate reward
        reward = self.calculate_reward(action, portfolio_return)

        # Check if done
        self.done = (
            self.current_step >= len(self.features) - 1
            or self.portfolio_value <= self.initial_balance * 0.1  # Stop loss
            or self.current_step - self.config.observation_window >= self.config.max_episode_steps
        )

        # Info dict
        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "current_price": next_price,
            "return": portfolio_return,
            "total_return": self.portfolio_value / self.initial_balance - 1,
        }

        return self.get_observation(), reward, self.done, info

    def _execute_discrete_action(self, action: int, current_price: float) -> float:
        """Execute discrete action (0: sell, 1: hold, 2: buy)."""
        position_change = 0.0

        if action == 0:  # Sell
            if self.position > 0:
                position_change = -min(self.position, 0.5)  # Sell up to 50%
        elif action == 2:  # Buy
            max_buy = (self.balance / current_price) * 0.5  # Buy up to 50% of available
            position_change = max_buy
        # action == 1 is hold (no change)

        # Apply transaction costs
        if abs(position_change) > 0:
            transaction_cost = abs(position_change * current_price * self.config.transaction_cost)
            self.balance -= transaction_cost

            # Record trade
            self.trades.append(
                {
                    "step": self.current_step,
                    "action": action,
                    "position_change": position_change,
                    "price": current_price,
                    "cost": transaction_cost,
                }
            )

        # Update position and balance
        self.position += position_change
        self.balance -= position_change * current_price

        return self.position

    def _execute_continuous_action(self, action: float, current_price: float) -> float:
        """Execute continuous action (position size between -1 and 1)."""
        # Clip action to bounds
        action = np.clip(action, self.config.action_bounds[0], self.config.action_bounds[1])

        # Calculate target position
        max_position = self.portfolio_value / current_price
        target_position = action * max_position

        # Position change
        position_change = target_position - self.position

        # Apply transaction costs
        if abs(position_change) > 0:
            transaction_cost = abs(position_change * current_price * self.config.transaction_cost)
            self.balance -= transaction_cost

            self.trades.append(
                {
                    "step": self.current_step,
                    "action": action,
                    "position_change": position_change,
                    "price": current_price,
                    "cost": transaction_cost,
                }
            )

        # Update position and balance
        self.position = target_position
        # Balance is adjusted by portfolio value changes, not individual trades in continuous case

        return self.position

    def calculate_reward(self, action: int | float, portfolio_return: float) -> float:
        """Calculate reward based on configuration."""
        base_reward = 0.0

        if self.config.reward_function == "profit":
            # Simple profit-based reward
            base_reward = portfolio_return * 100  # Scale up

        elif self.config.reward_function == "sharpe":
            # Sharpe ratio based reward
            if len(self.returns_history) >= 10:
                returns = np.array(self.returns_history[-10:])
                if np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                    base_reward = sharpe

        elif self.config.reward_function == "sortino":
            # Sortino ratio based reward
            if len(self.returns_history) >= 10:
                returns = np.array(self.returns_history[-10:])
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                    sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
                    base_reward = sortino
                else:
                    base_reward = portfolio_return * 100

        # Risk penalties
        risk_penalty = 0.0

        # Volatility penalty
        if len(self.returns_history) >= 5:
            recent_vol = np.std(self.returns_history[-5:])
            risk_penalty += recent_vol * self.config.risk_penalty

        # Position penalty (discourage extreme positions)
        position_penalty = abs(self.position) * self.config.position_penalty

        # Total reward
        total_reward = base_reward - risk_penalty - position_penalty

        return total_reward

    def get_portfolio_metrics(self) -> dict[str, float]:
        """Calculate portfolio performance metrics."""
        if len(self.returns_history) < 2:
            return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}

        returns = np.array(self.returns_history)
        portfolio_values = np.array(self.portfolio_history)

        # Total return
        total_return = self.portfolio_value / self.initial_balance - 1

        # Sharpe ratio
        if np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)

        # Win rate
        winning_trades = np.sum(returns > 0)
        win_rate = winning_trades / len(returns) if len(returns) > 0 else 0.0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": abs(max_drawdown),
            "win_rate": win_rate,
            "n_trades": len(self.trades),
            "portfolio_value": self.portfolio_value,
        }


class DQNNetwork(nn.Module):
    """Deep Q-Network for discrete action spaces."""

    def __init__(
        self, state_size: int, action_size: int, hidden_layers: list[int], dropout: float = 0.1
    ) -> None:
        super().__init__()

        # Build network layers
        layers = []
        input_size = state_size

        for hidden_size in hidden_layers:
            layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for continuous action spaces."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: list[int],
        action_bounds: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()

        self.action_bounds = action_bounds

        # Shared layers
        shared_layers = []
        input_size = state_size

        for hidden_size in hidden_layers[:-1]:
            shared_layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU()])
            input_size = hidden_size

        self.shared = nn.Sequential(*shared_layers)

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], action_size),
            nn.Tanh(),  # Output between -1 and 1
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_layers[-1]), nn.ReLU(), nn.Linear(hidden_layers[-1], 1)
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared(state)

        # Actor output (actions)
        actions = self.actor(shared_features)

        # Scale actions to bounds
        action_range = self.action_bounds[1] - self.action_bounds[0]
        actions = actions * action_range / 2 + (self.action_bounds[1] + self.action_bounds[0]) / 2

        # Critic output (value)
        value = self.critic(shared_features)

        return actions, value


class DQNAgent:
    """Deep Q-Network agent for discrete action trading."""

    def __init__(self, state_size: int, action_size: int, config: RLConfig) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        if not HAS_TORCH:
            raise ImportError("PyTorch is required for DQN agent")

        # Networks
        self.q_network = DQNNetwork(state_size, action_size, config.hidden_layers, config.dropout)
        self.target_network = DQNNetwork(
            state_size, action_size, config.hidden_layers, config.dropout
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

        # Initialize target network
        self.update_target_network()

        # Experience replay
        self.replay_buffer = ReplayBuffer(config.buffer_size, config.random_state)

        # Training state
        self.epsilon = config.epsilon_start
        self.training_step = 0

    def update_target_network(self) -> None:
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_experience(self, experience: Experience) -> None:
        """Store experience in replay buffer."""
        self.replay_buffer.push(experience)

    def train_step(self) -> float:
        """Perform one training step."""
        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0

        # Sample batch
        experiences = self.replay_buffer.sample(self.config.batch_size)
        batch = Experience(*zip(*experiences, strict=False))

        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state)
        action_batch = torch.LongTensor(batch.action)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(batch.next_state)
        done_batch = torch.BoolTensor(batch.done)

        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (self.config.gamma * next_q_values * ~done_batch)

        # Loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

        # Update target network
        self.training_step += 1
        if self.training_step % self.config.target_update_frequency == 0:
            self.update_target_network()

        return loss.item()


class A2CAgent:
    """Actor-Critic agent for continuous action trading."""

    def __init__(self, state_size: int, action_size: int, config: RLConfig) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        if not HAS_TORCH:
            raise ImportError("PyTorch is required for A2C agent")

        # Network
        self.network = ActorCriticNetwork(
            state_size, action_size, config.hidden_layers, config.action_bounds
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)

        # Training state
        self.training_step = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> tuple[float, float]:
        """Select action and return action + log probability."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action, value = self.network(state_tensor)

        if training:
            # Add noise for exploration
            noise = torch.randn_like(action) * 0.1
            action = torch.clamp(
                action + noise, self.config.action_bounds[0], self.config.action_bounds[1]
            )

        return action.squeeze().numpy(), value.squeeze().item()

    def train_step(self, experiences: list[tuple]) -> float:
        """Train using collected experiences."""
        if not experiences:
            return 0.0

        # Unpack experiences
        states, actions, rewards, next_states, dones, values = zip(*experiences, strict=False)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        old_values = torch.FloatTensor(values)

        # Calculate returns and advantages
        with torch.no_grad():
            _, next_values = self.network(next_states)
            returns = []
            advantages = []

            R = next_values[-1] * (1 - dones[-1])
            for i in reversed(range(len(rewards))):
                R = rewards[i] + self.config.gamma * R * (1 - dones[i])
                returns.insert(0, R)
                advantage = R - old_values[i]
                advantages.insert(0, advantage)

            returns = torch.FloatTensor(returns)
            advantages = torch.FloatTensor(advantages)

        # Forward pass
        pred_actions, pred_values = self.network(states)

        # Actor loss (policy gradient)
        action_errors = (actions - pred_actions) ** 2
        actor_loss = torch.mean(action_errors * advantages.unsqueeze(1))

        # Critic loss
        critic_loss = F.mse_loss(pred_values.squeeze(), returns)

        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        self.training_step += 1

        return total_loss.item()


class RLTrainer:
    """
    Reinforcement Learning trainer for trading agents.

    Handles training loop, evaluation, and performance tracking.
    """

    def __init__(self, agent, environment: BaseRLEnvironment, config: RLConfig) -> None:
        self.agent = agent
        self.environment = environment
        self.config = config

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.training_results = []
        self.best_performance = -np.inf

    def train(self, verbose: bool = True) -> list[TrainingResult]:
        """Train the RL agent."""
        logger.info(f"Starting RL training for {self.config.episodes} episodes...")

        for episode in range(self.config.episodes):
            result = self._train_episode()
            self.training_results.append(result)

            if verbose and episode % 50 == 0:
                logger.info(
                    f"Episode {episode}: Reward={result.total_reward:.4f}, "
                    f"Portfolio={result.portfolio_value:.2f}, "
                    f"Epsilon={result.epsilon:.4f}"
                )

            # Evaluation
            if episode % self.config.eval_frequency == 0:
                eval_result = self._evaluate_agent()

                if eval_result["avg_reward"] > self.best_performance:
                    self.best_performance = eval_result["avg_reward"]
                    logger.info(f"New best performance: {self.best_performance:.4f}")

        logger.info("RL training completed")
        return self.training_results

    def _train_episode(self) -> TrainingResult:
        """Train single episode."""
        state = self.environment.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_loss = 0.0

        # For A2C, collect experiences for batch update
        if hasattr(self.agent, "network"):  # A2C agent
            experiences = []

        while not self.environment.done and episode_steps < self.config.max_steps_per_episode:
            # Select action
            if hasattr(self.agent, "select_action"):
                if hasattr(self.agent, "network"):  # A2C
                    action, value = self.agent.select_action(state)
                else:  # DQN
                    action = self.agent.select_action(state)
                    value = 0.0
            else:
                action = 0  # Fallback
                value = 0.0

            # Execute action
            next_state, reward, done, info = self.environment.step(action)

            # Store experience
            if hasattr(self.agent, "store_experience"):  # DQN
                experience = Experience(state, action, reward, next_state, done)
                self.agent.store_experience(experience)

                # Train step
                loss = self.agent.train_step()
                episode_loss += loss

            elif hasattr(self.agent, "network"):  # A2C
                experiences.append((state, action, reward, next_state, done, value))

            episode_reward += reward
            episode_steps += 1
            state = next_state

        # A2C batch training
        if hasattr(self.agent, "network") and experiences:
            episode_loss = self.agent.train_step(experiences)

        # Get portfolio metrics
        metrics = self.environment.get_portfolio_metrics()

        self.episode += 1
        self.total_steps += episode_steps

        return TrainingResult(
            episode=self.episode,
            total_reward=episode_reward,
            episode_length=episode_steps,
            average_reward=episode_reward / episode_steps if episode_steps > 0 else 0,
            epsilon=getattr(self.agent, "epsilon", 0.0),
            loss=episode_loss,
            portfolio_value=metrics.get("portfolio_value", 0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0),
            max_drawdown=metrics.get("max_drawdown", 0),
        )

    def _evaluate_agent(self) -> dict[str, float]:
        """Evaluate trained agent."""
        eval_rewards = []
        eval_portfolios = []

        # Save current epsilon for DQN
        original_epsilon = getattr(self.agent, "epsilon", 0.0)
        if hasattr(self.agent, "epsilon"):
            self.agent.epsilon = 0.0  # No exploration during evaluation

        for _ in range(self.config.eval_episodes):
            state = self.environment.reset()
            episode_reward = 0.0

            while not self.environment.done:
                if hasattr(self.agent, "select_action"):
                    if hasattr(self.agent, "network"):  # A2C
                        action, _ = self.agent.select_action(state, training=False)
                    else:  # DQN
                        action = self.agent.select_action(state, training=False)
                else:
                    action = 0

                state, reward, done, info = self.environment.step(action)
                episode_reward += reward

            eval_rewards.append(episode_reward)

            metrics = self.environment.get_portfolio_metrics()
            eval_portfolios.append(metrics.get("portfolio_value", 0))

        # Restore epsilon
        if hasattr(self.agent, "epsilon"):
            self.agent.epsilon = original_epsilon

        return {
            "avg_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "avg_portfolio": np.mean(eval_portfolios),
            "best_portfolio": np.max(eval_portfolios),
        }


def create_rl_trading_system(
    data: pd.DataFrame, config: RLConfig | None = None
) -> tuple[RLTrainer, TradingEnvironment]:
    """Create complete RL trading system."""

    # Default configuration
    if config is None:
        config = RLConfig(
            algorithm="dqn",
            action_type="discrete",
            episodes=500,
            learning_rate=0.0003,
            batch_size=32,
            buffer_size=50000,
        )

    # Create environment
    environment = TradingEnvironment(data, config)

    # Get state and action dimensions
    state = environment.reset()
    state_size = len(state)

    if config.action_type == "discrete":
        action_size = config.n_discrete_actions
    else:
        action_size = 1  # Single continuous action (position size)

    # Create agent
    if not HAS_TORCH:
        logger.warning("PyTorch not available. Cannot create neural network agents.")
        return None, environment

    if config.algorithm == "dqn" and config.action_type == "discrete":
        agent = DQNAgent(state_size, action_size, config)
    elif config.algorithm == "a2c" and config.action_type == "continuous":
        agent = A2CAgent(state_size, action_size, config)
    else:
        logger.warning(f"Unsupported combination: {config.algorithm} with {config.action_type}")
        agent = (
            DQNAgent(state_size, action_size, config)
            if config.action_type == "discrete"
            else A2CAgent(state_size, action_size, config)
        )

    # Create trainer
    trainer = RLTrainer(agent, environment, config)

    return trainer, environment


# Example usage
def train_rl_trading_strategy(data: pd.DataFrame, episodes: int = 1000) -> dict[str, Any]:
    """Train RL trading strategy on provided data."""

    config = RLConfig(
        algorithm="dqn",
        action_type="discrete",
        episodes=episodes,
        learning_rate=0.0003,
        batch_size=32,
        epsilon_decay=0.995,
        reward_function="profit",
    )

    if not HAS_TORCH:
        logger.warning("PyTorch not available. Returning mock results.")
        return {
            "training_completed": False,
            "reason": "PyTorch not available",
            "episodes": 0,
            "final_performance": 0.0,
        }

    trainer, environment = create_rl_trading_system(data, config)

    if trainer is None:
        return {
            "training_completed": False,
            "reason": "Could not create trainer",
            "episodes": 0,
            "final_performance": 0.0,
        }

    # Train agent
    training_results = trainer.train(verbose=True)

    # Final evaluation
    final_eval = trainer._evaluate_agent()

    return {
        "training_completed": True,
        "episodes": len(training_results),
        "final_performance": final_eval["avg_portfolio"],
        "best_performance": trainer.best_performance,
        "training_results": training_results[-10:],  # Last 10 episodes
        "final_metrics": final_eval,
    }
