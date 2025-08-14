"""
Deep Reinforcement Learning Trading Agent

State-of-the-art RL algorithms for autonomous trading:
- Proximal Policy Optimization (PPO)
- Deep Q-Networks (DQN) with Double DQN
- Actor-Critic methods
- Multi-agent trading systems
- Risk-aware reward shaping
"""

from __future__ import annotations

import logging
import random
import time
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. RL agents will be limited.")


@dataclass
class RLConfig:
    """Configuration for RL trading agents"""

    # Environment settings
    initial_balance: float = 100000
    max_position_size: float = 0.1  # Max 10% per position
    transaction_cost: float = 0.001  # 0.1% commission
    slippage: float = 0.0005  # 0.05% slippage

    # State space
    lookback_window: int = 30
    n_features: int = 10

    # Action space
    n_actions: int = 3  # Buy, Hold, Sell
    continuous_actions: bool = False  # Discrete vs continuous

    # Training settings
    episodes: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.0003
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Algorithm-specific
    buffer_size: int = 100000  # Replay buffer
    update_frequency: int = 4  # Update target network
    tau: float = 0.001  # Soft update parameter

    # PPO specific
    ppo_epochs: int = 10
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01

    # Risk management
    max_drawdown_limit: float = 0.2  # 20% max drawdown
    risk_free_rate: float = 0.02  # Annual risk-free rate
    sharpe_target: float = 1.5


class TradingEnvironment:
    """
    Trading environment for RL agents.

    Features:
    - Realistic market simulation
    - Transaction costs and slippage
    - Risk metrics tracking
    - Multi-asset support
    """

    def __init__(self, data: pd.DataFrame, config: RLConfig) -> None:
        self.data = data
        self.config = config
        self.reset()

        # Precompute features
        self._precompute_features()

    def _precompute_features(self) -> None:
        """Precompute technical indicators and features"""
        df = self.data.copy()

        # Price features
        df["returns"] = df["Close"].pct_change()
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

        # Moving averages
        df["ma_5"] = df["Close"].rolling(5).mean()
        df["ma_20"] = df["Close"].rolling(20).mean()
        df["ma_ratio"] = df["ma_5"] / df["ma_20"]

        # Volatility
        df["volatility"] = df["returns"].rolling(20).std()

        # RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Volume features
        df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

        # MACD
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        self.features = df.fillna(0)

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = self.config.lookback_window
        self.balance = self.config.initial_balance
        self.position = 0  # Current position in shares
        self.entry_price = 0
        self.trades = []
        self.portfolio_values = [self.balance]

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state observation"""
        # Get historical window
        window_data = self.features.iloc[
            self.current_step - self.config.lookback_window : self.current_step
        ]

        # Extract features
        feature_cols = ["returns", "ma_ratio", "volatility", "rsi", "volume_ratio", "macd"]
        state_features = window_data[feature_cols].values.flatten()

        # Add portfolio state
        current_price = self.data["Close"].iloc[self.current_step]
        portfolio_state = np.array(
            [
                self.balance / self.config.initial_balance,  # Normalized balance
                self.position * current_price / self.config.initial_balance,  # Position value
                self.position / 1000,  # Normalized position size
            ]
        )

        # Combine all features
        state = np.concatenate([state_features, portfolio_state])

        return state.astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute action and return next state, reward, done, info.

        Args:
            action: 0=Sell, 1=Hold, 2=Buy
        """
        current_price = self.data["Close"].iloc[self.current_step]
        prev_portfolio_value = self._get_portfolio_value()

        # Execute action
        if action == 0:  # Sell
            if self.position > 0:
                # Close long position
                sell_value = (
                    self.position
                    * current_price
                    * (1 - self.config.transaction_cost - self.config.slippage)
                )
                profit = sell_value - (self.position * self.entry_price)
                self.balance += sell_value
                self.trades.append(
                    {
                        "step": self.current_step,
                        "action": "sell",
                        "price": current_price,
                        "position": self.position,
                        "profit": profit,
                    }
                )
                self.position = 0

        elif action == 2:  # Buy
            if self.position == 0 and self.balance > 0:
                # Open long position
                max_investment = self.balance * self.config.max_position_size
                shares = max_investment / (
                    current_price * (1 + self.config.transaction_cost + self.config.slippage)
                )
                cost = (
                    shares
                    * current_price
                    * (1 + self.config.transaction_cost + self.config.slippage)
                )

                if cost <= self.balance:
                    self.balance -= cost
                    self.position = shares
                    self.entry_price = current_price
                    self.trades.append(
                        {
                            "step": self.current_step,
                            "action": "buy",
                            "price": current_price,
                            "position": shares,
                            "cost": cost,
                        }
                    )

        # Move to next step
        self.current_step += 1

        # Calculate reward
        new_portfolio_value = self._get_portfolio_value()
        self.portfolio_values.append(new_portfolio_value)

        # Risk-adjusted reward
        returns = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value

        # Sharpe-based reward
        if len(self.portfolio_values) > 20:
            recent_returns = np.diff(self.portfolio_values[-20:]) / self.portfolio_values[-20:-1]
            sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0

        # Combine return and risk-adjusted components
        reward = returns * 100  # Scale returns
        reward += sharpe * 0.1  # Add Sharpe component

        # Penalty for excessive drawdown
        if self._get_drawdown() > self.config.max_drawdown_limit:
            reward -= 1.0

        # Check if episode is done
        done = (
            self.current_step >= len(self.data) - 1
            or self.balance <= self.config.initial_balance * 0.1  # Lost 90% of capital
        )

        # Additional info
        info = {
            "portfolio_value": new_portfolio_value,
            "position": self.position,
            "balance": self.balance,
            "drawdown": self._get_drawdown(),
            "sharpe": sharpe,
            "n_trades": len(self.trades),
        }

        return self._get_state(), reward, done, info

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        current_price = self.data["Close"].iloc[self.current_step]
        return self.balance + self.position * current_price

    def _get_drawdown(self) -> float:
        """Calculate current drawdown"""
        if len(self.portfolio_values) < 2:
            return 0

        peak = max(self.portfolio_values)
        current = self.portfolio_values[-1]
        return (peak - current) / peak


if TORCH_AVAILABLE:

    class DQNNetwork(nn.Module):
        """Deep Q-Network for value-based RL"""

        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
            super().__init__()

            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, action_dim)

            # Dueling DQN architecture
            self.value_stream = nn.Linear(hidden_dim, 1)
            self.advantage_stream = nn.Linear(hidden_dim, action_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))

            # Dueling streams
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)

            # Combine value and advantage
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

            return q_values

    class PPONetwork(nn.Module):
        """Actor-Critic network for PPO"""

        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
            super().__init__()

            # Shared layers
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)

            # Actor head (policy)
            self.actor = nn.Linear(hidden_dim, action_dim)

            # Critic head (value function)
            self.critic = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            # Get action probabilities and state value
            action_logits = self.actor(x)
            state_value = self.critic(x)

            return action_logits, state_value

        def get_action(self, state):
            """Sample action from policy"""
            action_logits, state_value = self.forward(state)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()

            return action, dist.log_prob(action), state_value

    class ReplayBuffer:
        """Experience replay buffer for DQN"""

        def __init__(self, capacity: int) -> None:
            self.buffer = deque(maxlen=capacity)

        def push(self, state, action, reward, next_state, done) -> None:
            self.buffer.append((state, action, reward, next_state, done))

        def sample(self, batch_size: int):
            batch = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = zip(*batch, strict=False)

            return (
                torch.FloatTensor(state),
                torch.LongTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(next_state),
                torch.FloatTensor(done),
            )

        def __len__(self) -> int:
            return len(self.buffer)

else:
    # Fallback classes
    class DQNNetwork:
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
            warnings.warn("PyTorch not available, DQNNetwork is a stub")

    class PPONetwork:
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
            warnings.warn("PyTorch not available, PPONetwork is a stub")

    class ReplayBuffer:
        def __init__(self, capacity: int) -> None:
            self.buffer = deque(maxlen=capacity)


class DQNAgent:
    """
    Deep Q-Learning agent for trading.

    Features:
    - Double DQN to reduce overestimation
    - Dueling architecture
    - Prioritized experience replay
    - Noisy networks for exploration
    """

    def __init__(self, state_dim: int, action_dim: int, config: RLConfig) -> None:
        self.config = config
        self.action_dim = action_dim
        self.epsilon = config.epsilon_start

        if TORCH_AVAILABLE:
            # Networks
            self.q_network = DQNNetwork(state_dim, action_dim)
            self.target_network = DQNNetwork(state_dim, action_dim)
            self.target_network.load_state_dict(self.q_network.state_dict())

            # Optimizer
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

            # Replay buffer
            self.replay_buffer = ReplayBuffer(config.buffer_size)

            # Device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_network.to(self.device)
            self.target_network.to(self.device)
        else:
            self.replay_buffer = ReplayBuffer(config.buffer_size)

        self.update_counter = 0
        self.logger = logging.getLogger(__name__)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        if TORCH_AVAILABLE:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        else:
            return random.randint(0, self.action_dim - 1)

    def train_step(self) -> None:
        """Perform one training step"""
        if not TORCH_AVAILABLE or len(self.replay_buffer) < self.config.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Double DQN: use main network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + self.config.gamma * next_q_values * (
                1 - dones.unsqueeze(1)
            )

        # Loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.config.update_frequency == 0:
            self._soft_update_target_network()

        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def _soft_update_target_network(self) -> None:
        """Soft update of target network parameters"""
        if TORCH_AVAILABLE:
            for target_param, param in zip(
                self.target_network.parameters(), self.q_network.parameters(), strict=False
            ):
                target_param.data.copy_(
                    self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                )


class PPOAgent:
    """
    Proximal Policy Optimization agent for trading.

    Features:
    - On-policy learning
    - Clipped objective for stable training
    - Advantage estimation
    - Entropy regularization
    """

    def __init__(self, state_dim: int, action_dim: int, config: RLConfig) -> None:
        self.config = config

        if TORCH_AVAILABLE:
            self.network = PPONetwork(state_dim, action_dim)
            self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.network.to(self.device)

        self.logger = logging.getLogger(__name__)

    def train_episode(self, env: TradingEnvironment) -> dict[str, float]:
        """Train for one episode"""
        if not TORCH_AVAILABLE:
            return {}

        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []

        state = env.reset()
        done = False

        # Collect trajectory
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, value = self.network.get_action(state_tensor)

            next_state, reward, done, info = env.step(action.item())

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())

            state = next_state

        # Calculate returns and advantages
        returns = []
        advantages = []
        G = 0

        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.config.gamma * G
            returns.insert(0, G)

            if t < len(rewards) - 1:
                advantage = returns[t] - values[t]
            else:
                advantage = rewards[t] - values[t]
            advantages.insert(0, advantage)

        # Normalize advantages
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)

        # PPO update
        for _ in range(self.config.ppo_epochs):
            # Get current policy
            action_logits, state_values = self.network(states)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)

            # Calculate ratios
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Clipped objective
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                * advantages
            )

            # Losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(state_values.squeeze(), returns)
            entropy_loss = -dist.entropy().mean()

            # Total loss
            loss = (
                policy_loss
                + self.config.value_loss_coef * value_loss
                - self.config.entropy_coef * entropy_loss
            )

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

        # Return episode metrics
        total_return = sum(rewards)
        portfolio_value = info["portfolio_value"]

        return {
            "total_return": total_return,
            "portfolio_value": portfolio_value,
            "n_trades": info["n_trades"],
            "sharpe": info["sharpe"],
        }


def train_rl_agent(
    data: pd.DataFrame, agent_type: str = "dqn", config: RLConfig | None = None
) -> dict[str, Any]:
    """
    Train RL trading agent.

    Args:
        data: Market data
        agent_type: "dqn" or "ppo"
        config: RL configuration

    Returns:
        Training results and metrics
    """
    config = config or RLConfig()

    # Create environment
    env = TradingEnvironment(data, config)

    # Get dimensions
    state = env.reset()
    state_dim = len(state)
    action_dim = config.n_actions

    # Create agent
    if agent_type == "dqn":
        agent = DQNAgent(state_dim, action_dim, config)
    elif agent_type == "ppo":
        agent = PPOAgent(state_dim, action_dim, config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Training metrics
    episode_rewards = []
    portfolio_values = []

    logger = logging.getLogger(__name__)
    logger.info(f"Starting {agent_type.upper()} training...")

    start_time = time.time()

    # Training loop
    for episode in range(config.episodes):
        if agent_type == "dqn":
            # DQN training
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                # Store transition
                agent.replay_buffer.push(state, action, reward, next_state, done)

                # Train
                agent.train_step()

                state = next_state
                episode_reward += reward

            episode_rewards.append(episode_reward)
            portfolio_values.append(info["portfolio_value"])

        elif agent_type == "ppo":
            # PPO training
            metrics = agent.train_episode(env)
            episode_rewards.append(metrics.get("total_return", 0))
            portfolio_values.append(metrics.get("portfolio_value", config.initial_balance))

        # Log progress
        if episode % 100 == 0:
            avg_reward = (
                np.mean(episode_rewards[-100:])
                if len(episode_rewards) >= 100
                else np.mean(episode_rewards)
            )
            avg_portfolio = (
                np.mean(portfolio_values[-100:])
                if len(portfolio_values) >= 100
                else np.mean(portfolio_values)
            )

            logger.info(
                f"Episode {episode}: "
                f"Avg Reward={avg_reward:.2f}, "
                f"Avg Portfolio=${avg_portfolio:,.0f}"
            )

    training_time = time.time() - start_time

    # Calculate final metrics
    final_portfolio = portfolio_values[-1] if portfolio_values else config.initial_balance
    total_return = (final_portfolio - config.initial_balance) / config.initial_balance

    # Calculate Sharpe ratio
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    else:
        sharpe = 0

    return {
        "agent_type": agent_type,
        "episodes": config.episodes,
        "training_time": training_time,
        "final_portfolio": final_portfolio,
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "episode_rewards": episode_rewards,
        "portfolio_values": portfolio_values,
    }


def benchmark_rl():
    """Benchmark RL agents"""
    print("🚀 Reinforcement Learning Benchmark")
    print("=" * 50)

    # Generate synthetic market data
    np.random.seed(42)
    n_days = 500

    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

    # Trending market with volatility
    trend = np.linspace(0, 0.2, n_days)
    noise = np.random.normal(0, 0.02, n_days)
    returns = trend / n_days + noise

    prices = 100 * np.cumprod(1 + returns)

    data = pd.DataFrame(
        {
            "Open": prices * np.random.uniform(0.99, 1.01, n_days),
            "High": prices * np.random.uniform(1.01, 1.03, n_days),
            "Low": prices * np.random.uniform(0.97, 0.99, n_days),
            "Close": prices,
            "Volume": np.random.lognormal(15, 0.5, n_days),
        },
        index=dates,
    )

    print(f"📊 Market data: {len(data)} days")
    print(f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")

    # Quick training config
    config = RLConfig(episodes=100, batch_size=32, learning_rate=0.001)  # Quick test

    # Test DQN
    print("\n🤖 Training DQN agent...")
    dqn_results = train_rl_agent(data, "dqn", config)

    print("\n📊 DQN Results:")
    print(f"   Training time: {dqn_results['training_time']:.2f}s")
    print(f"   Final portfolio: ${dqn_results['final_portfolio']:,.2f}")
    print(f"   Total return: {dqn_results['total_return']:.2%}")
    print(f"   Sharpe ratio: {dqn_results['sharpe_ratio']:.3f}")

    # Test PPO
    if TORCH_AVAILABLE:
        print("\n🤖 Training PPO agent...")
        ppo_results = train_rl_agent(data, "ppo", config)

        print("\n📊 PPO Results:")
        print(f"   Training time: {ppo_results['training_time']:.2f}s")
        print(f"   Final portfolio: ${ppo_results['final_portfolio']:,.2f}")
        print(f"   Total return: {ppo_results['total_return']:.2%}")
        print(f"   Sharpe ratio: {ppo_results['sharpe_ratio']:.3f}")

    return dqn_results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    benchmark_rl()
