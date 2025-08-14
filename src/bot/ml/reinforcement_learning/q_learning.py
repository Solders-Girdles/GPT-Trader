"""
RL-001: Q-Learning Implementation
Phase 4 - Week 3

Q-Learning for trading decisions:
- State representation (price, volume, indicators)
- Action space (buy, sell, hold)
- Reward shaping for risk-adjusted returns
- Epsilon-greedy exploration
- Experience replay buffer
"""

import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from ...utils.serialization import (
    load_json,
    save_json,
)

logger = logging.getLogger(__name__)


class TradingAction(Enum):
    """Trading actions"""

    HOLD = 0
    BUY = 1
    SELL = 2

    @classmethod
    def from_index(cls, index: int):
        return list(cls)[index]


class StateRepresentation(Enum):
    """Types of state representations"""

    RAW_PRICES = "raw_prices"
    TECHNICAL_INDICATORS = "technical_indicators"
    NORMALIZED_FEATURES = "normalized_features"
    MARKET_MICROSTRUCTURE = "market_microstructure"


@dataclass
class QLearningConfig:
    """Configuration for Q-Learning"""

    # Environment
    state_size: int = 50  # Number of features in state
    action_size: int = 3  # Buy, Sell, Hold
    lookback_window: int = 30  # Historical window for state

    # Q-Learning parameters
    learning_rate: float = 0.001
    discount_factor: float = 0.95  # Gamma
    epsilon_start: float = 1.0  # Initial exploration rate
    epsilon_end: float = 0.01  # Final exploration rate
    epsilon_decay: float = 0.995  # Decay rate

    # Experience replay
    replay_buffer_size: int = 10000
    batch_size: int = 32
    update_frequency: int = 4  # Update Q-values every N steps

    # Reward shaping
    profit_weight: float = 1.0
    risk_penalty: float = 0.1
    transaction_cost: float = 0.001  # 0.1% per trade
    holding_penalty: float = 0.0001  # Encourage action

    # Training
    episodes: int = 1000
    max_steps_per_episode: int = 1000
    target_update_frequency: int = 100  # For DQN

    # Performance
    min_performance_threshold: float = 0.1  # 10% return threshold
    save_frequency: int = 100  # Save model every N episodes


class StateEncoder:
    """Encode market state for Q-Learning"""

    def __init__(self, config: QLearningConfig):
        self.config = config
        self.feature_names = []
        self.normalization_params = {}

    def encode_state(self, market_data: dict[str, np.ndarray], position: float = 0.0) -> np.ndarray:
        """
        Encode current market state

        Args:
            market_data: Dictionary with price, volume, indicators
            position: Current position (-1 short, 0 flat, 1 long)

        Returns:
            State vector
        """
        features = []

        # Price features
        if "price" in market_data:
            prices = market_data["price"]
            features.extend(
                [
                    self._normalize(prices[-1], "price"),  # Current price
                    self._normalize(np.mean(prices[-5:]), "ma5"),  # MA5
                    self._normalize(np.mean(prices[-20:]), "ma20"),  # MA20
                    self._calculate_returns(prices, [1, 5, 20]),  # Returns
                ]
            )

        # Volume features
        if "volume" in market_data:
            volumes = market_data["volume"]
            features.extend(
                [
                    self._normalize(volumes[-1], "volume"),
                    self._normalize(np.mean(volumes[-5:]), "avg_volume"),
                    self._calculate_volume_ratio(volumes),
                ]
            )

        # Technical indicators
        if "rsi" in market_data:
            features.append(market_data["rsi"][-1] / 100.0)  # Normalize RSI

        if "macd" in market_data:
            features.append(self._normalize(market_data["macd"][-1], "macd"))

        # Market microstructure
        if "bid_ask_spread" in market_data:
            features.append(self._normalize(market_data["bid_ask_spread"][-1], "spread"))

        # Position encoding
        features.append(position)

        # Pad or truncate to state_size
        state = np.array(features).flatten()
        if len(state) < self.config.state_size:
            state = np.pad(state, (0, self.config.state_size - len(state)))
        else:
            state = state[: self.config.state_size]

        return state

    def _normalize(self, value: float, feature_name: str) -> float:
        """Normalize feature value"""
        if feature_name not in self.normalization_params:
            # Initialize with first value
            self.normalization_params[feature_name] = {"mean": value, "std": 1.0}

        params = self.normalization_params[feature_name]
        normalized = (value - params["mean"]) / (params["std"] + 1e-8)

        # Update running statistics
        alpha = 0.01
        params["mean"] = (1 - alpha) * params["mean"] + alpha * value
        params["std"] = (1 - alpha) * params["std"] + alpha * abs(value - params["mean"])

        return np.clip(normalized, -3, 3)  # Clip to prevent extreme values

    def _calculate_returns(self, prices: np.ndarray, periods: list[int]) -> np.ndarray:
        """Calculate returns over different periods"""
        returns = []
        for period in periods:
            if len(prices) > period:
                ret = (prices[-1] - prices[-period - 1]) / prices[-period - 1]
                returns.append(ret)
            else:
                returns.append(0.0)
        return np.array(returns)

    def _calculate_volume_ratio(self, volumes: np.ndarray) -> float:
        """Calculate volume ratio (current vs average)"""
        if len(volumes) > 20:
            return volumes[-1] / (np.mean(volumes[-20:]) + 1e-8)
        return 1.0


class RewardCalculator:
    """Calculate rewards for trading actions"""

    def __init__(self, config: QLearningConfig):
        self.config = config
        self.last_portfolio_value = None

    def calculate_reward(
        self,
        action: TradingAction,
        current_price: float,
        previous_price: float,
        position: float,
        portfolio_value: float,
    ) -> float:
        """
        Calculate reward for action taken

        Args:
            action: Trading action taken
            current_price: Current market price
            previous_price: Previous market price
            position: Current position
            portfolio_value: Current portfolio value

        Returns:
            Reward value
        """
        reward = 0.0
        price_change = (current_price - previous_price) / previous_price

        # Profit/Loss component
        if position != 0:
            pnl = position * price_change
            reward += self.config.profit_weight * pnl

        # Transaction cost penalty
        if action in [TradingAction.BUY, TradingAction.SELL]:
            reward -= self.config.transaction_cost

        # Risk penalty (volatility-based)
        if self.last_portfolio_value is not None:
            portfolio_change = (
                abs(portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
            )
            reward -= self.config.risk_penalty * portfolio_change

        # Holding penalty (encourage action)
        if action == TradingAction.HOLD and position == 0:
            reward -= self.config.holding_penalty

        # Sharpe ratio component (risk-adjusted returns)
        if hasattr(self, "return_history"):
            if len(self.return_history) > 20:
                sharpe = np.mean(self.return_history) / (np.std(self.return_history) + 1e-8)
                reward += 0.01 * sharpe

        self.last_portfolio_value = portfolio_value

        return reward

    def calculate_episode_metrics(
        self, rewards: list[float], actions: list[TradingAction], prices: list[float]
    ) -> dict[str, float]:
        """Calculate episode performance metrics"""

        total_return = (prices[-1] - prices[0]) / prices[0] if prices else 0
        total_reward = sum(rewards)

        # Calculate Sharpe ratio
        if len(rewards) > 1:
            returns = np.diff(prices) / prices[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0

        # Calculate win rate
        trades = [a for a in actions if a != TradingAction.HOLD]
        if trades:
            # Simplified win rate calculation
            win_rate = sum(1 for r in rewards if r > 0) / len(rewards)
        else:
            win_rate = 0

        return {
            "total_return": total_return,
            "total_reward": total_reward,
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "num_trades": len(trades),
        }


class ExperienceReplayBuffer:
    """Experience replay buffer for Q-Learning"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class QTable:
    """Q-Table for discrete state-action values"""

    def __init__(self, config: QLearningConfig):
        self.config = config
        self.q_values = defaultdict(lambda: np.zeros(config.action_size))
        self.visit_counts = defaultdict(lambda: np.zeros(config.action_size))

    def get_q_value(self, state: np.ndarray, action: int) -> float:
        """Get Q-value for state-action pair"""
        state_key = self._discretize_state(state)
        return self.q_values[state_key][action]

    def update_q_value(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ):
        """Update Q-value using Q-learning update rule"""
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)

        # Current Q-value
        current_q = self.q_values[state_key][action]

        # Maximum Q-value for next state
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_values[next_state_key])
            target_q = reward + self.config.discount_factor * max_next_q

        # Q-learning update
        self.q_values[state_key][action] = current_q + self.config.learning_rate * (
            target_q - current_q
        )

        # Update visit count
        self.visit_counts[state_key][action] += 1

    def get_best_action(self, state: np.ndarray) -> int:
        """Get best action for state (exploitation)"""
        state_key = self._discretize_state(state)
        return np.argmax(self.q_values[state_key])

    def _discretize_state(self, state: np.ndarray) -> tuple:
        """Discretize continuous state for Q-table lookup"""
        # Simple binning approach
        bins = 10
        discretized = np.floor(state * bins).astype(int)
        return tuple(discretized)

    def save(self, filepath: str):
        """Save Q-table to file using secure JSON format"""
        data = {
            "q_values": {str(k): v.tolist() for k, v in self.q_values.items()},
            "visit_counts": {str(k): v.tolist() for k, v in self.visit_counts.items()},
        }
        save_json(data, filepath)

    def load(self, filepath: str):
        """Load Q-table from file using secure JSON format"""
        data = load_json(filepath)
        # Convert string keys back to tuples and lists back to numpy arrays
        q_values = {eval(k): np.array(v) for k, v in data["q_values"].items()}
        visit_counts = {eval(k): np.array(v) for k, v in data["visit_counts"].items()}
        self.q_values = defaultdict(lambda: np.zeros(self.config.action_size), q_values)
        self.visit_counts = defaultdict(lambda: np.zeros(self.config.action_size), visit_counts)


class QLearningAgent:
    """Q-Learning agent for trading"""

    def __init__(self, config: QLearningConfig):
        self.config = config
        self.q_table = QTable(config)
        self.state_encoder = StateEncoder(config)
        self.reward_calculator = RewardCalculator(config)
        self.replay_buffer = ExperienceReplayBuffer(config.replay_buffer_size)

        self.epsilon = config.epsilon_start
        self.episode = 0
        self.training_history = []

    def select_action(self, state: np.ndarray, training: bool = True) -> TradingAction:
        """Select action using epsilon-greedy policy"""

        if training and random.random() < self.epsilon:
            # Exploration
            action_idx = random.randint(0, self.config.action_size - 1)
        else:
            # Exploitation
            action_idx = self.q_table.get_best_action(state)

        return TradingAction.from_index(action_idx)

    def train_step(
        self,
        state: np.ndarray,
        action: TradingAction,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Single training step"""

        # Store experience
        self.replay_buffer.push(state, action.value, reward, next_state, done)

        # Update Q-values
        if len(self.replay_buffer) >= self.config.batch_size:
            self._replay_experiences()

        # Direct Q-table update
        self.q_table.update_q_value(state, action.value, reward, next_state, done)

        # Decay epsilon
        if done:
            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def _replay_experiences(self):
        """Learn from replay buffer"""
        batch = self.replay_buffer.sample(self.config.batch_size)

        for state, action, reward, next_state, done in batch:
            self.q_table.update_q_value(state, action, reward, next_state, done)

    def train_episode(self, env) -> dict[str, float]:
        """Train for one episode"""

        state = env.reset()
        state = self.state_encoder.encode_state(env.get_market_data(), env.position)

        episode_rewards = []
        episode_actions = []
        episode_prices = []

        for step in range(self.config.max_steps_per_episode):
            # Select action
            action = self.select_action(state, training=True)
            episode_actions.append(action)

            # Take action in environment
            next_market_data, reward, done = env.step(action)
            episode_rewards.append(reward)
            episode_prices.append(env.current_price)

            # Encode next state
            next_state = self.state_encoder.encode_state(next_market_data, env.position)

            # Train
            self.train_step(state, action, reward, next_state, done)

            state = next_state

            if done:
                break

        # Calculate episode metrics
        metrics = self.reward_calculator.calculate_episode_metrics(
            episode_rewards, episode_actions, episode_prices
        )
        metrics["epsilon"] = self.epsilon
        metrics["episode"] = self.episode

        self.episode += 1
        self.training_history.append(metrics)

        return metrics

    def evaluate(self, env, episodes: int = 10) -> dict[str, float]:
        """Evaluate agent performance"""

        total_returns = []
        total_rewards = []

        for _ in range(episodes):
            state = env.reset()
            state = self.state_encoder.encode_state(env.get_market_data(), env.position)

            episode_reward = 0
            initial_price = env.current_price

            done = False
            while not done:
                # Select action (no exploration)
                action = self.select_action(state, training=False)

                # Take action
                next_market_data, reward, done = env.step(action)
                episode_reward += reward

                # Next state
                state = self.state_encoder.encode_state(next_market_data, env.position)

            final_price = env.current_price
            episode_return = (final_price - initial_price) / initial_price

            total_returns.append(episode_return)
            total_rewards.append(episode_reward)

        return {
            "avg_return": np.mean(total_returns),
            "std_return": np.std(total_returns),
            "avg_reward": np.mean(total_rewards),
            "sharpe_ratio": np.mean(total_returns) / (np.std(total_returns) + 1e-8) * np.sqrt(252),
        }

    def save(self, filepath: str):
        """Save agent"""
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.q_table.save(str(save_path / "q_table.json"))

        # Save config as JSON
        config_data = self.config.__dict__.copy()
        # Convert any non-serializable items
        for key, value in config_data.items():
            if hasattr(value, "__dict__"):
                config_data[key] = value.__dict__
        save_json(config_data, save_path / "config.json")

        # Save training history as JSON
        save_json(self.training_history, save_path / "history.json")

        logger.info(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent"""
        load_path = Path(filepath)

        self.q_table.load(str(load_path / "q_table.json"))

        # Load config from JSON
        config_data = load_json(load_path / "config.json")
        self.config = QLearningConfig(**config_data)

        # Load training history from JSON
        self.training_history = load_json(load_path / "history.json")

        logger.info(f"Agent loaded from {filepath}")


# Simple trading environment for testing
class TradingEnvironment:
    """Simple trading environment"""

    def __init__(self, price_data: np.ndarray):
        self.price_data = price_data
        self.reset()

    def reset(self):
        """Reset environment"""
        self.current_step = 30  # Start after lookback window
        self.position = 0  # 0: no position, 1: long, -1: short
        self.cash = 10000
        self.shares = 0
        self.current_price = self.price_data[self.current_step]
        return self.get_market_data()

    def step(self, action: TradingAction) -> tuple[dict, float, bool]:
        """Take action in environment"""

        # Execute action
        if action == TradingAction.BUY and self.position <= 0:
            self.shares = self.cash / self.current_price
            self.cash = 0
            self.position = 1
        elif action == TradingAction.SELL and self.position >= 0:
            self.cash = self.shares * self.current_price
            self.shares = 0
            self.position = -1

        # Move to next step
        self.current_step += 1
        previous_price = self.current_price
        self.current_price = self.price_data[self.current_step]

        # Calculate reward
        portfolio_value = self.cash + self.shares * self.current_price
        reward = (portfolio_value - 10000) / 10000  # Simplified reward

        # Check if done
        done = self.current_step >= len(self.price_data) - 1

        return self.get_market_data(), reward, done

    def get_market_data(self) -> dict[str, np.ndarray]:
        """Get current market data"""
        lookback = 30
        start_idx = max(0, self.current_step - lookback)

        return {
            "price": self.price_data[start_idx : self.current_step + 1],
            "volume": np.random.random(lookback + 1) * 1000000,  # Mock volume
            "rsi": np.random.random(lookback + 1) * 100,  # Mock RSI
        }


if __name__ == "__main__":
    # Test Q-Learning implementation
    config = QLearningConfig(state_size=50, action_size=3, episodes=100, learning_rate=0.001)

    # Create agent
    agent = QLearningAgent(config)

    # Create mock price data
    price_data = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    env = TradingEnvironment(price_data)

    print("Q-Learning Agent Configuration:")
    print(f"  State Size: {config.state_size}")
    print(f"  Action Size: {config.action_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Discount Factor: {config.discount_factor}")

    # Train for a few episodes
    print("\nTraining Q-Learning Agent...")
    for episode in range(10):
        metrics = agent.train_episode(env)
        if episode % 5 == 0:
            print(
                f"Episode {episode}: Return={metrics['total_return']:.3f}, "
                f"Reward={metrics['total_reward']:.3f}, "
                f"Epsilon={metrics['epsilon']:.3f}"
            )

    # Evaluate
    eval_metrics = agent.evaluate(env, episodes=5)
    print("\nEvaluation Results:")
    print(f"  Avg Return: {eval_metrics['avg_return']:.3f}")
    print(f"  Sharpe Ratio: {eval_metrics['sharpe_ratio']:.3f}")
