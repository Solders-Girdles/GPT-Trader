"""
RL-003: Policy Gradient Methods
Phase 4 - Week 3

Policy gradient methods for trading:
- REINFORCE algorithm
- Baseline variance reduction
- Advantage estimation
- Trust region methods
- Natural policy gradient
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical, Normal

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

logger = logging.getLogger(__name__)


class PolicyType(Enum):
    """Types of policies"""

    DISCRETE = "discrete"  # Categorical distribution
    CONTINUOUS = "continuous"  # Gaussian distribution
    MIXED = "mixed"  # Both discrete and continuous actions


@dataclass
class PolicyGradientConfig:
    """Configuration for policy gradient methods"""

    # Network architecture
    state_dim: int = 50
    action_dim: int = 3
    hidden_layers: list[int] = None
    activation: str = "tanh"

    # Policy type
    policy_type: PolicyType = PolicyType.DISCRETE
    action_std_init: float = 0.5  # For continuous actions

    # Training parameters
    learning_rate: float = 0.0003
    discount_factor: float = 0.99
    gae_lambda: float = 0.95  # GAE parameter

    # Variance reduction
    use_baseline: bool = True
    normalize_advantages: bool = True

    # Trust region
    use_trust_region: bool = False
    max_kl: float = 0.01  # KL divergence constraint
    damping: float = 0.1  # Conjugate gradient damping

    # Training schedule
    episodes: int = 1000
    max_steps_per_episode: int = 1000
    update_frequency: int = 1  # Episodes between updates

    # Entropy regularization
    entropy_coeff: float = 0.01
    entropy_decay: float = 0.999

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64]


if TORCH_AVAILABLE:

    class PolicyNetwork(nn.Module):
        """Policy network for discrete actions"""

        def __init__(self, config: PolicyGradientConfig):
            super().__init__()
            self.config = config

            # Build network layers
            layers = []
            input_dim = config.state_dim

            for hidden_dim in config.hidden_layers:
                layers.append(nn.Linear(input_dim, hidden_dim))

                if config.activation == "relu":
                    layers.append(nn.ReLU())
                elif config.activation == "tanh":
                    layers.append(nn.Tanh())
                elif config.activation == "leaky_relu":
                    layers.append(nn.LeakyReLU())

                input_dim = hidden_dim

            self.features = nn.Sequential(*layers)

            # Output layer for action probabilities
            self.action_head = nn.Linear(input_dim, config.action_dim)

            # Value head for baseline
            if config.use_baseline:
                self.value_head = nn.Linear(input_dim, 1)

        def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
            """Forward pass returning action logits and optional value"""
            features = self.features(state)
            action_logits = self.action_head(features)

            value = None
            if self.config.use_baseline:
                value = self.value_head(features)

            return action_logits, value

        def get_action(
            self, state: torch.Tensor
        ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor | None]:
            """Sample action from policy"""
            action_logits, value = self.forward(state)

            # Create categorical distribution
            dist = Categorical(logits=action_logits)

            # Sample action
            action = dist.sample()

            # Get log probability
            log_prob = dist.log_prob(action)

            # Get entropy for regularization
            entropy = dist.entropy()

            return action.item(), log_prob, entropy, value

    class ContinuousPolicyNetwork(nn.Module):
        """Policy network for continuous actions"""

        def __init__(self, config: PolicyGradientConfig):
            super().__init__()
            self.config = config

            # Build network layers
            layers = []
            input_dim = config.state_dim

            for hidden_dim in config.hidden_layers:
                layers.append(nn.Linear(input_dim, hidden_dim))

                if config.activation == "relu":
                    layers.append(nn.ReLU())
                elif config.activation == "tanh":
                    layers.append(nn.Tanh())

                input_dim = hidden_dim

            self.features = nn.Sequential(*layers)

            # Output layers for mean and std
            self.mean_head = nn.Linear(input_dim, config.action_dim)
            self.log_std = nn.Parameter(
                torch.ones(1, config.action_dim) * np.log(config.action_std_init)
            )

            # Value head for baseline
            if config.use_baseline:
                self.value_head = nn.Linear(input_dim, 1)

        def forward(
            self, state: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            """Forward pass returning action mean, std, and optional value"""
            features = self.features(state)

            mean = self.mean_head(features)
            std = self.log_std.exp().expand_as(mean)

            value = None
            if self.config.use_baseline:
                value = self.value_head(features)

            return mean, std, value

        def get_action(
            self, state: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
            """Sample action from Gaussian policy"""
            mean, std, value = self.forward(state)

            # Create normal distribution
            dist = Normal(mean, std)

            # Sample action
            action = dist.sample()

            # Get log probability
            log_prob = dist.log_prob(action).sum(dim=-1)

            # Get entropy
            entropy = dist.entropy().sum(dim=-1)

            return action, log_prob, entropy, value

    class REINFORCE:
        """REINFORCE algorithm with baseline"""

        def __init__(self, config: PolicyGradientConfig):
            self.config = config
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize policy network
            if config.policy_type == PolicyType.DISCRETE:
                self.policy = PolicyNetwork(config).to(self.device)
            else:
                self.policy = ContinuousPolicyNetwork(config).to(self.device)

            # Optimizer
            self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)

            # Episode storage
            self.states = []
            self.actions = []
            self.rewards = []
            self.log_probs = []
            self.values = []
            self.entropies = []

            # Training history
            self.training_history = []
            self.entropy_coeff = config.entropy_coeff

        def select_action(self, state: np.ndarray) -> int | np.ndarray:
            """Select action from policy"""
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if self.config.policy_type == PolicyType.DISCRETE:
                    action, log_prob, entropy, value = self.policy.get_action(state_tensor)
                else:
                    action, log_prob, entropy, value = self.policy.get_action(state_tensor)
                    action = action.squeeze(0).cpu().numpy()

            # Store for training
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.entropies.append(entropy)

            if value is not None:
                self.values.append(value)

            return action

        def store_reward(self, reward: float):
            """Store reward for current step"""
            self.rewards.append(reward)

        def compute_returns(self) -> torch.Tensor:
            """Compute discounted returns"""
            returns = []
            discounted_return = 0

            for reward in reversed(self.rewards):
                discounted_return = reward + self.config.discount_factor * discounted_return
                returns.insert(0, discounted_return)

            returns = torch.FloatTensor(returns).to(self.device)

            # Normalize returns
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            return returns

        def compute_advantages(self) -> torch.Tensor:
            """Compute advantage estimates using GAE"""
            if not self.config.use_baseline or not self.values:
                # Use returns as advantages
                return self.compute_returns()

            # Convert to tensors
            rewards = torch.FloatTensor(self.rewards).to(self.device)
            values = torch.cat(self.values).squeeze()

            # Compute TD errors
            advantages = []
            advantage = 0

            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]

                td_error = rewards[t] + self.config.discount_factor * next_value - values[t]
                advantage = (
                    td_error + self.config.discount_factor * self.config.gae_lambda * advantage
                )
                advantages.insert(0, advantage)

            advantages = torch.FloatTensor(advantages).to(self.device)

            # Normalize advantages
            if self.config.normalize_advantages and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return advantages

        def update(self) -> dict[str, float]:
            """Update policy using collected episode data"""

            if len(self.rewards) == 0:
                return {"loss": 0, "entropy": 0}

            # Compute returns or advantages
            if self.config.use_baseline:
                advantages = self.compute_advantages()
                returns = self.compute_returns()
            else:
                returns = self.compute_returns()
                advantages = returns

            # Convert to tensors
            log_probs = torch.cat(self.log_probs)
            entropies = torch.cat(self.entropies)

            # Policy gradient loss
            policy_loss = -(log_probs * advantages.detach()).mean()

            # Value loss if using baseline
            value_loss = 0
            if self.config.use_baseline and self.values:
                values = torch.cat(self.values).squeeze()
                value_loss = F.mse_loss(values, returns)

            # Entropy regularization
            entropy_loss = -self.entropy_coeff * entropies.mean()

            # Total loss
            total_loss = policy_loss + 0.5 * value_loss + entropy_loss

            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

            self.optimizer.step()

            # Decay entropy coefficient
            self.entropy_coeff *= self.config.entropy_decay

            # Clear episode data
            self.clear_episode()

            return {
                "total_loss": total_loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": (
                    value_loss.item() if isinstance(value_loss, torch.Tensor) else value_loss
                ),
                "entropy": entropies.mean().item(),
            }

        def clear_episode(self):
            """Clear episode storage"""
            self.states = []
            self.actions = []
            self.rewards = []
            self.log_probs = []
            self.values = []
            self.entropies = []

        def train_episode(self, env) -> dict[str, float]:
            """Train for one episode"""
            state = env.reset()
            episode_reward = 0

            done = False
            steps = 0

            while not done and steps < self.config.max_steps_per_episode:
                # Select action
                action = self.select_action(state)

                # Step environment
                next_state, reward, done, _ = env.step(action)

                # Store reward
                self.store_reward(reward)

                episode_reward += reward
                state = next_state
                steps += 1

            # Update policy
            update_metrics = self.update()

            metrics = {"episode_reward": episode_reward, "episode_length": steps, **update_metrics}

            self.training_history.append(metrics)

            return metrics

    class NaturalPolicyGradient(REINFORCE):
        """Natural Policy Gradient with Fisher Information Matrix"""

        def __init__(self, config: PolicyGradientConfig):
            super().__init__(config)
            self.config.use_trust_region = True

        def compute_fisher_vector_product(self, vector: torch.Tensor) -> torch.Tensor:
            """Compute Fisher-vector product for conjugate gradient"""

            # Sample states for Fisher estimation
            states = torch.FloatTensor(self.states).to(self.device)

            # Get policy distribution
            if self.config.policy_type == PolicyType.DISCRETE:
                action_logits, _ = self.policy(states)
                dist = Categorical(logits=action_logits)
            else:
                mean, std, _ = self.policy(states)
                dist = Normal(mean, std)

            # Sample actions
            actions = dist.sample()

            # Compute KL divergence
            with torch.no_grad():
                if self.config.policy_type == PolicyType.DISCRETE:
                    old_logits, _ = self.policy(states)
                    old_dist = Categorical(logits=old_logits)
                else:
                    old_mean, old_std, _ = self.policy(states)
                    old_dist = Normal(old_mean, old_std)

            kl = torch.distributions.kl_divergence(old_dist, dist).mean()

            # Compute gradient of KL
            grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_grad = torch.cat([g.view(-1) for g in grads])

            # Compute Fisher-vector product
            grad_vector_product = torch.sum(flat_grad * vector)
            fisher_vector_product = torch.autograd.grad(
                grad_vector_product, self.policy.parameters()
            )

            flat_fisher_vector = torch.cat([g.view(-1) for g in fisher_vector_product])

            return flat_fisher_vector + self.config.damping * vector

        def conjugate_gradient(self, b: torch.Tensor, max_iter: int = 10) -> torch.Tensor:
            """Solve Ax = b using conjugate gradient where A is Fisher matrix"""

            x = torch.zeros_like(b)
            r = b.clone()
            p = b.clone()
            rdotr = torch.dot(r, r)

            for _ in range(max_iter):
                Ap = self.compute_fisher_vector_product(p)
                alpha = rdotr / (torch.dot(p, Ap) + 1e-8)

                x = x + alpha * p
                r = r - alpha * Ap

                new_rdotr = torch.dot(r, r)
                beta = new_rdotr / (rdotr + 1e-8)

                p = r + beta * p
                rdotr = new_rdotr

                if rdotr < 1e-10:
                    break

            return x

        def update(self) -> dict[str, float]:
            """Update using natural policy gradient"""

            if not self.config.use_trust_region:
                return super().update()

            # Compute advantages
            advantages = self.compute_advantages()

            # Compute policy gradient
            states = torch.FloatTensor(self.states).to(self.device)
            actions = torch.LongTensor(self.actions).to(self.device)

            if self.config.policy_type == PolicyType.DISCRETE:
                action_logits, values = self.policy(states)
                dist = Categorical(logits=action_logits)
                log_probs = dist.log_prob(actions)
            else:
                mean, std, values = self.policy(states)
                dist = Normal(mean, std)
                log_probs = dist.log_prob(actions).sum(dim=-1)

            # Policy gradient
            policy_gradient = torch.autograd.grad(
                (log_probs * advantages.detach()).mean(), self.policy.parameters()
            )
            flat_gradient = torch.cat([g.view(-1) for g in policy_gradient])

            # Solve for natural gradient using conjugate gradient
            natural_gradient = self.conjugate_gradient(flat_gradient)

            # Compute step size
            shs = 0.5 * torch.dot(
                natural_gradient, self.compute_fisher_vector_product(natural_gradient)
            )
            step_size = torch.sqrt(2 * self.config.max_kl / (shs + 1e-8))

            # Update parameters
            param_idx = 0
            for param in self.policy.parameters():
                param_length = param.numel()
                param_update = natural_gradient[param_idx : param_idx + param_length].view(
                    param.shape
                )
                param.data += step_size * param_update
                param_idx += param_length

            # Clear episode data
            self.clear_episode()

            return {
                "total_loss": 0,  # Not directly computed
                "step_size": step_size.item(),
                "kl_divergence": shs.item(),
            }


def create_policy_gradient_agent(
    state_dim: int,
    action_dim: int,
    policy_type: PolicyType = PolicyType.DISCRETE,
    use_natural_gradient: bool = False,
) -> REINFORCE:
    """Factory function to create policy gradient agent"""

    config = PolicyGradientConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        policy_type=policy_type,
        use_baseline=True,
        normalize_advantages=True,
    )

    if use_natural_gradient:
        return NaturalPolicyGradient(config)
    else:
        return REINFORCE(config)


if __name__ == "__main__":
    print("Policy Gradient Methods Implementation")

    if TORCH_AVAILABLE:
        # Test REINFORCE
        config = PolicyGradientConfig(
            state_dim=50, action_dim=3, policy_type=PolicyType.DISCRETE, use_baseline=True
        )

        agent = REINFORCE(config)

        print("\nREINFORCE Configuration:")
        print(f"  State Dimension: {config.state_dim}")
        print(f"  Action Dimension: {config.action_dim}")
        print(f"  Policy Type: {config.policy_type.value}")
        print(f"  Using Baseline: {config.use_baseline}")
        print(f"  Device: {agent.device}")

        # Test action selection
        test_state = np.random.randn(config.state_dim)
        action = agent.select_action(test_state)
        print(f"\nTest action: {action}")

        # Test update with dummy data
        for _ in range(10):
            agent.select_action(np.random.randn(config.state_dim))
            agent.store_reward(np.random.randn())

        metrics = agent.update()
        print("\nUpdate metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    else:
        print("PyTorch not available")
