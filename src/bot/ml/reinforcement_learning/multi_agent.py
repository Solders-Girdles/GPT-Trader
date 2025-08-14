"""
RL-006: Multi-Agent Reinforcement Learning
Phase 4 - Week 4

Multi-agent systems for market simulation:
- Multiple trading agents competing/cooperating
- Market maker vs traders dynamics
- Nash equilibrium seeking
- Communication between agents
- Centralized training with decentralized execution
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import random
from pathlib import Path
import copy

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


class AgentType(Enum):
    """Types of agents in the market"""
    TRADER = "trader"  # Regular trading agent
    MARKET_MAKER = "market_maker"  # Provides liquidity
    ARBITRAGEUR = "arbitrageur"  # Exploits price differences
    NOISE_TRADER = "noise_trader"  # Random trading


class CommunicationType(Enum):
    """Types of communication between agents"""
    NONE = "none"  # No communication
    BROADCAST = "broadcast"  # All agents receive messages
    TARGETED = "targeted"  # Specific agent communication
    GRAPH = "graph"  # Communication through network topology


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent RL"""
    # Agent configuration
    n_agents: int = 4
    agent_types: List[AgentType] = None
    state_dim: int = 50
    action_dim: int = 3
    hidden_dims: List[int] = None
    
    # Communication
    communication_type: CommunicationType = CommunicationType.NONE
    message_dim: int = 16
    communication_rounds: int = 1
    
    # Training
    algorithm: str = "qmix"  # qmix, maddpg, commnet, independent
    learning_rate: float = 0.0003
    discount_factor: float = 0.99
    tau: float = 0.01  # Soft update
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    
    # Buffer
    buffer_size: int = 100000
    batch_size: int = 64
    
    # QMIX specific
    mixing_embed_dim: int = 32
    hypernet_embed_dim: int = 64
    
    # MADDPG specific
    use_gumbel_softmax: bool = True
    gumbel_temperature: float = 1.0
    
    def __post_init__(self):
        if self.agent_types is None:
            self.agent_types = [AgentType.TRADER] * self.n_agents
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]


if TORCH_AVAILABLE:
    
    class IndependentAgent(nn.Module):
        """Independent Q-learning agent"""
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
            super().__init__()
            
            layers = []
            input_dim = state_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                input_dim = hidden_dim
            
            layers.append(nn.Linear(input_dim, action_dim))
            
            self.q_network = nn.Sequential(*layers)
        
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            """Get Q-values for state"""
            return self.q_network(state)
        
        def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
            """Select action using epsilon-greedy"""
            if random.random() < epsilon:
                return random.randint(0, self.q_network[-1].out_features - 1)
            
            q_values = self.forward(state)
            return q_values.argmax(dim=-1).item()
    
    
    class QMIXNetwork(nn.Module):
        """QMIX mixing network for value decomposition"""
        
        def __init__(self, config: MultiAgentConfig):
            super().__init__()
            self.config = config
            self.n_agents = config.n_agents
            
            # Hypernetwork for weights
            self.hyper_w1 = nn.Sequential(
                nn.Linear(config.state_dim, config.hypernet_embed_dim),
                nn.ReLU(),
                nn.Linear(config.hypernet_embed_dim, config.n_agents * config.mixing_embed_dim)
            )
            
            self.hyper_w2 = nn.Sequential(
                nn.Linear(config.state_dim, config.hypernet_embed_dim),
                nn.ReLU(),
                nn.Linear(config.hypernet_embed_dim, config.mixing_embed_dim)
            )
            
            # Hypernetwork for biases
            self.hyper_b1 = nn.Linear(config.state_dim, config.mixing_embed_dim)
            self.hyper_b2 = nn.Sequential(
                nn.Linear(config.state_dim, config.mixing_embed_dim),
                nn.ReLU(),
                nn.Linear(config.mixing_embed_dim, 1)
            )
        
        def forward(self, agent_q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            """
            Mix agent Q-values to get total Q-value
            
            Args:
                agent_q_values: Q-values from all agents (batch, n_agents)
                state: Global state (batch, state_dim)
            """
            batch_size = agent_q_values.size(0)
            
            # Generate weights and biases
            w1 = torch.abs(self.hyper_w1(state))  # Ensure monotonicity
            w1 = w1.view(batch_size, self.n_agents, self.config.mixing_embed_dim)
            
            b1 = self.hyper_b1(state)
            b1 = b1.view(batch_size, 1, self.config.mixing_embed_dim)
            
            w2 = torch.abs(self.hyper_w2(state))
            w2 = w2.view(batch_size, self.config.mixing_embed_dim, 1)
            
            b2 = self.hyper_b2(state)
            b2 = b2.view(batch_size, 1, 1)
            
            # Mix Q-values
            hidden = F.elu(torch.bmm(agent_q_values.unsqueeze(1), w1) + b1)
            q_total = torch.bmm(hidden, w2) + b2
            
            return q_total.squeeze(-1).squeeze(-1)
    
    
    class CommunicationNetwork(nn.Module):
        """Communication module for agent interaction"""
        
        def __init__(self, config: MultiAgentConfig):
            super().__init__()
            self.config = config
            
            # Message encoder
            self.message_encoder = nn.Sequential(
                nn.Linear(config.state_dim, config.hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(config.hidden_dims[0], config.message_dim)
            )
            
            # Message aggregator
            if config.communication_type == CommunicationType.BROADCAST:
                self.aggregator = nn.GRU(
                    config.message_dim,
                    config.message_dim,
                    batch_first=True
                )
            elif config.communication_type == CommunicationType.GRAPH:
                self.graph_attention = nn.MultiheadAttention(
                    config.message_dim,
                    num_heads=4,
                    batch_first=True
                )
            
            # Message decoder
            self.message_decoder = nn.Sequential(
                nn.Linear(config.message_dim + config.state_dim, config.hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(config.hidden_dims[0], config.state_dim)
            )
        
        def forward(self, states: torch.Tensor, adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Process communication between agents
            
            Args:
                states: Agent states (batch, n_agents, state_dim)
                adjacency: Communication graph adjacency matrix
            """
            batch_size, n_agents = states.shape[:2]
            
            # Encode messages
            messages = self.message_encoder(states)
            
            # Aggregate messages based on communication type
            if self.config.communication_type == CommunicationType.BROADCAST:
                # All agents receive all messages
                aggregated, _ = self.aggregator(messages)
            elif self.config.communication_type == CommunicationType.GRAPH and adjacency is not None:
                # Graph-based communication
                aggregated, _ = self.graph_attention(messages, messages, messages, attn_mask=adjacency)
            else:
                # No aggregation
                aggregated = messages
            
            # Decode and combine with original states
            combined = torch.cat([states, aggregated], dim=-1)
            updated_states = self.message_decoder(combined)
            
            return states + updated_states  # Residual connection
    
    
    class MADDPG:
        """Multi-Agent Deep Deterministic Policy Gradient"""
        
        def __init__(self, config: MultiAgentConfig):
            self.config = config
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.n_agents = config.n_agents
            
            # Create actors and critics for each agent
            self.actors = []
            self.critics = []
            self.target_actors = []
            self.target_critics = []
            
            for i in range(self.n_agents):
                # Actor
                actor = self._build_actor(config.state_dim, config.action_dim).to(self.device)
                self.actors.append(actor)
                
                # Target actor
                target_actor = copy.deepcopy(actor)
                self.target_actors.append(target_actor)
                
                # Critic (sees all states and actions)
                critic_input_dim = config.state_dim * self.n_agents + config.action_dim * self.n_agents
                critic = self._build_critic(critic_input_dim).to(self.device)
                self.critics.append(critic)
                
                # Target critic
                target_critic = copy.deepcopy(critic)
                self.target_critics.append(target_critic)
            
            # Optimizers
            self.actor_optimizers = [
                optim.Adam(actor.parameters(), lr=config.learning_rate)
                for actor in self.actors
            ]
            self.critic_optimizers = [
                optim.Adam(critic.parameters(), lr=config.learning_rate)
                for critic in self.critics
            ]
            
            # Replay buffer
            self.replay_buffer = deque(maxlen=config.buffer_size)
            
            # Training state
            self.training_step = 0
        
        def _build_actor(self, state_dim: int, action_dim: int) -> nn.Module:
            """Build actor network"""
            return nn.Sequential(
                nn.Linear(state_dim, self.config.hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dims[1], action_dim),
                nn.Tanh()  # Bounded actions
            )
        
        def _build_critic(self, input_dim: int) -> nn.Module:
            """Build critic network"""
            return nn.Sequential(
                nn.Linear(input_dim, self.config.hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dims[1], 1)
            )
        
        def get_actions(self, states: List[np.ndarray], add_noise: bool = True) -> List[np.ndarray]:
            """Get actions for all agents"""
            actions = []
            
            for i, state in enumerate(states):
                state_tensor = torch.FloatTensor(state).to(self.device)
                
                with torch.no_grad():
                    action = self.actors[i](state_tensor)
                    
                    if add_noise:
                        # Add exploration noise
                        noise = torch.randn_like(action) * 0.1
                        action = torch.clamp(action + noise, -1, 1)
                    
                    actions.append(action.cpu().numpy())
            
            return actions
        
        def store_transition(self, states, actions, rewards, next_states, dones):
            """Store transition in replay buffer"""
            self.replay_buffer.append((states, actions, rewards, next_states, dones))
        
        def update(self) -> Dict[str, float]:
            """Update all agents using MADDPG"""
            
            if len(self.replay_buffer) < self.config.batch_size:
                return {}
            
            # Sample batch
            batch = random.sample(self.replay_buffer, self.config.batch_size)
            
            # Process batch
            states_batch = [[] for _ in range(self.n_agents)]
            actions_batch = [[] for _ in range(self.n_agents)]
            rewards_batch = [[] for _ in range(self.n_agents)]
            next_states_batch = [[] for _ in range(self.n_agents)]
            dones_batch = []
            
            for states, actions, rewards, next_states, dones in batch:
                for i in range(self.n_agents):
                    states_batch[i].append(states[i])
                    actions_batch[i].append(actions[i])
                    rewards_batch[i].append(rewards[i])
                    next_states_batch[i].append(next_states[i])
                dones_batch.append(dones)
            
            # Convert to tensors
            states_batch = [torch.FloatTensor(s).to(self.device) for s in states_batch]
            actions_batch = [torch.FloatTensor(a).to(self.device) for a in actions_batch]
            rewards_batch = [torch.FloatTensor(r).unsqueeze(1).to(self.device) for r in rewards_batch]
            next_states_batch = [torch.FloatTensor(s).to(self.device) for s in next_states_batch]
            dones_batch = torch.FloatTensor(dones_batch).unsqueeze(1).to(self.device)
            
            critic_losses = []
            actor_losses = []
            
            # Update each agent
            for i in range(self.n_agents):
                # Concatenate all states and actions for critic
                all_states = torch.cat(states_batch, dim=1)
                all_actions = torch.cat(actions_batch, dim=1)
                all_next_states = torch.cat(next_states_batch, dim=1)
                
                # Get target actions
                target_actions = []
                for j in range(self.n_agents):
                    with torch.no_grad():
                        target_action = self.target_actors[j](next_states_batch[j])
                        target_actions.append(target_action)
                all_target_actions = torch.cat(target_actions, dim=1)
                
                # Compute target Q-value
                with torch.no_grad():
                    target_critic_input = torch.cat([all_next_states, all_target_actions], dim=1)
                    target_q = self.target_critics[i](target_critic_input)
                    target_q = rewards_batch[i] + self.config.discount_factor * (1 - dones_batch) * target_q
                
                # Update critic
                critic_input = torch.cat([all_states, all_actions], dim=1)
                current_q = self.critics[i](critic_input)
                critic_loss = F.mse_loss(current_q, target_q)
                
                self.critic_optimizers[i].zero_grad()
                critic_loss.backward()
                self.critic_optimizers[i].step()
                critic_losses.append(critic_loss.item())
                
                # Update actor
                new_action = self.actors[i](states_batch[i])
                new_all_actions = actions_batch.copy()
                new_all_actions[i] = new_action
                new_all_actions = torch.cat(new_all_actions, dim=1)
                
                actor_loss = -self.critics[i](torch.cat([all_states, new_all_actions], dim=1)).mean()
                
                self.actor_optimizers[i].zero_grad()
                actor_loss.backward()
                self.actor_optimizers[i].step()
                actor_losses.append(actor_loss.item())
                
                # Soft update target networks
                for target_param, param in zip(self.target_actors[i].parameters(), 
                                              self.actors[i].parameters()):
                    target_param.data.copy_(
                        self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                    )
                
                for target_param, param in zip(self.target_critics[i].parameters(),
                                              self.critics[i].parameters()):
                    target_param.data.copy_(
                        self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                    )
            
            self.training_step += 1
            
            return {
                'critic_loss': np.mean(critic_losses),
                'actor_loss': np.mean(actor_losses)
            }
    
    
    class MarketEnvironment:
        """Multi-agent market environment"""
        
        def __init__(self, n_agents: int, initial_price: float = 100.0):
            self.n_agents = n_agents
            self.initial_price = initial_price
            self.reset()
        
        def reset(self):
            """Reset environment"""
            self.price = self.initial_price
            self.time_step = 0
            self.agent_positions = np.zeros(self.n_agents)
            self.agent_cash = np.ones(self.n_agents) * 10000
            self.order_book = {'bids': [], 'asks': []}
            
            return self._get_states()
        
        def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
            """
            Execute actions from all agents
            
            Actions: 0=Hold, 1=Buy, 2=Sell
            """
            rewards = []
            
            # Process orders
            for i, action in enumerate(actions):
                if action == 1:  # Buy
                    if self.agent_cash[i] >= self.price:
                        self.agent_positions[i] += 1
                        self.agent_cash[i] -= self.price
                        self.price *= 1.001  # Price impact
                elif action == 2:  # Sell
                    if self.agent_positions[i] > 0:
                        self.agent_positions[i] -= 1
                        self.agent_cash[i] += self.price
                        self.price *= 0.999  # Price impact
            
            # Calculate rewards (portfolio value change)
            for i in range(self.n_agents):
                portfolio_value = self.agent_cash[i] + self.agent_positions[i] * self.price
                reward = (portfolio_value - 10000) / 10000  # Normalized return
                rewards.append(reward)
            
            # Add market dynamics
            self.price += np.random.randn() * 0.5  # Random walk
            self.time_step += 1
            
            done = self.time_step >= 1000
            
            return self._get_states(), rewards, done, {}
        
        def _get_states(self) -> List[np.ndarray]:
            """Get state for each agent"""
            states = []
            
            # Common market features
            market_features = np.array([
                self.price / self.initial_price,  # Normalized price
                np.sin(2 * np.pi * self.time_step / 100),  # Time encoding
                np.cos(2 * np.pi * self.time_step / 100)
            ])
            
            for i in range(self.n_agents):
                # Agent-specific features
                agent_features = np.array([
                    self.agent_positions[i] / 10,  # Normalized position
                    self.agent_cash[i] / 10000,  # Normalized cash
                    (self.agent_cash[i] + self.agent_positions[i] * self.price) / 10000  # Portfolio value
                ])
                
                # Combine features
                state = np.concatenate([market_features, agent_features])
                
                # Pad to state dimension
                if len(state) < 50:
                    state = np.pad(state, (0, 50 - len(state)))
                
                states.append(state)
            
            return states


def create_multi_agent_system(n_agents: int = 4,
                             algorithm: str = "maddpg") -> Union[MADDPG, None]:
    """Factory function to create multi-agent system"""
    
    config = MultiAgentConfig(
        n_agents=n_agents,
        algorithm=algorithm
    )
    
    if algorithm == "maddpg":
        return MADDPG(config)
    else:
        logger.warning(f"Algorithm {algorithm} not fully implemented")
        return None


if __name__ == "__main__":
    print("Multi-Agent Reinforcement Learning Implementation")
    
    if TORCH_AVAILABLE:
        # Test multi-agent setup
        config = MultiAgentConfig(
            n_agents=4,
            state_dim=50,
            action_dim=3
        )
        
        print(f"\nMulti-Agent Configuration:")
        print(f"  Number of Agents: {config.n_agents}")
        print(f"  Agent Types: {[t.value for t in config.agent_types]}")
        print(f"  Communication: {config.communication_type.value}")
        
        # Test MADDPG
        print("\n=== Testing MADDPG ===")
        maddpg = MADDPG(config)
        
        # Create environment
        env = MarketEnvironment(n_agents=config.n_agents)
        states = env.reset()
        
        # Get actions
        actions = maddpg.get_actions(states)
        print(f"Actions shape: {[a.shape for a in actions]}")
        
        # Step environment
        next_states, rewards, done, _ = env.step([1, 0, 2, 1])  # Sample actions
        print(f"Rewards: {rewards}")
        
        # Store transition
        maddpg.store_transition(states, actions, rewards, next_states, done)
        
        # Test QMIX network
        print("\n=== Testing QMIX ===")
        qmix = QMIXNetwork(config)
        
        # Test forward pass
        agent_q_values = torch.randn(32, config.n_agents)
        global_state = torch.randn(32, config.state_dim)
        
        q_total = qmix(agent_q_values, global_state)
        print(f"Q-total shape: {q_total.shape}")
        
        # Test communication network
        print("\n=== Testing Communication ===")
        comm_config = MultiAgentConfig(
            n_agents=4,
            communication_type=CommunicationType.BROADCAST
        )
        comm_net = CommunicationNetwork(comm_config)
        
        agent_states = torch.randn(32, 4, 50)
        updated_states = comm_net(agent_states)
        print(f"Updated states shape: {updated_states.shape}")
    else:
        print("PyTorch not available")