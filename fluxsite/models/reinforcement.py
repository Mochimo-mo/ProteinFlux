import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import logging

# Set up logging
logger = logging.getLogger("reinforcement")


class ReplayBuffer:
    """Experience replay buffer for reinforcement learning."""
    
    def __init__(self, capacity=10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum capacity of the buffer.
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether the episode ended.
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size (int): Batch size.
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class PPOActorCritic(nn.Module):
    """
    Actor-Critic model for Proximal Policy Optimization (PPO).
    
    This can be used as a policy model for the final classification after 
    feature extraction using the encoders.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, use_layer_norm=False):
        """
        Initialize the PPO Actor-Critic model.
        
        Args:
            state_dim (int): Dimension of the state (feature) space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
            use_layer_norm (bool): Whether to use layer normalization.
        """
        super().__init__()
        
        # Store layer normalization flag
        self.use_layer_norm = use_layer_norm
        
        # Shared feature extractor
        feature_extractor_layers = []
        feature_extractor_layers.append(nn.Linear(state_dim, hidden_dim))
        if self.use_layer_norm:
            feature_extractor_layers.append(nn.LayerNorm(hidden_dim))
        feature_extractor_layers.append(nn.Tanh())
        feature_extractor_layers.append(nn.Linear(hidden_dim, hidden_dim))
        if self.use_layer_norm:
            feature_extractor_layers.append(nn.LayerNorm(hidden_dim))
        feature_extractor_layers.append(nn.Tanh())
        
        self.feature_extractor = nn.Sequential(*feature_extractor_layers)
        
        # Policy network (actor)
        policy_layers = []
        policy_layers.append(nn.Linear(hidden_dim, hidden_dim))
        if self.use_layer_norm:
            policy_layers.append(nn.LayerNorm(hidden_dim))
        policy_layers.append(nn.Tanh())
        policy_layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.policy = nn.Sequential(*policy_layers)
        
        # Value network (critic)
        value_layers = []
        value_layers.append(nn.Linear(hidden_dim, hidden_dim))
        if self.use_layer_norm:
            value_layers.append(nn.LayerNorm(hidden_dim))
        value_layers.append(nn.Tanh())
        value_layers.append(nn.Linear(hidden_dim, 1))
        
        self.value = nn.Sequential(*value_layers)
    
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: State (feature) tensor. Can be [batch_size, feature_dim] or [batch_size, seq_len, feature_dim]
            
        Returns:
            dict: Dictionary containing action_logits and state_value
        """
        # Handle both [batch_size, feature_dim] and [batch_size, seq_len, feature_dim] inputs
        original_shape_len = len(state.shape)
        if original_shape_len == 3:
            # If input is [batch_size, seq_len, feature_dim], reshape to [batch_size*seq_len, feature_dim]
            batch_size, seq_len, feature_dim = state.shape
            state = state.reshape(-1, feature_dim)
        
        # Process through feature extractor
        features = self.feature_extractor(state)
        
        # Get action logits and state value
        action_logits = self.policy(features)
        state_value = self.value(features)
        
        # Reshape back to original dimensions if needed
        if original_shape_len == 3:
            action_logits = action_logits.reshape(batch_size, seq_len, -1)
            state_value = state_value.reshape(batch_size, seq_len, -1)
        
        return {
            'action_logits': action_logits,
            'state_value': state_value
        }
    
    def get_action(self, state, deterministic=False):
        """
        Get an action from the policy.
        
        Args:
            state: State (feature) tensor.
            deterministic (bool): Whether to use deterministic action selection.
            
        Returns:
            tuple: (action, log_prob, action_logits)
        """
        # Forward pass
        outputs = self.forward(state)
        action_logits = outputs['action_logits']
        
        # Apply softmax to get action probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Use the Gumbel-Softmax trick for differentiable sampling
            action = torch.multinomial(action_probs, 1).squeeze(-1)
            
        log_prob = F.log_softmax(action_logits, dim=-1)
        
        return action, log_prob, action_logits


class RLClassifier(nn.Module):
    """
    Reinforcement learning-based classifier for acetylation prediction.
    
    This model treats the prediction as a sequential decision-making process,
    where it sequentially examines different aspects of a protein sequence and 
    structure before making a final decision.
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_steps=5):
        """
        Initialize the RL classifier.
        
        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden dimension.
            num_steps (int): Number of decision steps.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        
        # GRU for processing sequential information
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Policy network (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)  # 2 actions: 0=non-acetylated, 1=acetylated
        )
        
        # Value network (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, return_policy=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim].
            return_policy (bool): Whether to return policy outputs or final prediction.
            
        Returns:
            torch.Tensor or tuple: Prediction or (action_logits, state_value)
        """
        # Handle inputs with different dimensionalities
        if len(x.shape) == 2:
            # If input is 2D [batch_size, feature_dim], expand to 3D
            batch_size, feature_dim = x.shape
            seq_len = 1
            x = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
        else:
            # If input is 3D [batch_size, seq_len, feature_dim]
            batch_size, seq_len, _ = x.shape

        # Process sequence with GRU
        gru_out, hidden = self.gru(x)
        
        # Use attention to focus on important parts of the sequence
        attention_scores = self.attention(gru_out).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(1)
        context = torch.bmm(attention_weights, gru_out).squeeze(1)
        
        if return_policy:
            # Return policy and value for RL training
            action_logits = self.policy(context)
            state_value = self.value(context)
            return action_logits, state_value
        else:
            # Return final prediction (logits instead of probability)
            action_logits = self.policy(context)
            return action_logits[:, 1]  # Logits for acetylation (action 1)


class PPOTrainer:
    """
    Trainer for Proximal Policy Optimization (PPO).
    
    This class implements the PPO algorithm for reinforcement learning.
    """
    
    def __init__(self, actor_critic, optimizer, clip_param=0.2, value_coef=0.5, entropy_coef=0.01):
        """
        Initialize the PPO trainer.
        
        Args:
            actor_critic (nn.Module): Actor-critic model.
            optimizer: Optimizer.
            clip_param (float): PPO clipping parameter.
            value_coef (float): Value loss coefficient.
            entropy_coef (float): Entropy loss coefficient.
        """
        self.actor_critic = actor_critic
        self.optimizer = optimizer
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        """
        Update the actor-critic model using PPO.
        
        Args:
            states: Batch of states.
            actions: Batch of actions.
            old_log_probs: Batch of old log probabilities.
            returns: Batch of returns.
            advantages: Batch of advantages.
            
        Returns:
            dict: Dictionary of loss values.
        """
        # Forward pass through actor-critic
        outputs = self.actor_critic(states)
        
        # Extract action logits and value
        action_logits = outputs['action_logits']
        values = outputs['state_value']
        
        # Extract log probabilities for the actions that were taken
        new_log_probs = F.log_softmax(action_logits, dim=-1)
        new_log_probs = new_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute ratio (π_θ / π_θ_old)
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Compute surrogate loss
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        
        # Calculate losses
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        value_loss = F.mse_loss(values.squeeze(-1), returns)
        
        # Calculate entropy for exploration
        entropy = -(F.softmax(action_logits, dim=-1) * F.log_softmax(action_logits, dim=-1)).sum(-1).mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Update model - We need to clear the gradients before each update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)  # Add gradient clipping for stability
        self.optimizer.step()
        
        # Return metrics as scalars, not tensors that retain graph
        return {
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item()),
            'entropy': float(entropy.item()),
            'total_loss': float(loss.item())
        }


class BERTBiLSTMRLClassifier(nn.Module):
    """
    Reinforcement-learning classifier based on a BERT + BiLSTM architecture.

    Designed for protein post-translational modification (PTM) prediction by combining:
    - BERT-style long-range dependency modeling
    - BiLSTM-based local sequence feature extraction
    - Reinforcement-learning decision optimization
    """

    def __init__(self, input_dim, bert_hidden_dim=768, lstm_hidden_dim=128,
                 num_decision_steps=3, dropout=0.1):
        """
        Initialize the BERT+BiLSTM RL classifier.

        Args:
            input_dim (int): Input feature dimension.
            bert_hidden_dim (int): BERT hidden dimension.
            lstm_hidden_dim (int): LSTM hidden dimension.
            num_decision_steps (int): Number of decision steps.
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.input_dim = input_dim
        self.bert_hidden_dim = bert_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_decision_steps = num_decision_steps

        logger.info(f"InitializeBERT+BiLSTM ")
        logger.info(f"-: {input_dim}")
        logger.info(f"- BERT: {bert_hidden_dim}")
        logger.info(f"- LSTM: {lstm_hidden_dim}")
        logger.info(f"-: {num_decision_steps}")

        # 1. Input projection layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.Dropout(dropout)
        )

        # 2. Simplified BERT-style encoder (for long-range dependencies)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bert_hidden_dim,
            nhead=8,
            dim_feedforward=bert_hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.bert_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 3. BiLSTM layer (for local sequence features)
        self.bilstm = nn.LSTM(
            input_size=bert_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # BiLSTM output dimension
        self.lstm_output_dim = lstm_hidden_dim * 2

        # 4. Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(bert_hidden_dim + self.lstm_output_dim, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 5. Multi-step decision network
        self.decision_steps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(bert_hidden_dim, bert_hidden_dim // 2),
                nn.LayerNorm(bert_hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bert_hidden_dim // 2, bert_hidden_dim // 4),
                nn.LayerNorm(bert_hidden_dim // 4),
                nn.GELU()
            ) for _ in range(num_decision_steps)
        ])

        # 6. Attention mechanism (to integrate information across decision steps)
        self.step_attention = nn.MultiheadAttention(
            embed_dim=bert_hidden_dim // 4,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # 7. Policy network (Actor)
        self.policy_network = nn.Sequential(
            nn.Linear(bert_hidden_dim // 4, bert_hidden_dim // 8),
            nn.LayerNorm(bert_hidden_dim // 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bert_hidden_dim // 8, 2)  # Binary classification: 0=unmodified, 1=modified
        )

        # 8. Value network (Critic)
        self.value_network = nn.Sequential(
            nn.Linear(bert_hidden_dim // 4, bert_hidden_dim // 8),
            nn.LayerNorm(bert_hidden_dim // 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bert_hidden_dim // 8, 1)
        )

        # 9. Site-specific attention (for PTM prediction)
        self.site_attention = nn.MultiheadAttention(
            embed_dim=bert_hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        logger.info("BERT+BiLSTM InitializeCompleted")

    def forward(self, x, return_policy=False, return_attention=False):
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim].
            return_policy (bool): Whether to return policy outputs.
            return_attention (bool): Whether to return attention weights.

        Returns:
            torch.Tensor or dict: Prediction logits or a dict with policy/value outputs.
        """
        # Handle inputs with different dimensionalities
        if len(x.shape) == 2:
            # If input is 2D [batch_size, feature_dim], expand to 3D
            batch_size, feature_dim = x.shape
            seq_len = 1
            x = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
        else:
            # If input is 3D [batch_size, seq_len, feature_dim]
            batch_size, seq_len, _ = x.shape

        # 1. Input projection
        projected_input = self.input_projection(x)  # [batch_size, seq_len, bert_hidden_dim]

        # 2. BERT encoding (long-range dependencies)
        bert_features = self.bert_encoder(projected_input)  # [batch_size, seq_len, bert_hidden_dim]

        # 3. BiLSTM encoding (local sequence features)
        lstm_features, (hidden, cell) = self.bilstm(bert_features)
        # lstm_features: [batch_size, seq_len, lstm_output_dim]

        # 4. Feature fusion
        combined_features = torch.cat([bert_features, lstm_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)  # [batch_size, seq_len, bert_hidden_dim]

        # 5. Site-specific attention
        attended_features, site_attention_weights = self.site_attention(
            fused_features, fused_features, fused_features
        )

        # 6. Multi-step decision process
        decision_features = []
        current_features = attended_features.mean(dim=1)  # Global pooling [batch_size, bert_hidden_dim]

        for step_idx, decision_step in enumerate(self.decision_steps):
            step_output = decision_step(current_features)  # [batch_size, bert_hidden_dim // 4]
            decision_features.append(step_output.unsqueeze(1))

        # Stack decision features
        decision_features = torch.cat(decision_features, dim=1)  # [batch_size, num_steps, bert_hidden_dim // 4]

        # 7. Inter-step attention
        final_features, step_attention_weights = self.step_attention(
            decision_features, decision_features, decision_features
        )

        # 8. Final feature aggregation
        aggregated_features = final_features.mean(dim=1)  # [batch_size, bert_hidden_dim // 4]

        if return_policy:
            # Return policy and value for RL training
            action_logits = self.policy_network(aggregated_features)
            state_value = self.value_network(aggregated_features)

            result = {
                'action_logits': action_logits,
                'state_value': state_value
            }

            if return_attention:
                result.update({
                    'site_attention': site_attention_weights,
                    'step_attention': step_attention_weights
                })

            return result
        else:
            # Return final prediction (logits, not probabilities)
            action_logits = self.policy_network(aggregated_features)
            prediction_logits = action_logits[:, 1]  # Modification logit

            if return_attention:
                return prediction_logits, {
                    'site_attention': site_attention_weights,
                    'step_attention': step_attention_weights
                }
            else:
                return prediction_logits

    def get_action(self, x, deterministic=False):
        """
        Get an action (for RL training).

        Args:
            x: Input tensor.
            deterministic (bool): Whether to use deterministic action selection.

        Returns:
            tuple: (action, log_prob, action_logits)
        """
        outputs = self.forward(x, return_policy=True)
        action_logits = outputs['action_logits']

        # Apply softmax to obtain action probabilities
        action_probs = F.softmax(action_logits, dim=-1)

        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from a multinomial distribution
            action = torch.multinomial(action_probs, 1).squeeze(-1)

        log_prob = F.log_softmax(action_logits, dim=-1)

        return action, log_prob, action_logits
