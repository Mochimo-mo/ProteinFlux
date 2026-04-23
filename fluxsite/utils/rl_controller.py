import numpy as np
import torch
import logging
import json
from pathlib import Path

class RLHyperparameterController:
    """
    Reinforcement Learning Controller for Dynamic Hyperparameter Tuning.
    
    This agent observes the training state (Training Loss, Validation Loss, Accuracy, MCC)
    and adjusts regularization hyperparameters (Dropout, Weight Decay) and Augmentation parameters
    to mitigate overfitting and maximize performance.
    """
    def __init__(self, training_manager, config, output_dir):
        self.training_manager = training_manager
        self.model = training_manager.model
        self.optimizer = training_manager.optimizer
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger("RL_Controller")
        
        # RL Parameters
        rl_config = config.get('reinforcement_learning', {})
        self.learning_rate = rl_config.get('policy_learning_rate', 0.1)
        self.discount_factor = rl_config.get('discount_factor', 0.9)
        self.epsilon = rl_config.get('exploration_epsilon', 0.3)
        self.epsilon_decay = 0.95
        self.min_epsilon = 0.05
        
        # State Space: (Gap State, Loss Trend State, Metric Trend State)
        # Gap State: 0 (Low), 1 (Medium), 2 (High)
        # Loss Trend State: 0 (Improving), 1 (Stable), 2 (Degrading)
        # Metric Trend State: 0 (Improving), 1 (Stable), 2 (Degrading)
        self.q_table = np.zeros((3, 3, 3, 5)) # (Gap, LossTrend, MetricTrend, Action)
        
        # Actions: 
        # 0: Maintain
        # 1: Increase Regularization (Dropout++, WeightDecay++)
        # 2: Decrease Regularization (Dropout--, WeightDecay--)
        # 3: Increase Augmentation (MaskProb++, MixupProb++)
        # 4: Decrease Augmentation (MaskProb--, MixupProb--)
        self.action_space = [0, 1, 2, 3, 4]
        
        # History
        self.history = []
        self.prev_val_loss = float('inf')
        self.prev_val_metric = 0.0 # MCC or F1
        self.prev_state = None
        self.prev_action = None
        
        # Hyperparameter Bounds
        self.min_dropout = 0.0
        self.max_dropout = 0.8
        self.min_wd = 1e-6
        self.max_wd = 1e-1
        self.min_aug = 0.0
        self.max_aug = 0.8
        
        # Cache dropout layers
        self.dropout_layers = []
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                self.dropout_layers.append(module)
                
        self.logger.info(f"RL Controller initialized. Controlling {len(self.dropout_layers)} dropout layers.")
        
        # Load existing Q-table if available
        q_table_path = self.output_dir / "q_table.npy"
        if q_table_path.exists():
            try:
                self.q_table = np.load(q_table_path)
                self.logger.info("Loaded existing Q-table.")
            except Exception as e:
                self.logger.warning(f"Failed to load Q-table: {e}")

    def _get_gap_state(self, train_loss, val_loss):
        gap = val_loss - train_loss
        if gap < 0.02:
            return 0 # Low Overfitting
        elif gap < 0.10:
            return 1 # Medium Overfitting
        else:
            return 2 # High Overfitting

    def _get_trend_state(self, current_val, prev_val, mode='min'):
        delta = current_val - prev_val
        threshold = 0.001
        
        if mode == 'min': # For Loss
            if delta < -threshold: return 0 # Improving
            elif delta > threshold: return 2 # Degrading
            else: return 1 # Stable
        else: # For Metric (Max)
            if delta > threshold: return 0 # Improving
            elif delta < -threshold: return 2 # Degrading
            else: return 1 # Stable

    def _get_state(self, train_loss, val_loss, val_metric):
        gap_state = self._get_gap_state(train_loss, val_loss)
        loss_trend = self._get_trend_state(val_loss, self.prev_val_loss, mode='min')
        metric_trend = self._get_trend_state(val_metric, self.prev_val_metric, mode='max')
        return (gap_state, loss_trend, metric_trend)

    def _choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            gap, loss_trend, metric_trend = state
            # Add some noise to break ties
            q_values = self.q_table[gap, loss_trend, metric_trend]
            return np.argmax(q_values + np.random.uniform(0, 1e-6, size=len(q_values)))

    def _apply_action(self, action):
        action_name = "Maintain"
        
        if action == 1: # Increase Regularization
            action_name = "Increase Reg"
            self._adjust_dropout(1.1)
            self._adjust_weight_decay(1.2)
        elif action == 2: # Decrease Regularization
            action_name = "Decrease Reg"
            self._adjust_dropout(0.9)
            self._adjust_weight_decay(0.8)
        elif action == 3: # Increase Augmentation
            action_name = "Increase Aug"
            self._adjust_augmentation(1.2)
        elif action == 4: # Decrease Augmentation
            action_name = "Decrease Aug"
            self._adjust_augmentation(0.8)
            
        return action_name

    def _adjust_dropout(self, factor):
        for layer in self.dropout_layers:
            new_p = layer.p * factor
            new_p = max(self.min_dropout, min(self.max_dropout, new_p))
            layer.p = new_p

    def _adjust_weight_decay(self, factor):
        for group in self.optimizer.param_groups:
            if 'weight_decay' in group:
                new_wd = group['weight_decay'] * factor
                new_wd = max(self.min_wd, min(self.max_wd, new_wd))
                group['weight_decay'] = new_wd
                
    def _adjust_augmentation(self, factor):
        # Adjust random_mask_prob
        if hasattr(self.training_manager, 'random_mask_prob'):
            new_prob = self.training_manager.random_mask_prob * factor
            # If prob is 0, set to small value to allow increase
            if new_prob == 0 and factor > 1:
                new_prob = 0.05
            self.training_manager.random_mask_prob = max(self.min_aug, min(self.max_aug, new_prob))
            
        # Adjust mixup_prob
        if hasattr(self.training_manager, 'mixup_prob'):
            new_prob = self.training_manager.mixup_prob * factor
            if new_prob == 0 and factor > 1:
                new_prob = 0.05
            self.training_manager.mixup_prob = max(self.min_aug, min(self.max_aug, new_prob))

    def step(self, train_metrics, val_metrics, epoch):
        train_loss = train_metrics['loss']
        val_loss = val_metrics['val_loss']
        # Use MCC as primary metric, fallback to F1 or Acc
        val_metric = val_metrics.get('val_mcc', val_metrics.get('val_f1', val_metrics.get('val_acc', 0)))
        
        current_state = self._get_state(train_loss, val_loss, val_metric)
        
        # Calculate Reward for previous action
        reward = 0
        if self.prev_state is not None:
            # Reward components
            metric_improvement = val_metric - self.prev_val_metric
            gap = val_loss - train_loss
            
            # 1. Reward for Metric Improvement (Primary Goal)
            # Scale up to make it significant
            reward += metric_improvement * 10.0 
            
            # 2. Penalty for Overfitting Gap
            # If gap is large (>0.1), penalize.
            if gap > 0.1:
                reward -= (gap - 0.1) * 5.0
            
            # 3. Small penalty for degrading loss
            if val_loss > self.prev_val_loss:
                reward -= 0.1
            
            # Update Q-Table
            prev_gap, prev_lt, prev_mt = self.prev_state
            curr_gap, curr_lt, curr_mt = current_state
            
            best_next_action = np.argmax(self.q_table[curr_gap, curr_lt, curr_mt])
            td_target = reward + self.discount_factor * self.q_table[curr_gap, curr_lt, curr_mt, best_next_action]
            td_error = td_target - self.q_table[prev_gap, prev_lt, prev_mt, self.prev_action]
            
            self.q_table[prev_gap, prev_lt, prev_mt, self.prev_action] += self.learning_rate * td_error
            
            self.logger.info(f"RL Step: Reward={reward:.4f}, TD_Error={td_error:.4f}")

        # Choose new action
        action = self._choose_action(current_state)
        action_name = self._apply_action(action)
        
        # Log
        current_dropout = self.dropout_layers[0].p if self.dropout_layers else 0
        current_wd = self.optimizer.param_groups[0].get('weight_decay', 0)
        current_mask = getattr(self.training_manager, 'random_mask_prob', 0)
        current_mixup = getattr(self.training_manager, 'mixup_prob', 0)
        
        self.logger.info(f"RL State: Gap={current_state[0]}, LossTrend={current_state[1]}, MetricTrend={current_state[2]}")
        self.logger.info(f"RL Action: {action_name}")
        self.logger.info(f"  Params: Drop={current_dropout:.3f}, WD={current_wd:.5f}, Mask={current_mask:.3f}, Mixup={current_mixup:.3f}")
        
        # Update state
        self.prev_state = current_state
        self.prev_action = action
        self.prev_val_loss = val_loss
        self.prev_val_metric = val_metric
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Save Q-table
        np.save(self.output_dir / "q_table.npy", self.q_table)
