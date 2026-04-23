"""
Adaptive dropout module.
Dynamically adjusts dropout rates based on training progress and validation performance.
"""

import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)


class AdaptiveDropout(nn.Module):
    """
    Adaptive Dropout layer that can adjust the dropout rate based on training progress.
    """
    
    def __init__(self, initial_dropout=0.3, min_dropout=0.1, max_dropout=0.5, 
                 schedule='linear', warmup_epochs=10, total_epochs=50):
        """
        Initialize adaptive dropout.
        
        Args:
            initial_dropout (float): Initial dropout rate.
            min_dropout (float): Minimum dropout rate.
            max_dropout (float): Maximum dropout rate.
            schedule (str): Schedule strategy ('linear', 'exponential', 'cosine', 'adaptive').
            warmup_epochs (int): Number of warmup epochs.
            total_epochs (int): Total training epochs.
        """
        super().__init__()
        self.initial_dropout = initial_dropout
        self.min_dropout = min_dropout
        self.max_dropout = max_dropout
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
        # Current dropout state
        self.current_dropout = initial_dropout
        self.current_epoch = 0
        
        # Performance history (for adaptive schedule)
        self.performance_history = []
        self.patience_counter = 0
        self.best_performance = float('inf')
        
        logger.info(
            "Initialized AdaptiveDropout: initial=%.4f, range=[%.4f, %.4f], schedule=%s",
            float(initial_dropout),
            float(min_dropout),
            float(max_dropout),
            str(schedule),
        )
    
    def update_epoch(self, epoch, val_loss=None):
        """
        Update the current epoch and dropout rate.
        
        Args:
            epoch (int): Current epoch.
            val_loss (float): Validation loss (used for adaptive updates).
        """
        self.current_epoch = epoch
        
        if self.schedule == 'adaptive' and val_loss is not None:
            self._adaptive_update(val_loss)
        else:
            self._scheduled_update()
    
    def _scheduled_update(self):
        """Update dropout rate based on the configured schedule."""
        progress = min(self.current_epoch / self.total_epochs, 1.0)
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup: linearly decrease from max to initial value
            warmup_progress = self.current_epoch / self.warmup_epochs
            self.current_dropout = self.max_dropout - (self.max_dropout - self.initial_dropout) * warmup_progress
        else:
            # Main training phase
            adjusted_progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            adjusted_progress = min(adjusted_progress, 1.0)
            
            if self.schedule == 'linear':
                # Linear decay
                self.current_dropout = self.initial_dropout - (self.initial_dropout - self.min_dropout) * adjusted_progress
            elif self.schedule == 'exponential':
                # Exponential decay
                decay_factor = math.exp(-3 * adjusted_progress)  # Decay factor
                self.current_dropout = self.min_dropout + (self.initial_dropout - self.min_dropout) * decay_factor
            elif self.schedule == 'cosine':
                # Cosine decay
                self.current_dropout = self.min_dropout + (self.initial_dropout - self.min_dropout) *\
                                     (1 + math.cos(math.pi * adjusted_progress)) / 2
            else:
                # Keep initial value
                self.current_dropout = self.initial_dropout
        
        # Clamp to valid range
        self.current_dropout = max(self.min_dropout, min(self.max_dropout, self.current_dropout))
    
    def _adaptive_update(self, val_loss):
        """Adaptively update dropout rate based on validation performance."""
        self.performance_history.append(val_loss)

        # Check whether performance improved
        if val_loss < self.best_performance:
            self.best_performance = val_loss
            self.patience_counter = 0
            # Improvement: slightly reduce dropout
            self.current_dropout = max(self.min_dropout, self.current_dropout * 0.95)
        else:
            self.patience_counter += 1
            # No improvement: increase dropout to reduce overfitting
            if self.patience_counter >= 3:
                # Increase dropout, but cap at 0.95 as a safety margin
                self.current_dropout = min(0.95, self.current_dropout * 1.05)
                self.patience_counter = 0

        # Clamp to [0, 1]
        self.current_dropout = max(0.0, min(1.0, self.current_dropout))
        # Clamp to configured range
        self.current_dropout = max(self.min_dropout, min(self.max_dropout, self.current_dropout))
    
    def forward(self, x):
        """Forward pass."""
        if self.training:
            return nn.functional.dropout(x, p=self.current_dropout, training=True)
        else:
            return x
    
    def get_current_dropout(self):
        """Get the current dropout rate."""
        return self.current_dropout
    
    def get_dropout_info(self):
        """Get a summary of current dropout state."""
        return {
            'current_dropout': self.current_dropout,
            'current_epoch': self.current_epoch,
            'schedule': self.schedule,
            'performance_history_length': len(self.performance_history),
            'best_performance': self.best_performance if self.performance_history else None
        }


class DropoutScheduler:
    """
    Dropout scheduler managing all AdaptiveDropout layers in a model.
    """
    
    def __init__(self, model, config):
        """
        Initialize the dropout scheduler.
        
        Args:
            model: Model that may contain AdaptiveDropout layers.
            config: Configuration dictionary.
        """
        self.model = model
        self.config = config
        self.adaptive_dropouts = []
        
        # Find all AdaptiveDropout layers
        self._find_adaptive_dropouts()
        
        logger.info("Detected %d AdaptiveDropout layer(s).", len(self.adaptive_dropouts))
    
    def _find_adaptive_dropouts(self):
        """Find all AdaptiveDropout layers in the model."""
        for name, module in self.model.named_modules():
            if isinstance(module, AdaptiveDropout):
                self.adaptive_dropouts.append((name, module))
    
    def step(self, epoch, val_loss=None):
        """
        Update all AdaptiveDropout layers.
        
        Args:
            epoch (int): Current epoch.
            val_loss (float): Validation loss.
        """
        for name, dropout_layer in self.adaptive_dropouts:
            dropout_layer.update_epoch(epoch, val_loss)
            
            if epoch % 10 == 0:  # Log every 10 epochs
                info = dropout_layer.get_dropout_info()
                logger.info("Dropout %s: current=%.4f", name, float(info["current_dropout"]))
    
    def get_all_dropout_rates(self):
        """Get current dropout rates for all AdaptiveDropout layers."""
        rates = {}
        for name, dropout_layer in self.adaptive_dropouts:
            rates[name] = dropout_layer.get_current_dropout()
        return rates
    
    def set_dropout_rates(self, rates_dict):
        """Manually set dropout rates."""
        for name, dropout_layer in self.adaptive_dropouts:
            if name in rates_dict:
                dropout_layer.current_dropout = rates_dict[name]


def create_adaptive_dropout_layer(config, layer_name="default"):
    """
    Factory function to create an AdaptiveDropout layer.
    
    Args:
        config: Configuration dictionary.
        layer_name: Layer name used to select a specific configuration.
        
    Returns:
        AdaptiveDropout: Adaptive dropout layer.
    """
    # Base configuration
    base_dropout = config.get('dropout', 0.3)
    
    # Per-layer overrides
    layer_configs = {
        'input': {
            'initial_dropout': config.get('input_dropout', base_dropout * 0.7),
            'min_dropout': 0.05,
            'max_dropout': base_dropout
        },
        'hidden': {
            'initial_dropout': config.get('hidden_dropout', base_dropout),
            'min_dropout': 0.1,
            'max_dropout': base_dropout * 1.5
        },
        'output': {
            'initial_dropout': config.get('output_dropout', base_dropout * 0.5),
            'min_dropout': 0.02,
            'max_dropout': base_dropout * 0.8
        },
        'attention': {
            'initial_dropout': config.get('attention_dropout', base_dropout * 0.3),
            'min_dropout': 0.01,
            'max_dropout': base_dropout * 0.5
        },
        'lstm': {
            'initial_dropout': config.get('lstm_dropout', base_dropout * 0.7),
            'min_dropout': 0.05,
            'max_dropout': base_dropout
        }
    }
    
    # Get layer-specific configuration or fall back to defaults
    layer_config = layer_configs.get(layer_name, {
        'initial_dropout': base_dropout,
        'min_dropout': 0.1,
        'max_dropout': min(0.95, base_dropout * 1.1)  # Ensure max_dropout does not exceed 0.95
    })

    # Ensure max_dropout does not exceed 1.0
    layer_config['max_dropout'] = min(0.95, layer_config['max_dropout'])

    return AdaptiveDropout(
        initial_dropout=layer_config['initial_dropout'],
        min_dropout=layer_config['min_dropout'],
        max_dropout=layer_config['max_dropout'],
        schedule=config.get('dropout_schedule', 'linear'),
        warmup_epochs=config.get('warmup_epochs', 10),
        total_epochs=config.get('epochs', 50)
    )
