"""
Configuration management module.
Responsible for loading, validating, merging, and saving configuration files.
"""

import os
import json
import argparse
from typing import Dict, Any, Optional


class ConfigManager:
    """Configuration manager for loading, validating, and dynamically adjusting configuration values."""
    
    def __init__(self, config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None):
        self.config_path = config_path
        self.args = args
        self.config = self._load_config()
        self._validate_config()
        self._merge_args()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load a configuration file."""
        if self.config_path and os.path.exists(self.config_path):
            print(f" Config LoadParameters: {self.config_path}")
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        else:
            print("Use ConfigParameters")
            config = self._get_default_config()

        # Flatten nested configs into the top level
        if 'model_config' in config and isinstance(config['model_config'], dict):
            for key, value in config['model_config'].items():
                if key not in config:  # Avoid overriding existing top-level keys
                    config[key] = value

        if 'training_config' in config and isinstance(config['training_config'], dict):
            for key, value in config['training_config'].items():
                if key not in config:
                    config[key] = value

        if 'scheduler_config' in config and isinstance(config['scheduler_config'], dict):
            for key, value in config['scheduler_config'].items():
                if key not in config:
                    config[key] = value

        if 'early_stopping_config' in config and isinstance(config['early_stopping_config'], dict):
            for key, value in config['early_stopping_config'].items():
                if key not in config:
                    config[key] = value

        if 'regularization_config' in config and isinstance(config['regularization_config'], dict):
            for key, value in config['regularization_config'].items():
                if key not in config:
                    config[key] = value

        if 'data_config' in config and isinstance(config['data_config'], dict):
            for key, value in config['data_config'].items():
                if key not in config:
                    config[key] = value

        if 'augmentation_config' in config and isinstance(config['augmentation_config'], dict):
            for key, value in config['augmentation_config'].items():
                if key not in config:
                    config[key] = value

        if 'evaluation_config' in config and isinstance(config['evaluation_config'], dict):
            for key, value in config['evaluation_config'].items():
                if key not in config:
                    config[key] = value

        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration values."""
        return {
            # Model architecture
            "hidden_dim": 512,
            "projection_dim": 512,
            "bert_hidden_dim": 768,
            "force_simple_bert": False,
            "dropout": 0.9,
            "use_layer_norm": True,
            
            # Training hyperparameters
            "learning_rate": 2e-5,  # Tuned initial learning rate for cosine-style schedulers
            "weight_decay": 0.3,
            "epochs": 200,
            "batch_size": 64,
            "grad_accum_steps": 2,
            
            # Optimizer and scheduler
            "optimizer": "adamw",  # adamw, adam, sgd
            "scheduler": "cosine_with_warmup",  # cosine, cosine_with_warmup, cosine_restarts, step, plateau, exponential, none
            "warmup_steps": 500,  # Warmup steps (for cosine_with_warmup)
            "scheduler_cycles": 0.5,  # Number of cycles for cosine schedulers
            "scheduler_T_max": 150,  # T_max for CosineAnnealingLR; uses epochs when None
            "scheduler_eta_min": 1e-6,  # Minimum LR for CosineAnnealingLR
            "scheduler_T_0": 50,  # Initial restart period for CosineAnnealingWarmRestarts
            "scheduler_T_mult": 2,  # Period multiplier for CosineAnnealingWarmRestarts
            "scheduler_step_size": 30,  # Step size for StepLR
            "scheduler_gamma": 0.1,  # Decay factor for StepLR/ExponentialLR
            "scheduler_patience": 15,  # Patience for ReduceLROnPlateau
            "scheduler_factor": 0.5,  # Factor for ReduceLROnPlateau
            "scheduler_threshold": 1e-4,  # Threshold for ReduceLROnPlateau
            "scheduler_mode": "max",  # ReduceLROnPlateau mode (min/max)
            "scheduler_min_lr": 1e-7,  # Minimum LR for ReduceLROnPlateau
            
            # Optimizer parameters
            "optimizer_betas": [0.9, 0.999],
            "optimizer_eps": 1e-8,
            "optimizer_momentum": 0.9,
            "optimizer_nesterov": True,
            
            # Regularization and early stopping
            "gradient_clip_val": 1.1,
            "use_early_stopping": True,
            "early_stopping_patience": 15,
            "early_stopping_min_delta": 0.0001,
            "early_stopping_monitor": "val_loss",  # val_loss, val_acc, val_f1, val_auc
            "early_stopping_mode": "min",
            "restore_best_weights": True,
            "early_stopping_verbose": True,

            # Dropout
            "input_dropout": 0.2,
            "hidden_dropout": 0.3,
            "output_dropout": 0.1,
            "attention_dropout": 0.1,
            "lstm_dropout": 0.2,
            "adaptive_dropout": True,
            "dropout_schedule": "linear",  # linear, exponential, cosine
            
            # Data augmentation
            "use_enhanced_augmentation": True,
            "augment_prob": 0.7,
            "feature_noise_level": 0.25,
            "feature_mask_prob": 0.35,
            
            # Loss function
            "use_focal_loss": True,
            "focal_alpha": 0.3,
            "focal_gamma": 2.0,
            "use_label_smoothing": True,
            "label_smoothing": 0.1,
            "loss_combination_weight": 0.8,
            
            # ================== Advanced modules ==================
            
            # Advanced module toggles
            "use_reinforcement": False,
            "use_moe": False,
            "integration_method": "ensemble",  # serial, ensemble, dynamic

            # Reinforcement learning configuration
            "reinforcement_learning": {
                "reward_type": "performance_based",  # performance_based, confidence_based, hybrid
                "exploration_epsilon": 0.1,
                "policy_learning_rate": 1e-5,
                "discount_factor": 0.95,
                "experience_buffer_size": 10000,
                "ppo_clip_param": 0.2,
                "value_coef": 0.5,
                "entropy_coef": 0.01,
                "num_decision_steps": 3
            },

            # Mixture-of-experts configuration
            "mixture_of_experts": {
                "num_experts": 8,
                "top_k": 2,
                "routing_method": "top_k",  # top_k, sparse
                "expert_type": "feed_forward",  # feed_forward, transformer
                "expert_hidden_dim": 256,
                "load_balancing_loss_coef": 0.01
            },
            
            # Multi-module integration
            "module_integration": {
                "fusion_method": "attention",  # attention, concatenation, addition, gating
                "attention_heads": 8,
                "fusion_hidden_dim": 512,
                "module_weights": [1.0, 1.0, 1.0, 1.0],
                "adaptive_weighting": True,
                "weight_learning_rate": 1e-5,
                "temperature": 1.0,
                "use_residual_connection": True,
                "dropout_integration": 0.1,
                "cross_module_attention": True,
                "synchronization_method": "gradual",  # gradual, alternating, simultaneous
                "module_communication": True,
            },
            
            # Data processing
            "use_structure": True,
            "use_interpretability": True,
            "use_class_weights": True,
            "pos_weight": 1.0,
            "val_ratio": 0.2,
            "test_ratio": 0.1,
            "stratified_split": True,
            "balanced_sampling": True,
            "optimize_threshold": True,
            "threshold_metric": "f1",
            "threshold_min_precision": 0.0,
            "threshold_min_specificity": 0.0,
            "threshold_min_recall": 0.0,
            "target_ptm_type": "phosphorylation",
            
            # ================== K-Fold cross-validation ==================
            "use_kfold": True,
            "kfold_splits": 5,
            "kfold_stratified": True,
            "kfold_shuffle": True,
            "kfold_random_state": 42,
            "kfold_save_models": True,
            "kfold_ensemble_method": "average",  # average, voting, weighted
            "kfold_early_stopping": True,
            "kfold_parallel": False,
            "kfold_save_predictions": True,
            "kfold_detailed_metrics": True,
        }
    
    def _validate_config(self):
        """Validate configuration values."""
        print("ValidationConfigParameters...")
        
        # Validate learning rate settings
        if self.config.get('max_learning_rate', 0) <= self.config.get('learning_rate', 0):
            print("Warning: max_learning_rate learning_rate, ")
            self.config['max_learning_rate'] = self.config.get('learning_rate', 1e-4) * 5
        
        # Validate dropout range
        dropout = self.config.get('dropout', 0.2)
        if not 0 <= dropout <= 1:
            print(f"Warning: dropout {dropout} [0,1], 0.2")
            self.config['dropout'] = 0.2
        
        # Validate batch size
        batch_size = self.config.get('batch_size', 32)
        if batch_size <= 0:
            print(f"Warning: batch_size {batch_size}, 32")
            self.config['batch_size'] = 32
        
        # Validate epoch count
        epochs = self.config.get('epochs', 50)
        if epochs <= 0:
            print(f"Warning: epochs {epochs}, 50")
            self.config['epochs'] = 50
        
        print("ConfigParametersValidationCompleted")
    
    def _merge_args(self):
        """Merge CLI arguments into the configuration."""
        if self.args is None:
            return
            
        specified_args = getattr(self.args, '_specified_args', None)
        # CLI arguments override config file values when provided
        arg_to_config_mapping = {
            # Base parameters
            'epochs': 'epochs',
            'batch_size': 'batch_size',
            'grad_accum_steps': 'grad_accum_steps',
            'optimizer': 'optimizer',
            'scheduler': 'scheduler',
            'learning_rate': 'learning_rate',
            'weight_decay': 'weight_decay',
            'warmup_steps': 'warmup_steps',
            'use_reinforcement': 'use_reinforcement',
            'use_moe': 'use_moe',
            'integration_method': 'integration_method',
            'target_ptm_type': 'target_ptm_type',
            'core_model_type': 'core_model_type',
            'seq_encoder_type': 'seq_encoder_type',
            'struct_encoder_type': 'struct_encoder_type',

            # K-Fold cross-validation
            'use_kfold': 'use_kfold',
            'kfold_splits': 'kfold_splits',
            'kfold_stratified': 'kfold_stratified',
            
            # Reinforcement learning parameters
            'reward_type': ['reinforcement_learning', 'reward_type'],
            'exploration_epsilon': ['reinforcement_learning', 'exploration_epsilon'],
            'policy_learning_rate': ['reinforcement_learning', 'policy_learning_rate'],
            'discount_factor': ['reinforcement_learning', 'discount_factor'],
            
            # Mixture-of-experts parameters
            'num_experts': ['mixture_of_experts', 'num_experts'],
            'top_k': ['mixture_of_experts', 'top_k'],
            'routing_method': ['mixture_of_experts', 'routing_method'],
            'expert_type': ['mixture_of_experts', 'expert_type'],
            'load_balancing_weight': ['mixture_of_experts', 'load_balancing_weight'],
            
            # Integration parameters
            'fusion_method': ['module_integration', 'fusion_method'],
            'attention_heads': ['module_integration', 'attention_heads'],
            'adaptive_weighting': ['module_integration', 'adaptive_weighting'],
        }
        
        for arg_name, config_key in arg_to_config_mapping.items():
            if not hasattr(self.args, arg_name):
                continue
            if specified_args is not None and arg_name not in specified_args:
                continue

            arg_value = getattr(self.args, arg_name)
            if arg_value is None:
                continue

            if isinstance(config_key, list):
                # Nested config
                parent_key, child_key = config_key
                if parent_key not in self.config or not isinstance(self.config[parent_key], dict):
                    self.config[parent_key] = {}

                old_value = self.config[parent_key].get(child_key)
                if old_value != arg_value:
                    print(f" Parameters: {parent_key}.{child_key} {old_value} -> {arg_value}")
                    self.config[parent_key][child_key] = arg_value
            else:
                # Top-level config
                old_value = self.config.get(config_key)
                if old_value != arg_value:
                    print(f" Parameters: {config_key} {old_value} -> {arg_value}")
                    self.config[config_key] = arg_value

    def get_config(self) -> Dict[str, Any]:
        """Return the final merged configuration."""
        return self.config
    
    def print_config(self):
        """Print the final configuration in a grouped format."""
        print("="*80)
        print(" ConfigParameters:")
        print("="*80)
        
        # Group and print by category
        categories = {
            "Model ": ["hidden_dim", "projection_dim", "bert_hidden_dim", "force_simple_bert", "dropout", "use_layer_norm"],
            "Training Parameters": ["learning_rate", "weight_decay", "epochs", "batch_size", "grad_accum_steps"],
            " ": ["optimizer", "optimizer_betas", "optimizer_eps", "optimizer_momentum"],
            " Parameters": ["scheduler", "warmup_steps", "scheduler_cycles", "scheduler_T_max"],
            " ": ["gradient_clip_val", "use_early_stopping", "early_stopping_patience"],
            "Data ": ["use_enhanced_augmentation", "augment_prob", "feature_noise_level"],
            " ": ["use_focal_loss", "focal_alpha", "focal_gamma", "use_label_smoothing"],
            " ": ["use_reinforcement", "use_moe", "integration_method"],
            "Data ": ["use_structure", "use_interpretability", "val_ratio", "test_ratio", "target_ptm_type"]
        }
        
        for category, keys in categories.items():
            print(f"\n{category}:")
            for key in keys:
                if key in self.config:
                    print(f"  {key}: {self.config[key]}")
        
        # Print advanced module configs
        advanced_modules = ["reinforcement_learning", "mixture_of_experts", "module_integration"]
        
        for module_name in advanced_modules:
            if module_name in self.config:
                print(f"\n{module_name.replace('_', ' ').title()} Parameters:")
                module_config = self.config[module_name]
                for key, value in module_config.items():
                    print(f"  {key}: {value}")
        
        print("="*80)
    
    def save_config(self, output_path: str):
        """Save the configuration to disk."""
        # Create a JSON-serializable copy and skip non-serializable objects
        serializable_config = {}
        non_serializable_keys = ['tensorboard_writer']
        
        for key, value in self.config.items():
            if key not in non_serializable_keys:
                # Check JSON serializability
                try:
                    json.dumps(value)
                    serializable_config[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable items
                    print(f"Skip Config: {key}")
        
        with open(output_path, 'w') as f:
            json.dump(serializable_config, f, indent=2)
        print(f"Config Save: {output_path}")


def demonstrate_config_usage():
    """Demonstrate how to use advanced configuration options."""
    print("\n" + "="*80)
    print(" Config Use ")
    print("="*80)
    
    print("\n1. Use (UseAdamW + Cosine with Warmup):")
    print("   python train.py")
    
    print("\n2. Enable:")
    print("   python train.py \\")
    print("          --use_reinforcement \\")
    print("          --reward_type performance_based \\")
    print("          --exploration_epsilon 0.1 \\")
    print("          --policy_learning_rate 1e-5")
    
    print("\n3. Enable:")
    print("   python train.py \\")
    print("          --use_moe \\")
    print("          --num_experts 8 \\")
    print("          --top_k 2 \\")
    print("          --routing_method top_k \\")
    print("          --expert_type feed_forward")
    
    print("\n4. Training ():")
    print("   python train.py \\")
    print("          --use_moe --use_reinforcement \\")
    print("          --integration_method ensemble \\")
    print("          --fusion_method attention \\")
    print("          --attention_heads 8 \\")
    print("          --adaptive_weighting \\")
    print("          --num_experts 6 \\")
    print("          --top_k 2 \\")
    print("          --reward_type hybrid \\")
    print("          --optimizer adamw \\")
    print("          --scheduler cosine_with_warmup \\")
    print("          --learning_rate 1e-4")
    
    print("\n5. UseConfig Training:")
    print("   python train.py \\")
    print("          --config config/advanced_modules_config.json \\")
    print("          --num_experts 12 --top_k 3")  # Override parameters from the config file

    print("\n6. TestConfig:")
    print("   python train.py \\")
    print("          --use_moe --num_experts 4 --top_k 2 \\")
    print("          --batch_size 16 --epochs 10")

    print("\n" + "="*80)
    print(" Parameters:")
    print("="*80)
    
    module_descriptions = {
        " ": {
            "reward_type": ":performance_based(),confidence_based(),hybrid()",
            "exploration_epsilon": " Probabilities:, 0.01-0.3",
            "policy_learning_rate": ": 1-2 ",
            "discount_factor": ":, 0.9-0.99",
            "experience_buffer_size": ":, "
        },
        " ": {
            "num_experts": ":, 4-64",
            "routing_method": ":top_k(top-k),sparse(),soft(),switch()",
            "top_k": ":, 1-4",
            "load_balancing_weight": ": Use, 0.001-0.1",
            "expert_specialization": ":domain(),feature(),task(),adaptive()"
        },
        " ": {
            "fusion_method": ":attention(),concatenation(),addition(),gating()",
            "attention_heads": ":, 4-16",
            "adaptive_weighting": ": ",
            "temperature": ": "
        }
    }
    
    for module, params in module_descriptions.items():
        print(f"\n{module}:")
        for param, desc in params.items():
            print(f"  • {param}: {desc}")

    print("\n" + "="*80)
    print("Parameters:")
    print("="*80)
    print("\n•: Use Config, ")
    print("•: num_experts,batch_size, grad_accum_steps")
    print("•: learning_rate,weight_decay,schedulerParameters")
    print("•: Training, ")
    print("• Parameters: UseConfig Test Parameters ")
    print("\n" + "="*80)
