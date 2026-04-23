"""
Model management utilities.
Responsible for model creation and optimizer/scheduler setup.
"""

import logging
import torch

from fluxsite.models.acetylation_predictor import AcetylationPredictor, DualBranchFusionPredictor
from .optimizer_utils import create_scheduler


class ModelManager:
    """Model manager responsible for creating the model and optimizer."""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
    
    def create_model(self):
        """Create a model instance."""
        logger = logging.getLogger("model_manager")

        core_model_type = self.config.get('core_model_type', 'ptm_bert_bilstm')
        if core_model_type == 'dual_branch_fusion':
            logger.info("Creating DualBranchFusionPredictor for K-fold training.")
            model = DualBranchFusionPredictor(config=self.config)
        else:
            logger.info("Creating AcetylationPredictor (core_model_type=%s) for K-fold training.", core_model_type)
            seq_encoder_type = self.config.get('seq_encoder_type', 'esm2_t33_650M_UR50D')
            struct_encoder_type = self.config.get('struct_encoder_type', 'esm_if1_gvp4_t16_142M_UR50')
            model = AcetylationPredictor(
                config=self.config,
                seq_encoder_type=seq_encoder_type,
                struct_encoder_type=struct_encoder_type,
                core_model_type=core_model_type,
                use_moe=self.config.get('use_moe', False),
                use_reinforcement=self.config.get('use_reinforcement', False),
                use_interpretability=self.config.get('use_interpretability', True),
                use_precomputed_features=self.config.get('use_precomputed_features', True)
            )

        # Move model to device
        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Model created. Total parameters: %s; trainable parameters: %s", f"{total_params:,}", f"{trainable_params:,}")
        self._print_model_info()

        return model
    
    def _print_model_info(self):
        """Print model information."""
        print("Model initialized with the following configuration:")
        print(f"- Reinforcement learning: {self.config.get('use_reinforcement', False)}")
        print(f"- Mixture-of-Experts: {self.config.get('use_moe', False)}")
        print(f"- Integration method: {self.config.get('integration_method', 'serial')}")
        print(f"- Layer normalization: {self.config.get('use_layer_norm', False)}")
        print(f"- Dropout: {self.config.get('dropout', 0.3)}")
    
    def create_optimizer_and_scheduler(self, model, train_loader):
        """Create optimizer and scheduler."""
        print("Creating optimizer and scheduler.")
        
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer, train_loader)
        
        return optimizer, scheduler
    
    def _create_optimizer(self, model):
        """Create optimizer."""
        # Parameter groups: apply different learning rates to different components
        param_groups = self._create_parameter_groups(model)
        
        optimizer_type = self.config.get('optimizer_type', 'adamw')
        
        if optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-2),
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-2)
            )
        elif optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-2),
                momentum=0.9
            )
        else:
            print(f": {optimizer_type},Use AdamW")
            optimizer = torch.optim.AdamW(param_groups)
        
        print(f": {optimizer_type}, Parameters: {len(param_groups)}")
        return optimizer
    
    def _create_parameter_groups(self, model):
        """Create parameter groups."""
        return [{'params': model.parameters()}]
    
    def _create_scheduler(self, optimizer, train_loader):
        """Create learning-rate scheduler."""
        scheduler, scheduler_type = create_scheduler(
            optimizer,
            self.config,
            train_loader=train_loader,
            grad_accum_steps=self.config.get('grad_accum_steps', 1)
        )

        if scheduler_type and scheduler is not None:
            print(f": {scheduler_type}")
        elif scheduler_type:
            print(f" Config {scheduler_type}, Create ")
        else:
            print(" Use ")

        # Keep normalized scheduler name for later use
        self.config['scheduler_normalized'] = scheduler_type

        return scheduler
