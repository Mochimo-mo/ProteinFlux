import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

try:
    from fluxsite.utils.adaptive_dropout import create_adaptive_dropout_layer
except ImportError:
    create_adaptive_dropout_layer = None
from .dual_branch_modules import (
    SequenceTower,
    StructureTower,
    GatedFusionModule,
    SequenceContextEncoder,
    StructureContextEncoder,
    CrossModalFusionLayer,
    AdaptiveGating,
    ProjectionHead,
    masked_mean
)
from .enhanced_components import (
    EnhancedAttentionModule,
    MultiScaleFusionNetwork,
    DynamicFeatureSelector
)
from .encoders import (
    MultiModalFusion,
    ProjectionLayer,
    ProteinSequenceEncoder,
    ProteinStructureEncoder,
    AcetylationInfoEncoder
)
from .reinforcement import RLClassifier, BERTBiLSTMRLClassifier
from fluxsite.utils.enhanced_loss import FocalLoss, SupervisedContrastiveLoss
import numpy as np
import math
import warnings
import logging
from typing import List, Optional

# Import matplotlib and seaborn only when needed to avoid dependency issues
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError as e:
    HAS_PLOTTING = False
    plt = None
    sns = None
    logging.warning("Failed to import visualization libraries: %s", e)
except Exception as e:
    HAS_PLOTTING = False
    plt = None
    sns = None
    logging.warning("Error importing visualization libraries: %s", e)

# Set up logging
logger = logging.getLogger("acetylation_predictor")


class InterpretabilityModule(nn.Module):
    """Interpretability module for analyzing why the model makes a prediction."""
    
    def __init__(self, input_dim, hidden_dim=128):
        """
        Initialize the interpretability module.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden dimension.
        """
        super().__init__()
        
        self.importance_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, return_importance=False):
        """
        Forward pass.

        Args:
            x: Input features [batch_size, seq_len, input_dim].
            return_importance (bool): Whether to return importance scores.

        Returns:
            torch.Tensor or tuple: Context vector, optionally with attention weights.
        """
        # Importance score per position
        importance = self.importance_network(x).squeeze(-1)  # [batch_size, seq_len]
        
        # Attention weights over residue positions
        attention_weights = F.softmax(importance, dim=1).unsqueeze(2)  # [batch_size, seq_len, 1]
        
        # Weighted sum to obtain context vector
        context = torch.sum(x * attention_weights, dim=1)  # [batch_size, input_dim]
        
        if return_importance:
            return context, attention_weights.squeeze(2)
        else:
            return context


class AcetylationPredictor(nn.Module):
    """
    Modular acetylation prediction model that can use different encoders and architectures.
    Optimized for PTM prediction by integrating sequence/structure encoders and acetylation-specific features.
    """
    
    def __init__(self, config=None, seq_encoder_type="esm2_t33_650M_UR50D", struct_encoder_type="esm_if1_gvp4_t16_142M_UR50",
                 core_model_type="bert_bilstm", use_moe=False, use_reinforcement=False,
                 use_interpretability=True, use_precomputed_features=True):
        """
        Initialize the acetylation predictor.

        Args:
            config: Configuration object or dict.
            seq_encoder_type: Sequence encoder type.
            struct_encoder_type: Structure encoder type.
            core_model_type: Core model type ('transformer', 'bert_bilstm', 'ptm_bert_bilstm').
            use_moe: Enable Mixture-of-Experts.
            use_reinforcement: Enable reinforcement-learning components.
            use_interpretability: Enable interpretability utilities.
            use_precomputed_features: Use precomputed features (skip encoder loading).
        """
        super().__init__()
        
        self.device = torch.device("cpu")
        logger.info("AcetylationPredictor initialized (device placeholder: %s)", self.device)
        
        # Store configuration
        self.config = config or {}
        self.use_moe = use_moe
        self.use_reinforcement = use_reinforcement
        self.use_interpretability = use_interpretability
        self.use_precomputed_features = use_precomputed_features
        self.use_multitask_loss = False  # Set to True when using the acetylation_transformer core model
        
        # [New] Whether to concatenate one-hot residue encoding (hybrid encoding).
        # Defaults to True when use_explicit_features is enabled.
        self.use_one_hot = self.config.get("use_one_hot", self.config.get("use_explicit_features", False))
        if self.use_one_hot:
            logger.info("EnableHybrid Encoding: One-hot ESM ")

        # Base dimensions
        self.hidden_dim = self.config.get("hidden_dim", 256)
        self.projection_dim = self.config.get("projection_dim", 256)
        self.dropout = self.config.get("dropout", 0.1)
        self.bert_hidden_dim = self.config.get("bert_hidden_dim", 384)
        self.config.setdefault("bert_hidden_dim", self.bert_hidden_dim)

        # Resolve PTM, residue, and position configuration
        ptm_mapping = self.config.get("ptm_type_to_id", {})
        inferred_num_ptm_types = max(1, len(ptm_mapping))
        self.num_ptm_types = max(self.config.get("num_ptm_types", inferred_num_ptm_types), inferred_num_ptm_types)
        self.config["num_ptm_types"] = self.num_ptm_types

        self.num_residue_types = self.config.get("num_residue_types", 35)
        self.config.setdefault("num_residue_types", self.num_residue_types)

        self.max_position_embeddings = self.config.get("max_position_embeddings", 512)
        self.config.setdefault("max_position_embeddings", self.max_position_embeddings)
            
        # Initialize encoders (depending on whether precomputed features are used)
        if self.use_precomputed_features:
            # Use precomputed features; skip encoder loading
            logger.info("Using precomputed features; skipping encoder loading.")
            self.seq_encoder = None
            self.struct_encoder = None
            # Default feature dimensions for precomputed ESM-2 and ESM-IF embeddings
            self.seq_dim = 1280  # ESM-2 t33 650M feature dimension
            
            # [New] Add one-hot dimension when enabled
            if self.use_one_hot:
                self.seq_dim += self.num_residue_types
                logger.info(f"EnableHybrid Encoding,: {self.seq_dim} (1280 + {self.num_residue_types})")

            self.struct_dim = 512  # ESM-IF feature dimension
            self.seq_projection = ProjectionLayer(self.seq_dim, self.projection_dim)
            self.struct_projection = ProjectionLayer(self.struct_dim, self.projection_dim)
            self.use_structure = True
            logger.info("Precomputed feature dimensions - sequence: %d, structure: %d", self.seq_dim, self.struct_dim)
        else:
            # Sequence encoder
            self.seq_encoder = ProteinSequenceEncoder(model_name=seq_encoder_type, freeze=True)
            self.seq_projection = ProjectionLayer(self.seq_encoder.embed_dim, self.projection_dim)
            self.seq_dim = self.seq_encoder.embed_dim
            logger.info("Sequence encoder initialized (dim=%d)", self.seq_dim)

            # Structure encoder (optional; used only when structure data is available)
            try:
                self.struct_encoder = ProteinStructureEncoder(model_name=struct_encoder_type, freeze=True)
                self.struct_projection = ProjectionLayer(self.struct_encoder.embed_dim, self.projection_dim)
                self.struct_dim = self.struct_encoder.embed_dim
                self.use_structure = True
                logger.info("Structure encoder initialized (dim=%d)", self.struct_dim)
            except Exception as e:
                logger.warning("Failed to initialize structure encoder: %s", e)
                self.use_structure = False
                self.struct_dim = 0
        
        # Acetylation-specific feature encoder
        self.acetylation_encoder = AcetylationInfoEncoder(
            input_dim=21,  # 20 amino acids + 1 padding token
            hidden_dim=128,
            output_dim=self.projection_dim
        )
        logger.info("Acetylation-specific feature encoder initialized.")
        
        # Multi-modal fusion
        if self.use_structure:
            self.fusion = MultiModalFusion(feature_dim=self.projection_dim, num_heads=4)
            logger.info("Multi-modal fusion layer initialized.")
        else:
            self.fusion = None
            logger.info("Multi-modal fusion disabled (use_structure=False).")

        # Explicit feature embeddings
        self.position_embedding_layer = nn.Embedding(self.max_position_embeddings, self.projection_dim)
        self.residue_type_embedding = nn.Embedding(self.num_residue_types, self.projection_dim)
        # Note: ptm_type_embedding is kept for backward compatibility and checkpoint loading, but is no longer used
        # in forward to avoid label leakage (see comments in forward()).
        self.ptm_type_embedding = nn.Embedding(self.num_ptm_types, self.projection_dim)
        self.relative_position_projection = nn.Sequential(
            nn.Linear(1, self.projection_dim),
            nn.Tanh()
        )
        self.site_indicator_projection = nn.Sequential(
            nn.Linear(1, self.projection_dim),
            nn.Sigmoid()
        )
        
        # Local sequence projection.
        # Note: when hybrid encoding is enabled, seq_dim includes the one-hot dimension, but we currently apply the
        # one-hot concatenation only to global sequence features.
        local_input_dim = self.seq_dim
        if self.use_one_hot and self.use_precomputed_features:
             local_input_dim = self.seq_dim - self.num_residue_types
             
        self.local_seq_projection = ProjectionLayer(local_input_dim, self.projection_dim)

        # Global context projection (to fuse additional global features).
        # Global features are typically pooled from raw ESM embeddings and do not include one-hot encodings.
        self.global_feature_dim = self.config.get("global_feature_dim", local_input_dim)
        self.global_context_projection = ProjectionLayer(self.global_feature_dim, self.projection_dim)
        
        # Interpretability modules
        if self.use_interpretability:
            self.local_interpretability = InterpretabilityModule(self.projection_dim)
            self.global_interpretability = InterpretabilityModule(self.projection_dim)
            logger.info("Interpretability modules initialized.")
        else:
            self.local_interpretability = None
            self.global_interpretability = None

        # Center-feature regularization (hard masking + curriculum schedule)
        self._init_center_regularization()
        
        # Core model (supports multiple architectures)
        requested_core_model_type = core_model_type
        has_transformer_model = isinstance(globals().get("TransformerModel"), type)
        has_bert_bilstm_model = isinstance(globals().get("BERTBiLSTMModel"), type)
        has_ptm_bert_bilstm_model = isinstance(globals().get("PTMSpecificBERTBiLSTM"), type)
        if core_model_type == "transformer" and not has_transformer_model:
            logger.warning(
                "core_model_type='transformer' is not available in this open-source snapshot; falling back to 'dual_branch_fusion'."
            )
            core_model_type = "dual_branch_fusion"
        elif core_model_type == "bert_bilstm" and not has_bert_bilstm_model:
            logger.warning(
                "core_model_type='bert_bilstm' is not available in this open-source snapshot; falling back to 'dual_branch_fusion'."
            )
            core_model_type = "dual_branch_fusion"
        elif core_model_type == "ptm_bert_bilstm" and not has_ptm_bert_bilstm_model:
            logger.warning(
                "core_model_type='ptm_bert_bilstm' is not available in this open-source snapshot; falling back to 'dual_branch_fusion'."
            )
            core_model_type = "dual_branch_fusion"

        self.requested_core_model_type = requested_core_model_type
        self.core_model_type = core_model_type

        if core_model_type == "transformer":
            if use_moe:
                # Transformer with MoE
                logger.info("UseMoE Transformer Model")
                self.core_model = nn.TransformerEncoder(
                    MoETransformerLayer(
                        d_model=self.projection_dim,
                        nhead=self.config.get("num_heads", 8),
                        num_experts=self.config.get("num_experts", 4),
                        top_k=self.config.get("top_k", 2),
                        dropout=self.dropout
                    ),
                    num_layers=self.config.get("num_layers", 6)
                )
            else:
                # Standard Transformer
                logger.info("Use Transformer Model")
                self.core_model = TransformerModel(
                    input_dim=self.projection_dim,
                    nhead=self.config.get("num_heads", 8),
                    num_layers=self.config.get("num_layers", 6),
                    dim_feedforward=self.config.get("dim_feedforward", 2048),
                    dropout=self.dropout
                )
        elif core_model_type == "bert_bilstm":
            # BERT + BiLSTM hybrid architecture
            logger.info("UseBERT+BiLSTM Model")
            self.core_model = BERTBiLSTMModel(
                input_dim=self.projection_dim,
                bert_model_name=self.config.get("bert_model_name", "bert-base-uncased"),
                lstm_hidden_dim=self.config.get("lstm_hidden_dim", 256),
                lstm_num_layers=self.config.get("lstm_num_layers", 2),
                fusion_dim=self.config.get("fusion_dim", 512),
                dropout=self.dropout,
                use_bert_pooler=self.config.get("use_bert_pooler", False),
                freeze_bert=self.config.get("freeze_bert", False),
                config=self.config,
                transformer_layers=self.config.get("transformer_layers", 6),
                struct_dim=self.projection_dim if self.use_structure else 0
            )
        elif core_model_type == "ptm_bert_bilstm":
            # PTM-specific BERT + BiLSTM architecture
            logger.info("UsePTM BERT+BiLSTM Model")
            self.core_model = PTMSpecificBERTBiLSTM(
                input_dim=self.projection_dim,
                num_ptm_types=self.config.get("num_ptm_types", 1),
                aa_embedding_dim=self.config.get("aa_embedding_dim", 64),
                position_embedding_dim=self.config.get("position_embedding_dim", 32),
                bert_model_name=self.config.get("bert_model_name", "bert-base-uncased"),
                lstm_hidden_dim=self.config.get("lstm_hidden_dim", 256),
                lstm_num_layers=self.config.get("lstm_num_layers", 2),
                fusion_dim=self.config.get("fusion_dim", 512),
                dropout=self.dropout,
                freeze_bert=self.config.get("freeze_bert", False),
                config=self.config,
                transformer_layers=self.config.get("transformer_layers", 6),
                struct_dim=self.projection_dim if self.use_structure else 0
            )
        elif core_model_type in ["acetylation_transformer", "dual_branch_fusion", "dual_branch_attention"]:
            # Acetylation Transformer / dual-branch fusion architectures optimized for acetylation prediction
            logger.info(f"Use {core_model_type} Model (AcetylationPredictionModel)")
            self.core_model = AcetylationPredictionModel(
                seq_feature_dim=self.seq_dim,
                struct_feature_dim=self.struct_dim,
                hidden_dim=self.config.get("hidden_dim", 512),
                context_window=self.config.get("context_window", 15),
                dropout=self.dropout,
                config=self.config
            )
            # Multi-task loss for acetylation transformer variants
            self.multitask_loss = MultiTaskLoss(
                alpha=self.config.get("multitask_alpha", 0.7),
                class_weights=self.config.get("class_weights", [1.0, 3.0]),
                alpha_bounds=self.config.get("multitask_alpha_bounds")
            )
            self.use_multitask_loss = True
            logger.info("Core model initialized; multitask loss enabled.")
        else:
            raise ValueError(
                f"Unsupported core_model_type: {core_model_type}. "
                f"Expected one of: 'transformer', 'bert_bilstm', 'ptm_bert_bilstm', "
                f"'acetylation_transformer', 'dual_branch_fusion', 'dual_branch_attention'."
            )
            
        # Reinforcement-learning classifier (optional)
        if use_reinforcement:
            logger.info("Initialize ")

            # Choose an RL classifier based on the core model type
            if core_model_type in ["bert_bilstm", "ptm_bert_bilstm"]:
                logger.info("UseBERT+BiLSTM ")
                self.rl_classifier = BERTBiLSTMRLClassifier(
                    input_dim=self.projection_dim,
                    bert_hidden_dim=self.bert_hidden_dim,
                    lstm_hidden_dim=self.config.get("rl_lstm_hidden_dim", 128),
                    num_decision_steps=self.config.get("rl_num_steps", 3),
                    dropout=self.dropout
                )
            else:
                # Traditional RL classifier
                self.rl_classifier = RLClassifier(
                    input_dim=self.projection_dim,
                    hidden_dim=self.config.get("rl_hidden_dim", 128),
                    num_steps=self.config.get("rl_num_steps", 5)
                )
            self.use_rl_classifier = True
        else:
            self.use_rl_classifier = False
            
        # Global and local feature fusion
        # Fusion dimension: global + local
        fusion_input_dim = self.projection_dim * 2
        
        # Center-site focused fusion
        self.use_center_focus = self.config.get("use_center_focus", True)
        if self.use_center_focus:
            logger.info("Enable Site (Center-Focused Fusion)")
            fusion_input_dim += self.projection_dim
            # [New] Targeted dropout applied to center features to reduce memorization of the center position
            self.center_dropout = nn.Dropout(self.config.get("center_dropout", 0.5))

        self.context_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
            
        # Final classification head (simplified for stability)
        if not self.use_rl_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(self.projection_dim, 64),  # Significantly reduce intermediate layer size
                nn.ReLU(),                           # Use a more stable ReLU
                nn.Dropout(0.1),                     # Reduce dropout
                nn.Linear(64, 1)                     # Output layer
            )

            # Stable weight initialization
            for layer in self.classifier:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)  # Small gain
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        logger.info("AcetylationPredictor initialization completed.")
        
    def _init_center_regularization(self) -> None:
        """Initialize masking tokens and curriculum config for center features."""
        self.center_mask_prob = float(self.config.get("center_mask_prob", 0.0))
        self.center_mask_noise_std = float(self.config.get("center_mask_noise_std", 0.0))
        self.center_mask_token_seq = None
        self.center_mask_token_struct = None
        if self.center_mask_prob > 0.0:
            self.center_mask_token_seq = nn.Parameter(torch.zeros(self.seq_dim))
            nn.init.normal_(self.center_mask_token_seq, mean=0.0, std=0.02)
            if self.use_structure:
                self.center_mask_token_struct = nn.Parameter(torch.zeros(self.struct_dim))
                nn.init.normal_(self.center_mask_token_struct, mean=0.0, std=0.02)

        self.center_curriculum_cfg = self.config.get("center_curriculum", {}) or {}
        self.center_curriculum_enabled = bool(self.center_curriculum_cfg)
        default_alpha = float(
            self.center_curriculum_cfg.get(
                "initial_alpha",
                0.0 if self.center_curriculum_enabled else 1.0,
            )
        )
        self.center_curriculum_eval_alpha = float(self.center_curriculum_cfg.get("eval_alpha", 1.0))
        self.current_center_alpha = default_alpha if self.center_curriculum_enabled else 1.0

    def update_center_curriculum(self, epoch: float) -> None:
        """External hook for training loop to update curriculum epoch."""
        if not self.center_curriculum_enabled:
            return
        self.current_center_alpha = self._compute_center_alpha(float(epoch))

    def set_center_alpha(self, alpha: float) -> None:
        """Allow manual override of the current center feature scaling."""
        self.current_center_alpha = float(alpha)

    def _resolve_center_alpha(self, epoch_override=None, alpha_override=None) -> float:
        if alpha_override is not None:
            self.set_center_alpha(alpha_override)
        elif epoch_override is not None:
            self.update_center_curriculum(epoch_override)
        elif not self.training and self.center_curriculum_enabled:
            self.current_center_alpha = self.center_curriculum_eval_alpha
        return float(self.current_center_alpha)

    def _compute_center_alpha(self, epoch_value: float) -> float:
        cfg = self.center_curriculum_cfg
        start_alpha = float(cfg.get("start_alpha", cfg.get("initial_alpha", 0.0)))
        target_alpha = float(cfg.get("target_alpha", cfg.get("alpha_target", 1.0)))
        warmup = max(0, int(cfg.get("warmup_epochs", 0)))
        ramp = max(1, int(cfg.get("ramp_epochs", 1)))
        if epoch_value < warmup:
            return start_alpha
        progress = min(1.0, (epoch_value - warmup) / float(ramp))
        return start_alpha + (target_alpha - start_alpha) * progress

    @staticmethod
    def _apply_center_alpha(tensor: Optional[torch.Tensor], alpha: float) -> Optional[torch.Tensor]:
        if tensor is None or alpha >= 0.9999:
            return tensor
        return tensor * alpha

    def _apply_center_mask_if_needed(
        self,
        seq_tensor: Optional[torch.Tensor],
        local_tensor: Optional[torch.Tensor],
        struct_tensor: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        relative_offsets: Optional[torch.Tensor],
    ):
        if (
            not self.training
            or self.center_mask_prob <= 0.0
            or seq_tensor is None
            or not isinstance(seq_tensor, torch.Tensor)
            or seq_tensor.dim() < 3
        ):
            return seq_tensor, local_tensor, struct_tensor, position_ids, relative_offsets

        batch_size = seq_tensor.size(0)
        device = seq_tensor.device
        mask_flags = torch.rand(batch_size, device=device) < self.center_mask_prob
        if not mask_flags.any():
            return seq_tensor, local_tensor, struct_tensor, position_ids, relative_offsets

        seq_tensor = self._mask_center_tensor(seq_tensor, mask_flags, self.center_mask_token_seq)
        if isinstance(local_tensor, torch.Tensor) and local_tensor.dim() >= 3:
            local_tensor = self._mask_center_tensor(local_tensor, mask_flags, self.center_mask_token_seq)
        if (
            isinstance(struct_tensor, torch.Tensor)
            and struct_tensor.dim() >= 3
            and self.center_mask_token_struct is not None
        ):
            struct_tensor = self._mask_center_tensor(struct_tensor, mask_flags, self.center_mask_token_struct)

        mask_indices = mask_flags.nonzero(as_tuple=False).view(-1)
        if mask_indices.numel() > 0:
            if isinstance(position_ids, torch.Tensor) and position_ids.dim() >= 2:
                position_ids = position_ids.clone()
                center_idx = min(position_ids.size(1) // 2, position_ids.size(1) - 1)
                if self.center_mask_noise_std > 0:
                    noise = torch.normal(
                        mean=0.0,
                        std=self.center_mask_noise_std,
                        size=(mask_indices.numel(),),
                        device=position_ids.device,
                    ).round().long()
                else:
                    noise = torch.zeros(mask_indices.numel(), device=position_ids.device, dtype=torch.long)
                center_vals = position_ids[mask_indices, center_idx] + noise
                position_ids[mask_indices, center_idx] = center_vals.clamp_(0, self.max_position_embeddings - 1)

            if isinstance(relative_offsets, torch.Tensor) and relative_offsets.dim() >= 2:
                relative_offsets = relative_offsets.clone()
                center_idx = min(relative_offsets.size(1) // 2, relative_offsets.size(1) - 1)
                relative_offsets[mask_indices, center_idx] = 0

        return seq_tensor, local_tensor, struct_tensor, position_ids, relative_offsets

    @staticmethod
    def _mask_center_tensor(
        tensor: torch.Tensor,
        mask_flags: torch.Tensor,
        mask_token: Optional[torch.nn.Parameter],
    ) -> torch.Tensor:
        if mask_token is None or tensor.dim() < 3:
            return tensor
        center_idx = tensor.size(1) // 2
        masked = tensor.clone()
        token = mask_token.to(dtype=masked.dtype, device=masked.device)
        masked[mask_flags, center_idx, :] = token
        return masked

    def forward(self, seq_data=None, local_seq_data=None, struct_data=None,
                acetylation_positions=None, site_positions=None, **kwargs):
        """
        Forward pass.

        Args:
            seq_data: Sequence data or a batch dict.
                - torch.Tensor: used directly as sequence features.
                - dict: batch dict containing keys like 'sequence', 'structure', 'label', etc.
            local_seq_data: Local sequence data (optional).
            struct_data: Structure data (optional).
            acetylation_positions: Acetylation site positions [batch_size] (optional).
            site_positions: Modification site positions [batch_size] (optional; used by acetylation_transformer).
            **kwargs: Extra features from DataLoader (e.g., window_features, local_features).

        Returns:
            torch.Tensor or tuple: Prediction outputs.
        """
        curriculum_epoch = kwargs.pop('center_curriculum_epoch', None)
        center_alpha_override = kwargs.pop('center_alpha', None)
        if center_alpha_override is None:
            center_alpha_override = kwargs.pop('center_alpha_override', None)
        active_center_alpha = self._resolve_center_alpha(curriculum_epoch, center_alpha_override)

        attention_mask = kwargs.pop('attention_mask', None)
        ptm_type_ids = None
        position_ids = None
        residue_types_tensor = None
        relative_position_offsets = None
        site_indicator = None
        extra_context = None

        # Accept common keyword aliases from external callers
        if seq_data is None and 'window_features' in kwargs:
            seq_data = kwargs.pop('window_features')
        if local_seq_data is None and 'local_features' in kwargs:
            local_seq_data = kwargs.pop('local_features')
        if struct_data is None and 'structure' in kwargs:
            struct_data = kwargs.pop('structure')
        if struct_data is None and 'structure_features' in kwargs:
            struct_data = kwargs.pop('structure_features')
        if acetylation_positions is None and 'positions' in kwargs:
            acetylation_positions = kwargs.pop('positions')
        if site_positions is None and 'site_positions' in kwargs:
            site_positions = kwargs.pop('site_positions')
        if 'position_ids' in kwargs and position_ids is None:
            position_ids = kwargs.pop('position_ids')
        if 'ptm_type_ids' in kwargs and ptm_type_ids is None:
            ptm_type_ids = kwargs.pop('ptm_type_ids')
        if 'site_indicator' in kwargs and site_indicator is None:
            site_indicator = kwargs.pop('site_indicator')
        if 'relative_position_offsets' in kwargs and relative_position_offsets is None:
            relative_position_offsets = kwargs.pop('relative_position_offsets')
        if 'global_features' in kwargs:
            extra_context = kwargs.pop('global_features')
        
        micro_env_data = None
        if 'micro_environment' in kwargs:
            micro_env_data = kwargs.pop('micro_environment')

        # Handle batch dict input
        if isinstance(seq_data, dict):
            batch = seq_data
            # Extract fields from batch dict
            if 'sequence' in batch:
                if isinstance(batch['sequence'], dict):
                    # New format: sequence is a dict containing multiple feature types
                    if 'window_features' in batch['sequence']:
                        seq_data = batch['sequence']['window_features']
                    elif 'sequence_features' in batch['sequence']:
                        seq_data = batch['sequence']['sequence_features']
                    elif 'local_features' in batch['sequence']:
                        seq_data = batch['sequence']['local_features']
                    else:
                        # Take the first available feature
                        seq_data = list(batch['sequence'].values())[0]

                    # Extract local sequence features
                    if local_seq_data is None and 'local_features' in batch['sequence']:
                        local_seq_data = batch['sequence']['local_features']
                else:
                    # Legacy format: sequence is a tensor
                    seq_data = batch['sequence']
            elif 'sequence_features' in batch:
                seq_data = batch['sequence_features']
                if local_seq_data is None and 'local_features' in batch:
                    local_seq_data = batch['local_features']
            else:
                raise ValueError(" Data")

            # Extract structure data
            if struct_data is None:
                if 'structure' in batch:
                    struct_data = batch['structure']
                elif 'structure_features' in batch:
                    struct_data = batch['structure_features']

            # Extract acetylation positions
            if acetylation_positions is None and 'positions' in batch:
                acetylation_positions = batch['positions']

            # Extract residue types
            if 'residue_types' in batch:
                residue_types_tensor = batch['residue_types']
            elif 'sequence' in batch and isinstance(batch['sequence'], dict) and 'residue_types' in batch['sequence']:
                residue_types_tensor = batch['sequence']['residue_types']

            # Extract position information
            if 'position_ids' in batch:
                position_ids = batch['position_ids']
            elif 'sequence' in batch and isinstance(batch['sequence'], dict) and 'position_ids' in batch['sequence']:
                position_ids = batch['sequence']['position_ids']

            if 'relative_position_offsets' in batch:
                relative_position_offsets = batch['relative_position_offsets']
            elif 'sequence' in batch and isinstance(batch['sequence'], dict) and 'relative_position_offsets' in batch['sequence']:
                relative_position_offsets = batch['sequence']['relative_position_offsets']

            # Extract PTM type information
            if 'ptm_type_ids' in batch:
                ptm_type_ids = batch['ptm_type_ids']
            elif 'sequence' in batch and isinstance(batch['sequence'], dict) and 'ptm_type_ids' in batch['sequence']:
                ptm_type_ids = batch['sequence']['ptm_type_ids']
            elif 'ptm_type' in batch:
                ptm_type_ids = batch['ptm_type']

            # Target-site indicator
            if 'site_indicator' in batch:
                site_indicator = batch['site_indicator']
            elif 'sequence' in batch and isinstance(batch['sequence'], dict) and 'site_indicator' in batch['sequence']:
                site_indicator = batch['sequence']['site_indicator']

            if extra_context is None and 'global_features' in batch:
                extra_context = batch['global_features']

            if attention_mask is None:
                if 'attention_mask' in batch:
                    attention_mask = batch['attention_mask']
                elif 'sequence' in batch and isinstance(batch['sequence'], dict) and 'attention_mask' in batch['sequence']:
                    attention_mask = batch['sequence']['attention_mask']


        # Normalize PTM type IDs
        if ptm_type_ids is not None and not isinstance(ptm_type_ids, torch.Tensor):
            mapping = self.config.get('ptm_type_to_id') if isinstance(self.config, dict) else None
            if isinstance(ptm_type_ids, (list, tuple)):
                normalized_ids = []
                for value in ptm_type_ids:
                    if isinstance(value, torch.Tensor):
                        normalized_ids.append(int(value.item()))
                    elif isinstance(value, str):
                        if mapping and value in mapping:
                            normalized_ids.append(mapping[value])
                        else:
                            normalized_ids.append(0)
                    else:
                        normalized_ids.append(int(value))
                ptm_type_ids = normalized_ids
            elif isinstance(ptm_type_ids, str):
                if mapping and ptm_type_ids in mapping:
                    ptm_type_ids = [mapping[ptm_type_ids]]
                else:
                    ptm_type_ids = [0]
            else:
                ptm_type_ids = [int(ptm_type_ids)]
            ptm_type_ids = torch.tensor(ptm_type_ids, dtype=torch.long)

        try:
            resolved_device = next(self.parameters()).device
        except StopIteration:
            resolved_device = None

        if resolved_device is None:
            if isinstance(seq_data, torch.Tensor):
                resolved_device = seq_data.device
            elif isinstance(struct_data, torch.Tensor):
                resolved_device = struct_data.device
            else:
                resolved_device = torch.device("cpu")

        self.device = resolved_device

        if isinstance(seq_data, torch.Tensor):
            seq_data = seq_data.to(resolved_device)
        if local_seq_data is not None and isinstance(local_seq_data, torch.Tensor):
            local_seq_data = local_seq_data.to(resolved_device)
        if struct_data is not None and isinstance(struct_data, torch.Tensor):
            struct_data = struct_data.to(resolved_device)
        if ptm_type_ids is not None and isinstance(ptm_type_ids, torch.Tensor):
            ptm_type_ids = ptm_type_ids.to(resolved_device)
        if position_ids is not None and isinstance(position_ids, torch.Tensor):
            position_ids = position_ids.to(resolved_device)
        if residue_types_tensor is not None and isinstance(residue_types_tensor, torch.Tensor):
            residue_types_tensor = residue_types_tensor.to(resolved_device)
        if relative_position_offsets is not None and isinstance(relative_position_offsets, torch.Tensor):
            relative_position_offsets = relative_position_offsets.to(resolved_device)
        if site_indicator is not None and isinstance(site_indicator, torch.Tensor):
            site_indicator = site_indicator.to(resolved_device)
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(resolved_device)
                
        seq_data, local_seq_data, struct_data, position_ids, relative_position_offsets = self._apply_center_mask_if_needed(
            seq_data,
            local_seq_data,
            struct_data,
            position_ids,
            relative_position_offsets,
        )

        # Validate required inputs
        if seq_data is None:
            raise ValueError("forward: seq_data window_features")

        # Log device placement on the first forward pass
        if not hasattr(self, '_first_forward_done'):
            logger.info(f"\n===== AcetylationPredictorDevice =====")
            if isinstance(seq_data, torch.Tensor):
                logger.info(f"seq_dataDevice: {seq_data.device}")
            if struct_data is not None:
                if isinstance(struct_data, torch.Tensor):
                    logger.info(f"struct_dataDevice: {struct_data.device}")
                else:
                    logger.info(f"struct_data: {type(struct_data)}")
            logger.info(f"ModelParametersDevice: {next(self.parameters()).device}")
            self._first_forward_done = True
            logger.info("===== Device =====\n")
        
        # 1. Sequence features
        if self.use_precomputed_features:
            # With precomputed features, seq_data should already be a feature tensor
            if isinstance(seq_data, torch.Tensor):
                per_residue_seq = seq_data  # Use precomputed features directly
                
                # [Fix] Prepare residue-type IDs early so they are valid before one-hot encoding
                if residue_types_tensor is not None:
                     if not isinstance(residue_types_tensor, torch.Tensor):
                        residue_types_tensor = torch.tensor(residue_types_tensor, dtype=torch.long, device=self.device)
                     else:
                        residue_types_tensor = residue_types_tensor.to(self.device)
                        if residue_types_tensor.dtype != torch.long:
                             residue_types_tensor = residue_types_tensor.long()
                     
                     if residue_types_tensor.dim() == 1:
                         batch_size, seq_len = per_residue_seq.shape[:2]
                         residue_types_tensor = residue_types_tensor.unsqueeze(1).expand(-1, seq_len)

                     # Clamp residue IDs to avoid out-of-range indices
                     max_residue_id = self.num_residue_types - 1
                     if residue_types_tensor.max() > max_residue_id:
                        if not hasattr(self, '_warned_residue_clamp'):
                             logger.warning(f" ResidueID ({residue_types_tensor.max()}) Config num_residue_types-1 ({max_residue_id}),. Config num_residue_types.")
                             self._warned_residue_clamp = True
                        residue_types_tensor = torch.clamp(residue_types_tensor, min=0, max=max_residue_id)

                # [Modified] Restore One-hot concatenation if needed (Hybrid Encoding)
                # If the model expects higher dimensionality (e.g. 1315) but we have 1280 (ESM),
                # and residue types are available, we should concatenate One-hot encoding.
                if hasattr(self, 'seq_dim') and per_residue_seq.shape[-1] < self.seq_dim:
                    diff = self.seq_dim - per_residue_seq.shape[-1]
                    # Check if the difference matches num_residue_types (approx 35)
                    # We only apply if residue_types_tensor is available
                    if residue_types_tensor is not None:
                        one_hot = F.one_hot(residue_types_tensor, num_classes=self.num_residue_types).float()
                        # Ensure dimension match
                        if one_hot.shape[-1] != diff:
                            if one_hot.shape[-1] > diff:
                                one_hot = one_hot[:, :, :diff]
                            else:
                                pad_len = diff - one_hot.shape[-1]
                                one_hot = F.pad(one_hot, (0, pad_len))
                        
                        per_residue_seq = torch.cat([per_residue_seq, one_hot], dim=-1)
                        # logger.debug(f"Applied Hybrid Encoding: Concatenated One-hot ({diff} dims) to ESM features.")
                
                seq_repr = per_residue_seq.mean(dim=1)  # Global representation
            else:
                raise ValueError("Use,seq_data torch.Tensor")
        else:
            # Extract features with encoder
            seq_repr, per_residue_seq = self.seq_encoder(seq_data)

        # acetylation_transformer variants operate on raw per-residue features (no projection)
        if self.use_multitask_loss:
            seq_features = per_residue_seq
        else:
            # Strict dimension check; re-initialize the projection layer when feature dims change
            if hasattr(self, 'seq_projection'):
                input_dim = per_residue_seq.shape[-1]
                expected_dim = self.seq_projection.projection[0].in_features
                if expected_dim != input_dim:
                     logger.warning(f" ({expected_dim} -> {input_dim}), Initialize seq_projection")
                     device = per_residue_seq.device
                     self.seq_projection = ProjectionLayer(input_dim, self.projection_dim).to(device)
                
                seq_features = self.seq_projection(per_residue_seq)
            else:
                # This should not happen unless seq_projection was never initialized
                pass

        # 2. Local sequence features (optional)
        # Note: acetylation_transformer variants do not use the local sequence branch here.
        if local_seq_data is not None and not self.use_multitask_loss:
            if self.use_precomputed_features:
                # Precomputed local features
                if isinstance(local_seq_data, torch.Tensor):
                    local_per_residue_seq = local_seq_data
                else:
                    raise ValueError("Use,local_seq_data torch.Tensor")
            else:
                # Extract local features with encoder
                local_seq_repr, local_per_residue_seq = self.seq_encoder(local_seq_data)

            # Re-initialize projection when input dim changes
            if hasattr(self, 'local_seq_projection'):
                input_dim = local_per_residue_seq.shape[-1]
                if self.local_seq_projection.projection[0].in_features != input_dim:
                    logger.warning(f" ({self.local_seq_projection.projection[0].in_features} -> {input_dim}), Initialize local_seq_projection")
                    device = local_per_residue_seq.device
                    self.local_seq_projection = ProjectionLayer(input_dim, self.projection_dim).to(device)

            local_features = self.local_seq_projection(local_per_residue_seq)
        else:
            local_features = None

        # 3. Structure features (optional)
        struct_features = None
        per_residue_struct = None  # Initialize per_residue_struct
        if struct_data is not None and self.use_structure:
            if self.use_precomputed_features:
                # Precomputed structure features
                if isinstance(struct_data, torch.Tensor):
                    per_residue_struct = struct_data
                else:
                    raise ValueError("Use,struct_data torch.Tensor")
            else:
                # Extract structure features with encoder
                # [Optimization] Pass distance matrix to enable SpatialGNN
                dist_mat = kwargs.get('distance_matrix', None)
                if hasattr(self, 'device') and dist_mat is not None and isinstance(dist_mat, torch.Tensor):
                    dist_mat = dist_mat.to(self.device)
                
                # Check if encoder accepts distance_matrices (it should be StructureContextEncoder)
                try:
                    struct_repr, per_residue_struct = self.struct_encoder(struct_data, distance_matrices=dist_mat)
                except TypeError:
                    # Fallback for encoders that don't support distance_matrices
                    struct_repr, per_residue_struct = self.struct_encoder(struct_data)

            # acetylation_transformer variants operate on raw per-residue structure features (no projection)
            if self.use_multitask_loss:
                struct_features = per_residue_struct
            else:
                # Re-initialize projection when input dim changes
                if hasattr(self, 'struct_projection'):
                    input_dim = per_residue_struct.shape[-1]
                    if self.struct_projection.projection[0].in_features != input_dim:
                        logger.warning(f" ({self.struct_projection.projection[0].in_features} -> {input_dim}), Initialize struct_projection")
                        device = per_residue_struct.device
                        self.struct_projection = ProjectionLayer(input_dim, self.projection_dim).to(device)
                
                struct_features = self.struct_projection(per_residue_struct)

        # 4. Acetylation-specific features
        if isinstance(seq_data, torch.Tensor) and seq_data.dim() > 1:
            if self.use_precomputed_features:
                # With precomputed features, skip acetylation encoding for now
                acetyl_features = None
            else:
                acetyl_features = self.acetylation_encoder(seq_data, acetylation_positions)
        else:
            acetyl_features = None
            
        # 5. Multi-modal feature fusion
        # Note: acetylation_transformer variants do not use this fusion step.
        if self.use_multitask_loss:
            # For acetylation_transformer, use raw features directly
            combined_features = seq_features
        else:
            available_features = []

            # Sequence features (always available)
            available_features.append(seq_features)

            # Structure features (optional)
            if struct_features is not None:
                available_features.append(struct_features)

            # Fuse features.
            # Optimization: BERTBiLSTMModel variants handle structure fusion internally, so only pass sequence features
            # here to avoid double fusion.
            bilstm_types: Tuple[type, ...] = tuple(
                t for t in (globals().get("BERTBiLSTMModel"), globals().get("PTMSpecificBERTBiLSTM"))
                if isinstance(t, type)
            )
            if bilstm_types and isinstance(self.core_model, bilstm_types):
                # Pass micro-environment features to the core model
                if micro_env_data is not None:
                    # Use kwargs to avoid changing the core_model signature.
                    # Note: core_model.__call__ may not accept kwargs; ensure it supports micro_env_features.
                    combined_features = seq_features
                    # Do not call core_model here; we only prepare inputs. core_model will be called later.
                else:
                    combined_features = seq_features
            else:
                # For other models, apply early fusion
                if self.fusion is not None:
                    combined_features = self.fusion(*available_features)
                else:
                    combined_features = available_features[0]

        batch_size, seq_len, _ = combined_features.shape

        # 5.1 Add explicit position/residue/PTM information.
        # Optimization: PTMSpecificBERTBiLSTM handles these embeddings internally (via concatenation), so skip adding
        # them here to avoid double embedding.
        ptm_specific_cls = globals().get("PTMSpecificBERTBiLSTM")
        is_ptm_specific_model = bool(isinstance(ptm_specific_cls, type) and isinstance(self.core_model, ptm_specific_cls))
        
        # Note: acetylation_transformer does not need position embeddings here; AcetylationPredictionModel handles it.
        if position_ids is not None and not self.use_multitask_loss and not is_ptm_specific_model:
            if not isinstance(position_ids, torch.Tensor):
                position_ids_tensor = torch.tensor(position_ids, dtype=torch.long, device=self.device)
            else:
                position_ids_tensor = position_ids.to(self.device)
                if position_ids_tensor.dtype != torch.long:
                    position_ids_tensor = position_ids_tensor.long()
            if position_ids_tensor.dim() == 1:
                position_ids_tensor = position_ids_tensor.unsqueeze(1).expand(-1, seq_len)
            min_val = position_ids_tensor.min()
            if min_val < 0:
                position_ids_tensor = position_ids_tensor - min_val
            max_allowed = self.position_embedding_layer.num_embeddings - 1
            max_val = position_ids_tensor.max()
            if max_val > max_allowed:
                logger.warning(
                    "position_ids,. max_position_embeddings Config."
                )
                position_ids_tensor = torch.clamp(position_ids_tensor, max=max_allowed)
            position_embedding = self.position_embedding_layer(position_ids_tensor)
            combined_features = combined_features + position_embedding
        else:
            position_ids_tensor = None

        # Note: acetylation_transformer does not need residue-type embeddings here.
        if residue_types_tensor is not None and not self.use_multitask_loss and not is_ptm_specific_model:
            if not isinstance(residue_types_tensor, torch.Tensor):
                residue_tensor = torch.tensor(residue_types_tensor, dtype=torch.long, device=self.device)
            else:
                residue_tensor = residue_types_tensor.to(self.device)
                if residue_tensor.dtype != torch.long:
                    residue_tensor = residue_tensor.long()
            if residue_tensor.dim() == 1:
                residue_tensor = residue_tensor.unsqueeze(1).expand(-1, seq_len)
            
            # Clamp residue IDs to avoid out-of-range indices
            max_residue_id = self.num_residue_types - 1
            if residue_tensor.max() > max_residue_id:
                if not hasattr(self, '_warned_residue_clamp'):
                    logger.warning(f" ResidueID ({residue_tensor.max()}) Config num_residue_types-1 ({max_residue_id}),. Config num_residue_types.")
                    self._warned_residue_clamp = True
                residue_tensor = torch.clamp(residue_tensor, min=0, max=max_residue_id)
                
            residue_embedding = self.residue_type_embedding(residue_tensor)
            combined_features = combined_features + residue_embedding
        else:
            residue_tensor = None

        # Avoid label leakage: do not use ptm_type_embedding.
        # ptm_type is used only as a task identifier and must not be an input feature, to prevent the model from
        # trivially separating positives/negatives via ptm_type_id.
        ptm_tensor = None
        if ptm_type_ids is not None:
            if not isinstance(ptm_type_ids, torch.Tensor):
                ptm_tensor = torch.tensor(ptm_type_ids, dtype=torch.long, device=self.device)
            else:
                ptm_tensor = ptm_type_ids.to(self.device)
                if ptm_tensor.dtype != torch.long:
                    ptm_tensor = ptm_tensor.long()
            # Note: do not use ptm_type_embedding to avoid label leakage
            # ptm_embedding = self.ptm_type_embedding(ptm_tensor)
            # combined_features = combined_features + ptm_embedding

        site_indicator_tensor = None
        # Note: acetylation_transformer does not use relative-position or site-indicator projections here.
        if relative_position_offsets is not None and not self.use_multitask_loss:
            if not isinstance(relative_position_offsets, torch.Tensor):
                rel_tensor = torch.tensor(relative_position_offsets, dtype=torch.long, device=self.device)
            else:
                rel_tensor = relative_position_offsets.to(self.device)
                if rel_tensor.dtype != torch.long:
                    rel_tensor = rel_tensor.long()
            if rel_tensor.dim() == 1:
                rel_tensor = rel_tensor.unsqueeze(1).expand(-1, seq_len)
            rel_float = rel_tensor.to(combined_features.dtype).unsqueeze(-1)
            rel_projection = self.relative_position_projection(rel_float)
            combined_features = combined_features + rel_projection
            site_indicator_tensor = (rel_tensor == 0).float()
        if site_indicator is not None and not self.use_multitask_loss:
            if not isinstance(site_indicator, torch.Tensor):
                site_indicator_tensor = torch.tensor(site_indicator, dtype=torch.float32, device=self.device)
            else:
                site_indicator_tensor = site_indicator.to(self.device)
                if site_indicator_tensor.dtype != torch.float32:
                    site_indicator_tensor = site_indicator_tensor.float()
        if site_indicator_tensor is not None and not self.use_multitask_loss:
            if site_indicator_tensor.dim() == 1:
                site_indicator_tensor = site_indicator_tensor.unsqueeze(1).expand(-1, seq_len)
            site_emb_input = site_indicator_tensor.unsqueeze(-1)
            site_embedding = self.site_indicator_projection(site_emb_input)
            combined_features = combined_features + site_embedding

        # Cache processed tensors for interpretability outputs
        if residue_tensor is not None:
            residue_types_tensor = residue_tensor
        if position_ids_tensor is not None:
            position_ids = position_ids_tensor
        if ptm_tensor is not None:
            ptm_type_ids = ptm_tensor
        if relative_position_offsets is not None and 'rel_tensor' in locals():
            relative_position_offsets = rel_tensor
        if site_indicator_tensor is not None:
            site_indicator = site_indicator_tensor
        
        # 6. Core model
        core_kwargs = {}

        # Pass structure features to core models that support them (e.g., BERTBiLSTMModel)
        if struct_features is not None and self.use_structure:
            core_kwargs['struct_features'] = struct_features
            
        # Pass micro-environment features to core models that support them
        if micro_env_data is not None:
            core_kwargs['micro_env_features'] = micro_env_data

        # acetylation_transformer branch
        if self.use_multitask_loss:
            # acetylation_transformer requires raw features and site_positions
            logger.debug(f"[acetylation_transformer] Use ")
            logger.debug(f"[acetylation_transformer] per_residue_seq shape: {per_residue_seq.shape}")
            logger.debug(f"[acetylation_transformer] per_residue_struct: {per_residue_struct.shape if per_residue_struct is not None else 'None'}")

            # Resolve site position indices
            if site_positions is None:
                # If site_positions is missing, derive it from relative_position_offsets
                if relative_position_offsets is not None:
                    if isinstance(relative_position_offsets, torch.Tensor):
                        # Find positions where relative offset is 0
                        site_pos = (relative_position_offsets == 0).nonzero(as_tuple=True)[1]
                    else:
                        site_pos = torch.tensor(relative_position_offsets, dtype=torch.long, device=self.device)
                        site_pos = (site_pos == 0).nonzero(as_tuple=True)[1]
                else:
                    # Default to sequence center
                    batch_size = per_residue_seq.shape[0]
                    site_pos = torch.full((batch_size,), per_residue_seq.shape[1] // 2, dtype=torch.long, device=self.device)
            else:
                if not isinstance(site_positions, torch.Tensor):
                    site_pos = torch.tensor(site_positions, dtype=torch.long, device=self.device)
                else:
                    site_pos = site_positions.to(self.device).long()

            # Ensure site_pos is integer dtype
            if site_pos.dtype != torch.long:
                site_pos = site_pos.long()

            # Ensure per_residue_struct is defined
            if per_residue_struct is None:
                # If structure features are missing, use a zero tensor
                batch_size, seq_len = per_residue_seq.shape[0], per_residue_seq.shape[1]
                per_residue_struct = torch.zeros(batch_size, seq_len, self.struct_dim, device=self.device, dtype=per_residue_seq.dtype)
                logger.debug(f"[acetylation_transformer] None,Use,shape: {per_residue_struct.shape}")

            logger.debug(f"[acetylation_transformer] site_pos shape: {site_pos.shape}")
            logger.debug(f"[acetylation_transformer] acetylation_transformerModel")

            # Convert attention_mask to a padding mask (True means padding)
            padding_mask = None
            if attention_mask is not None:
                padding_mask = (attention_mask == 0)
                # Ensure padding_mask length matches the sequence length
                if padding_mask.shape[1] != per_residue_seq.shape[1]:
                    logger.warning(f"attention_mask length ({padding_mask.shape[1]}) != sequence length ({per_residue_seq.shape[1]}). Truncating/Padding mask.")
                    if padding_mask.shape[1] > per_residue_seq.shape[1]:
                         padding_mask = padding_mask[:, :per_residue_seq.shape[1]]
                    else:
                         pad_len = per_residue_seq.shape[1] - padding_mask.shape[1]
                         padding_mask = F.pad(padding_mask, (0, pad_len), value=True) # Pad with True (padding)

            # Call the acetylation_transformer core model
            core_output = self.core_model(per_residue_seq, per_residue_struct, site_pos, mask=padding_mask)
            logger.debug(f"[acetylation_transformer] ModelOutput: {type(core_output)}")
            if isinstance(core_output, tuple):
                logger.debug(f"[acetylation_transformer] ModelOutput,: {len(core_output)}")
                for i, out in enumerate(core_output):
                    if isinstance(out, torch.Tensor):
                        logger.debug(f"[acetylation_transformer] Output{i} shape: {out.shape}")
            elif isinstance(core_output, torch.Tensor):
                logger.debug(f"[acetylation_transformer] ModelOutputshape: {core_output.shape}")
        elif is_ptm_specific_model:
            if ptm_type_ids is not None:
                if not isinstance(ptm_type_ids, torch.Tensor):
                    ptm_type_ids = torch.tensor(ptm_type_ids, dtype=torch.long, device=self.device)
                elif ptm_type_ids.dtype != torch.long:
                    ptm_type_ids = ptm_type_ids.long()
                core_kwargs['ptm_type_ids'] = ptm_type_ids

            if position_ids is not None:
                if not isinstance(position_ids, torch.Tensor):
                    position_ids = torch.tensor(position_ids, dtype=torch.long, device=self.device)
                elif position_ids.dtype != torch.long:
                    position_ids = position_ids.long()
                core_kwargs['position_ids'] = position_ids

            if residue_types_tensor is not None:
                if not isinstance(residue_types_tensor, torch.Tensor):
                    residue_types_tensor = torch.tensor(residue_types_tensor, dtype=torch.long, device=self.device)
                elif residue_types_tensor.dtype != torch.long:
                    residue_types_tensor = residue_types_tensor.long()
                core_kwargs['residue_types'] = residue_types_tensor

            core_output = self.core_model(combined_features, **core_kwargs)
        else:
            core_output = self.core_model(combined_features, **core_kwargs)
        
        # 7. Special-case outputs from acetylation_transformer
        if self.use_multitask_loss:
            # acetylation_transformer returns (acetylation_prob, strength_pred); return as-is.
            return core_output

        # 8. Store interpretability scores
        importance_scores = {}

        # 9. Local sequence features
        if local_features is not None:
            # If center-focus is enabled, prefer the local center feature (similar to DualBranchFusion behavior).
            # This preserves the most direct center-residue signal and reduces over-smoothing.
            if self.use_center_focus:
                local_center_idx = local_features.size(1) // 2
                local_context = local_features[:, local_center_idx, :]
                # [New] Apply stronger dropout to the local center feature
                if hasattr(self, 'center_dropout'):
                    local_context = self.center_dropout(local_context)
                local_context = self._apply_center_alpha(local_context, active_center_alpha)
            elif self.use_interpretability:
                local_context, local_importance = self.local_interpretability(local_features, return_importance=True)
                importance_scores['local_importance'] = local_importance
            else:
                local_context = torch.mean(local_features, dim=1)  # Mean pooling
        else:
            local_context = None

        # 10. Apply interpretability module to global features
        if self.use_interpretability:
            global_context, global_importance = self.global_interpretability(core_output, return_importance=True)
            importance_scores['global_importance'] = global_importance
            if residue_types_tensor is not None:
                importance_scores['residue_types'] = residue_types_tensor
            if position_ids is not None:
                importance_scores['position_ids'] = position_ids
            if relative_position_offsets is not None:
                importance_scores['relative_position_offsets'] = relative_position_offsets
            if ptm_type_ids is not None:
                importance_scores['ptm_type_ids'] = ptm_type_ids
            if site_indicator is not None:
                importance_scores['site_indicator'] = site_indicator
        else:
            global_context = core_output.mean(dim=1)  # Mean pooling
            
        # 10. Fuse global and local features
        fusion_inputs = [global_context]
        
        # Prepare local context (zero-fill when missing to match fusion dims)
        if local_context is None:
            local_context = torch.zeros_like(global_context)
        fusion_inputs.append(local_context)
            
        # Center-site feature (center-focused fusion)
        if self.use_center_focus:
            # core_output shape: [batch_size, seq_len, dim]
            seq_len = core_output.size(1)
            center_idx = seq_len // 2
            center_feature = core_output[:, center_idx, :]
            # [New] Apply stronger dropout to the global center feature
            if hasattr(self, 'center_dropout'):
                center_feature = self.center_dropout(center_feature)
            center_feature = self._apply_center_alpha(center_feature, active_center_alpha)
            fusion_inputs.append(center_feature)
            
        # Fuse
        # Note: context_fusion input dim is configured in __init__.
        if len(fusion_inputs) > 1:
            fused_context = self.context_fusion(torch.cat(fusion_inputs, dim=1))
        else:
            fused_context = global_context
        
        # 11. Add acetylation features (optional)
        if acetyl_features is not None:
            fused_context = fused_context + 0.2 * acetyl_features  # Weighted sum

        # 11.a Add micro-environment features (optional)
        if micro_env_data is not None:
            # Project micro-environment features
            micro_env_proj = self.micro_env_projection(micro_env_data)
            
            # If input is 3D [batch, seq_len, 6], reduce to [batch, dim]
            if micro_env_proj.dim() == 3:
                # Micro-environment features typically describe the target residue and are expected to be
                # [batch, 6] or [batch, 1, 6]. If a per-residue tensor is provided, we reduce it below.
                pass 
            
            # Ensure [batch, projection_dim]
            if micro_env_proj.dim() == 3:
                 # [batch, seq_len, dim] -> [batch, dim]
                 micro_env_proj = micro_env_proj.mean(dim=1)
            
            # Fuse
            fused_context = fused_context + micro_env_proj
            
        # 11.1 Add external context features (optional)
        if extra_context is not None:
            if isinstance(extra_context, torch.Tensor):
                extra_context = extra_context.to(fused_context.device)
                
                # Re-initialize projection when input dim changes
                if hasattr(self, 'global_context_projection'):
                    input_dim = extra_context.shape[-1]
                    if self.global_context_projection.projection[0].in_features != input_dim:
                        logger.warning(f" ({self.global_context_projection.projection[0].in_features} -> {input_dim}), Initialize global_context_projection")
                        device = extra_context.device
                        self.global_context_projection = ProjectionLayer(input_dim, self.projection_dim).to(device)

                if extra_context.dim() == 2:
                    projected_global = self.global_context_projection(extra_context)
                    fused_context = fused_context + projected_global
                elif extra_context.dim() == 3:
                    projected_global = self.global_context_projection(extra_context)
                    fused_context = fused_context + projected_global.mean(dim=1)
                else:
                    logger.warning("global_features,: %s", extra_context.shape)
            else:
                logger.warning("global_features,: %s", type(extra_context))
            
        # 12. Reinforcement-learning classifier (optional)
        if self.use_rl_classifier:
            rl_output = self.rl_classifier(fused_context)
            if self.use_interpretability:
                return rl_output, importance_scores
            else:
                return rl_output
            
        # 13. Standard classification
        if self.use_multitask_loss:
            # acetylation_transformer already returned predictions; no further classification.
            if self.use_interpretability:
                return core_output, importance_scores
            else:
                return core_output
        else:
            logits = self.classifier(fused_context)

            # [Fix] Check for NaNs in logits
            if torch.isnan(logits).any():
                logger.warning(f"NaN detected in logits! Batch size: {logits.size(0)}. Replacing with zeros.")
                logits = torch.nan_to_num(logits, nan=0.0)

            if self.use_interpretability:
                return logits, importance_scores
            else:
                return logits
            
    def extract_features(self, seq_data, struct_data=None, return_separate=False):
        """
        Extract features without classification.

        Args:
            seq_data: Sequence data.
            struct_data: Structure data (optional).
            return_separate: Whether to return separate feature components.

        Returns:
            torch.Tensor or dict: Extracted features.
        """
        # Ensure tensors are on the correct device
        if hasattr(self, 'device'):
            seq_data = seq_data.to(self.device)
            if struct_data is not None:
                struct_data = struct_data.to(self.device)
                
        # Sequence features
        seq_repr, per_residue_seq = self.seq_encoder(seq_data)
        seq_features = self.seq_projection(per_residue_seq)
        
        # Structure features (optional)
        if struct_data is not None and self.use_structure:
            struct_repr, per_residue_struct = self.struct_encoder(struct_data)
            struct_features = self.struct_projection(per_residue_struct)
        else:
            struct_features = None
            
        if return_separate:
            features = {
                'seq_repr': seq_repr,
                'seq_features': seq_features
            }
            if struct_features is not None:
                features['struct_repr'] = struct_repr
                features['struct_features'] = struct_features
            return features
        
        # Fuse features
        if struct_features is not None:
            # Multi-modal fusion
            fused_features = self.fusion(seq_features, struct_features)
        else:
            fused_features = seq_features
            
        # Run core model and pool globally
        return self.core_model(fused_features).mean(dim=1)  # Global pooling
            
    def visualize_importance(
        self,
        local_sequence,
        importance_scores,
        save_path=None,
        title: Optional[str] = None,
        residue_offsets: Optional[List[int]] = None,
        cmap: str = "YlOrRd",
        global_sequence: Optional[str] = None,
        global_importance: Optional[torch.Tensor] = None,
        global_offsets: Optional[List[int]] = None,
        global_top_k: int = 20,
    ):
        """
        Visualize residue importance.

        Args:
            local_sequence (str): Local amino-acid sequence.
            importance_scores (torch.Tensor): Importance scores [seq_len].
            save_path (str, optional): Output path.
            title (str, optional): Figure title.
            residue_offsets (List[int], optional): Local offsets relative to the center.
            cmap (str): Colormap for the heatmap.
            global_sequence (str, optional): Global sequence window for showing global contributions.
            global_importance (torch.Tensor, optional): Global importance scores.
            global_offsets (List[int], optional): Global offsets relative to the center.
            global_top_k (int): Number of top global positions to display.

        Returns:
            matplotlib.figure.Figure: Figure object.
        """
        if not HAS_PLOTTING:
            logger.warning(", ")
            return None

        if local_sequence is None:
            raise ValueError("local_sequence ")

        # Prepare local importance scores
        seq_len = len(local_sequence)
        if isinstance(importance_scores, torch.Tensor):
            scores = importance_scores[:seq_len].detach().cpu().numpy()
        else:
            scores = np.asarray(importance_scores)[:seq_len]

        if scores.shape[-1] != seq_len:
            raise ValueError(f" {scores.shape[-1]} {seq_len} ")

        center_pos = seq_len // 2

        if residue_offsets is None:
            residue_offsets = list(range(-center_pos, seq_len - center_pos))
        if len(residue_offsets) != seq_len:
            residue_offsets = list(range(-center_pos, seq_len - center_pos))

        # Global contribution panel (optional)
        has_global = False
        global_fig_offsets: Optional[List[int]] = None
        global_scores: Optional[np.ndarray] = None
        if global_sequence is not None and global_importance is not None:
            global_seq = str(global_sequence)
            if isinstance(global_importance, torch.Tensor):
                global_scores = global_importance[:len(global_seq)].detach().cpu().numpy()
            else:
                global_scores = np.asarray(global_importance)[:len(global_seq)]

            if global_scores.size == len(global_seq) and global_scores.size > 0:
                has_global = True
                if global_offsets is not None and len(global_offsets) == len(global_seq):
                    global_fig_offsets = list(global_offsets)
                else:
                    center = len(global_seq) // 2
                    global_fig_offsets = list(range(-center, len(global_seq) - center))
            else:
                logger.debug(
                    "Skip: global_sequence =%s, global_scores =%s",
                    len(global_seq),
                    None if global_scores is None else global_scores.size,
                )
                has_global = False

        base_width = max(14.0, seq_len * 0.5)
        local_height = 3.8

        # Create figure
        if has_global:
            available_global = len(global_scores) if global_scores is not None else 0
            top_count = max(1, min(global_top_k if global_top_k > 0 else available_global, available_global))
            global_height = max(2.8, 0.32 * top_count)
            fig = plt.figure(figsize=(base_width, local_height + global_height))
            gs = fig.add_gridspec(2, 1, height_ratios=[global_height, local_height], hspace=0.5)
            ax_global = fig.add_subplot(gs[0, 0])
            ax = fig.add_subplot(gs[1, 0])
        else:
            fig, ax = plt.subplots(figsize=(base_width, local_height))
            ax_global = None

        fig.subplots_adjust(left=0.12, right=0.97)
        
        # Heatmap
        sns.heatmap(
            [scores],
            cmap=cmap,
            annot=np.array([list(local_sequence)]),
            fmt='',
            annot_kws={"fontsize": 12, "fontweight": "bold"},
            cbar_kws={'label': 'Importance weight'},
            ax=ax
        )
        
        # Title and labels
        display_title = title or 'Local residue importance'
        ax.set_title(display_title)
        ax.set_yticks([])
        tick_positions = np.arange(seq_len) + 0.5
        tick_labels = [f"{res}\n{offset:+d}" for res, offset in zip(local_sequence, residue_offsets)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0, fontsize=10)
        ax.set_xlabel('Relative position (offset from center)')
        
        # Highlight the center position
        ax.add_patch(plt.Rectangle((center_pos, 0), 1, 1, fill=False, edgecolor='black', lw=2))
        
        if ax_global is not None and global_scores is not None and global_fig_offsets is not None:
            abs_scores = np.abs(global_scores)
            if global_top_k > 0 and global_top_k < len(abs_scores):
                top_indices = np.argsort(abs_scores)[-global_top_k:][::-1]
            else:
                top_indices = np.argsort(abs_scores)[::-1]
            residues = [
                f"{global_sequence[i]} ({global_fig_offsets[i]:+d})"
                for i in top_indices
            ]
            values = abs_scores[top_indices]
            signed_values = global_scores[top_indices]

            cmap_obj = plt.get_cmap(cmap)
            if values.size:
                vmin = float(values.min())
                vmax = float(values.max())
                if math.isclose(vmax, vmin):
                    vmax = vmin + 1e-6
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                colors = [cmap_obj(norm(v)) for v in values]
            else:
                colors = None

            bar_colors = colors[::-1] if colors else None
            ax_global.barh(residues[::-1], values[::-1], color=bar_colors, height=0.8)
            ax_global.invert_yaxis()
            ax_global.set_xlabel('Absolute importance')
            ax_global.set_title('Top global residues by importance')
            ax_global.tick_params(axis='y', labelsize=11)
            ax_global.margins(y=0.02)
            ax_global.grid(axis='x', linestyle='--', alpha=0.3)

            if values.size:
                max_val = values.max()
                ax_global.set_xlim(0, max_val * 1.2 if max_val > 0 else 1.0)
                offset = max(max_val * 0.03, 0.01)
                for idx, (val, signed) in enumerate(zip(values[::-1], signed_values[::-1])):
                    ax_global.text(val + offset, idx, f"{signed:.4f}", va='center', fontsize=9)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            fig.tight_layout()

        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None

        return fig


class AblationWrapper(nn.Module):
    """Wrapper for ablation studies on different components of the acetylation predictor."""
    
    def __init__(self, base_model_config, ablation_configs=None):
        """
        Initialize the ablation wrapper.
        
        Args:
            base_model_config: Configuration for the base model.
            ablation_configs (dict): Dictionary of configurations for ablation variants.
        """
        super().__init__()
        
        # Initialize base model
        self.base_model = AcetylationPredictor(**base_model_config)
        
        # Initialize ablation variants
        self.ablation_models = nn.ModuleDict()
        
        if ablation_configs:
            for name, config in ablation_configs.items():
                self.ablation_models[name] = AcetylationPredictor(**config)
                
    def forward(self, seq_data, local_seq_data=None, struct_data=None, model_name=None):
        """
        Forward pass.
        
        Args:
            seq_data: Global sequence data.
            local_seq_data: Local sequence data.
            struct_data: Structure data (optional).
            model_name (str): Ablation model name to use (None for the base model).
            
        Returns:
            torch.Tensor or dict: Model outputs.
        """
        if model_name is None:
            return self.base_model(seq_data, local_seq_data, struct_data)
        else:
            return self.ablation_models[model_name](seq_data, local_seq_data, struct_data)
            
    def evaluate_all(self, seq_data, local_seq_data=None, struct_data=None):
        """
        Evaluate all model variants.
        
        Args:
            seq_data: Global sequence data.
            local_seq_data: Local sequence data.
            struct_data: Structure data (optional).
            
        Returns:
            dict: Mapping from model name to outputs.
        """
        results = {
            'base': self.base_model(seq_data, local_seq_data, struct_data)
        }
        
        for name in self.ablation_models:
            results[name] = self.ablation_models[name](seq_data, local_seq_data, struct_data)
            
        return results


class DualBranchFusionPredictor(nn.Module):
    """Dual-branch PTM predictor with adaptive fusion between sequence and structure features."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.device = torch.device("cpu")

        # Core hyper-parameters
        self.seq_input_dim = int(self.config.get("seq_feature_dim", self.config.get("seq_dim", 1280)))
        self.local_input_dim = self.seq_input_dim  # Default to seq_input_dim
        self.global_input_dim = self.seq_input_dim # Default to seq_input_dim
        self.struct_input_dim = int(self.config.get("struct_feature_dim", self.config.get("struct_dim", 512)))
        self.branch_hidden_dim = int(self.config.get("branch_hidden_dim", 256))
        self.fusion_hidden_dim = int(self.config.get("fusion_hidden_dim", 384))
        self.dropout = float(self.config.get("dropout", 0.1))
        self.modality_dropout = float(self.config.get("modality_dropout", 0.1))  # Modality dropout probability
        self.use_structure = bool(self.config.get("use_structure", True))
        self.branch_warmup_use_branch_logits = bool(self.config.get("branch_warmup_use_branch_logits", False))
        self.branch_warmup_freeze_fusion = bool(self.config.get("branch_warmup_freeze_fusion", False))
        self.branch_warmup_aux_multiplier = float(self.config.get("branch_warmup_aux_multiplier", 1.0))
        self.use_legacy_fusion = bool(self.config.get("use_legacy_fusion", False))

        # [NEW] Explicit Feature Embeddings for Dual Branch
        self.use_explicit_features = bool(self.config.get("use_explicit_features", True))
        self.residue_embedding_dim = int(self.config.get("aa_embedding_dim", 64))
        self.position_embedding_dim = int(self.config.get("position_embedding_dim", 32))
        
        if self.use_explicit_features:
            # 25 for standard AA + special tokens
            self.residue_embedding = nn.Embedding(25, self.residue_embedding_dim)
            # 512 for relative position (0..window_size or similar)
            self.position_embedding = nn.Embedding(512, self.position_embedding_dim)
            self._residue_embedding_repaired = False
            self._position_embedding_repaired = False
            
            # Increase input dimension for sequence tower
            self.seq_input_dim += (self.residue_embedding_dim + self.position_embedding_dim)
            
            # Also increase structure input dimension for position embedding if structure is used
            if self.use_structure:
                # [Modified] Add residue embedding (64) + position embedding (32) + micro-env (6) to structure branch
                self.struct_input_dim += (self.residue_embedding_dim + self.position_embedding_dim + 6)
                
            logger.info(f"DualBranchFusionPredictor: Enabled explicit features. Seq dim increased to {self.seq_input_dim}, Struct dim increased to {self.struct_input_dim}")

        seq_layers = int(self.config.get("seq_transformer_layers", 2))
        struct_layers = int(self.config.get("struct_transformer_layers", 2))
        seq_heads = int(self.config.get("seq_attention_heads", 4))
        struct_heads = int(self.config.get("struct_attention_heads", 4))
        cross_heads = int(self.config.get("cross_attention_heads", 4))
        cross_bidirectional = bool(self.config.get("cross_attention_bidirectional", True))
        use_attention_pooling = bool(self.config.get("use_attention_pooling", True))
        window_radius = int(self.config.get("center_window_radius", 3))

        # [Enhanced Components - Attention]
        self.use_enhanced_attention = bool(self.config.get("use_enhanced_attention", False))
        seq_tower_input_dim = self.seq_input_dim
        struct_tower_input_dim = self.struct_input_dim
        
        if self.use_enhanced_attention:
            self.enhanced_dim = int(self.config.get("enhanced_dim", 256))
            logger.info(f"DualBranchFusionPredictor: Enabling Enhanced Attention Module (dim={self.enhanced_dim})")
            
            self.seq_enhanced_proj = nn.Linear(self.seq_input_dim, self.enhanced_dim)
            self.seq_enhanced_norm = nn.LayerNorm(self.enhanced_dim)
            
            self.enhanced_seq_attention = EnhancedAttentionModule(
                input_dim=self.enhanced_dim,
                num_heads=int(self.config.get("enhanced_attention_heads", 4)),
                num_layers=int(self.config.get("enhanced_attention_layers", 2)),
                dropout=self.dropout
            )
            seq_tower_input_dim = self.enhanced_dim
            
            if self.use_structure:
                self.struct_enhanced_proj = nn.Linear(self.struct_input_dim, self.enhanced_dim)
                self.struct_enhanced_norm = nn.LayerNorm(self.enhanced_dim)
                
                self.enhanced_struct_attention = EnhancedAttentionModule(
                    input_dim=self.enhanced_dim,
                    num_heads=int(self.config.get("enhanced_attention_heads", 4)),
                    num_layers=int(self.config.get("enhanced_attention_layers", 2)),
                    dropout=self.dropout
                )
                struct_tower_input_dim = self.enhanced_dim

        self.sequence_tower = SequenceTower(
            input_dim=seq_tower_input_dim,
            hidden_dim=self.branch_hidden_dim,
            num_layers=seq_layers,
            num_heads=seq_heads,
            dropout=self.dropout,
            use_attention_pooling=use_attention_pooling,
            window_radius=window_radius,
        )

        self.structure_tower = None
        if self.use_structure:
            self.structure_tower = StructureTower(
                input_dim=struct_tower_input_dim,
                hidden_dim=self.branch_hidden_dim,
                dropout=self.dropout,
                num_layers=struct_layers,
                num_heads=struct_heads,
                use_attention_pooling=use_attention_pooling,
                window_radius=window_radius,
            )

        self.cross_modal = None
        if self.structure_tower is not None:
            self.cross_modal = CrossModalFusionLayer(
                hidden_dim=self.branch_hidden_dim,
                num_heads=cross_heads,
                dropout=self.dropout,
                bidirectional=cross_bidirectional,
            )

        self.gating = AdaptiveGating(self.branch_hidden_dim)

        # Projections for auxiliary features
        self.local_projection = nn.Sequential(
            nn.Linear(self.local_input_dim, self.branch_hidden_dim),
            nn.LayerNorm(self.branch_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(self.global_input_dim, self.branch_hidden_dim),
            nn.LayerNorm(self.branch_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        if self.use_legacy_fusion:
            self.bilinear_interaction = nn.Bilinear(
                self.branch_hidden_dim,
                self.branch_hidden_dim,
                self.branch_hidden_dim,
            )
            fusion_input_dim = self.branch_hidden_dim * 7
        else:
            self.bilinear_interaction = nn.Bilinear(
                self.branch_hidden_dim,
                self.branch_hidden_dim,
                self.branch_hidden_dim,
            )
            # Enhanced: Added back abs_diff and interaction, plus micro_env
            fusion_input_dim = self.branch_hidden_dim * 7 + 6 # fused + seq + struct + diff + interaction + local + global + micro_env
            
        # [NEW] Residual-style fusion network to better preserve branch features and improve gradient flow
        self.fusion_proj = nn.Linear(fusion_input_dim, self.fusion_hidden_dim)
        self.fusion_residual_block = nn.Sequential(
            nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim),
            nn.LayerNorm(self.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim),
            nn.LayerNorm(self.fusion_hidden_dim),
        )
        self.fusion_bottleneck = nn.Sequential(
            nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim // 2),
            nn.LayerNorm(self.fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout / 2.0),
        )
        self.classifier = nn.Linear(self.fusion_hidden_dim // 2, 1)

        # Auxiliary objectives
        self.alignment_weight = float(self.config.get("alignment_weight", 0.0))
        self.contrastive_weight = float(self.config.get("contrastive_weight", 0.0))
        self.similarity_weight = float(self.config.get("structure_similarity_weight", 0.0))
        self.aux_cls_weight = float(self.config.get("aux_cls_weight", 0.0))  # New: Auxiliary classification weight
        self.temperature = float(self.config.get("contrastive_temperature", 0.07))

        enable_branch_cls_default = self.aux_cls_weight > 0 or self.branch_warmup_use_branch_logits
        self.enable_branch_classifiers = bool(self.config.get("enable_branch_classifiers", enable_branch_cls_default))

        if self.enable_branch_classifiers:
            self.seq_classifier = nn.Linear(self.branch_hidden_dim, 1)
            if self.use_structure:
                self.struct_classifier = nn.Linear(self.branch_hidden_dim, 1)
            else:
                self.struct_classifier = None
        else:
            self.seq_classifier = None
            self.struct_classifier = None

        if self.contrastive_weight > 0:
            projection_dim = int(self.config.get("contrastive_projection_dim", 128))
            self.projection_head = ProjectionHead(
                input_dim=self.branch_hidden_dim,
                projection_dim=projection_dim,
                dropout=self.dropout,
            )
        else:
            self.projection_head = None

        # Learning rate scaling for stage-wise optimization
        self.seq_lr_scale = float(self.config.get("lr_scale_seq", 0.3))
        self.struct_lr_scale = float(self.config.get("lr_scale_struct", 1.0))
        self.fusion_lr_scale = float(self.config.get("lr_scale_fusion", 1.0))
        self.training_stage = "joint"
        self.branch_warmup_active = False
        self._structure_all_mask_warning_emitted = False

        self._init_center_regularization_dual()

        # [Enhanced Components Initialization]
        self.use_multiscale_fusion = bool(self.config.get("use_multiscale_fusion", False))
        self.use_dynamic_feature_selection = bool(self.config.get("use_dynamic_feature_selection", False))

        if self.use_multiscale_fusion:
            logger.info("DualBranchFusionPredictor: Enabling Multi-Scale Fusion Network")
            # MultiScaleFusionNetwork expects sequence inputs, so we use branch_hidden_dim (output of towers)
            self.multiscale_fusion_net = MultiScaleFusionNetwork(
                seq_dim=self.branch_hidden_dim,
                struct_dim=self.branch_hidden_dim,
                hidden_dims=[512, 256, 128], # Configurable?
                dropout=self.dropout
            )
            # Update fusion dimension because MultiScaleFusionNetwork returns hidden_dims[0] (512)
            # We will use this INSTEAD OF or IN ADDITION TO the standard gated fusion?
            # Let's add it to the fusion inputs.
            # fusion_input_dim calculation needs to be updated in forward or here.
            # But self.fusion_proj is already initialized above. We might need to adjust it.
            # To avoid breaking existing weights if not used, we'll handle dimension mismatch dynamically or re-init if needed.
            # Actually, better to re-init fusion_proj here if we change the input size.
            
            ms_out_dim = 512 # hidden_dims[0]
            # We will replace 'fused_global' (branch_hidden_dim) with ms_out_dim or append it?
            # Let's append it.
            # New fusion_input_dim = old_fusion_input_dim + ms_out_dim
            # But wait, fusion_proj is already created. 
            # I should re-create fusion_proj if this is enabled.
            
            # Recalculate fusion_input_dim
            # Standard: fused(256) + seq(256) + struct(256) + diff(256) + inter(256) + local(256) + global(256) + micro(6) = 1798
            # With MS Fusion: + 512 = 2310
            
            current_dim = self.fusion_proj.in_features
            new_dim = current_dim + ms_out_dim
            
            self.fusion_proj = nn.Linear(new_dim, self.fusion_hidden_dim)
            logger.info(f"DualBranchFusionPredictor: Re-initialized fusion_proj with dim {new_dim}")

        if self.use_dynamic_feature_selection:
            logger.info("DualBranchFusionPredictor: Enabling Dynamic Feature Selection")
            # We will select from: Seq, Struct, Local, Global, (optional MS Fusion)
            # Let's define the feature dims
            feature_dims = {
                'seq': self.branch_hidden_dim,
                'struct': self.branch_hidden_dim,
                'local': self.branch_hidden_dim,
                'global': self.branch_hidden_dim,
            }
            if self.use_multiscale_fusion:
                feature_dims['multiscale'] = 512 # ms_out_dim
            else:
                feature_dims['gated'] = self.branch_hidden_dim # standard gated fusion
                
            self.feature_selector = DynamicFeatureSelector(
                feature_dims=feature_dims,
                hidden_dim=256
            )
            
            # If using selector, the fusion_proj input dim becomes sum(feature_dims.values()) + others?
            # Or we replace the concatenation logic.
            # The selector returns fused features (concatenated weighted).
            # Then we pass this to fusion_proj.
            # So fusion_input_dim = sum(feature_dims.values()) + others (like diff, interaction, micro_env)
            
            # Let's assume we keep diff, interaction, micro_env outside of selection (or include them?)
            # Report says: "Dynamic Feature Selector" for "sequence, structure, local, global".
            # So we select these, and then concat with interaction terms.
            
            selector_out_dim = sum(feature_dims.values())
            # Interaction terms: diff(256) + inter(256) + micro(6) = 518
            # Total = selector_out_dim + 518
            
            new_dim_selector = selector_out_dim + self.branch_hidden_dim * 2 + 6
            if self.use_legacy_fusion:
                 # Legacy didn't have micro_env? 
                 pass
                 
            self.fusion_proj = nn.Linear(new_dim_selector, self.fusion_hidden_dim)
            logger.info(f"DualBranchFusionPredictor: Re-initialized fusion_proj for Feature Selector with dim {new_dim_selector}")

    def _init_center_regularization_dual(self) -> None:
        self.center_mask_prob = float(self.config.get("center_mask_prob", 0.0))
        self.center_mask_noise_std = float(self.config.get("center_mask_noise_std", 0.0))
        self.center_mask_token_seq = None
        self.center_mask_token_struct = None
        if self.center_mask_prob > 0.0:
            self.center_mask_token_seq = nn.Parameter(torch.zeros(self.seq_input_dim))
            nn.init.normal_(self.center_mask_token_seq, mean=0.0, std=0.02)
            if self.use_structure:
                self.center_mask_token_struct = nn.Parameter(torch.zeros(self.struct_input_dim))
                nn.init.normal_(self.center_mask_token_struct, mean=0.0, std=0.02)

        self.center_curriculum_cfg = self.config.get("center_curriculum", {}) or {}
        self.center_curriculum_enabled = bool(self.center_curriculum_cfg)
        default_alpha = float(
            self.center_curriculum_cfg.get(
                "initial_alpha",
                0.0 if self.center_curriculum_enabled else 1.0,
            )
        )
        self.center_curriculum_eval_alpha = float(self.center_curriculum_cfg.get("eval_alpha", 1.0))
        self._active_center_alpha = default_alpha if self.center_curriculum_enabled else 1.0

    def update_center_curriculum(self, epoch: float) -> None:
        if not self.center_curriculum_enabled:
            return
        self._active_center_alpha = self._compute_center_alpha_dual(float(epoch))

    def set_center_alpha(self, alpha: float) -> None:
        self._active_center_alpha = float(alpha)

    def _resolve_center_alpha(self, epoch_override=None, alpha_override=None) -> float:
        if alpha_override is not None:
            self.set_center_alpha(alpha_override)
        elif epoch_override is not None:
            self.update_center_curriculum(epoch_override)
        elif not self.training and self.center_curriculum_enabled:
            self._active_center_alpha = self.center_curriculum_eval_alpha
        return float(self._active_center_alpha)

    def _compute_center_alpha_dual(self, epoch_value: float) -> float:
        cfg = self.center_curriculum_cfg
        start_alpha = float(cfg.get("start_alpha", cfg.get("initial_alpha", 0.0)))
        target_alpha = float(cfg.get("target_alpha", cfg.get("alpha_target", 1.0)))
        warmup = max(0, int(cfg.get("warmup_epochs", 0)))
        ramp = max(1, int(cfg.get("ramp_epochs", 1)))
        if epoch_value < warmup:
            return start_alpha
        progress = min(1.0, (epoch_value - warmup) / float(ramp))
        return start_alpha + (target_alpha - start_alpha) * progress

    def _apply_center_mask_if_needed_dual(
        self,
        seq_tensor: torch.Tensor,
        local_tensor: Optional[torch.Tensor],
        struct_tensor: Optional[torch.Tensor],
    ):
        # [NaN Debug] Check inputs
        if torch.isnan(seq_tensor).any():
            logger.warning("NaN detected in seq_tensor BEFORE center mask application. Applying nan_to_num.")
            seq_tensor = torch.nan_to_num(seq_tensor)

        if (
            not self.training
            or self.center_mask_prob <= 0.0
            or seq_tensor is None
            or not isinstance(seq_tensor, torch.Tensor)
            or seq_tensor.dim() < 3
        ):
            return seq_tensor, local_tensor, struct_tensor

        # Ensure mask probability is valid
        prob = max(0.0, min(1.0, self.center_mask_prob))
        
        mask_flags = torch.rand(seq_tensor.size(0), device=seq_tensor.device) < prob
        if not mask_flags.any():
            return seq_tensor, local_tensor, struct_tensor

        seq_tensor = self._mask_center_tensor_dual(seq_tensor, mask_flags, self.center_mask_token_seq)
        if isinstance(local_tensor, torch.Tensor) and local_tensor.dim() >= 3:
            local_tensor = self._mask_center_tensor_dual(local_tensor, mask_flags, self.center_mask_token_seq)
        if (
            isinstance(struct_tensor, torch.Tensor)
            and struct_tensor.dim() >= 3
            and self.center_mask_token_struct is not None
        ):
            struct_tensor = self._mask_center_tensor_dual(struct_tensor, mask_flags, self.center_mask_token_struct)

        # [NaN Debug] Check outputs
        if torch.isnan(seq_tensor).any():
            logger.warning("NaN detected in seq_tensor AFTER center mask application.")
            
        return seq_tensor, local_tensor, struct_tensor

    @staticmethod
    def _mask_center_tensor_dual(
        tensor: torch.Tensor,
        mask_flags: torch.Tensor,
        mask_token: Optional[torch.nn.Parameter],
    ) -> torch.Tensor:
        if mask_token is None:
            return tensor
        center_idx = tensor.size(1) // 2
        masked = tensor.clone()
        token = mask_token.to(dtype=masked.dtype, device=masked.device)
        masked[mask_flags, center_idx, :] = token
        return masked

    def _extract_masks(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor.abs().sum(dim=-1) == 0)

    def forward(self, batch, return_importance=False):
        sequence_block = batch.get('sequence', {})
        if isinstance(sequence_block, dict):
            seq_window = sequence_block.get('window_features')
        else:
            seq_window = sequence_block
            sequence_block = {}

        if seq_window is None:
            seq_window = batch.get('sequence_window')
        if seq_window is None:
            seq_window = batch.get('sequence_features')

        # [Fix] Extract micro-environment features early
        micro_env = batch.get('micro_env_features')
        if micro_env is None:
            # Fallback if missing (should not happen with correct data processor)
            if seq_window is not None:
                 micro_env = torch.zeros(seq_window.size(0), 6, device=seq_window.device)
            else:
                 # Should fail later, but provide a safe fallback here
                 micro_env = None

        if seq_window is None:
            raise ValueError("Batch sequence['window_features'] ")

        if isinstance(seq_window, torch.Tensor):
            self.device = seq_window.device
        else:
            try:
                self.device = next(self.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        labels = batch.get('label', None)

        center_epoch = batch.get('center_curriculum_epoch')
        center_alpha_override = batch.get('center_alpha')
        active_center_alpha = self._resolve_center_alpha(center_epoch, center_alpha_override)

        struct_tensor = batch.get('structure')
        if struct_tensor is None:
            struct_tensor = batch.get('structure_features')
        if not self.use_structure:
            struct_tensor = None

        # [Optimization] Extract distance matrix for structure branch
        distance_matrices = batch.get('distance_matrix')
        if distance_matrices is None:
             sequence_block = batch.get('sequence', {})
             if isinstance(sequence_block, dict):
                 distance_matrices = sequence_block.get('distance_matrix')
        
        # Ensure distance matrix is on correct device if present
        if distance_matrices is not None and isinstance(distance_matrices, torch.Tensor):
            distance_matrices = distance_matrices.to(self.device)

        # [Fix] Calculate masks BEFORE injecting explicit features (which make padding non-zero)
        seq_mask = self._extract_masks(seq_window)
        struct_mask = None
        if struct_tensor is not None:
            struct_mask = self._extract_masks(struct_tensor)

        # [NEW] Inject Explicit Features (Residue + Position)
        if self.use_explicit_features:
            # 1. Residue Embeddings
            residue_types = sequence_block.get('residue_types')
            if residue_types is None:
                residue_types = batch.get('residue_types')
            if residue_types is not None:
                if residue_types.device != seq_window.device:
                    residue_types = residue_types.to(seq_window.device)
                if residue_types.dtype not in (torch.int32, torch.int64):
                    residue_types = torch.nan_to_num(residue_types.float(), nan=0.0, posinf=0.0, neginf=0.0).long()
                # Ensure valid range [0, 24]
                residue_types = residue_types.clamp(0, 24)

                if not torch.isfinite(self.residue_embedding.weight).all():
                    if not self._residue_embedding_repaired:
                        logger.warning("Non-finite values detected in residue_embedding weights. Repairing with nan_to_num.")
                        self._residue_embedding_repaired = True
                    with torch.no_grad():
                        self.residue_embedding.weight.data = torch.nan_to_num(
                            self.residue_embedding.weight.data,
                            nan=0.0,
                            posinf=1.0,
                            neginf=-1.0,
                        )
                res_emb = self.residue_embedding(residue_types)
                # Check for NaNs in residue embedding
                if torch.isnan(res_emb).any():
                    logger.warning("NaN detected in res_emb. Applying nan_to_num.")
                    res_emb = torch.nan_to_num(res_emb)
            else:
                # Fallback: zeros
                res_emb = torch.zeros(
                    seq_window.size(0), seq_window.size(1), self.residue_embedding_dim,
                    device=seq_window.device, dtype=seq_window.dtype
                )
            
            # 2. Position Embeddings
            # Use simple 0..L-1 position encoding
            pos_ids = torch.arange(seq_window.size(1), device=seq_window.device).expand(seq_window.size(0), -1)
            # Clamp to embedding size limit
            pos_ids = pos_ids.clamp(0, 511)

            if not torch.isfinite(self.position_embedding.weight).all():
                if not self._position_embedding_repaired:
                    logger.warning("Non-finite values detected in position_embedding weights. Repairing with nan_to_num.")
                    self._position_embedding_repaired = True
                with torch.no_grad():
                    self.position_embedding.weight.data = torch.nan_to_num(
                        self.position_embedding.weight.data,
                        nan=0.0,
                        posinf=1.0,
                        neginf=-1.0,
                    )
            pos_emb = self.position_embedding(pos_ids)
            # Check for NaNs in position embedding
            if torch.isnan(pos_emb).any():
                logger.warning("NaN detected in pos_emb. Applying nan_to_num.")
                pos_emb = torch.nan_to_num(pos_emb)
            
            # 3. Concatenate
            # Check seq_window before concat
            if torch.isnan(seq_window).any():
                 logger.warning("NaN detected in seq_window BEFORE concat. Applying nan_to_num.")
                 seq_window = torch.nan_to_num(seq_window)
                 
            seq_window = torch.cat([seq_window, res_emb, pos_emb], dim=-1)

            # 4. Inject Position Embedding into Structure (Moved here to fix dimension check)
            if struct_tensor is not None:
                # Check struct_tensor before concat
                if torch.isnan(struct_tensor).any():
                     logger.warning("NaN detected in struct_tensor BEFORE concat. Applying nan_to_num.")
                     struct_tensor = torch.nan_to_num(struct_tensor)
                
                # [Modified] Add residue embedding (64) + position embedding (32) + micro-env (3) to structure branch
                # micro_env: [B, 3] -> expand to [B, L, 3]
                if micro_env is not None:
                    micro_env_expanded = micro_env.unsqueeze(1).expand(-1, struct_tensor.size(1), -1)
                    struct_tensor = torch.cat([struct_tensor, res_emb, pos_emb, micro_env_expanded], dim=-1)
                else:
                    # Fallback if micro_env is somehow missing
                    dummy_micro_env = torch.zeros(struct_tensor.size(0), struct_tensor.size(1), 6, device=struct_tensor.device, dtype=struct_tensor.dtype)
                    struct_tensor = torch.cat([struct_tensor, res_emb, pos_emb, dummy_micro_env], dim=-1)
                
                # [FIX] Zero out padding regions again because embeddings made them non-zero
                # struct_mask is True for padding (zeros). We want to keep valid data (False).
                if struct_mask is not None:
                    # struct_mask: [B, L] -> True for padding
                    # We want 1.0 for valid, 0.0 for padding
                    valid_mask = (~struct_mask).unsqueeze(-1).to(dtype=struct_tensor.dtype)
                    struct_tensor = struct_tensor * valid_mask

        # Dynamic dimension check
        struct_tensor_check = struct_tensor
        struct_dim = struct_tensor_check.shape[-1] if struct_tensor_check is not None else None
        
        local_features = sequence_block.get('local_features')
        if local_features is None:
            local_features = batch.get('local_features')
        local_dim = local_features.shape[-1] if local_features is not None else None
        
        global_features = sequence_block.get('global_features')
        if global_features is None:
            global_features = batch.get('global_features')
        global_dim = global_features.shape[-1] if global_features is not None else None
        
        self._check_and_fix_projection(seq_window.shape[-1], struct_dim, local_dim, global_dim)

        seq_window, local_features, struct_tensor = self._apply_center_mask_if_needed_dual(
            seq_window,
            local_features,
            struct_tensor,
        )

        # [Enhanced Components] Apply Enhanced Attention if enabled
        if getattr(self, 'use_enhanced_attention', False):
            # Project inputs first
            seq_window = self.seq_enhanced_norm(self.seq_enhanced_proj(seq_window))
            seq_window = self.enhanced_seq_attention(seq_window, mask=seq_mask)
            
            if struct_tensor is not None and getattr(self, 'use_structure', False):
                struct_tensor = self.struct_enhanced_norm(self.struct_enhanced_proj(struct_tensor))
                struct_tensor = self.enhanced_struct_attention(struct_tensor, mask=struct_mask)

        # seq_mask is already calculated
        seq_weights = None
        if return_importance:
            seq_context, seq_global, seq_weights = self.sequence_tower(seq_window, seq_mask, center_alpha=active_center_alpha, return_weights=True)
        else:
            seq_context, seq_global = self.sequence_tower(seq_window, seq_mask, center_alpha=active_center_alpha)

        labels_flat = None

        labels_column = None
        if isinstance(labels, torch.Tensor):
            labels_flat = labels.view(-1).to(seq_global.device)
            labels_column = labels_flat.view(-1, 1).float()

        struct_context = None
        struct_global = None
        struct_mask = None
        struct_all_masked = None
        seq_all_masked = None

        if seq_mask is not None and seq_mask.ndim == 2:
            seq_all_masked = seq_mask.all(dim=1)

        if self.structure_tower is not None and struct_tensor is not None:
            # struct_mask is already calculated
            if struct_mask is None:
                 # Fallback if something went wrong, though it shouldn't
                 struct_mask = self._extract_masks(struct_tensor)
            
            struct_all_masked = struct_mask.all(dim=1) if struct_mask.ndim == 2 else None

            if struct_all_masked is not None and struct_all_masked.any():
                if not self._structure_all_mask_warning_emitted:
                    logger.warning(
                        " %d Samples, Skip, Use, /NaN",
                        int(struct_all_masked.sum().item()),
                    )
                    self._structure_all_mask_warning_emitted = True
                # To avoid NaNs when the structure branch is fully masked, clear the mask first and then explicitly zero it.
                struct_mask = struct_mask.clone()
                struct_mask[struct_all_masked] = False

            if return_importance:
                struct_context, struct_global, struct_weights = self.structure_tower(
                    struct_tensor, 
                    struct_mask, 
                    center_alpha=active_center_alpha, 
                    return_weights=True,
                    distance_matrices=distance_matrices
                )
            else:
                struct_context, struct_global = self.structure_tower(
                    struct_tensor, 
                    struct_mask, 
                    center_alpha=active_center_alpha,
                    distance_matrices=distance_matrices
                )
            
            # Capture raw features for independent auxiliary classification
            seq_global_raw = seq_global
            struct_global_raw = struct_global

            if self.cross_modal is not None:
                # If all structure samples are masked, skip cross-modal interaction to reduce noise.
                if struct_all_masked is not None and struct_all_masked.all():
                    seq_context_fused, struct_context_fused = seq_context, struct_context
                else:
                    seq_context_fused, struct_context_fused = self.cross_modal(
                        seq_context,
                        struct_context,
                        seq_mask,
                        struct_mask,
                    )
                seq_context = seq_context_fused
                struct_context = struct_context_fused
                seq_global = masked_mean(seq_context, seq_mask)
                struct_global = masked_mean(struct_context, struct_mask)

            if struct_all_masked is not None and struct_all_masked.any():
                struct_context = struct_context.clone()
                struct_global = struct_global.clone()
                struct_global_raw = struct_global_raw.clone() # Also mask raw
                struct_context[struct_all_masked] = 0.0
                struct_global[struct_all_masked] = 0.0
                struct_global_raw[struct_all_masked] = 0.0
        else:
            seq_global_raw = seq_global
            struct_global_raw = None

        if struct_global is None:
            struct_global = seq_global.new_zeros(seq_global.shape)
            struct_global_raw = seq_global.new_zeros(seq_global.shape)

        # If the sequence branch is also fully masked, explicitly zero it to reduce noise.
        if seq_all_masked is not None and seq_all_masked.any():
            seq_context = seq_context.clone()
            seq_global = seq_global.clone()
            seq_global_raw = seq_global_raw.clone()
            seq_context[seq_all_masked] = 0.0
            seq_global[seq_all_masked] = 0.0
            seq_global_raw[seq_all_masked] = 0.0

        # Modality dropout: randomly drop one modality to improve robustness (training only)
        if self.training and self.modality_dropout > 0:
            # Randomly decide whether to drop a modality
            if torch.rand(1).item() < self.modality_dropout:
                # Choose which modality to drop: 0=sequence, 1=structure
                # Ensure at least one modality remains. If structure is absent, do not drop sequence.
                if struct_tensor is not None and torch.rand(1).item() < 0.5:
                    # Drop sequence
                    seq_global = torch.zeros_like(seq_global)
                else:
                    # Drop structure
                    struct_global = torch.zeros_like(struct_global)

        fused_global, gate_weights = self.gating(seq_global, struct_global)

        if local_features is not None:
            # [Modified] Use average pooling over the local window instead of just center pixel
            # This captures the local motif context better than a single residue
            local_repr = self.local_projection(local_features.mean(dim=1))
        else:
            local_repr = seq_global.new_zeros(seq_global.shape)

        if global_features is not None:
            global_repr = self.global_projection(global_features)
        else:
            global_repr = seq_global.new_zeros(seq_global.shape)

        # [Enhanced Components] Multi-Scale Fusion
        ms_fused = None
        if getattr(self, 'use_multiscale_fusion', False):
             ms_fused = self.multiscale_fusion_net(
                 seq_context,
                 struct_context,
                 seq_mask=seq_mask,
                 struct_mask=struct_mask
             )

        if self.use_legacy_fusion:
            interaction = self.bilinear_interaction(seq_global, struct_global)
            abs_diff = torch.abs(seq_global - struct_global)
            
            fusion_inputs = torch.cat(
                [
                    fused_global,  # Gated combination of Seq and Struct
                    seq_global,    # Sequence features
                    struct_global, # Structure features
                    abs_diff,      # Difference
                    interaction,   # Interaction
                    local_repr,    # Local sequence context
                    global_repr,   # Global protein context
                ],
                dim=-1,
            )
        elif getattr(self, 'use_dynamic_feature_selection', False):
            # [Enhanced Components] Dynamic Feature Selection
            features_to_select = {
                 'seq': seq_global,
                 'struct': struct_global,
                 'local': local_repr,
                 'global': global_repr
            }
            if ms_fused is not None:
                 features_to_select['multiscale'] = ms_fused
            else:
                 features_to_select['gated'] = fused_global
            
            fused_selected, selection_weights = self.feature_selector(features_to_select)
            
            # Recalculate interaction terms
            struct_confidence = 1.0 - gate_weights
            interaction = self.bilinear_interaction(seq_global, struct_global) * struct_confidence
            abs_diff = torch.abs(seq_global - struct_global) * struct_confidence
            
            fusion_inputs = torch.cat(
                [
                    fused_selected,
                    abs_diff,
                    interaction,
                    micro_env
                ],
                dim=-1
            )
        else:
            # Enhanced: Added explicit interaction and difference to modern fusion path
            # [Fix] Apply gating to interaction and difference to suppress noise from missing/unreliable structure
            # gate_weights: [B, H], higher means more focus on sequence (less on structure)
            struct_confidence = 1.0 - gate_weights
            interaction = self.bilinear_interaction(seq_global, struct_global) * struct_confidence
            abs_diff = torch.abs(seq_global - struct_global) * struct_confidence
            
            fusion_list = [
                    fused_global,  # Gated combination of Seq and Struct
                    seq_global,    # Sequence features (enhanced by structure attention)
                    struct_global, # Structure features
                    abs_diff,      # Explicit difference (weighted)
                    interaction,   # Explicit bilinear interaction (weighted)
                    local_repr,    # Local sequence context
                    global_repr,   # Global protein context
                    micro_env,     # [NEW] Micro-environment geometric features
            ]
            
            if ms_fused is not None:
                fusion_list.append(ms_fused)

            fusion_inputs = torch.cat(fusion_list, dim=-1)

        # [NEW] Residual-style fusion forward pass
        x = self.fusion_proj(fusion_inputs)
        residual = x
        x = self.fusion_residual_block(x)
        x = x + residual # Residual connection
        fused_representation = self.fusion_bottleneck(x)
        
        logits = self.classifier(fused_representation).squeeze(-1)

        auxiliary_losses = {}
        auxiliary_logs = {}

        if self.alignment_weight > 0 and struct_tensor is not None:
            alignment_raw = F.mse_loss(seq_global, struct_global, reduction='none').mean(dim=-1)
            # Use average of gate_weights (which is feature-wise weight for sequence branch)
            # 1 - gate_weights is the weight for the structure branch.
            # If structure is deemed unreliable (low weight), we reduce the alignment constraint.
            struct_gate_weight = (1.0 - gate_weights).mean(dim=-1)
            alignment_loss = (alignment_raw * struct_gate_weight).mean()
            
            auxiliary_losses['alignment'] = alignment_loss * self.alignment_weight
            auxiliary_logs['alignment'] = alignment_loss.detach()

        if self.similarity_weight > 0 and struct_tensor is not None:
            # [Fix] Avoid NaN in cosine similarity when structure is fully masked (zero vector)
            if struct_all_masked is not None and struct_all_masked.any():
                valid_mask = ~struct_all_masked
                if valid_mask.sum() > 0:
                    cosine_sim = F.cosine_similarity(seq_global[valid_mask], struct_global[valid_mask], dim=-1)
                    similarity_loss = (1.0 - cosine_sim).mean()
                else:
                    similarity_loss = torch.tensor(0.0, device=seq_global.device)
            else:
                cosine_sim = F.cosine_similarity(seq_global, struct_global, dim=-1)
                similarity_loss = (1.0 - cosine_sim).mean()
                
            auxiliary_losses['structure_similarity'] = similarity_loss * self.similarity_weight
            auxiliary_logs['structure_similarity'] = similarity_loss.detach()

        if self.contrastive_weight > 0 and self.projection_head is not None and labels_flat is not None:
            if torch.unique(labels_flat).numel() > 1:
                # 1. Sequence Contrastive
                seq_projected = self.projection_head(seq_global)
                seq_con_loss = self._supervised_contrastive(seq_projected, labels_flat)
                
                # 2. Structure Contrastive
                struct_con_loss = None
                if struct_global is not None:
                    # [Fix] Exclude masked structures from contrastive learning to avoid noise
                    if struct_all_masked is not None and struct_all_masked.any():
                        valid_mask = ~struct_all_masked
                        if valid_mask.sum() > 1: # Need at least 2 samples for contrastive
                            struct_projected_valid = self.projection_head(struct_global[valid_mask])
                            struct_con_loss = self._supervised_contrastive(struct_projected_valid, labels_flat[valid_mask])
                    else:
                        struct_projected = self.projection_head(struct_global)
                        struct_con_loss = self._supervised_contrastive(struct_projected, labels_flat)
                
                # Combine losses
                total_con_loss = 0.0
                if seq_con_loss is not None:
                    total_con_loss += seq_con_loss
                if struct_con_loss is not None:
                    total_con_loss += struct_con_loss
                    # Average if both present
                    if seq_con_loss is not None:
                        total_con_loss /= 2.0
                        
                if total_con_loss is not None and (isinstance(total_con_loss, float) and total_con_loss > 0 or isinstance(total_con_loss, torch.Tensor)):
                     auxiliary_losses['contrastive'] = total_con_loss * self.contrastive_weight
                     auxiliary_logs['contrastive'] = total_con_loss.detach()

        aux_seq_logits = None
        aux_struct_logits = None

        if self.seq_classifier is not None:
            aux_seq_logits = self.seq_classifier(seq_global_raw)
            if labels_column is not None and self.aux_cls_weight > 0:
                aux_seq_loss = F.binary_cross_entropy_with_logits(
                    aux_seq_logits,
                    labels_column.to(aux_seq_logits.device),
                )
                auxiliary_losses['aux_seq'] = aux_seq_loss * self.aux_cls_weight
                auxiliary_logs['aux_seq'] = auxiliary_losses['aux_seq'].detach()

        if self.struct_classifier is not None and struct_global_raw is not None:
            aux_struct_logits = self.struct_classifier(struct_global_raw)
            if labels_column is not None and self.aux_cls_weight > 0:
                aux_struct_loss = F.binary_cross_entropy_with_logits(
                    aux_struct_logits,
                    labels_column.to(aux_struct_logits.device),
                )
                auxiliary_losses['aux_struct'] = aux_struct_loss * self.aux_cls_weight
                auxiliary_logs['aux_struct'] = auxiliary_losses['aux_struct'].detach()

        if self.branch_warmup_active and self.branch_warmup_aux_multiplier > 1.0:
            for aux_key in ('aux_seq', 'aux_struct'):
                if aux_key in auxiliary_losses:
                    auxiliary_losses[aux_key] = auxiliary_losses[aux_key] * self.branch_warmup_aux_multiplier
                    auxiliary_logs[aux_key] = auxiliary_losses[aux_key].detach()

        if self.branch_warmup_active and self.branch_warmup_use_branch_logits:
            branch_logits = []
            if aux_seq_logits is not None:
                branch_logits.append(aux_seq_logits.squeeze(-1))
            if aux_struct_logits is not None:
                branch_logits.append(aux_struct_logits.squeeze(-1))
            if branch_logits:
                logits = torch.stack(branch_logits, dim=0).mean(dim=0)

        outputs = {
            'logits': logits,
            'fusion_weights': gate_weights,
            'sequence_global': seq_global,
            'structure_global': struct_global,
            'auxiliary_losses': auxiliary_losses,
            'auxiliary_logs': auxiliary_logs,
            'aux_seq_logits': aux_seq_logits,
            'aux_struct_logits': aux_struct_logits,
            'struct_masked_count': struct_all_masked.sum().detach() if struct_all_masked is not None else torch.tensor(0.0).to(logits.device),
        }
        
        if return_importance:
            return logits, {
                'window_importance': seq_weights if seq_weights is not None else torch.zeros(logits.shape[0], seq_window.shape[1]),
                'structure_importance': struct_weights if 'struct_weights' in locals() and struct_weights is not None else None,
                'gate_weights': gate_weights
            }
            
        return outputs

    def _supervised_contrastive(self, features: torch.Tensor, labels: torch.Tensor) -> Optional[torch.Tensor]:
        labels = labels.contiguous().view(-1)
        if labels.numel() < 2:
            return None

        device = features.device
        temperature = max(self.temperature, 1e-6)

        sim_matrix = torch.div(torch.matmul(features, features.t()), temperature)
        logits_max = torch.max(sim_matrix, dim=1, keepdim=True).values.detach()
        logits = sim_matrix - logits_max

        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.t()).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positives = mask.sum(dim=1)
        valid_mask = positives > 0
        if valid_mask.sum() == 0:
            return None

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / torch.clamp(positives, min=1.0)
        loss = -mean_log_prob_pos[valid_mask].mean()
        return loss

    def _set_requires_grad(self, module: Optional[nn.Module], requires_grad: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = requires_grad

    def _apply_branch_warmup_freeze(self) -> None:
        freeze_targets = [
            self.cross_modal,
            self.gating,
            self.fusion_network,
            self.classifier,
            self.local_projection,
            self.global_projection,
            self.projection_head,
        ]
        if self.use_legacy_fusion:
            freeze_targets.append(self.bilinear_interaction)
            
        for module in freeze_targets:
            self._set_requires_grad(module, False)

    def get_grouped_parameters(self, base_lr, weight_decay):
        param_groups = []
        
        # Sequence tower
        param_groups.append({
            'params': list(self.sequence_tower.parameters()),
            'lr': base_lr * self.seq_lr_scale,
            'weight_decay': weight_decay,
        })

        # Structure tower
        if self.structure_tower is not None:
            param_groups.append({
                'params': list(self.structure_tower.parameters()),
                'lr': base_lr * self.struct_lr_scale,
                'weight_decay': weight_decay,
            })

        fusion_modules = [
            self.cross_modal,
            self.gating,
            self.fusion_network,
            self.classifier,
            self.local_projection,
            self.global_projection,
            self.projection_head,
            self.seq_classifier,
            self.struct_classifier,
        ]
        fusion_params = []
        for module in fusion_modules:
            if module is None:
                continue
            fusion_params.extend(list(module.parameters()))

        if fusion_params:
            param_groups.append({
                'params': fusion_params,
                'lr': base_lr * self.fusion_lr_scale,
                'weight_decay': weight_decay,
            })

        return param_groups if param_groups else [{'params': self.parameters(), 'lr': base_lr, 'weight_decay': weight_decay}]

    def set_training_stage(self, stage: str):
        """
        Set the training stage and freeze/unfreeze parameters accordingly.
        Supported stages: 'structure_warmup', 'joint', 'full_finetune'
        """
        self.training_stage = stage
        self.branch_warmup_active = stage == 'structure_warmup'
        logger.info(f"Setting training stage to: {stage}")

        # Always unfreeze everything first to ensure clean state
        for param in self.parameters():
            param.requires_grad = True

        if self.branch_warmup_active and self.branch_warmup_freeze_fusion:
            logger.info("Stage 'structure_warmup': Freezing fusion/gating modules for branch warmup")
            self._apply_branch_warmup_freeze()

        if stage == 'structure_warmup':
            # Branch warmup lets towers specialize before joint fusion updates.
            return

        elif stage == 'joint':
            # In joint stage, check config for freezing
            freeze_struct = self.config.get('joint_freeze_structure', False)
            freeze_seq = self.config.get('joint_freeze_sequence', False)
            
            if freeze_struct and self.structure_tower is not None:
                logger.info("Stage 'joint': Freezing structure tower")
                for param in self.structure_tower.parameters():
                    param.requires_grad = False
            
            if freeze_seq:
                logger.info("Stage 'joint': Freezing sequence tower")
                for param in self.sequence_tower.parameters():
                    param.requires_grad = False
                    
        elif stage == 'full_finetune':
            # Unfreeze everything (already done)
            logger.info("Stage 'full_finetune': Unfreezing all parameters")
    
    def _check_and_fix_projection(self, seq_dim, struct_dim, local_dim=None, global_dim=None):
        # Check sequence dimension
        if seq_dim != self.seq_input_dim:
            logger.warning(f"DualBranchFusionPredictor: Input sequence dim {seq_dim} != config dim {self.seq_input_dim}. Re-initializing sequence layers.")
            self.seq_input_dim = seq_dim
            
            # Re-initialize Sequence Tower
            seq_layers = int(self.config.get("seq_transformer_layers", 2))
            seq_heads = int(self.config.get("seq_attention_heads", 4))
            use_attention_pooling = bool(self.config.get("use_attention_pooling", True))
            
            self.sequence_tower = SequenceTower(
                input_dim=self.seq_input_dim,
                hidden_dim=self.branch_hidden_dim,
                num_layers=seq_layers,
                num_heads=seq_heads,
                dropout=self.dropout,
                use_attention_pooling=use_attention_pooling,
            ).to(self.device)
            
        # Check local dimension
        if local_dim is not None and local_dim != self.local_input_dim:
            logger.warning(f"DualBranchFusionPredictor: Input local dim {local_dim} != config dim {self.local_input_dim}. Re-initializing local projection.")
            self.local_input_dim = local_dim
            self.local_projection = nn.Sequential(
                nn.Linear(self.local_input_dim, self.branch_hidden_dim),
                nn.LayerNorm(self.branch_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
            ).to(self.device)
            
        # Check global dimension
        if global_dim is not None and global_dim != self.global_input_dim:
            logger.warning(f"DualBranchFusionPredictor: Input global dim {global_dim} != config dim {self.global_input_dim}. Re-initializing global projection.")
            self.global_input_dim = global_dim
            self.global_projection = nn.Sequential(
                nn.Linear(self.global_input_dim, self.branch_hidden_dim),
                nn.LayerNorm(self.branch_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
            ).to(self.device)

        # Check structure dimension
        if self.use_structure and struct_dim is not None and struct_dim != self.struct_input_dim:
            logger.warning(f"DualBranchFusionPredictor: Input structure dim {struct_dim} != config dim {self.struct_input_dim}. Re-initializing structure layers.")
            self.struct_input_dim = struct_dim
            
            struct_layers = int(self.config.get("struct_transformer_layers", 2))
            struct_heads = int(self.config.get("struct_attention_heads", 4))
            use_attention_pooling = bool(self.config.get("use_attention_pooling", True))
            
            self.structure_tower = StructureTower(
                input_dim=self.struct_input_dim,
                hidden_dim=self.branch_hidden_dim,
                dropout=self.dropout,
                num_layers=struct_layers,
                num_heads=struct_heads,
                use_attention_pooling=use_attention_pooling,
            ).to(self.device)


class AcetylationPredictionModel(nn.Module):
    """Full acetylation prediction model (legacy implementation)."""
    
    def __init__(self,
                 seq_feature_dim=1280,
                 struct_feature_dim=512,
                 hidden_dim=512,
                 context_window=15,
                 dropout=0.1,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.config = config or {}
        self.use_adaptive_dropout = bool(
            self.config.get("adaptive_dropout", False) and create_adaptive_dropout_layer is not None
        )

        self.seq_encoder = SequenceContextEncoder(
            input_dim=seq_feature_dim,
            hidden_dim=hidden_dim,
            context_window=context_window
        )

        self.struct_encoder = StructureContextEncoder(
            input_dim=struct_feature_dim,
            hidden_dim=hidden_dim // 2,
            context_window=context_window,
            adaptive_config=self.config if self.use_adaptive_dropout else None,
            use_adaptive_dropout=self.use_adaptive_dropout
        )

        seq_out_dim = self.seq_encoder.output_dim
        struct_out_dim = self.struct_encoder.output_dim

        self.fusion_module = GatedFusionModule(
            seq_dim=seq_out_dim,
            struct_dim=struct_out_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim // 2,
            dropout=self.config.get("fusion_dropout", 0.2),
            adaptive_config=self.config if self.use_adaptive_dropout else None,
            use_adaptive_dropout=self.use_adaptive_dropout
        )
        
        if self.use_adaptive_dropout and create_adaptive_dropout_layer is not None:
            classifier_dropout = create_adaptive_dropout_layer(self.config, 'output')
        else:
            classifier_dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            classifier_dropout,
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self._warned_missing_structures = False
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self,
                sequences: torch.Tensor,
                structures: torch.Tensor,
                site_positions: torch.Tensor,
                distance_matrices: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        seq_site_feat, seq_context_feat, seq_padding_mask = self.seq_encoder(sequences, site_positions, mask=mask)

        if structures is None:
            if not self._warned_missing_structures:
                logger.warning("AcetylationPredictionModel:,Use Skip Info.")
                self._warned_missing_structures = True
            batch_size, seq_len, _ = sequences.shape
            structures = sequences.new_zeros(batch_size, seq_len, self.struct_encoder.input_dim)
            distance_matrices = None
        elif distance_matrices is None:
            with torch.no_grad():
                distance_matrices = torch.cdist(structures.detach(), structures.detach(), p=2)

        struct_site_feat, struct_context_feat, struct_padding_mask = self.struct_encoder(
            structures,
            site_positions,
            distance_matrices
        )

        fused_features = self.fusion_module(
            seq_site_feat, 
            struct_site_feat,
            seq_context=seq_context_feat,
            struct_context=struct_context_feat,
            seq_padding_mask=seq_padding_mask,
            struct_padding_mask=struct_padding_mask
        )

        acetylation_logits = self.classifier(fused_features)
        strength_logits = self.regression_head(fused_features)

        return acetylation_logits, strength_logits


class MultiTaskLoss(nn.Module):
    """Multi-task loss supporting dynamic weight updates and focal loss."""

    def __init__(self, alpha: float = 0.7, class_weights=None, alpha_bounds: Optional[Tuple[float, float]] = None, use_focal_loss: bool = True):
        super().__init__()
        if class_weights is None:
            class_weights = torch.tensor([1.0, 3.0], dtype=torch.float32)
        else:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        self.register_buffer('class_weights', class_weights)
        self.alpha = float(alpha)
        self.alpha_min = float(alpha_bounds[0]) if alpha_bounds else float(alpha)
        self.alpha_max = float(alpha_bounds[1]) if alpha_bounds else float(alpha)

        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            # Focal Loss parameters can be tuned or passed via config if needed
            self.classification_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        else:
            self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)

        self.mse_loss = nn.MSELoss(reduction='mean')

    def set_alpha(self, new_alpha: float):
        self.alpha = float(new_alpha)

    def configure_alpha_schedule(self, min_alpha: float, max_alpha: float):
        self.alpha_min = float(min_alpha)
        self.alpha_max = float(max(self.alpha_min, max_alpha))

    def update_alpha_by_progress(self, progress: float):
        progress_tensor = torch.clamp(torch.tensor(progress, dtype=torch.float32), 0.0, 1.0)
        progress = float(progress_tensor.item())
        self.alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * progress

    def set_class_weights(self, class_weights):
        if not torch.is_tensor(class_weights):
            class_weights = torch.tensor(class_weights, dtype=self.class_weights.dtype, device=self.class_weights.device)
        class_weights = class_weights.to(self.class_weights.device).to(self.class_weights.dtype)
        self.class_weights.copy_(class_weights)
        
        # Update CrossEntropyLoss weights if not using Focal Loss
        if not self.use_focal_loss:
             self.classification_loss.weight = self.class_weights

    def forward(self,
                predictions: Tuple[torch.Tensor, torch.Tensor],
                targets: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        class_pred, strength_pred = predictions
        class_target, strength_target = targets

        if self.use_focal_loss:
            # Focal Loss handles class imbalance internally via alpha parameter or we can pass weights if modified
            class_loss = self.classification_loss(class_pred, class_target.long())
        else:
            class_weights = self.class_weights.to(class_pred.device)
            class_loss = F.cross_entropy(class_pred, class_target.long(), weight=class_weights)

        mask = (class_target == 1).float()
        if mask.sum().item() > 0:
            strength_pred = strength_pred.squeeze(-1)
            reg_loss = self.mse_loss(strength_pred * mask, strength_target * mask)
        else:
            reg_loss = torch.zeros(1, device=class_pred.device, dtype=class_pred.dtype).squeeze()

        total_loss = self.alpha * class_loss + (1 - self.alpha) * reg_loss
        return total_loss, class_loss, reg_loss
