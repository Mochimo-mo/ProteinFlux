import math
from typing import Optional, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .attention_pooling import AttentionPooling
from .cnn_model import MultiScaleCNNBlock
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


def _ensure_bool_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        return mask.to(dtype=torch.bool)
    return mask


def _sanitize_padding_mask(mask: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return a safe padding mask and rows that were fully masked.

    Some attention kernels can produce NaN if a row has all keys masked.
    For such rows we temporarily unmask all keys and handle fallback downstream.
    """
    mask_bool = _ensure_bool_mask(mask)
    if mask_bool is None or mask_bool.ndim != 2:
        return mask_bool, None

    all_masked = mask_bool.all(dim=1)
    if all_masked.any():
        mask_bool = mask_bool.clone()
        mask_bool[all_masked] = False
    return mask_bool, all_masked


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute mean across sequence length with optional padding mask."""
    if mask is None:
        return tensor.mean(dim=1)

    mask_bool = _ensure_bool_mask(mask)
    valid = (~mask_bool).to(tensor.dtype)
    denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    
    # [NaN Fix] Use masked_fill to zero out padded positions. 
    # Direct multiplication (tensor * valid) propagates NaNs if they exist in padded regions.
    tensor_safe = tensor.masked_fill(mask_bool.unsqueeze(-1), 0.0)
    
    return tensor_safe.sum(dim=1) / denom


class SequenceTower(nn.Module):
    """Transformer-based sequence branch with bottleneck projection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_multiplier: int = 4,
        use_attention_pooling: bool = True,
        window_radius: int = 5,
    ) -> None:
        super().__init__()
        self.use_attention_pooling = use_attention_pooling
        self.window_radius = window_radius
        self.input_dim = input_dim
        
        # [Modified] Enhanced projection to retain high-dimensional features longer
        mid_dim = max(hidden_dim, input_dim // 2)
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # [New] Multi-scale CNN for local motif capture (ψKxE etc.)
        self.cnn_encoder = MultiScaleCNNBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_sizes=[3, 5, 7],
            dropout=dropout
        )
        
        # [New] Positional encoding: provide explicit positional information for the Transformer
        self.pos_encoder = PositionalEncoding(hidden_dim)

        try:
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * ff_multiplier,
                dropout=dropout,
                batch_first=True,
                enable_nested_tensor=False,
            )
        except TypeError:
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * ff_multiplier,
                dropout=dropout,
                batch_first=True,
            )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(hidden_dim)
        
        if self.use_attention_pooling:
            self.attention_pool = AttentionPooling(hidden_dim, max(32, hidden_dim // 4))
            # Fusion: Center + Context -> Global
            self.global_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
            self.global_norm = nn.LayerNorm(hidden_dim)
            self.dropout_layer = nn.Dropout(dropout)
            self.center_dropout = nn.Dropout(dropout) # Added for consistency with StructureTower
    def forward(
        self,
        sequence_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        center_alpha: float = 1.0,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.projection(sequence_features)
        
        # [New] Apply CNN for local features
        # Input: [Batch, SeqLen, Dim] -> [Batch, Dim, SeqLen]
        hidden = hidden.transpose(1, 2)
        hidden = self.cnn_encoder(hidden)
        # Output: [Batch, Dim, SeqLen] -> [Batch, SeqLen, Dim]
        hidden = hidden.transpose(1, 2)
        
        # [New] Add positional encoding
        hidden = self.pos_encoder(hidden)
        
        padding_mask, all_masked = _sanitize_padding_mask(mask)
        contextual = self.encoder(hidden, src_key_padding_mask=padding_mask)
        contextual = torch.nan_to_num(contextual, nan=0.0, posinf=0.0, neginf=0.0)
        if all_masked is not None and all_masked.any():
            contextual = contextual.clone()
            contextual[all_masked] = 0.0
        contextual = self.post_norm(contextual + hidden)
        
        attn_weights = None
        
        if self.use_attention_pooling:
            # 1. Attention Pooling for Context
            pooled_context, weights = self.attention_pool(contextual, padding_mask)
            attn_weights = weights
            
            # 2. Center Residue Feature (Explicitly focus on the PTM site)
            seq_len = contextual.size(1)
            center_idx = seq_len // 2
            # Ensure center index is valid
            if center_idx < seq_len:
                # Use configured window around center
                radius = getattr(self, "window_radius", 5)
                start = max(0, center_idx - radius)
                end = min(seq_len, center_idx + radius + 1)
                center_feature = contextual[:, start:end, :].mean(dim=1)
            else:
                center_feature = pooled_context # Fallback
            
            # [New] Apply targeted dropout to the central features
            if hasattr(self, 'center_dropout'):
                center_feature = self.center_dropout(center_feature)
            if center_alpha < 0.9999:
                center_feature = center_feature * center_alpha
                
            # 3. Fuse Center and Context
            combined = torch.cat([center_feature, pooled_context], dim=-1)
            global_repr = self.global_fusion(combined)
            global_repr = self.global_norm(self.dropout_layer(global_repr))
        else:
            global_repr = masked_mean(contextual, padding_mask)
            if return_weights:
                # Create uniform weights if attention pooling is not used
                if padding_mask is not None:
                    valid_mask = (~padding_mask).float()
                    attn_weights = valid_mask / valid_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
                else:
                    bs, sl = contextual.shape[:2]
                    attn_weights = torch.ones(bs, sl, device=contextual.device) / sl
        
        if return_weights:
            return contextual, global_repr, attn_weights
            
        return contextual, global_repr


class SpatialGraphModule(nn.Module):
    """
    Graph Attention Network (GAT) based spatial encoding module.
    Uses protein structure distance matrix to guide attention.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Attention scores
        self.att_src = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        
        # [Optimization] Learnable distance bias scale
        self.dist_bias_scale = nn.Parameter(torch.tensor(0.5))
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, distance_matrix: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [Batch, SeqLen, HiddenDim]
        distance_matrix: [Batch, SeqLen, SeqLen] (optional, lower is closer)
        mask: [Batch, SeqLen] (optional, True for padding)
        """
        residual = x
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.linear_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.linear_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.linear_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute attention logits
        # Simple GAT style: (a^T [Wh_i || Wh_j])
        # Here we use dot product attention similar to Transformer but biased by distance
        
        # Standard attention scores
        # [B, H, L, D] @ [B, H, D, L] -> [B, H, L, L]
        q = q.permute(0, 2, 1, 3) # [B, H, L, D]
        k = k.permute(0, 2, 3, 1) # [B, H, D, L]
        v = v.permute(0, 2, 1, 3) # [B, H, L, D]
        
        attn_logits = torch.matmul(q, k) / math.sqrt(self.head_dim)
        
        # Incorporate spatial information
        if distance_matrix is not None:
            distance_matrix = torch.nan_to_num(distance_matrix, nan=0.0, posinf=50.0, neginf=0.0)
            # Assume distance_matrix is [B, L, L]
            # Convert distances to bias: closer = higher attention
            # Simple Gaussian kernel or threshold
            # Bias = -distance (so closer is larger) or 1/(1+dist)
            # Let's assume pre-processed distances or apply a kernel here
            # For now, assuming distance_matrix contains valid distances (e.g. Angstroms)
            # Mask out far interactions (> 10A) or learn a bias
            
            # Add bias: shape [B, 1, L, L]
            # [Optimization] Use learnable scale for distance bias
            spatial_bias = -1.0 * distance_matrix.unsqueeze(1) 
            attn_logits = attn_logits + self.dist_bias_scale * spatial_bias

        # [Optimization] Apply padding mask to prevent attention to padding tokens
        if mask is not None:
            # mask is True for padding. Expand to [B, 1, 1, L] (masking keys)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            # Use a large finite negative value instead of -inf to avoid NaN when a row is fully masked.
            attn_logits = attn_logits.masked_fill(mask_expanded, -1e4)
            
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=1.0, neginf=0.0)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v) # [B, H, L, D]
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        
        out = self.out_proj(out)
        out = self.norm(residual + self.dropout(out))
        
        return out


class StructureTower(nn.Module):
    """Structure branch: Simplified to match SequenceTower (Pure Transformer)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_multiplier: int = 4,
        dilations: Tuple[int, ...] = (1, 2, 3), # Kept for compatibility but unused
        use_attention_pooling: bool = True,
        window_radius: int = 3,
    ) -> None:
        super().__init__()
        self.use_attention_pooling = use_attention_pooling
        self.window_radius = window_radius
        self.input_dim = input_dim
        
        # 1. Projection: Input Dim -> Hidden Dim
        # [Modified] Enhanced projection to retain high-dimensional features longer
        mid_dim = max(hidden_dim, input_dim // 2)
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # [New] Spatial Graph Module
        self.spatial_gnn = SpatialGraphModule(hidden_dim, num_heads=num_heads, dropout=dropout)

        # 3. Transformer Encoder (Unrolled for local residuals)
        try:
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * ff_multiplier,
                dropout=dropout,
                batch_first=True,
                enable_nested_tensor=False,
            )
        except TypeError:
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * ff_multiplier,
                dropout=dropout,
                batch_first=True,
            )
            
        # Replaced nn.TransformerEncoder with ModuleList to allow custom forward logic
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.post_norm = nn.LayerNorm(hidden_dim)
        
        # [Optimization] Learnable residual weight for spatial context
        self.spatial_residual_weight = nn.Parameter(torch.tensor(0.5))
        
        if self.use_attention_pooling:
            self.attention_pool = AttentionPooling(hidden_dim, max(32, hidden_dim // 4))
            self.global_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
            self.global_norm = nn.LayerNorm(hidden_dim)
            self.dropout_layer = nn.Dropout(dropout)
            self.center_dropout = nn.Dropout(dropout)

    def forward(
        self,
        structure_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        center_alpha: float = 1.0,
        return_weights: bool = False,
        distance_matrices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Projection
        hidden = self.projection(structure_features)
        
        # 2. Positional Encoding
        hidden = self.pos_encoder(hidden)

        # 3. Transformer Encoding with Local Residuals
        padding_mask, all_masked = _sanitize_padding_mask(mask)

        # [New] Spatial GNN
        # Capture local geometry before Transformer
        # [Optimization] Pass mask to SpatialGNN to avoid attending to padding
        spatial_context = self.spatial_gnn(hidden, distance_matrix=distance_matrices, mask=padding_mask)
        
        curr = spatial_context
        for layer in self.layers:
            # Standard Transformer Layer
            curr = layer(curr, src_key_padding_mask=padding_mask)
            curr = torch.nan_to_num(curr, nan=0.0, posinf=0.0, neginf=0.0)
            # [New] Add local residual connection (from spatial context) to preserve geometry
            # [Optimization] Use learnable weight
            curr = curr + self.spatial_residual_weight * spatial_context
            
        contextual = self.post_norm(curr)
        contextual = torch.nan_to_num(contextual, nan=0.0, posinf=0.0, neginf=0.0)
        if all_masked is not None and all_masked.any():
            contextual = contextual.clone()
            contextual[all_masked] = 0.0
        
        attn_weights = None
        
        if self.use_attention_pooling:
            # 1. Attention Pooling
            pooled_context, weights = self.attention_pool(contextual, padding_mask)
            attn_weights = weights
            
            # 2. Center Residue Feature
            seq_len = contextual.size(1)
            center_idx = seq_len // 2
            
            # [Optimization] Use EXACT center residue feature for structure as well
            if center_idx < seq_len:
                center_feature = contextual[:, center_idx, :]
            else:
                center_feature = pooled_context
            
            # Apply dropout
            if hasattr(self, 'center_dropout'):
                center_feature = self.center_dropout(center_feature)
            if center_alpha < 0.9999:
                center_feature = center_feature * center_alpha
                
            # 3. Fusion
            combined = torch.cat([center_feature, pooled_context], dim=-1)
            global_repr = self.global_fusion(combined)
            global_repr = self.global_norm(self.dropout_layer(global_repr))
        else:
            global_repr = masked_mean(contextual, padding_mask)
            if return_weights:
                # Create uniform weights
                if padding_mask is not None:
                    valid_mask = (~padding_mask).float()
                    attn_weights = valid_mask / valid_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
                else:
                    bs, sl = contextual.shape[:2]
                    attn_weights = torch.ones(bs, sl, device=contextual.device) / sl
                    
        if return_weights:
            return contextual, global_repr, attn_weights
            
        return contextual, global_repr


class CrossModalFusionLayer(nn.Module):
    """Bi-directional or Uni-directional cross-attention to align sequence and structure modalities."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1, bidirectional: bool = True) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.seq_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        if self.bidirectional:
            self.struct_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.struct_norm = nn.LayerNorm(hidden_dim)
            
        self.seq_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        seq_repr: torch.Tensor,
        struct_repr: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        struct_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        struct_padding, struct_all_masked = _sanitize_padding_mask(struct_mask)
        seq_padding, seq_all_masked = _sanitize_padding_mask(seq_mask)

        # Sequence attends to Structure (Q=Seq, K=Struct, V=Struct)
        seq_context, _ = self.seq_attn(
            seq_repr,
            struct_repr,
            struct_repr,
            key_padding_mask=struct_padding,
        )
        seq_context = torch.nan_to_num(seq_context, nan=0.0, posinf=0.0, neginf=0.0)
        if struct_all_masked is not None and struct_all_masked.any():
            seq_context = seq_context.clone()
            seq_context[struct_all_masked] = 0.0
        seq_fused = self.seq_norm(seq_repr + seq_context)

        if self.bidirectional:
            # Structure attends to Sequence (Q=Struct, K=Seq, V=Seq)
            struct_context, _ = self.struct_attn(
                struct_repr,
                seq_repr,
                seq_repr,
                key_padding_mask=seq_padding,
            )
            struct_context = torch.nan_to_num(struct_context, nan=0.0, posinf=0.0, neginf=0.0)
            if seq_all_masked is not None and seq_all_masked.any():
                struct_context = struct_context.clone()
                struct_context[seq_all_masked] = 0.0
            struct_fused = self.struct_norm(struct_repr + struct_context)
        else:
            # In uni-directional mode, structure features remain unchanged
            struct_fused = struct_repr

        return seq_fused, struct_fused


class AdaptiveGating(nn.Module):
    """Learnable gating between sequence and structure representations with feature-wise control."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # Use a more expressive gating network that outputs feature-wise weights
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        seq_global: torch.Tensor,
        struct_global: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if struct_global is None:
            batch_size, dim = seq_global.shape
            # Return weights as 1.0 for sequence, 0.0 for structure
            weights = seq_global.new_ones(batch_size, dim)
            return seq_global, weights

        combined = torch.cat([seq_global, struct_global], dim=-1)
        # gate_vector will be in [0, 1] range, representing the weight for the sequence branch
        # 1-gate_vector will be the weight for the structure branch
        gate_vector = self.gate(combined)
        
        fused = gate_vector * seq_global + (1.0 - gate_vector) * struct_global
        return fused, gate_vector


class ProjectionHead(nn.Module):
    """Projection head for contrastive objectives."""

    def __init__(self, input_dim: int, projection_dim: int = 128, hidden_dim: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.net(features)
        return F.normalize(projected, dim=-1)


class SequenceContextEncoder(SequenceTower):
    """Wrapper around SequenceTower to match the interface expected by AcetylationPredictionModel."""
    def __init__(self, input_dim: int, hidden_dim: int, context_window: int = 15):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            window_radius=context_window // 2
        )
        self.output_dim = hidden_dim

    def forward(self, x, site_positions=None, mask=None):
        # SequenceTower.forward returns (contextual, global_repr)
        contextual, global_repr = super().forward(x, mask=mask)
        # Return (site_feature, context_features, padding_mask)
        return global_repr, contextual, mask


class StructureContextEncoder(StructureTower):
    """Wrapper around StructureTower to match the interface expected by AcetylationPredictionModel."""
    def __init__(self, input_dim: int, hidden_dim: int, context_window: int = 15, adaptive_config=None, use_adaptive_dropout=False):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            window_radius=context_window // 2
        )
        self.output_dim = hidden_dim

    def forward(self, x, site_positions=None, distance_matrices=None):
        # StructureTower currently doesn't use distance_matrices, but we accept it for compatibility
        # We assume x is structure features. Mask is not provided usually for structure here?
        # If x has padding, we might need a mask. Assuming x is [batch, len, dim]
        
        # If we want to generate a mask from x (assuming zero padding):
        mask = (x.sum(dim=-1) == 0) if x is not None else None
        
        # [Fix] If all tokens are masked (e.g. dummy zero structures), unmask them to avoid Transformer crash
        if mask is not None and mask.all():
            mask = None
        
        # [Fix] Pass distance_matrices to super().forward to enable SpatialGNN
        contextual, global_repr = super().forward(x, mask=mask, distance_matrices=distance_matrices)
        return global_repr, contextual, mask


class GatedFusionModule(nn.Module):
    """
    Advanced fusion module with Context-Aware Cross-Attention.
    Query = Site Feature
    Key/Value = Context Features from OTHER modality (or same?)
    """
    def __init__(self, seq_dim: int, struct_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2, adaptive_config=None, use_adaptive_dropout=False):
        super().__init__()
        
        # Projections to common dimension for attention
        self.seq_site_proj = nn.Linear(seq_dim, hidden_dim)
        self.struct_site_proj = nn.Linear(struct_dim, hidden_dim)
        
        self.seq_ctx_proj = nn.Linear(seq_dim, hidden_dim)
        self.struct_ctx_proj = nn.Linear(struct_dim, hidden_dim)
        
        # Cross Attention: Seq Site attends to Structure Context
        self.cross_attn_seq_to_struct = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross Attention: Structure Site attends to Sequence Context
        self.cross_attn_struct_to_seq = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm_seq = nn.LayerNorm(hidden_dim)
        self.norm_struct = nn.LayerNorm(hidden_dim)
        
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # [New] Confidence-based Adaptive Gating
        # Estimate confidence of structure branch
        self.struct_conf_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # Learnable parameters for gating function: sigmoid(alpha * conf + beta)
        self.conf_alpha = nn.Parameter(torch.tensor(5.0))
        self.conf_beta = nn.Parameter(torch.tensor(-2.0))
        
        # Final fusion
        self.fusion_proj = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, seq_site, struct_site, seq_context, struct_context, seq_padding_mask=None, struct_padding_mask=None):
        # 1. Project inputs
        seq_s = self.seq_site_proj(seq_site)       # [B, H]
        struct_s = self.struct_site_proj(struct_site) # [B, H]
        
        seq_c = self.seq_ctx_proj(seq_context)     # [B, L, H]
        struct_c = self.struct_ctx_proj(struct_context) # [B, L, H]
        
        # 2. Cross Attention
        # Seq Site (Q) -> Struct Context (K, V)
        # Q needs to be [B, 1, H] for batch_first=True
        seq_s_unsqueezed = seq_s.unsqueeze(1)
        
        struct_padding_mask = _ensure_bool_mask(struct_padding_mask)
        seq_padding_mask = _ensure_bool_mask(seq_padding_mask)

        # Attn: Q=SeqSite, K=StructContext, V=StructContext
        seq_cross, _ = self.cross_attn_seq_to_struct(
            seq_s_unsqueezed, 
            struct_c, 
            struct_c, 
            key_padding_mask=struct_padding_mask
        )
        seq_cross = seq_cross.squeeze(1) # [B, H]
        seq_enhanced = self.norm_seq(seq_s + seq_cross)
        
        # Struct Site (Q) -> Seq Context (K, V)
        struct_s_unsqueezed = struct_s.unsqueeze(1)
        struct_cross, _ = self.cross_attn_struct_to_seq(
            struct_s_unsqueezed, 
            seq_c, 
            seq_c, 
            key_padding_mask=seq_padding_mask
        )
        struct_cross = struct_cross.squeeze(1) # [B, H]
        struct_enhanced = self.norm_struct(struct_s + struct_cross)
        
        # [New] Adaptive Gating based on Confidence
        # 1. Calculate confidence score for structure features
        raw_conf = self.struct_conf_estimator(struct_enhanced) # [B, 1]
        
        # 2. Apply gating function: sigmoid(alpha * conf + beta)
        confidence = torch.sigmoid(self.conf_alpha * raw_conf + self.conf_beta)
        
        # 3. Re-weight structure features based on confidence
        # If confidence is low, reduce contribution of structure branch
        struct_weighted = struct_enhanced * confidence
        
        combined = torch.cat([seq_enhanced, struct_weighted], dim=-1)
        gate_vector = self.gate_net(combined)
        fused = torch.cat([gate_vector * seq_enhanced, (1.0 - gate_vector) * struct_weighted], dim=-1)
        output = self.fusion_proj(fused)
        output = self.layer_norm(self.dropout(output))
        
        return output
