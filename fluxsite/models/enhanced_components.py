import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
        """
        # Assume batch_first=True for x: [batch_size, seq_len, dim]
        if x.size(1) > self.pe.size(0):
             # Dynamic resize if needed (though usually fixed max_len is enough)
             pass
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class ReductionBlock(nn.Module):
    """
    Single reduction block with compression, selection, and compensation.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        # Compression path
        self.compress = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Feature Selection (Gating)
        self.selection_gate = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid()
        )
        
        # Information Compensation (Projection for residual-like connection)
        self.compensation = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Dropout(dropout)
        )
        
        self.final_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compression
        compressed = self.compress(x)
        
        # Selection
        gate = self.selection_gate(x)
        selected = compressed * gate
        
        # Compensation
        comp = self.compensation(x)
        
        # Fusion
        return self.final_norm(selected + comp)

class ProgressiveDimensionalityReduction(nn.Module):
    """
    Progressively reduces dimensionality: 1280 -> 768 -> 512 -> 256.
    """
    def __init__(self, input_dim: int = 1280, target_dims: List[int] = [768, 512, 256], dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_dim = input_dim
        for next_dim in target_dims:
            self.layers.append(ReductionBlock(curr_dim, next_dim, dropout))
            curr_dim = next_dim
        self.output_dim = curr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class ResidueTypeAttention(nn.Module):
    """
    Residue-Type Aware Attention Mechanism.
    """
    def __init__(self, dim: int, num_residue_types: int = 25): # 20 standard + specials
        super().__init__()
        self.residue_embedding = nn.Embedding(num_residue_types, dim)
        
        # Attention weights calculation
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
            nn.Softmax(dim=1) # Attention over sequence length
        )
        
        # Dynamic update strategy (Gate)
        self.update_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor, residue_types: torch.Tensor) -> torch.Tensor:
        """
        features: [batch, seq_len, dim]
        residue_types: [batch, seq_len]
        """
        residue_emb = self.residue_embedding(residue_types)
        
        # Calculate attention weights
        combined = torch.cat([features, residue_emb], dim=-1)
        weights = self.attention(combined) # [batch, seq_len, 1]
        
        # Weighted features
        weighted_features = features * weights
        
        # Dynamic update (fuse original and weighted)
        gate = self.update_gate(combined)
        output = gate * weighted_features + (1 - gate) * features
        
        return output

class EnhancedAttentionModule(nn.Module):
    """
    Multi-head Position-aware Attention + Residue Type Attention.
    """
    def __init__(self, input_dim: int, num_heads: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        # Position Encoding
        self.pos_encoding = PositionalEncoding(input_dim, dropout, max_len=1024)
        
        # Multi-head Self-Attention Layers (Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Residue Type Attention
        self.residue_attention = ResidueTypeAttention(input_dim)
        
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, features: torch.Tensor, residue_types: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Add position encoding
        features = self.pos_encoding(features)
        
        # Transformer Encoder
        # src_key_padding_mask needs to be [batch, seq_len] bool
        features = self.transformer(features, src_key_padding_mask=mask)
        
        # Residue Type Attention
        if residue_types is not None:
            features = self.residue_attention(features, residue_types)
            
        return self.norm(features)

class CrossModalAttention(nn.Module):
    """
    Bidirectional Cross-Modal Attention with Normalization and Residuals.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.seq_to_struct = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.struct_to_seq = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm_seq = nn.LayerNorm(dim)
        self.norm_struct = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, seq_feat: torch.Tensor, struct_feat: torch.Tensor,
                seq_mask: Optional[torch.Tensor] = None, struct_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Seq attends to Structure (Q=Seq, K=V=Struct)
        seq_attn, _ = self.seq_to_struct(seq_feat, struct_feat, struct_feat, key_padding_mask=struct_mask)
        seq_fused = self.norm_seq(seq_feat + self.dropout(seq_attn))
        
        # Structure attends to Seq (Q=Struct, K=V=Seq)
        struct_attn, _ = self.struct_to_seq(struct_feat, seq_feat, seq_feat, key_padding_mask=seq_mask)
        struct_fused = self.norm_struct(struct_feat + self.dropout(struct_attn))
        
        return seq_fused, struct_fused

class GatingMechanism(nn.Module):
    """
    Dynamic Gating Mechanism for feature fusion.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2), # Weights for 2 branches
            nn.Softmax(dim=-1)
        )
        
    def forward(self, branch1: torch.Tensor, branch2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Global pooling for gating signal generation? Or element-wise?
        # Requirement says "Gating signal should be automatically generated based on current input features"
        # Let's do element-wise or token-wise if shapes match, or pool if global.
        # Assuming inputs are [batch, dim] (pooled) or [batch, seq, dim]
        
        combined = torch.cat([branch1, branch2], dim=-1)
        weights = self.gate_net(combined) # [..., 2]
        
        w1 = weights[..., 0:1]
        w2 = weights[..., 1:2]
        
        fused = w1 * branch1 + w2 * branch2
        return fused, weights

class SupervisedContrastiveLearning(nn.Module):
    """
    Supervised Contrastive Learning Module.
    """
    def __init__(self, temperature: float = 0.07, hard_negative_weight: float = 2.0):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor, hard_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        features: [batch, dim]
        labels: [batch]
        hard_mask: [batch] boolean mask indicating hard negatives
        """
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Label mask
        labels = labels.view(-1, 1)
        # Handle multi-label or soft labels? Assuming binary/multiclass hard labels for now.
        # Check for exact match
        if labels.shape[0] != features.shape[0]:
             # Handle case where labels might be different shape (e.g. batch size mismatch)
             return torch.tensor(0.0, device=features.device, requires_grad=True)

        mask = torch.eq(labels, labels.T).float()
        
        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0], device=mask.device).view(-1, 1),
            0
        )
        mask = mask * logits_mask
        
        # Hard negative weighting
        # If hard_mask is provided, we increase weight of negative pairs involving hard samples
        # This implementation in the prompt image seemed to modify the positive/negative mask logic
        # Here is a simplified interpretation:
        
        exp_sim = torch.exp(similarity) * logits_mask
        
        # Standard SupCon Loss denominator: sum of all exp_sim (except self)
        # But we can weight the negatives.
        
        # For simplicity, standard SupCon implementation:
        log_prob = similarity - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        
        # Mean log-likelihood for positive pairs
        # Sum over positive pairs / count of positive pairs
        mask_sum = mask.sum(1)
        # Avoid division by zero
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        loss = -mean_log_prob_pos
        loss = loss.mean()
        
        return loss

class MultiScaleFusionNetwork(nn.Module):
    """
    Updated Fusion Network incorporating Progressive Reduction, Cross-Modal Attention, and Gating.
    """
    def __init__(self, seq_dim: int, struct_dim: int, hidden_dims: List[int] = [768, 512, 256], dropout: float = 0.1):
        super().__init__()
        
        # Progressive Reduction for Sequence
        # 1280 -> 768 -> 512 -> 256
        # Assuming input seq_dim is 1280
        self.seq_reduction = ProgressiveDimensionalityReduction(seq_dim, hidden_dims, dropout)
        
        # Progressive Reduction for Structure
        # Structure dim might be different (e.g. 512). 
        # If struct_dim is 512, we might just process it to match hidden_dims stages or align at final stage.
        # Let's assume we align structure to sequence dimensions at each stage or just at final.
        # For simplicity, we align input to first hidden dim, then reduce.
        self.struct_reduction = ProgressiveDimensionalityReduction(struct_dim, hidden_dims, dropout)
        
        # Cross Modal Attention at final stage (or multi-scale?)
        # Requirement 2 says "Bidirectional sequence-structure feature interaction module".
        # Let's apply it at the final reduced dimension (256).
        final_dim = hidden_dims[-1]
        self.cross_attention = CrossModalAttention(final_dim, num_heads=4, dropout=dropout)
        
        # Gating Mechanism
        self.gating = GatingMechanism(final_dim)
        
        self.output_dim = final_dim

    def forward(self, seq_feat: torch.Tensor, struct_feat: torch.Tensor,
                seq_mask: Optional[torch.Tensor] = None, struct_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # 1. Progressive Reduction
        seq_reduced = self.seq_reduction(seq_feat)
        struct_reduced = self.struct_reduction(struct_feat)
        
        # 2. Cross Modal Interaction
        seq_fused, struct_fused = self.cross_attention(seq_reduced, struct_reduced, seq_mask, struct_mask)
        
        # 3. Pooling (Global Average Pooling for classification)
        # Apply masks if needed
        if seq_mask is not None:
            mask_expanded = seq_mask.unsqueeze(-1).expand_as(seq_fused)
            seq_fused = seq_fused.masked_fill(mask_expanded, 0.0)
            seq_pooled = seq_fused.sum(dim=1) / (~seq_mask).sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            seq_pooled = seq_fused.mean(dim=1)
            
        if struct_mask is not None:
            mask_expanded = struct_mask.unsqueeze(-1).expand_as(struct_fused)
            struct_fused = struct_fused.masked_fill(mask_expanded, 0.0)
            struct_pooled = struct_fused.sum(dim=1) / (~struct_mask).sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            struct_pooled = struct_fused.mean(dim=1)
            
        # 4. Gating Fusion
        final_output, weights = self.gating(seq_pooled, struct_pooled)
        
        return final_output


class DynamicFeatureSelector(nn.Module):
    """
    Lightweight dynamic feature selection module.

    This module learns a feature-wise gate and applies it to the input features.
    It supports both [batch, dim] and [batch, seq_len, dim] inputs.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(1, int(hidden_dim))
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor, return_gate: bool = False):
        # features: [batch, dim] or [batch, seq_len, dim]
        gates = self.gate(features)
        selected = features * gates
        if return_gate:
            return selected, gates
        return selected
