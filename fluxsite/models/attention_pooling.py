
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer that learns to weight different positions in the sequence.
    Useful for focusing on the center residue and other key motifs instead of simple averaging.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional boolean mask of shape (batch_size, seq_len), True indicates padding/masked
            
        Returns:
            pooled: Weighted sum of input features (batch_size, input_dim)
            weights: Attention weights (batch_size, seq_len)
        """
        # Calculate attention scores
        scores = self.attention_net(x).squeeze(-1) # (batch_size, seq_len)
        
        # Apply mask if provided (mask positions with a very small value)
        if mask is not None:
            # Use float('-inf') or the minimum representable value for scores.dtype to avoid float16 overflow under AMP
            fill_value = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(mask, fill_value)
            
        # Softmax to get weights
        weights = F.softmax(scores, dim=1) # (batch_size, seq_len)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Weighted sum
        # x: (B, L, D), weights: (B, L) -> (B, L, 1)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1) # (batch_size, input_dim)
        
        return pooled, weights

class CenterFocusedPooling(nn.Module):
    """
    Pooling strategy that explicitly concatenates the center residue feature 
    with an attention-pooled context.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attention_pooling = AttentionPooling(input_dim, hidden_dim)
        # Output dimension will be input_dim (center) + input_dim (context)
        self.output_dim = input_dim * 2
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            mask: Padding mask
        """
        batch_size, seq_len, _ = x.shape
        center_idx = seq_len // 2
        
        # Extract center feature
        center_feature = x[:, center_idx, :]
        
        # Get context feature via attention
        context_feature, _ = self.attention_pooling(x, mask)
        
        # Concatenate
        combined = torch.cat([center_feature, context_feature], dim=-1)
        
        return combined
