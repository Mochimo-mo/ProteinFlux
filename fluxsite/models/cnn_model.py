import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCNNBlock(nn.Module):
    """
    Multi-scale CNN Block that captures motifs of different lengths using parallel convolutions.
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, k, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        # Fusion of multi-scale features
        self.fusion = nn.Conv1d(out_channels * len(kernel_sizes), out_channels, 1)
        
        # Residual connection
        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
            
        self.final_relu = nn.ReLU()

    def forward(self, x):
        # x: [B, C, L]
        outs = [conv(x) for conv in self.convs]
        out = torch.cat(outs, dim=1)
        out = self.fusion(out)
        res = self.residual(x)
        return self.final_relu(out + res)

class CNNDualStreamPredictor(nn.Module):
    """
    A Dual-Stream CNN model for PTM prediction.
    Stream 1: Sequence Features (e.g., ESM-2)
    Stream 2: Structure Features (e.g., ESM-IF)
    
    Features are processed by multi-scale 1D convolutions and then fused.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.seq_dim = config.get("seq_feature_dim", 1280)
        self.struct_dim = config.get("struct_feature_dim", 512)
        # Prefer branch_hidden_dim (config key); fall back to hidden_dim or default 256 when missing
        self.hidden_dim = config.get("branch_hidden_dim", config.get("hidden_dim", 256))
        self.dropout = config.get("dropout", 0.2)
        
        # Sequence Branch
        self.seq_proj = nn.Conv1d(self.seq_dim, self.hidden_dim, 1)
        self.seq_encoder = nn.Sequential(
            MultiScaleCNNBlock(self.hidden_dim, self.hidden_dim, kernel_sizes=[3, 5, 7], dropout=self.dropout),
            MultiScaleCNNBlock(self.hidden_dim, self.hidden_dim, kernel_sizes=[3, 5, 9], dropout=self.dropout),
            nn.MaxPool1d(2, stride=1, padding=1) # Slight pooling to increase receptive field without losing too much resolution
        )
        
        # Structure Branch
        self.struct_proj = nn.Conv1d(self.struct_dim, self.hidden_dim, 1)
        self.struct_encoder = nn.Sequential(
            MultiScaleCNNBlock(self.hidden_dim, self.hidden_dim, kernel_sizes=[3, 5, 7], dropout=self.dropout),
            MultiScaleCNNBlock(self.hidden_dim, self.hidden_dim, kernel_sizes=[3, 5, 9], dropout=self.dropout),
            nn.MaxPool1d(2, stride=1, padding=1)
        )
        
        # Attention-based Fusion
        # Compute attention weights for Sequence vs Structure at each position
        self.fusion_gate = nn.Sequential(
            nn.Conv1d(self.hidden_dim * 2, self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Post-Fusion Processing
        self.fusion_block = nn.Sequential(
            nn.Conv1d(self.hidden_dim * 2, self.hidden_dim, 1), # Reduce dimension
            MultiScaleCNNBlock(self.hidden_dim, self.hidden_dim, kernel_sizes=[3, 5], dropout=self.dropout)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        # Auxiliary Strength Head (for compatibility with MultiTaskLoss)
        self.strength_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, batch):
        # Extract data from batch
        sequences = batch['sequence_features'] # [B, L, D_seq]
        structures = batch['structure_features'] # [B, L, D_struct]
        
        # Transpose for CNN: [B, D, L]
        x_seq = sequences.transpose(1, 2)
        x_struct = structures.transpose(1, 2)
        
        # --- Sequence Stream ---
        x_seq = self.seq_proj(x_seq)
        x_seq = self.seq_encoder(x_seq)
        
        # --- Structure Stream ---
        x_struct = self.struct_proj(x_struct)
        x_struct = self.struct_encoder(x_struct)
        
        # Ensure lengths match (in case of padding/pooling differences, though here they should match)
        min_len = min(x_seq.size(2), x_struct.size(2))
        x_seq = x_seq[:, :, :min_len]
        x_struct = x_struct[:, :, :min_len]
        
        # --- Fusion ---
        combined = torch.cat([x_seq, x_struct], dim=1) # [B, 2*H, L]
        
        # Gated Fusion
        # gate = self.fusion_gate(combined) # [B, H, L]
        # x_fused = x_seq * gate + x_struct * (1 - gate) # Weighted sum
        # Alternatively, just process the concatenated features which is more flexible
        
        x_fused = self.fusion_block(combined) # [B, H, L]
        
        # --- Prediction ---
        # Extract center position feature
        # Note: MaxPool1d(2, stride=1, padding=1) keeps the sequence length roughly unchanged.
        # Recompute the center index defensively.
        L = x_fused.size(2)
        center_idx = L // 2
        
        # Extract center-position features
        center_features = x_fused[:, :, center_idx] # [B, H]
        
        # If use_attention_pooling is enabled, use global average pooling or attention pooling to enhance the center features
        if self.config.get('use_attention_pooling', False):
            # Global average pooling
            global_features = torch.mean(x_fused, dim=2) # [B, H]
            center_features = center_features + global_features
        
        logits = self.classifier(center_features)
        strength_logits = self.strength_head(center_features)
        
        return logits, strength_logits

class CNNSequencePredictor(nn.Module):
    """
    A Single-Stream CNN model for PTM prediction using only Sequence Features (e.g., ESM-2).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.seq_dim = config.get("seq_feature_dim", 1280)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.dropout = config.get("dropout", 0.2)
        
        # Sequence Branch
        self.seq_proj = nn.Conv1d(self.seq_dim, self.hidden_dim, 1)
        self.seq_encoder = nn.Sequential(
            MultiScaleCNNBlock(self.hidden_dim, self.hidden_dim, kernel_sizes=[3, 5, 7], dropout=self.dropout),
            MultiScaleCNNBlock(self.hidden_dim, self.hidden_dim, kernel_sizes=[3, 5, 9], dropout=self.dropout),
            nn.MaxPool1d(2, stride=1, padding=1)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, batch):
        # Extract data from batch
        sequences = batch['sequence_features'] # [B, L, D_seq]
        
        # Transpose for CNN: [B, D, L]
        x_seq = sequences.transpose(1, 2)
        
        # --- Sequence Stream ---
        x_seq = self.seq_proj(x_seq)
        x_seq = self.seq_encoder(x_seq)
        
        # --- Prediction ---
        L = x_seq.size(2)
        center_idx = L // 2
        
        # Extract center position feature
        center_features = x_seq[:, :, center_idx] # [B, H]
        
        # Global pooling option
        if self.config.get('use_attention_pooling', False):
            global_features = torch.mean(x_seq, dim=2) # [B, H]
            center_features = center_features + global_features
        
        logits = self.classifier(center_features)
        
        return logits

class DeepMVPSequencePredictor(nn.Module):
    """
    Enhanced DeepMVP-style model for PTM prediction using ESM-2 features.
    Combines Multi-scale CNN (local motifs) with BiGRU (global dependencies) and Attention.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.seq_dim = config.get("seq_feature_dim", 1280)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.rnn_hidden_dim = config.get("rnn_hidden_dim", 128)
        self.dropout = config.get("dropout", 0.2)
        
        # 1. Projection & CNN Encoder (Local Motifs)
        self.seq_proj = nn.Conv1d(self.seq_dim, self.hidden_dim, 1)
        self.cnn_encoder = nn.Sequential(
            MultiScaleCNNBlock(self.hidden_dim, self.hidden_dim, kernel_sizes=[3, 5, 7], dropout=self.dropout),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            MultiScaleCNNBlock(self.hidden_dim, self.hidden_dim, kernel_sizes=[3, 5, 9], dropout=self.dropout),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            # Optional: MaxPool to reduce length for RNN? DeepMVP doesn't pool before RNN.
            # We keep resolution high.
        )
        
        # 2. BiGRU Encoder (Global Dependencies)
        # Input to RNN: (B, L, H)
        self.bigru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.rnn_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.dropout > 0 else 0
        )
        
        # 3. Attention Mechanism
        # Attention weights for each position
        self.attention = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # 4. Classification Head
        # Input: Center Feature + Weighted Sum (Context)
        self.classifier = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim * 2 * 2, 128), # *2 for BiDirectional, *2 for (Center + Context)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1) # Binary classification (Logits)
        )
        
    def forward(self, batch):
        # Extract data from batch
        sequences = batch['sequence_features'] # [B, L, D_seq]
        
        # Transpose for CNN: [B, D, L]
        x = sequences.transpose(1, 2)
        
        # --- CNN Stream ---
        x = self.seq_proj(x)
        x = self.cnn_encoder(x) # [B, H, L]
        
        # --- RNN Stream ---
        # Transpose back for RNN: [B, L, H]
        x_rnn_in = x.transpose(1, 2)
        rnn_out, _ = self.bigru(x_rnn_in) # [B, L, 2*RNN_H]
        
        # --- Feature Aggregation ---
        L = rnn_out.size(1)
        center_idx = L // 2
        
        # 1. Center Feature (Direct evidence at the site)
        center_feature = rnn_out[:, center_idx, :] # [B, 2*RNN_H]
        
        # 2. Attention Context (Global evidence)
        attn_weights = self.attention(rnn_out) # [B, L, 1]
        context_feature = torch.sum(attn_weights * rnn_out, dim=1) # [B, 2*RNN_H]
        
        # Combine
        combined_feature = torch.cat([center_feature, context_feature], dim=1) # [B, 4*RNN_H]
        
        # --- Prediction ---
        logits = self.classifier(combined_feature)
        
        return logits

class DeepMVPFlattenPredictor(nn.Module):
    """
    DeepMVP-style model that strictly follows the original architecture:
    Stacked Conv1D -> BiGRU -> Flatten -> Dense.
    Adapted for ESM-2 features.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.seq_dim = config.get("seq_feature_dim", 1280)
        # DeepMVP uses 512 filters for the main CNN blocks
        self.hidden_dim = config.get("hidden_dim", 512) 
        # DeepMVP uses 128 filters for the reduction CNN block
        self.reduction_dim = 128
        # DeepMVP uses 50 units for GRU (bidirectional -> 100)
        self.rnn_hidden_dim = config.get("rnn_hidden_dim", 50)
        self.dropout = config.get("dropout", 0.5)
        self.window_size = config.get("window_size", 61)
        
        # Amino Acid Embedding (Explicit Identity)
        # Adds explicit "One-Hot-like" information to the rich ESM-2 features
        self.vocab_size = 25 # Standard 20 + special tokens
        self.embedding_dim = 21 # Similar to One-Hot dimension
        self.aa_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # Input dimension is now ESM dim + Embedding dim
        self.input_dim = self.seq_dim + self.embedding_dim
        
        # 1. Deep CNN Stack (Strictly matching DeepMVP log)
        # Note: We removed the 1x1 projection to allow the first Conv layer 
        # to extract motifs directly from the full ESM-2 feature space (1280 dim).
        
        # Block 1: Conv(1280+21 -> 512) -> BN -> LeakyReLU -> Dropout
        self.block1 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Block 2: Conv(512) -> BN -> ReLU -> Dropout
        self.block2 = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Block 3: Conv(512) -> BN -> LeakyReLU -> Dropout
        self.block3 = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Block 4 (Reduction): Conv(128) -> ReLU -> Dropout
        # Note: Log shows Activation after Conv1D_3, likely ReLU. No BN in log for this layer?
        # Log: conv1d_3 -> activation_1 -> dropout_3. No BN listed.
        self.block4 = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.reduction_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 3. BiGRU
        # Log: bidirectional (None, 57, 100) -> Dropout
        self.bigru = nn.GRU(
            input_size=self.reduction_dim,
            hidden_size=self.rnn_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.gru_dropout = nn.Dropout(self.dropout)
        
        # 4. Flatten & Dense
        # Output of BiGRU is [B, L, 2*rnn_hidden_dim] -> [B, 57, 100]
        flatten_dim = self.window_size * (self.rnn_hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Dense(64) -> BN -> ReLU -> Dropout
            nn.Linear(flatten_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            # Dense(1)
            nn.Linear(64, 1)
        )
        
    def forward(self, batch):
        # Extract data from batch
        sequences = batch['sequence_features'] # [B, L, D_seq]
        residue_types = batch['residue_types'] # [B, L]
        
        # Embedding (Explicit Identity)
        aa_emb = self.aa_embedding(residue_types) # [B, L, D_emb]
        
        # Concatenate ESM features with Explicit Identity
        x = torch.cat([sequences, aa_emb], dim=2) # [B, L, D_seq + D_emb]
        
        # Transpose for CNN: [B, D, L]
        x = x.transpose(1, 2)
        
        # CNN Stack
        # Direct input to Block 1 (no projection)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Transpose for RNN: [B, L, H']
        x_rnn_in = x.transpose(1, 2)
        
        # BiGRU
        rnn_out, _ = self.bigru(x_rnn_in) # [B, L, 2*RNN_H]
        rnn_out = self.gru_dropout(rnn_out)
        
        # Flatten & Classify
        logits = self.classifier(rnn_out)
        
        return logits
