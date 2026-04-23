"""
Enhanced Loss Functions for PTM Prediction
Enhanced losses: Focal Loss + supervised contrastive learning + hard negative mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Paper: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Class-balancing factor.
            gamma: Focusing parameter that down-weights easy examples.
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, num_classes] logits
            targets: [batch_size] class labels
        Returns:
            focal_loss: scalar or [batch_size]
        """
        # Cross-entropy
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # pt = exp(-CE)
        pt = torch.exp(-ce_loss)
        
        # Focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss
    Paper: https://arxiv.org/abs/2004.11362
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, feature_dim] normalized feature vectors.
            labels: [batch_size] class labels.
        Returns:
            contrastive_loss: scalar
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Label mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove diagonal (self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # exp(similarity)
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        
        # log_prob
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean log-likelihood per sample
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1.0)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class HardNegativeMiningLoss(nn.Module):
    """
    Hard Negative Mining Loss
    Dynamically mines hard negatives and increases their weight.
    """
    
    def __init__(self, neg_pos_ratio: float = 3.0, hard_ratio: float = 0.5):
        """
        Args:
            neg_pos_ratio: Negative-to-positive ratio.
            hard_ratio: Fraction of hard negatives among selected negatives.
        """
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.hard_ratio = hard_ratio
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, num_classes] logits
            targets: [batch_size] class labels
        Returns:
            loss: scalar
        """
        # Per-sample loss
        loss_per_sample = F.cross_entropy(predictions, targets, reduction='none')
        
        # Split positives/negatives
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        pos_loss = loss_per_sample[pos_mask]
        neg_loss = loss_per_sample[neg_mask]
        
        # Fallback when either class is missing
        if pos_loss.numel() == 0 or neg_loss.numel() == 0:
            return loss_per_sample.mean()
        
        # Number of negatives to keep
        num_pos = pos_loss.numel()
        num_neg_total = neg_loss.numel()
        num_neg_keep = min(int(num_pos * self.neg_pos_ratio), num_neg_total)
        
        # Split into hard/easy negatives
        num_hard_neg = int(num_neg_keep * self.hard_ratio)
        num_easy_neg = num_neg_keep - num_hard_neg
        
        # Sort by loss and pick hardest negatives
        neg_loss_sorted, neg_indices = torch.sort(neg_loss, descending=True)
        hard_neg_loss = neg_loss_sorted[:num_hard_neg]
        
        # Randomly sample some easy negatives
        if num_easy_neg > 0:
            easy_neg_indices = torch.randperm(num_neg_total - num_hard_neg)[:num_easy_neg]
            easy_neg_loss = neg_loss_sorted[num_hard_neg:][easy_neg_indices]
            selected_neg_loss = torch.cat([hard_neg_loss, easy_neg_loss])
        else:
            selected_neg_loss = hard_neg_loss
        
        # Final loss
        total_loss = torch.cat([pos_loss, selected_neg_loss]).mean()
        
        return total_loss


class EnhancedPTMLoss(nn.Module):
    """
    Enhanced PTM Prediction Loss
    Combines focal loss, contrastive learning, and hard negative mining.
    """
    
    def __init__(self, 
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 contrastive_temperature: float = 0.07,
                 contrastive_weight: float = 0.1,
                 use_hard_mining: bool = True,
                 hard_mining_ratio: float = 0.5):
        super().__init__()
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=contrastive_temperature)
        self.hard_mining_loss = HardNegativeMiningLoss(hard_ratio=hard_mining_ratio) if use_hard_mining else None
        
        self.contrastive_weight = contrastive_weight
        self.use_hard_mining = use_hard_mining
        
    def forward(self, predictions: torch.Tensor, features: torch.Tensor, 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: [batch_size, num_classes] logits
            features: [batch_size, feature_dim] feature vectors (for contrastive learning)
            targets: [batch_size] class labels
        Returns:
            loss_dict: {'total_loss', 'focal_loss', 'contrastive_loss', 'hard_mining_loss'}
        """
        # Focal Loss
        if self.use_hard_mining and self.hard_mining_loss is not None:
            classification_loss = self.hard_mining_loss(predictions, targets)
            loss_type = 'hard_mining_loss'
        else:
            classification_loss = self.focal_loss(predictions, targets)
            loss_type = 'focal_loss'
        
        # Contrastive Loss
        contrastive = self.contrastive_loss(features, targets)
        
        # Total Loss
        total_loss = classification_loss + self.contrastive_weight * contrastive
        
        loss_dict = {
            'total_loss': total_loss,
            loss_type: classification_loss,
            'contrastive_loss': contrastive
        }
        
        return loss_dict


class AdaptiveWeightScheduler:
    """
    Adaptive weight scheduler.
    Dynamically adjusts loss weights based on per-class performance.
    """
    
    def __init__(self, initial_weights: list = None, adaptation_rate: float = 0.01):
        if initial_weights is None:
            initial_weights = [1.0, 3.0]  # [neg_weight, pos_weight]
        
        self.weights = torch.tensor(initial_weights, dtype=torch.float32)
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        
    def update_weights(self, class_accuracies: list) -> torch.Tensor:
        """
        Dynamically adjust weights based on per-class accuracy.
        
        Args:
            class_accuracies: [neg_acc, pos_acc]
        Returns:
            updated_weights: torch.Tensor
        """
        for i, acc in enumerate(class_accuracies):
            if acc < 0.8:  # Low accuracy: increase weight
                self.weights[i] *= (1 + self.adaptation_rate)
            elif acc > 0.95:  # Very high accuracy: decrease weight
                self.weights[i] *= (1 - self.adaptation_rate)
        
        # Normalize weights
        self.weights = self.weights / self.weights.sum() * len(self.weights)
        
        return self.weights.clone()
    
    def get_weights(self) -> torch.Tensor:
        return self.weights.clone()


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss
    Prevents over-confidence and can improve generalization.
    """
    
    def __init__(self, num_classes: int = 2, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, num_classes] logits
            targets: [batch_size] class labels
        Returns:
            loss: scalar
        """
        log_probs = F.log_softmax(predictions, dim=-1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
        
        return loss


# Test code
if __name__ == "__main__":
    # Test enhanced loss functions
    batch_size = 32
    num_classes = 2
    feature_dim = 128
    
    # Synthetic data
    predictions = torch.randn(batch_size, num_classes)
    features = torch.randn(batch_size, feature_dim)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test focal loss
    focal_loss = FocalLoss()
    loss1 = focal_loss(predictions, targets)
    print(f"Focal Loss: {loss1.item():.4f}")
    
    # Test contrastive loss
    contrastive_loss = SupervisedContrastiveLoss()
    loss2 = contrastive_loss(features, targets)
    print(f"Contrastive Loss: {loss2.item():.4f}")
    
    # Test hard negative mining
    hard_mining_loss = HardNegativeMiningLoss()
    loss3 = hard_mining_loss(predictions, targets)
    print(f"Hard Mining Loss: {loss3.item():.4f}")
    
    # Test enhanced PTM loss
    enhanced_loss = EnhancedPTMLoss()
    loss_dict = enhanced_loss(predictions, features, targets)
    print(f"\nEnhanced PTM Loss:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test label smoothing
    label_smoothing = LabelSmoothingLoss()
    loss4 = label_smoothing(predictions, targets)
    print(f"\nLabel Smoothing Loss: {loss4.item():.4f}")
