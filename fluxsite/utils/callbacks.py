import os
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import pandas as pd
import seaborn as sns


class VisualizationCallback(Callback):
    """
    PyTorch Lightning callback for generating visualizations during training.
    
    Creates various plots to monitor training progress, including:
    - Loss curves
    - Metric curves
    - Confusion matrices
    - ROC curves
    - Precision-Recall curves
    - Learning rate schedule
    """
    
    def __init__(self, output_dir='visualizations', log_every_n_epochs=5):
        """
        Initialize the visualization callback.
        
        Args:
            output_dir: Directory to save visualizations
            log_every_n_epochs: How often to create visualizations
        """
        super().__init__()
        self.output_dir = output_dir
        self.log_every_n_epochs = log_every_n_epochs
        
        # Create output directory
        self.viz_dir = os.path.join(output_dir, 'training_visualizations')
        Path(self.viz_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'learning_rate': [],
            'epoch': []
        }
        
        # For confusion matrix and ROC/PR curves
        self.current_labels = None
        self.current_preds = None
        self.current_probs = None
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""
        # Store current epoch metrics
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        # Store learning rate
        optimizers = trainer.optimizers
        if len(optimizers) > 0:
            for param_group in optimizers[0].param_groups:
                self.history['learning_rate'].append(param_group['lr'])
                break
        
        # Update history
        self.history['epoch'].append(epoch)
        
        for metric in self.history:
            if metric in ['epoch', 'learning_rate']:
                continue
                
            # Try to get the metric from callback_metrics
            if metric in metrics:
                self.history[metric].append(metrics[metric].item())
            else:
                # If not found, try alternative names or set as NaN
                alt_names = [
                    metric,
                    metric.replace('train_', 'train_epoch_'),
                    metric.replace('val_', 'val_epoch_')
                ]
                
                found = False
                for name in alt_names:
                    if name in metrics:
                        self.history[metric].append(metrics[name].item())
                        found = True
                        break
                
                if not found:
                    self.history[metric].append(float('nan'))
        
        # Create visualizations every n epochs
        if epoch % self.log_every_n_epochs == 0 or epoch == trainer.max_epochs - 1:
            self._create_visualizations(epoch)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the validation batch ends."""
        # If outputs is a dictionary, look for relevant keys
        if isinstance(outputs, dict) and 'loss' in outputs:
            # Store predictions and labels for confusion matrix and ROC/PR curves
            if 'preds' in outputs and 'labels' in outputs:
                preds = outputs['preds']
                labels = outputs['labels']
                
                # Initialize if first batch
                if self.current_preds is None:
                    self.current_preds = preds.detach()
                    self.current_labels = labels.detach()
                    if 'probs' in outputs:
                        self.current_probs = outputs['probs'].detach()
                else:
                    # Concatenate with previous batches
                    self.current_preds = torch.cat([self.current_preds, preds.detach()], dim=0)
                    self.current_labels = torch.cat([self.current_labels, labels.detach()], dim=0)
                    if 'probs' in outputs and self.current_probs is not None:
                        self.current_probs = torch.cat([self.current_probs, outputs['probs'].detach()], dim=0)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""
        # Reset validation batch data
        self.current_labels = None
        self.current_preds = None
        self.current_probs = None
    
    def _create_visualizations(self, epoch):
        """Create visualizations based on current history."""
        # Create DataFrame for easier plotting
        history_df = pd.DataFrame(self.history)
        
        # 1. Loss Curve
        plt.figure(figsize=(10, 6))
        plt.plot(history_df['epoch'], history_df['train_loss'], label='Training Loss')
        plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss (Epoch {epoch})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.viz_dir, f'loss_curve_epoch_{epoch}.png'), dpi=300)
        plt.close()
        
        # 2. Accuracy Curve
        plt.figure(figsize=(10, 6))
        plt.plot(history_df['epoch'], history_df['train_acc'], label='Training Accuracy')
        plt.plot(history_df['epoch'], history_df['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Training and Validation Accuracy (Epoch {epoch})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.viz_dir, f'accuracy_curve_epoch_{epoch}.png'), dpi=300)
        plt.close()
        
        # 3. F1 Score Curve
        plt.figure(figsize=(10, 6))
        plt.plot(history_df['epoch'], history_df['train_f1'], label='Training F1')
        plt.plot(history_df['epoch'], history_df['val_f1'], label='Validation F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title(f'Training and Validation F1 Score (Epoch {epoch})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.viz_dir, f'f1_curve_epoch_{epoch}.png'), dpi=300)
        plt.close()
        
        # 4. Precision & Recall Curve
        plt.figure(figsize=(10, 6))
        plt.plot(history_df['epoch'], history_df['train_precision'], label='Training Precision')
        plt.plot(history_df['epoch'], history_df['val_precision'], label='Validation Precision')
        plt.plot(history_df['epoch'], history_df['train_recall'], label='Training Recall')
        plt.plot(history_df['epoch'], history_df['val_recall'], label='Validation Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title(f'Training and Validation Precision/Recall (Epoch {epoch})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.viz_dir, f'precision_recall_curve_epoch_{epoch}.png'), dpi=300)
        plt.close()
        
        # 5. Learning Rate Curve
        if self.history['learning_rate']:
            plt.figure(figsize=(10, 6))
            plt.plot(history_df['epoch'], history_df['learning_rate'])
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title(f'Learning Rate Schedule (Epoch {epoch})')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.viz_dir, f'lr_curve_epoch_{epoch}.png'), dpi=300)
            plt.close()
        
        # 6. Save history to CSV
        history_df.to_csv(os.path.join(self.viz_dir, 'training_history.csv'), index=False)


class PredictionSamplingCallback(Callback):
    """
    Callback to sample and visualize model predictions during training.
    
    This callback will:
    1. Sample a few examples from validation data
    2. Track model predictions on these examples over time
    3. Visualize how predictions change during training
    """
    
    def __init__(self, output_dir='visualizations', num_samples=5, log_every_n_epochs=5):
        """
        Initialize the prediction sampling callback.
        
        Args:
            output_dir: Directory to save visualizations
            num_samples: Number of examples to sample and track
            log_every_n_epochs: How often to log predictions
        """
        super().__init__()
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        
        # Create output directory
        self.samples_dir = os.path.join(output_dir, 'prediction_samples')
        Path(self.samples_dir).mkdir(parents=True, exist_ok=True)
        
        # Store samples
        self.sampled_inputs = None
        self.sampled_labels = None
        self.sampled_metadata = []
        
        # Track predictions over time
        self.prediction_history = []
        self.epochs = []
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """Called at the start of validation."""
        # Sample examples on the first epoch
        if trainer.current_epoch == 0:
            self._sample_examples(trainer, pl_module)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of validation."""
        epoch = trainer.current_epoch
        
        # Record predictions every n epochs
        if epoch % self.log_every_n_epochs == 0 or epoch == trainer.max_epochs - 1:
            self._record_predictions(trainer, pl_module, epoch)
    
    def on_train_end(self, trainer, pl_module):
        """Called when training ends."""
        # Generate final visualization
        self._visualize_prediction_evolution()
    
    def _sample_examples(self, trainer, pl_module):
        """Sample examples from validation data."""
        # Get validation dataloader
        val_dataloader = trainer.val_dataloaders[0]
        
        # Get a batch
        batch = next(iter(val_dataloader))
        
        # Select a subset of examples
        indices = torch.randperm(len(batch['label']))[:self.num_samples]
        
        # Store inputs and labels
        self.sampled_inputs = {
            'sequence': {k: v[indices].to(pl_module.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch['sequence'].items()} if isinstance(batch['sequence'], dict) 
                        else batch['sequence'][indices].to(pl_module.device)
        }
        
        if 'structure' in batch and batch['structure'] is not None:
            self.sampled_inputs['structure'] = batch['structure'][indices].to(pl_module.device)
            
        self.sampled_labels = batch['label'][indices].to(pl_module.device)
        
        # Store metadata if available
        metadata = {}
        if 'protein_id' in batch:
            metadata['protein_ids'] = [batch['protein_id'][i] for i in indices]
        if 'position' in batch:
            metadata['positions'] = batch['position'][indices].cpu().numpy()
        if 'seq' in batch:
            metadata['sequences'] = [batch['seq'][i] for i in indices]
            
        self.sampled_metadata = metadata
    
    def _record_predictions(self, trainer, pl_module, epoch):
        """Record predictions for sampled examples."""
        if self.sampled_inputs is None:
            return
        
        # Put model in eval mode
        pl_module.eval()
        
        # Get predictions
        with torch.no_grad():
            outputs = pl_module(**self.sampled_inputs)
            
            # Handle different output types
            if isinstance(outputs, dict):
                probs = outputs.get('outcome_prob', outputs.get('predicted_outcome', None))
                if probs is None:
                    probs = torch.sigmoid(outputs)
            else:
                probs = outputs
                
            # Convert to probabilities if needed
            if not torch.all((probs >= 0) & (probs <= 1)):
                probs = torch.sigmoid(probs)
        
        # Store predictions
        self.prediction_history.append(probs.cpu().numpy())
        self.epochs.append(epoch)
        
        # Put model back in train mode
        pl_module.train()
    
    def _visualize_prediction_evolution(self):
        """Visualize how predictions evolve during training."""
        if not self.prediction_history:
            return
        
        # Convert to numpy arrays
        predictions = np.array(self.prediction_history)
        labels = self.sampled_labels.cpu().numpy()
        
        # Create visualization for each sample
        for i in range(self.num_samples):
            plt.figure(figsize=(10, 6))
            
            # Plot prediction evolution
            plt.plot(self.epochs, predictions[:, i], 'b-', label='Prediction')
            
            # Plot true label as horizontal line
            plt.axhline(y=labels[i], color='r', linestyle='--', label='True Label')
            
            # Plot threshold at 0.5
            plt.axhline(y=0.5, color='g', linestyle=':', label='Threshold (0.5)')
            
            # Add details to plot
            sample_name = f"Sample {i+1}"
            if 'protein_ids' in self.sampled_metadata and len(self.sampled_metadata['protein_ids']) > i:
                sample_name = self.sampled_metadata['protein_ids'][i]
            if 'positions' in self.sampled_metadata and len(self.sampled_metadata['positions']) > i:
                sample_name += f" (Pos: {self.sampled_metadata['positions'][i]})"
                
            plt.title(f'Prediction Evolution for {sample_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Prediction Probability')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(-0.1, 1.1)
            
            # Save visualization
            plt.savefig(os.path.join(self.samples_dir, f'prediction_evolution_sample_{i+1}.png'), dpi=300)
            plt.close()
        
        # Create visualization of all samples together
        plt.figure(figsize=(12, 8))
        
        for i in range(self.num_samples):
            sample_name = f"Sample {i+1}"
            if 'protein_ids' in self.sampled_metadata and len(self.sampled_metadata['protein_ids']) > i:
                sample_name = self.sampled_metadata['protein_ids'][i]
            if 'positions' in self.sampled_metadata and len(self.sampled_metadata['positions']) > i:
                sample_name += f" (Pos: {self.sampled_metadata['positions'][i]})"
                
            plt.plot(self.epochs, predictions[:, i], '-', label=f'{sample_name} (True: {labels[i]})')
        
        # Add threshold at 0.5
        plt.axhline(y=0.5, color='g', linestyle=':', label='Threshold (0.5)')
        
        plt.title('Prediction Evolution for All Samples')
        plt.xlabel('Epoch')
        plt.ylabel('Prediction Probability')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        
        # Save visualization
        plt.savefig(os.path.join(self.samples_dir, 'prediction_evolution_all_samples.png'), dpi=300)
        plt.close() 