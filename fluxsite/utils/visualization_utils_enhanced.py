
import os
import numpy as np
from pathlib import Path
import json

# Import matplotlib and seaborn only when available to avoid dependency issues
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    plt = None
    sns = None
    mpatches = None
    Rectangle = None


class EnhancedVisualizationManager:
    """Enhanced visualization manager that generates high-quality plots."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        Path(self.viz_dir).mkdir(parents=True, exist_ok=True)

        # Check whether plotting is available
        if not HAS_PLOTTING:
            print("Warning:matplotlibUnavailable, Disable")
            return

        # Configure a publication-style matplotlib theme
        self._setup_matplotlib_style()

        # Define a consistent color palette
        self.colors = {
            'train': '#2E86AB',      # Deep blue
            'val': '#A23B72',        # Magenta
            'test': '#F18F01',       # Orange
            'best': '#C73E1D',       # Red
            'good': '#2D5016',       # Dark green
            'highlight': '#FFD23F',  # Yellow
            'bg_light': '#F8F9FA',   # Light gray background
            'bg_dark': '#E9ECEF',    # Dark gray background
            'grid': '#E5E5E5',       # Grid line color
            'text': '#333333'        # Text color
        }
    
    def _setup_matplotlib_style(self):
        """Configure a publication-style matplotlib theme."""
        if not HAS_PLOTTING:
            return

        plt.style.use('seaborn-v0_8-whitegrid')

        # Custom rcParams
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': '#333333',
            'axes.linewidth': 1.2,
            'grid.color': '#E5E5E5',
            'grid.linewidth': 0.8,
            'grid.alpha': 0.6,
            'font.family': 'DejaVu Sans',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'lines.linewidth': 2.5,
            'lines.markersize': 6,
            'patch.linewidth': 0.5,
            'patch.facecolor': '#F0F0F0'
        })
    
    def create_training_plots(self, training_history):
        """Create the full set of training visualizations."""
        print("🎨 Training.")

        # Check whether plotting is available
        if not HAS_PLOTTING:
            print("⚠️ matplotlibUnavailable,Skip ")
            return

        # Validate training history
        if not self._validate_history(training_history):
            print("⚠️ Training Data,Skip ")
            return
        
        try:
            # 1. Loss curves
            self._plot_loss_curves(training_history)
            
            # 2. Accuracy curves
            self._plot_accuracy_curves(training_history)
            
            # 3. F1-score curves
            self._plot_f1_curves(training_history)
            
            # 4. MCC curves
            self._plot_mcc_curves(training_history)
            
            # 5. AUC curves
            self._plot_auc_curves(training_history)
            
            # 6. Precision/recall curves
            self._plot_precision_recall_curves(training_history)
            
            # 7. Multi-metric dashboard
            self._plot_metrics_dashboard(training_history)
            
            # 8. Training overview
            self._plot_training_overview(training_history)
            
            # 9. Performance trend analysis
            self._plot_performance_trends(training_history)
            
            print(f"✨ Save: {self.viz_dir}")
            
        except Exception as e:
            print(f"❌: {e}")
            import traceback
            traceback.print_exc()
    
    def _validate_history(self, history):
        """Validate training history completeness."""
        required_keys = ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'val_f1']
        
        for key in required_keys:
            if key not in history or not history[key]:
                print(f"❌ Data: {key}")
                return False
        
        return True
    
    def _plot_loss_curves(self, history):
        """Plot publication-style loss curves."""
        fig, ax = plt.subplots(figsize=(12, 8))
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot main curves
        ax.plot(epochs, history['train_loss'], color=self.colors['train'], 
               linewidth=3, label='Training Loss', alpha=0.9, marker='o', markersize=5)
        ax.plot(epochs, history['val_loss'], color=self.colors['val'], 
               linewidth=3, label='Validation Loss', alpha=0.9, marker='s', markersize=5)
        
        # Add filled band
        ax.fill_between(epochs, history['train_loss'], alpha=0.1, color=self.colors['train'])
        ax.fill_between(epochs, history['val_loss'], alpha=0.1, color=self.colors['val'])
        
        # Mark best point
        best_idx = np.argmin(history['val_loss'])
        best_val = min(history['val_loss'])
        ax.scatter(best_idx + 1, best_val, color=self.colors['best'], s=200, 
                 zorder=5, edgecolors='white', linewidth=3)
        ax.annotate(f'Best: {best_val:.4f} (Epoch {best_idx + 1})', 
                   xy=(best_idx + 1, best_val),
                   xytext=(20, 30), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor=self.colors['highlight'], 
                            alpha=0.9, edgecolor=self.colors['best'], linewidth=2),
                   arrowprops=dict(arrowstyle='->', color=self.colors['best'], lw=2))
        
        # Style the plot
        ax.set_title('📉 Training & Validation Loss Curves', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Training Epochs', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor(self.colors['bg_light'])
        
        # Add performance band annotations
        if len(epochs) > 10:
            ax.axvspan(1, len(epochs)//4, alpha=0.1, color='red', label='Early Training')
            ax.axvspan(len(epochs)//4, 3*len(epochs)//4, alpha=0.1, color='orange', label='Mid Training')
            ax.axvspan(3*len(epochs)//4, len(epochs), alpha=0.1, color='green', label='Late Training')
        
        self._add_watermark(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'enhanced_loss_curves.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_accuracy_curves(self, history):
        """Plot publication-style accuracy curves."""
        fig, ax = plt.subplots(figsize=(12, 8))
        epochs = range(1, len(history['train_acc']) + 1)
        
        # Plot main curves
        ax.plot(epochs, history['train_acc'], color=self.colors['train'], 
               linewidth=3, label='Training Accuracy', alpha=0.9, marker='o', markersize=5)
        ax.plot(epochs, history['val_acc'], color=self.colors['val'], 
               linewidth=3, label='Validation Accuracy', alpha=0.9, marker='s', markersize=5)
        
        # Add filled band
        ax.fill_between(epochs, history['train_acc'], alpha=0.1, color=self.colors['train'])
        ax.fill_between(epochs, history['val_acc'], alpha=0.1, color=self.colors['val'])
        
        # Mark best point
        best_idx = np.argmax(history['val_acc'])
        best_val = max(history['val_acc'])
        ax.scatter(best_idx + 1, best_val, color=self.colors['best'], s=200, 
                 zorder=5, edgecolors='white', linewidth=3)
        ax.annotate(f'Best: {best_val:.4f} (Epoch {best_idx + 1})', 
                   xy=(best_idx + 1, best_val),
                   xytext=(20, -40), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor=self.colors['highlight'], 
                            alpha=0.9, edgecolor=self.colors['best'], linewidth=2),
                   arrowprops=dict(arrowstyle='->', color=self.colors['best'], lw=2))
        
        # Add target line
        target_acc = 0.95
        ax.axhline(y=target_acc, color=self.colors['good'], linestyle='--', alpha=0.8, 
                  linewidth=2, label=f'Target Accuracy ({target_acc:.0%})')
        
        # Style the plot
        ax.set_title('🎯 Training & Validation Accuracy Curves', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Training Epochs', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor(self.colors['bg_light'])
        
        # Format y-axis as percentages
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        self._add_watermark(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'enhanced_accuracy_curves.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_f1_curves(self, history):
        """Plot F1-score curves."""
        if 'val_f1' not in history or not history['val_f1']:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        epochs = range(1, len(history['val_f1']) + 1)
        
        # Plot F1 curves
        ax.plot(epochs, history['val_f1'], color=self.colors['val'], 
               linewidth=3, label='Validation F1 Score', alpha=0.9, marker='D', markersize=6)
        
        if 'train_f1' in history and history['train_f1']:
            ax.plot(epochs, history['train_f1'], color=self.colors['train'], 
                   linewidth=3, label='Training F1 Score', alpha=0.9, marker='o', markersize=5)
        
        # Add filled band
        ax.fill_between(epochs, history['val_f1'], alpha=0.1, color=self.colors['val'])
        
        # Mark best point
        best_idx = np.argmax(history['val_f1'])
        best_val = max(history['val_f1'])
        ax.scatter(best_idx + 1, best_val, color=self.colors['best'], s=200, 
                 zorder=5, edgecolors='white', linewidth=3)
        ax.annotate(f'Best F1: {best_val:.4f} (Epoch {best_idx + 1})', 
                   xy=(best_idx + 1, best_val),
                   xytext=(20, 30), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor=self.colors['highlight'], 
                            alpha=0.9, edgecolor=self.colors['best'], linewidth=2),
                   arrowprops=dict(arrowstyle='->', color=self.colors['best'], lw=2))
        
        # Add reference line for strong performance
        excellent_f1 = 0.9
        ax.axhline(y=excellent_f1, color=self.colors['good'], linestyle='--', alpha=0.8, 
                  linewidth=2, label=f'Excellent F1 ({excellent_f1:.1f})')
        
        # Style the plot
        ax.set_title('🏆 F1 Score Performance', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Training Epochs', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor(self.colors['bg_light'])
        
        self._add_watermark(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'enhanced_f1_curves.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_mcc_curves(self, history):
        """Plot MCC curves."""
        if 'val_mcc' not in history or not history['val_mcc']:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        epochs = range(1, len(history['val_mcc']) + 1)
        
        # Plot MCC curves
        ax.plot(epochs, history['val_mcc'], color='purple', 
               linewidth=3, label='Validation MCC', alpha=0.9, marker='H', markersize=6)
        
        # Add filled band
        ax.fill_between(epochs, history['val_mcc'], alpha=0.1, color='purple')
        
        # Mark best point
        best_idx = np.argmax(history['val_mcc'])
        best_val = max(history['val_mcc'])
        ax.scatter(best_idx + 1, best_val, color=self.colors['best'], s=200, 
                 zorder=5, edgecolors='white', linewidth=3)
        ax.annotate(f'Best MCC: {best_val:.4f} (Epoch {best_idx + 1})', 
                   xy=(best_idx + 1, best_val),
                   xytext=(20, 30), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor=self.colors['highlight'], 
                            alpha=0.9, edgecolor=self.colors['best'], linewidth=2),
                   arrowprops=dict(arrowstyle='->', color=self.colors['best'], lw=2))
        
        # Style the plot
        ax.set_title('📊 Matthews Correlation Coefficient (MCC)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Training Epochs', fontsize=12, fontweight='bold')
        ax.set_ylabel('MCC Score', fontsize=12, fontweight='bold')
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor(self.colors['bg_light'])
        
        # Add performance bands
        ax.axhspan(0.8, 1.0, alpha=0.1, color='green', label='Excellent')
        ax.axhspan(0.6, 0.8, alpha=0.1, color='orange', label='Good')
        ax.axhspan(0.0, 0.6, alpha=0.1, color='red', label='Poor')
        
        self._add_watermark(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'enhanced_mcc_curves.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_auc_curves(self, history):
        """Plot AUC curves."""
        if 'val_auc' not in history or not history['val_auc']:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        epochs = range(1, len(history['val_auc']) + 1)
        
        # Plot AUC curves
        ax.plot(epochs, history['val_auc'], color='darkgreen', 
               linewidth=3, label='Validation AUC', alpha=0.9, marker='v', markersize=6)
        
        # Add filled band
        ax.fill_between(epochs, history['val_auc'], alpha=0.1, color='darkgreen')
        
        # Mark best point
        best_idx = np.argmax(history['val_auc'])
        best_val = max(history['val_auc'])
        ax.scatter(best_idx + 1, best_val, color=self.colors['best'], s=200, 
                 zorder=5, edgecolors='white', linewidth=3)
        ax.annotate(f'Best AUC: {best_val:.4f} (Epoch {best_idx + 1})', 
                   xy=(best_idx + 1, best_val),
                   xytext=(20, -40), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor=self.colors['highlight'], 
                            alpha=0.9, edgecolor=self.colors['best'], linewidth=2),
                   arrowprops=dict(arrowstyle='->', color=self.colors['best'], lw=2))
        
        # Add random-classifier baseline
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.8, 
                  linewidth=2, label='Random Classifier (0.5)')
        
        # Style the plot
        ax.set_title('📈 ROC-AUC Performance', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Training Epochs', fontsize=12, fontweight='bold')
        ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor(self.colors['bg_light'])
        
        self._add_watermark(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'enhanced_auc_curves.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_precision_recall_curves(self, history):
        """Plot precision and recall curves."""
        if 'val_precision' not in history or 'val_recall' not in history:
            return
        if not history['val_precision'] or not history['val_recall']:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        epochs = range(1, len(history['val_precision']) + 1)
        
        # Plot precision and recall curves
        ax.plot(epochs, history['val_precision'], color='darkblue', 
               linewidth=3, label='Validation Precision', alpha=0.9, marker='^', markersize=6)
        ax.plot(epochs, history['val_recall'], color='darkred', 
               linewidth=3, label='Validation Recall', alpha=0.9, marker='v', markersize=6)
        
        # Add filled band
        ax.fill_between(epochs, history['val_precision'], alpha=0.1, color='darkblue')
        ax.fill_between(epochs, history['val_recall'], alpha=0.1, color='darkred')
        
        # Style the plot
        ax.set_title('⚖️ Precision & Recall Performance', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Training Epochs', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor(self.colors['bg_light'])
        
        self._add_watermark(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'enhanced_precision_recall_curves.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_metrics_dashboard(self, history):
        """Plot a multi-metric dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('📊 Training Metrics Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        # Subplot configuration
        plots_config = [
            ('train_loss', 'val_loss', 'Loss', '📉', False),
            ('train_acc', 'val_acc', 'Accuracy', '🎯', True),
            ('train_f1', 'val_f1', 'F1 Score', '🏆', True),
            ('val_mcc', None, 'MCC', '📊', True),
            ('val_auc', None, 'AUC', '📈', True),
            ('val_auprc', None, 'AUPRC', '📋', True)
        ]
        
        for i, (train_key, val_key, title, emoji, higher_better) in enumerate(plots_config):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            if train_key in history and history[train_key]:
                epochs = range(1, len(history[train_key]) + 1)
                ax.plot(epochs, history[train_key], color=self.colors['train'], 
                       linewidth=2, label='Training', alpha=0.9)
                
                if val_key and val_key in history and history[val_key]:
                    ax.plot(epochs, history[val_key], color=self.colors['val'], 
                           linewidth=2, label='Validation', alpha=0.9)
                    
            # Mark best point
                    values = history[val_key]
                    if higher_better:
                        best_idx = np.argmax(values)
                    else:
                        best_idx = np.argmin(values)
                    best_val = values[best_idx]
                    ax.scatter(best_idx + 1, best_val, color=self.colors['best'], s=100, zorder=5)
                elif not val_key:
        # Case: only validation series is available
                    values = history[train_key]
                    if higher_better:
                        best_idx = np.argmax(values)
                    else:
                        best_idx = np.argmin(values)
                    best_val = values[best_idx]
                    ax.scatter(best_idx + 1, best_val, color=self.colors['best'], s=100, zorder=5)
                
                ax.set_title(f'{emoji} {title}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Epochs', fontsize=10)
                ax.set_ylabel(title, fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_facecolor(self.colors['bg_light'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'metrics_dashboard.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_training_overview(self, history):
        """Plot a training overview figure."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🔍 Training Process Overview', fontsize=18, fontweight='bold', y=0.98)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss comparison
        ax1.plot(epochs, history['train_loss'], color=self.colors['train'], 
                linewidth=2, label='Training Loss')
        ax1.plot(epochs, history['val_loss'], color=self.colors['val'], 
                linewidth=2, label='Validation Loss')
        ax1.set_title('📉 Loss Comparison', fontweight='bold')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy comparison
        ax2.plot(epochs, history['train_acc'], color=self.colors['train'], 
                linewidth=2, label='Training Accuracy')
        ax2.plot(epochs, history['val_acc'], color=self.colors['val'], 
                linewidth=2, label='Validation Accuracy')
        ax2.set_title('🎯 Accuracy Comparison', fontweight='bold')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Validation metric summary
        if 'val_f1' in history and history['val_f1']:
            ax3.plot(epochs, history['val_f1'], color='green', linewidth=2, label='F1 Score')
        if 'val_auc' in history and history['val_auc']:
            ax3.plot(epochs, history['val_auc'], color='blue', linewidth=2, label='AUC')
        if 'val_mcc' in history and history['val_mcc']:
        # Map MCC from [-1, 1] to [0, 1] for comparison
            mcc_normalized = [(x + 1) / 2 for x in history['val_mcc']]
            ax3.plot(epochs, mcc_normalized, color='purple', linewidth=2, label='MCC (normalized)')
        ax3.set_title('🏆 Validation Metrics', fontweight='bold')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning-curve trends
        train_acc_trend = np.polyfit(epochs, history['train_acc'], 1)
        val_acc_trend = np.polyfit(epochs, history['val_acc'], 1)
        ax4.plot(epochs, history['train_acc'], color=self.colors['train'], 
                alpha=0.7, linewidth=1, label='Training Accuracy')
        ax4.plot(epochs, history['val_acc'], color=self.colors['val'], 
                alpha=0.7, linewidth=1, label='Validation Accuracy')
        ax4.plot(epochs, np.poly1d(train_acc_trend)(epochs), 
                color=self.colors['train'], linewidth=3, linestyle='--', label='Train Trend')
        ax4.plot(epochs, np.poly1d(val_acc_trend)(epochs), 
                color=self.colors['val'], linewidth=3, linestyle='--', label='Val Trend')
        ax4.set_title('📈 Learning Trends', fontweight='bold')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'training_overview.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _plot_performance_trends(self, history):
        """Plot performance trend analysis."""
        val_acc = np.array(history.get('val_acc', []), dtype=float)
        if val_acc.size == 0:
            print("⚠️ Validation,Skip ")
            return

        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Compute moving average to show trend; window must not exceed series length
        window_size = max(3, val_acc.size // 10)
        window_size = min(window_size, val_acc.size)
        window_size = max(window_size, 1)
        
        def moving_average(data, window):
            if window <= 1 or len(data) < window:
                return np.array(data, dtype=float)
            smoothed = np.convolve(data, np.ones(window) / window, mode='same')
            if smoothed.shape[0] != len(data):
                smoothed = smoothed[:len(data)]
            return smoothed
        
        epochs = np.arange(1, val_acc.size + 1)
        
        # Plot raw series
        ax.plot(epochs, val_acc, color=self.colors['val'], 
               alpha=0.3, linewidth=1, label='Validation Accuracy (Raw)')
        
        # Plot moving average
        val_acc_smooth = moving_average(val_acc, window_size)
        ax.plot(epochs, val_acc_smooth, color=self.colors['val'], 
               linewidth=3, label=f'Validation Accuracy (MA-{window_size})')
        
        # Annotate performance phases
        total_epochs = len(epochs)
        phase_1 = min(max(total_epochs // 3, 1), total_epochs)
        if total_epochs >= 3:
            phase_2 = max(2 * total_epochs // 3, phase_1 + 1)
        else:
            phase_2 = total_epochs
        phase_2 = min(phase_2, total_epochs)

        phase_spans = [
            (1, phase_1, 'red', 'Early Phase'),
            (phase_1, phase_2, 'orange', 'Middle Phase'),
            (phase_2, total_epochs, 'green', 'Late Phase'),
        ]
        for start, end, color, label in phase_spans:
            end = min(end, total_epochs)
            if start < end:
                ax.axvspan(start, end, alpha=0.1, color=color, label=label)
        
        # Mark key points
        max_acc_idx = np.argmax(val_acc)
        max_acc = float(val_acc[max_acc_idx])
        ax.scatter(max_acc_idx + 1, max_acc, color=self.colors['best'], s=200, 
                 zorder=5, edgecolors='white', linewidth=3)
        ax.annotate(f'Peak Performance\n{max_acc:.4f} (Epoch {max_acc_idx + 1})', 
                   xy=(max_acc_idx + 1, max_acc),
                   xytext=(50, 50), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor=self.colors['highlight'], 
                            alpha=0.9, edgecolor=self.colors['best'], linewidth=2),
                   arrowprops=dict(arrowstyle='->', color=self.colors['best'], lw=2))
        
        ax.set_title('📊 Performance Trend Analysis', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Training Epochs', fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor(self.colors['bg_light'])
        
        # Format y-axis as percentages
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        self._add_watermark(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'performance_trends.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _add_watermark(self, ax):
        """Add a subtle watermark."""
        ax.text(0.99, 0.01, 'PTM-GPT-MoE Advanced Training System', 
                transform=ax.transAxes, fontsize=8, alpha=0.6,
                ha='right', va='bottom', style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def save_training_summary(self, training_results):
        """Save training summary information."""
        summary = {
            'best_validation_loss': training_results.get('best_val_loss', 'N/A'),
            'best_validation_accuracy': training_results.get('best_val_acc', 'N/A'),
            'best_validation_f1': training_results.get('best_val_f1', 'N/A'),
            'best_validation_mcc': training_results.get('best_val_mcc', 'N/A'),
            'best_epoch': training_results.get('best_epoch', 'N/A'),
            'total_epochs': len(training_results.get('history', {}).get('train_loss', [])),
        }
        
        summary_path = os.path.join(self.output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 TrainingSummary Save: {summary_path}")
        return summary
