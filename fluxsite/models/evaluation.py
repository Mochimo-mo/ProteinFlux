import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef,
    precision_recall_curve, auc, roc_curve
)


class EvaluationMetrics:
    """Metric utilities for evaluating protein acetylation-site prediction models."""
    
    def __init__(self):
        """Initialize the metrics helper."""
        pass
        
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_scores=None, threshold=0.5):
        """
        Compute evaluation metrics.
        
        Args:
            y_true (numpy.ndarray): Ground-truth labels.
            y_pred (numpy.ndarray): Predicted labels.
            y_scores (numpy.ndarray, optional): Predicted probabilities/scores.
            threshold (float): Classification threshold.
            
        Returns:
            dict: Dictionary of metrics.
        """
        # Ensure NumPy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Apply threshold if scores are provided
        if y_scores is not None:
            y_scores = np.array(y_scores)
            binary_preds = (y_scores >= threshold).astype(int)
        else:
            binary_preds = y_pred
            y_scores = y_pred  # Fallback: use predictions as scores
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, binary_preds),
            'precision': precision_score(y_true, binary_preds, zero_division=0),
            'recall': recall_score(y_true, binary_preds, zero_division=0),
            'f1': f1_score(y_true, binary_preds, zero_division=0),
            'mcc': matthews_corrcoef(y_true, binary_preds),
        }
        
        # AUC/AUPRC (only meaningful when scores are not strictly binary)
        if not np.all(np.isin(y_scores, [0, 1])):
            try:
                metrics['auc'] = roc_auc_score(y_true, y_scores)
                
                # AUPRC
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                metrics['auprc'] = auc(recall, precision)
            except:
                metrics['auc'] = 0.5
                metrics['auprc'] = 0.5
        
        # Confusion matrix-derived metrics
        tn, fp, fn, tp = confusion_matrix(y_true, binary_preds, labels=[0, 1]).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        return metrics
    
    @staticmethod
    def threshold_optimization(y_true, y_scores, metric='f1'):
        """
        Find the best classification threshold.
        
        Args:
            y_true (numpy.ndarray): Ground-truth labels.
            y_scores (numpy.ndarray): Predicted scores/probabilities.
            metric (str): Optimization metric ('f1', 'mcc', 'balanced_accuracy').
            
        Returns:
            float: Best threshold.
        """
        thresholds = np.linspace(0.01, 0.99, 99)
        best_score = -np.inf
        best_threshold = 0.5
        
        for threshold in thresholds:
            # Convert scores to binary predictions
            y_pred = (y_scores >= threshold).astype(int)
            
            # Compute score for the chosen metric
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'mcc':
                score = matthews_corrcoef(y_true, y_pred)
            elif metric == 'balanced_accuracy':
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                score = (sensitivity + specificity) / 2
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
            # Update best score/threshold
            if score > best_score:
                best_score = score
                best_threshold = threshold
                
        return best_threshold
    
    @staticmethod
    def bootstrap_confidence_interval(y_true, y_scores, n_bootstraps=1000, ci=0.95, metric='auc'):
        """
        Compute confidence intervals via bootstrap resampling.
        
        Args:
            y_true (numpy.ndarray): Ground-truth labels.
            y_scores (numpy.ndarray): Predicted scores/probabilities.
            n_bootstraps (int): Number of bootstrap samples.
            ci (float): Confidence level.
            metric (str): Metric name.
            
        Returns:
            tuple: (lower, upper) confidence interval.
        """
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        bootstrap_scores = []
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        
        for i in range(n_bootstraps):
            # Bootstrap resampling (with replacement)
            indices = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[indices])) < 2:
                # Skip samples with only one class
                continue
                
            # Compute metric
            if metric == 'auc':
                score = roc_auc_score(y_true[indices], y_scores[indices])
            elif metric == 'mcc':
                y_pred = (y_scores[indices] >= 0.5).astype(int)
                score = matthews_corrcoef(y_true[indices], y_pred)
            elif metric == 'f1':
                y_pred = (y_scores[indices] >= 0.5).astype(int)
                score = f1_score(y_true[indices], y_pred, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
            bootstrap_scores.append(score)
            
        # Confidence interval
        alpha = (1.0 - ci) / 2.0
        lower_bound = np.percentile(bootstrap_scores, 100 * alpha)
        upper_bound = np.percentile(bootstrap_scores, 100 * (1 - alpha))
        
        return lower_bound, upper_bound


class VisualizationTool:
    """Utility class for visualizing evaluation results."""
    
    def __init__(self, save_dir="./visualizations"):
        """
        Initialize the visualization tool.

        Args:
            save_dir (str): Directory to save generated figures.
        """
        self.save_dir = save_dir
        
    def plot_roc_curve(self, y_true, y_scores, model_names=None, save_path=None, show=True):
        """
        Plot ROC curves.

        Args:
            y_true (list or numpy.ndarray): Ground-truth labels (list of arrays or a single array).
            y_scores (list or numpy.ndarray): Predicted scores/probabilities (list of arrays or a single array).
            model_names (list, optional): Model names.
            save_path (str, optional): Output path.
            show (bool): Whether to display the figure.

        Returns:
            matplotlib.figure.Figure: Figure object.
        """
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # If a single array is provided, wrap it as a list
        if not isinstance(y_scores[0], (list, np.ndarray)):
            y_true = [y_true]
            y_scores = [y_scores]
            
        # Generate default names when not provided
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(y_scores))]
            
        # Plot ROC curve for each model
        for i, (y_t, y_s, name) in enumerate(zip(y_true, y_scores, model_names)):
            fpr, tpr, _ = roc_curve(y_t, y_s)
            roc_auc = roc_auc_score(y_t, y_s)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
            
        # Random-guess baseline
        plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
        
        # Styling
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR (FPR)')
        plt.ylabel('TPR (TPR)')
        plt.title(' (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Show figure
        if show:
            plt.show()
            
        return plt.gcf()
        
    def plot_precision_recall_curve(self, y_true, y_scores, model_names=None, save_path=None, show=True):
        """
        Plot precision-recall curves.

        Args:
            y_true (list or numpy.ndarray): Ground-truth labels (list of arrays or a single array).
            y_scores (list or numpy.ndarray): Predicted scores/probabilities (list of arrays or a single array).
            model_names (list, optional): Model names.
            save_path (str, optional): Output path.
            show (bool): Whether to display the figure.

        Returns:
            matplotlib.figure.Figure: Figure object.
        """
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # If a single array is provided, wrap it as a list
        if not isinstance(y_scores[0], (list, np.ndarray)):
            y_true = [y_true]
            y_scores = [y_scores]
            
        # Generate default names when not provided
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(y_scores))]
            
        # Plot PR curve for each model
        for i, (y_t, y_s, name) in enumerate(zip(y_true, y_scores, model_names)):
            precision, recall, _ = precision_recall_curve(y_t, y_s)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f'{name} (AUPRC = {pr_auc:.3f})')
            
        # Styling
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Recall)')
        plt.ylabel('Precision (Precision)')
        plt.title('Precision-Recall (PR)')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Show figure
        if show:
            plt.show()
            
        return plt.gcf()
        
    def plot_confusion_matrix(self, y_true, y_pred, normalize=True, save_path=None, show=True):
        """
        Plot a confusion matrix.

        Args:
            y_true (numpy.ndarray): Ground-truth labels.
            y_pred (numpy.ndarray): Predicted labels.
            normalize (bool): Whether to normalize rows.
            save_path (str, optional): Output path.
            show (bool): Whether to display the figure.

        Returns:
            matplotlib.figure.Figure: Figure object.
        """
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
            
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Heatmap via seaborn
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=['Non-acetylation', 'Acetylation'],
                   yticklabels=['Non-acetylation', 'Acetylation'])
        
        # Styling
        plt.ylabel(' Labels')
        plt.xlabel(' Labels')
        plt.title('Confusion Matrix')
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Show figure
        if show:
            plt.show()
            
        return plt.gcf()
        
    def plot_metrics_comparison(self, metrics_dict, metric_names=None, model_names=None, save_path=None, show=True):
        """
        Compare multiple metrics across different models.

        Args:
            metrics_dict (dict): Dictionary of per-model metrics.
                Format: {model_name: {metric_name: value}}
            metric_names (list, optional): Metrics to compare.
            model_names (list, optional): Models to compare.
            save_path (str, optional): Output path.
            show (bool): Whether to display the figure.

        Returns:
            matplotlib.figure.Figure: Figure object.
        """
        # Use metrics from the first model when metric_names is not provided
        if metric_names is None:
            first_model = list(metrics_dict.keys())[0]
            metric_names = list(metrics_dict[first_model].keys())
            
        # Use all models when model_names is not provided
        if model_names is None:
            model_names = list(metrics_dict.keys())
            
        # Prepare data
        data = []
        for model in model_names:
            for metric in metric_names:
                data.append({
                    'Model': model,
                    'Metric': metric,
                    'Value': metrics_dict[model][metric]
                })
                
        df = pd.DataFrame(data)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Bar plot via seaborn
        sns.barplot(x='Metric', y='Value', hue='Model', data=df)
        
        # Styling
        plt.title('Model ')
        plt.ylabel(' ')
        plt.ylim(0, 1.05)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Model')
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Show figure
        if show:
            plt.show()
            
        return plt.gcf()
        
    def plot_ablation_study(self, ablation_results, metric='f1', save_path=None, show=True):
        """
        Plot ablation study results.

        Args:
            ablation_results (dict): Ablation results.
                Format: {model_variant: score}
            metric (str): Metric name.
            save_path (str, optional): Output path.
            show (bool): Whether to display the figure.

        Returns:
            matplotlib.figure.Figure: Figure object.
        """
        # Sort results
        sorted_results = sorted(ablation_results.items(), key=lambda x: x[1], reverse=True)
        model_names = [x[0] for x in sorted_results]
        scores = [x[1] for x in sorted_results]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Horizontal bar chart
        bars = plt.barh(model_names, scores, color=plt.cm.viridis(np.linspace(0, 0.8, len(scores))))
        
        # Add value labels on bars
        for i, (value, bar) in enumerate(zip(scores, bars)):
            plt.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
                    va='center', ha='left', fontsize=10)
        
        # Styling
        plt.xlabel(f'{metric.upper()} ')
        plt.title(' Results')
        plt.xlim(0, max(scores) * 1.15)  # Leave room for value labels
        plt.grid(axis='x', alpha=0.3)
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Show figure
        if show:
            plt.show()
            
        return plt.gcf()
        
    def plot_learning_curve(self, train_scores, val_scores, epochs, metric='loss', save_path=None, show=True):
        """
        Plot learning curves.

        Args:
            train_scores (list): Training scores.
            val_scores (list): Validation scores.
            epochs (list or int): Epoch indices or total epoch count.
            metric (str): Metric name.
            save_path (str, optional): Output path.
            show (bool): Whether to display the figure.

        Returns:
            matplotlib.figure.Figure: Figure object.
        """
        # If a total epoch count is provided, generate an epoch list
        if isinstance(epochs, int):
            epochs = list(range(1, epochs + 1))
            
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot learning curves
        plt.plot(epochs, train_scores, label=f'Training {metric}', marker='o', markersize=4)
        plt.plot(epochs, val_scores, label=f'Validation {metric}', marker='s', markersize=4)
        
        # Styling
        plt.xlabel('Training ')
        plt.ylabel(metric.capitalize())
        plt.title(f' - {metric.capitalize()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Show figure
        if show:
            plt.show()
            
        return plt.gcf()
        
    def visualize_importance_across_proteins(self, protein_ids, importance_scores, residue_positions, save_path=None, show=True):
        """
        Compare residue importance scores across proteins.

        Args:
            protein_ids (list): Protein IDs.
            importance_scores (list): Importance scores per protein.
            residue_positions (list): Residue positions per protein.
            save_path (str, optional): Output path.
            show (bool): Whether to display the figure.

        Returns:
            matplotlib.figure.Figure: Figure object.
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Scatter plot for each protein
        for i, (protein_id, scores, positions) in enumerate(zip(protein_ids, importance_scores, residue_positions)):
            plt.scatter(positions, scores, label=protein_id, s=30, alpha=0.7)
            
        # Styling
        plt.xlabel('Residue ')
        plt.ylabel(' ')
        plt.title(' Protein Site ')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Show figure
        if show:
            plt.show()
            
        return plt.gcf()


def create_results_report(metrics, model_name, dataset_info=None, save_path=None):
    """
    Generate an evaluation results report.

    Args:
        metrics (dict): Metrics dictionary.
        model_name (str): Model name.
        dataset_info (dict, optional): Dataset information.
        save_path (str, optional): Output path.

    Returns:
        str: Report text.
    """
    # Report header
    report = f"# {model_name} Evaluation Report\n\n"
    
    # Dataset section
    if dataset_info:
        report += "## Dataset Information\n\n"
        report += f"- Dataset name: {dataset_info.get('name', 'Not specified')}\n"
        report += f"- Number of samples: {dataset_info.get('num_samples', 'Not specified')}\n"
        report += f"- Positive ratio: {dataset_info.get('positive_ratio', 'Not specified')}\n"
        report += f"- Number of features: {dataset_info.get('num_features', 'Not specified')}\n\n"
    
    # Metrics section
    report += "## Metrics\n\n"
    report += "| Metric | Value |\n"
    report += "| --- | --- |\n"
    
    # List key metrics in priority order
    key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc', 'auprc', 
                  'specificity', 'balanced_accuracy']
    
    for metric in key_metrics:
        if metric in metrics:
            report += f"| {metric.replace('_', ' ').title()} | {metrics[metric]:.4f} |\n"
    
    # Add remaining metrics
    for metric, value in metrics.items():
        if metric not in key_metrics:
            report += f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n"
    
    # Summary section
    report += "\n## Summary\n\n"
    
    # Heuristic performance summary based on F1 and MCC
    f1 = metrics.get('f1', 0)
    mcc = metrics.get('mcc', 0)
    
    if f1 > 0.9 and mcc > 0.8:
        performance = "excellent"
    elif f1 > 0.8 and mcc > 0.6:
        performance = "very good"
    elif f1 > 0.7 and mcc > 0.4:
        performance = "good"
    elif f1 > 0.6 and mcc > 0.2:
        performance = "fair"
    else:
        performance = "needs improvement"
        
    report += f"Overall model performance is **{performance}**.\n\n"
    
    # Timestamp
    from datetime import datetime
    report += f"\n\n*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    
    # Save report
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
    return report 
