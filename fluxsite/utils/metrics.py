import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve, auc, log_loss, brier_score_loss
)


def calculate_metrics(y_true, y_pred, y_prob=None, threshold=0.5, prefix=""):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels (if None, will be derived from y_prob)
        y_prob: Predicted probabilities (if None, metrics requiring probabilities will be skipped)
        threshold: Threshold for converting probabilities to binary predictions
        prefix: Prefix for metric names (e.g., 'train_', 'val_', 'test_')
        
    Returns:
        Dictionary of metrics
    """
    # If y_prob is provided but y_pred is not, derive y_pred from y_prob
    if y_pred is None and y_prob is not None:
        y_pred = (y_prob > threshold).astype(int)
    
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_prob, torch.Tensor) and y_prob is not None:
        y_prob = y_prob.cpu().numpy()
    
    # Ensure proper shape
    if len(y_true.shape) > 1 and y_true.shape[1] == 1:
        y_true = y_true.reshape(-1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.reshape(-1)
    if y_prob is not None and len(y_prob.shape) > 1 and y_prob.shape[1] == 1:
        y_prob = y_prob.reshape(-1)
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Basic metrics
    metrics[f"{prefix}accuracy"] = accuracy_score(y_true, y_pred)
    metrics[f"{prefix}precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics[f"{prefix}recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics[f"{prefix}f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics[f"{prefix}mcc"] = matthews_corrcoef(y_true, y_pred)
    
    # Calculate specificity (true negative rate)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics[f"{prefix}specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Sensitivity (Recall) alias
    metrics[f"{prefix}sensitivity"] = metrics[f"{prefix}recall"]
    
    # Balanced accuracy (average of sensitivity and specificity)
    metrics[f"{prefix}balanced_accuracy"] = (metrics[f"{prefix}recall"] + metrics[f"{prefix}specificity"]) / 2
    
    # Probability-based metrics (if y_prob is provided)
    if y_prob is not None:
        # Check for NaNs in y_prob
        if np.isnan(y_prob).any():
             y_prob = np.nan_to_num(y_prob, nan=0.0)

        # Check if both classes exist in the dataset
        if len(np.unique(y_true)) > 1:
            metrics[f"{prefix}roc_auc"] = roc_auc_score(y_true, y_prob)
            metrics[f"{prefix}pr_auc"] = average_precision_score(y_true, y_prob)
            metrics[f"{prefix}log_loss"] = log_loss(y_true, y_prob)
            metrics[f"{prefix}brier_score"] = brier_score_loss(y_true, y_prob)
        else:
            # Set default values if only one class exists
            metrics[f"{prefix}roc_auc"] = 0.5
            metrics[f"{prefix}pr_auc"] = y_true[0]  # If all positive, AUPRC=1; if all negative, AUPRC=0
            metrics[f"{prefix}log_loss"] = -np.log(0.5)
            metrics[f"{prefix}brier_score"] = 0.25
    
    return metrics


def calculate_optimal_threshold(
    y_true,
    y_prob,
    metric='f1',
    min_precision: float = 0.0,
    min_specificity: float = 0.0,
    min_recall: float = 0.0
):
    """
    Calculate the optimal threshold for converting probabilities to binary predictions.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
    metric: Metric to optimize ('f1', 'mcc', 'balanced_accuracy', 'precision', 'recall', 'specificity', 'f0.5', ...)
    min_precision: Minimum precision constraint a threshold must satisfy
    min_specificity: Minimum specificity (true negative rate) constraint
    min_recall: Minimum recall (sensitivity) constraint
        
    Returns:
        Optimal threshold, metric value at optimal threshold
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    # Check for NaNs in y_prob
    if np.isnan(y_prob).any():
        y_prob = np.nan_to_num(y_prob, nan=0.0)
    
    # Ensure proper shape
    if len(y_true.shape) > 1 and y_true.shape[1] == 1:
        y_true = y_true.reshape(-1)
    if len(y_prob.shape) > 1 and y_prob.shape[1] == 1:
        y_prob = y_prob.reshape(-1)
    
    # Sort predictions
    sorted_indices = np.argsort(y_prob)
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]
    
    # Normalize metric name for comparison
    metric = (metric or 'f1').lower()

    # Calculate performance metric for different thresholds
    thresholds = np.unique(y_prob_sorted)
    
    if len(thresholds) > 1000:
        # If there are too many unique thresholds, sample a reasonable number
        percentiles = np.linspace(0, 100, 1000)
        thresholds = np.percentile(y_prob_sorted, percentiles)
    
    # Add 0 and 1 to thresholds if not already present
    if thresholds[0] > 0:
        thresholds = np.append([0], thresholds)
    if thresholds[-1] < 1:
        thresholds = np.append(thresholds, [1])
    
    # Calculate metric for each threshold
    metric_values = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if precision_val < min_precision or recall_val < min_recall or specificity_val < min_specificity:
            score = -np.inf
        else:
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'mcc':
                score = matthews_corrcoef(y_true, y_pred)
            elif metric in {'balanced_accuracy', 'balanced', 'balanced_acc'}:
                score = (recall_val + specificity_val) / 2
            elif metric in {'recall', 'sensitivity', 'tpr'}:
                score = recall_val
            elif metric in {'precision', 'ppv'}:
                score = precision_val
            elif metric in {'specificity', 'tnr'}:
                score = specificity_val
            elif metric in {'accuracy', 'acc'}:
                score = accuracy_score(y_true, y_pred)
            elif metric in {'auroc', 'roc_auc'}:
                # Use Youden's J statistic (TPR - FPR) to approximate the optimal cutoff on the ROC curve
                score = recall_val + specificity_val - 1.0
            elif metric.startswith('f') and metric[1:].replace('.', '', 1).isdigit():
                try:
                    beta = float(metric[1:])
                    if beta <= 0:
                        raise ValueError
                except ValueError:
                    beta = 1.0
                score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        metric_values.append(score)
    
    # Find threshold with highest metric value
    best_idx = np.argmax(metric_values)
    best_threshold = thresholds[best_idx]
    best_metric = metric_values[best_idx]

    if best_metric == -np.inf:
        warnings.warn(
            "No threshold satisfied the provided precision/recall/specificity constraints; falling back to 0.5",
            RuntimeWarning
        )
        return 0.5, -np.inf
    
    return best_threshold, best_metric


def generate_threshold_curve(y_true, y_prob, metric='f1', num_thresholds=100):
    """
    Generate a curve of metric values for different thresholds.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        metric: Metric to calculate ('f1', 'mcc', 'balanced_accuracy')
        num_thresholds: Number of thresholds to evaluate
        
    Returns:
        DataFrame with thresholds and corresponding metric values
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    # Ensure proper shape
    if len(y_true.shape) > 1 and y_true.shape[1] == 1:
        y_true = y_true.reshape(-1)
    if len(y_prob.shape) > 1 and y_prob.shape[1] == 1:
        y_prob = y_prob.reshape(-1)
    
    # Generate thresholds
    thresholds = np.linspace(0, 1, num_thresholds)
    
    # Calculate metrics for each threshold
    results = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate various metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_acc = (recall + specificity) / 2
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'specificity': specificity,
            'balanced_accuracy': balanced_acc
        })
    
    return pd.DataFrame(results)


def calculate_per_class_metrics(y_true, y_pred, labels=None, target_names=None):
    """
    Calculate per-class classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of labels to include in the report (default: None)
        target_names: List of target names (default: None)
        
    Returns:
        DataFrame with per-class metrics
    """
    # Get classification report as dictionary
    report = classification_report(y_true, y_pred, labels=labels, 
                                  target_names=target_names, output_dict=True)
    
    # Convert to DataFrame and return
    report_df = pd.DataFrame(report).transpose()
    
    return report_df


def calculate_expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        Expected calibration error (ECE)
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    # Ensure proper shape
    if len(y_true.shape) > 1 and y_true.shape[1] == 1:
        y_true = y_true.reshape(-1)
    if len(y_prob.shape) > 1 and y_prob.shape[1] == 1:
        y_prob = y_prob.reshape(-1)
    
    # Define bins and bin the predicted probabilities
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    # Calculate average predicted probability in each bin
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_counts = np.bincount(binids, minlength=len(bins))
    
    # Avoid division by zero
    nonzero = bin_counts != 0
    bin_avgs = np.divide(bin_sums, bin_counts, out=np.zeros_like(bin_sums, dtype=float), where=nonzero)
    bin_true_avgs = np.divide(bin_true, bin_counts, out=np.zeros_like(bin_true, dtype=float), where=nonzero)
    
    # Calculate expected calibration error
    ece = np.sum(np.abs(bin_avgs - bin_true_avgs) * (bin_counts / len(y_true)))
    
    return ece


def calculate_metrics_over_time(y_true_list, y_prob_list, threshold=0.5):
    """
    Calculate metrics over time (e.g., over epochs).
    
    Args:
        y_true_list: List of true binary labels for each time point
        y_prob_list: List of predicted probabilities for each time point
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        DataFrame with metrics over time
    """
    metrics_list = []
    
    for i, (y_true, y_prob) in enumerate(zip(y_true_list, y_prob_list)):
        y_pred = (y_prob > threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        metrics['time_point'] = i
        metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list) 