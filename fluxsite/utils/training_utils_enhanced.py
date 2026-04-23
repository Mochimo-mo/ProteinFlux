"""
Enhanced training management utilities.

Provides training loop, validation, early stopping, and related helpers, with verbose logging aligned
to the original script style.
"""

import os
import time
import json
import math
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import TensorBoard only when available to avoid dependency issues
HAS_TENSORBOARD = False
SummaryWriter = None

SCHEDULER_ALIASES = {
    'cosineannealing': 'cosine',
    'cosineannealinglr': 'cosine',
    'cosine_warm_restart': 'cosine_warm_restarts',
    'cosine_warm_restarts': 'cosine_warm_restarts',
    'cosinewarmrestarts': 'cosine_warm_restarts',
    'cosine_with_warmup': 'warmup_cosine',
    'cosine-warmup': 'warmup_cosine',
    'cosinewarmup': 'warmup_cosine',
    'cosine_warmup': 'warmup_cosine',
    'one_cycle': 'onecycle'
}

def _try_import_tensorboard():
    """Try importing TensorBoard; return False when unavailable."""
    global HAS_TENSORBOARD, SummaryWriter
    try:
        from torch.utils.tensorboard import SummaryWriter as TB_SummaryWriter
        HAS_TENSORBOARD = True
        SummaryWriter = TB_SummaryWriter
        return True
    except Exception as e:
        HAS_TENSORBOARD = False
        SummaryWriter = None
        return False
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score, roc_auc_score,
    matthews_corrcoef, precision_recall_curve, auc, confusion_matrix,
    balanced_accuracy_score
)

# Optional adaptive-dropout scheduler.
try:
    from .adaptive_dropout import DropoutScheduler
    HAS_DROPOUT_SCHEDULER = True
except ImportError:
    HAS_DROPOUT_SCHEDULER = False
    DropoutScheduler = None


def check_gpu_memory():
    """Check current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "CUDA not available"


class TensorRangeObserver:
    """Collect tensor statistics to help locate NaN/Inf sources during training."""

    def __init__(self, model, logger, output_dir, config=None):
        self.logger = logger
        self.config = config or {}
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.enabled = bool(self.config.get('enable_nan_debugging', False))
        self.global_step = 0
        self.context_stack = []
        self.handles = []
        self.module_counters = defaultdict(int)
        self.batch_counter = 0
        self.saved_tensor_count = 0

        # Early exit if disabled, but keep attributes callable.
        if not self.enabled or model is None:
            return

        self.frequency = max(1, int(self.config.get('nan_debug_frequency', 50)))
        self.log_finite = bool(self.config.get('nan_debug_log_finite', False))
        self.break_on_error = bool(self.config.get('nan_debug_break_on_error', False))
        self.save_on_issue = bool(self.config.get('nan_debug_save_tensors_on_anomaly', True))
        self.max_saved_tensors = max(0, int(self.config.get('nan_debug_max_saved_tensors', 10)))
        self.batch_check_enabled = bool(self.config.get('nan_debug_check_batches', False))
        self.batch_frequency = max(1, int(self.config.get('nan_debug_batch_frequency', 1)))
        self.max_records_per_call = max(0, int(self.config.get('nan_debug_max_tensors_per_call', 6)))

        self.targets = self._normalize_targets(self.config.get('nan_debug_targets'))
        self.watch_all = self.targets is None

        save_dir_cfg = self.config.get('nan_debug_save_dir')
        self.save_dir = Path(save_dir_cfg) if save_dir_cfg else (self.output_dir / 'nan_debug')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.save_dir / 'nan_events.jsonl'

        self.logger.info(
            "NaN Enable: watch_all=%s, targets=%s, log_finite=%s, frequency=%s, save_dir=%s",
            self.watch_all,
            self.targets if not self.watch_all else 'ALL',
            self.log_finite,
            self.frequency,
            self.save_dir,
        )

        self._register_hooks(model)

    # ------------------------------------------------------------------
    # Public helper API
    # ------------------------------------------------------------------
    def update_global_step(self, step):
        if not self.enabled:
            return
        self.global_step = int(step)

    def push_context(self, **context):
        if not self.enabled:
            return
        merged = dict(context)
        merged.setdefault('global_step', self.global_step)
        self.context_stack.append(merged)

    def pop_context(self):
        if not self.enabled or not self.context_stack:
            return
        self.context_stack.pop()

    def inspect_batch(self, batch, *, phase, epoch=None, batch_idx=None, global_step=None):
        if not self.enabled or not self.batch_check_enabled:
            return

        self.batch_counter += 1
        if self.batch_counter % self.batch_frequency != 0:
            return

        context = {
            'phase': phase,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'global_step': self.global_step if global_step is None else global_step,
            'scope': 'batch',
        }

        log_snapshot = self.log_finite and (self.batch_counter % self.frequency == 0)
        logged_snapshots = 0

        for tensor_index, (name, tensor) in enumerate(self._iter_named_tensors(batch)):
            stats = self._tensor_stats(tensor)
            if stats is None:
                continue

            stats.update({
                'module': name,
                'kind': 'batch_input',
                'tensor_index': tensor_index,
                'call_index': self.batch_counter,
            })

            snapshot_allowed = log_snapshot and (
                self.max_records_per_call <= 0 or logged_snapshots < self.max_records_per_call
            )

            logged = self._handle_stats(
                stats,
                context,
                tensor if (stats.get('has_nan') or stats.get('has_inf')) else None,
                log_snapshot=snapshot_allowed,
                anomaly_event='batch_anomaly',
            )

            if logged and not (stats.get('has_nan') or stats.get('has_inf')):
                logged_snapshots += 1

    def observe_tensor(
        self,
        name,
        tensor,
        *,
        phase=None,
        epoch=None,
        batch_idx=None,
        global_step=None,
        reason='value',
        log_snapshot=False,
    ):
        if not self.enabled:
            return

        stats = self._tensor_stats(tensor)
        if stats is None:
            return

        context = {
            'phase': phase,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'global_step': self.global_step if global_step is None else global_step,
            'scope': 'observe_tensor',
            'module_name': name,
        }

        stats.update({
            'module': name,
            'kind': reason,
            'tensor_index': 0,
            'call_index': None,
        })

        self._handle_stats(
            stats,
            context,
            tensor if (stats.get('has_nan') or stats.get('has_inf')) else None,
            log_snapshot=log_snapshot,
        )

    def observe_scalar(
        self,
        name,
        value,
        *,
        phase=None,
        epoch=None,
        batch_idx=None,
        global_step=None,
    ):
        if not self.enabled:
            return

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return
            scalar = float(value.detach().cpu().item())
        else:
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                return

        stats = {
            'module': name,
            'kind': 'scalar',
            'tensor_index': 0,
            'call_index': None,
            'dtype': 'float',
            'shape': [],
            'numel': 1,
            'min': scalar,
            'max': scalar,
            'mean': scalar,
            'std': 0.0,
            'abs_max': abs(scalar),
            'has_nan': math.isnan(scalar),
            'has_inf': math.isinf(scalar),
            'device': 'cpu',
            'requires_grad': False,
        }

        context = {
            'phase': phase,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'global_step': self.global_step if global_step is None else global_step,
            'scope': 'scalar',
            'module_name': name,
        }

        self._handle_stats(
            stats,
            context,
            tensor=None,
            log_snapshot=False,
            anomaly_event='scalar_anomaly',
        )

    def close(self):
        if not self.handles:
            return
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                continue
        self.handles.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_targets(self, targets):
        if targets is None:
            return []
        if isinstance(targets, str):
            text = targets.strip()
            if not text:
                return []
            if text.lower() in {'*', 'all', 'any'}:
                return None
            return [text.lower()]
        normalized = []
        watch_all = False
        for item in targets if isinstance(targets, (list, tuple, set)) else [targets]:
            if item is None:
                continue
            text = str(item).strip()
            if not text:
                continue
            if text.lower() in {'*', 'all', 'any'}:
                watch_all = True
            else:
                normalized.append(text.lower())
        if watch_all:
            return None
        return normalized

    def _register_hooks(self, model):
        for name, module in model.named_modules():
            module_name = name or module.__class__.__name__
            if not self._should_watch(module_name, module):
                continue
            try:
                handle = module.register_forward_hook(self._make_hook(module_name))
                self.handles.append(handle)
            except Exception as exc:
                self.logger.warning(" NaN Failed: module=%s, error=%s", module_name, exc)

        if not self.handles:
            self.logger.warning("NaN, nan_debug_targetsConfig")

    def _should_watch(self, module_name, module):
        if self.watch_all:
            return True
        if not self.targets:
            # Default to watching only the final model output.
            return module_name in {'model', module.__class__.__name__}

        label = module_name.lower()
        class_name = module.__class__.__name__.lower()
        for target in self.targets:
            if target in label or target in class_name:
                return True
        return False

    def _make_hook(self, module_name):
        def hook(module, inputs, output):
            if not self.enabled:
                return

            call_index = self.module_counters[module_name] + 1
            self.module_counters[module_name] = call_index

            context = dict(self._current_context())
            context.setdefault('scope', 'forward_hook')
            context['module_name'] = module_name
            context['call_index'] = call_index

            log_snapshot = self.log_finite and (call_index % self.frequency == 0)
            logged_snapshots = 0

            for tensor_index, tensor in enumerate(self._iter_tensors(inputs)):
                stats = self._tensor_stats(tensor)
                if stats is None:
                    continue
                stats.update({
                    'module': module_name,
                    'kind': 'input',
                    'tensor_index': tensor_index,
                    'call_index': call_index,
                })
                snapshot_allowed = log_snapshot and (
                    self.max_records_per_call <= 0 or logged_snapshots < self.max_records_per_call
                )
                logged = self._handle_stats(
                    stats,
                    context,
                    tensor if (stats.get('has_nan') or stats.get('has_inf')) else None,
                    log_snapshot=snapshot_allowed,
                )
                if logged and not (stats.get('has_nan') or stats.get('has_inf')):
                    logged_snapshots += 1

            for tensor_index, tensor in enumerate(self._iter_tensors(output)):
                stats = self._tensor_stats(tensor)
                if stats is None:
                    continue
                stats.update({
                    'module': module_name,
                    'kind': 'output',
                    'tensor_index': tensor_index,
                    'call_index': call_index,
                })
                snapshot_allowed = log_snapshot and (
                    self.max_records_per_call <= 0 or logged_snapshots < self.max_records_per_call
                )
                logged = self._handle_stats(
                    stats,
                    context,
                    tensor if (stats.get('has_nan') or stats.get('has_inf')) else None,
                    log_snapshot=snapshot_allowed,
                )
                if logged and not (stats.get('has_nan') or stats.get('has_inf')):
                    logged_snapshots += 1

        return hook

    def _current_context(self):
        if self.context_stack:
            return self.context_stack[-1]
        return {'global_step': self.global_step}

    def _iter_tensors(self, value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from self._iter_tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from self._iter_tensors(item)

    def _iter_named_tensors(self, value, prefix=None):
        if isinstance(value, torch.Tensor):
            yield prefix or 'tensor', value
        elif isinstance(value, dict):
            for key, item in value.items():
                new_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from self._iter_named_tensors(item, new_prefix)
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                new_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                yield from self._iter_named_tensors(item, new_prefix)

    def _tensor_stats(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            return None

        with torch.no_grad():
            data = tensor.detach()

            is_nested = bool(getattr(data, "is_nested", False))
            if is_nested:
                try:
                    # Attempt to convert to padded tensor for stats
                    # Use 0.0 as padding value which is standard for stats
                    padded_data = data.to_padded_tensor(0.0)
                    
                    # Recursively call _tensor_stats on the padded version
                    # But mark it as nested in the result
                    nested_stats = self._tensor_stats(padded_data)
                    if nested_stats:
                        nested_stats['is_nested'] = True
                        nested_stats['notes'] = 'nested_tensor_converted_to_padded'
                        return nested_stats
                        
                except Exception as exc:
                    self.logger.warning(
                        "NaN: NestedTensor (%s), ",
                        exc,
                    )
                    return {
                        'dtype': str(data.dtype),
                        'shape': None, # Shape might be ragged
                        'numel': None,
                        'device': str(data.device),
                        'requires_grad': bool(tensor.requires_grad),
                        'has_nan': False, # Cannot determine easily without conversion
                        'has_inf': False,
                        'min': None,
                        'max': None,
                        'mean': None,
                        'std': None,
                        'abs_max': None,
                        'notes': 'nested_tensor_conversion_failed',
                    }

            stats = {
                'dtype': str(data.dtype),
                'shape': list(data.shape),
                'numel': int(data.numel()),
                'device': str(data.device),
                'requires_grad': bool(tensor.requires_grad),
                'is_nested': is_nested,
            }

            if data.numel() == 0:
                stats.update({
                    'has_nan': False,
                    'has_inf': False,
                    'min': None,
                    'max': None,
                    'mean': None,
                    'std': None,
                    'abs_max': None,
                })
                return stats

            try:
                if data.is_floating_point() or data.dtype in {torch.float16, torch.bfloat16}:
                    has_nan = bool(torch.isnan(data).any().item())
                    has_inf = bool(torch.isinf(data).any().item())
                    finite_mask = torch.isfinite(data)
                    if finite_mask.any():
                        finite_values = data[finite_mask].float()
                        stats['min'] = float(finite_values.min().item())
                        stats['max'] = float(finite_values.max().item())
                        stats['mean'] = float(finite_values.mean().item())
                        stats['abs_max'] = float(torch.max(torch.abs(finite_values)).item())
                        if finite_values.numel() > 1:
                            stats['std'] = float(finite_values.std(unbiased=False).item())
                        else:
                            stats['std'] = 0.0
                    else:
                        stats['min'] = stats['max'] = stats['mean'] = stats['std'] = stats['abs_max'] = None
                    stats['has_nan'] = has_nan
                    stats['has_inf'] = has_inf
                else:
                    data_fp = data.float()
                    stats['min'] = float(data_fp.min().item())
                    stats['max'] = float(data_fp.max().item())
                    stats['mean'] = float(data_fp.mean().item())
                    stats['abs_max'] = float(torch.max(torch.abs(data_fp)).item())
                    if data_fp.numel() > 1:
                        stats['std'] = float(data_fp.std(unbiased=False).item())
                    else:
                        stats['std'] = 0.0
                    stats['has_nan'] = False
                    stats['has_inf'] = False
            except Exception as exc:
                stats.update({
                    'has_nan': False,
                    'has_inf': False,
                    'min': None,
                    'max': None,
                    'mean': None,
                    'std': None,
                    'abs_max': None,
                    'error': str(exc),
                })

            return stats

    def _handle_stats(self, stats, context, tensor, log_snapshot, *, anomaly_event='anomaly', snapshot_event='snapshot'):
        logged = False
        has_nan = bool(stats.get('has_nan'))
        has_inf = bool(stats.get('has_inf'))

        if has_nan or has_inf:
            self._emit_event(stats, context, anomaly_event, tensor=tensor)
            logged = True
        elif log_snapshot and self.log_finite:
            self._emit_event(stats, context, snapshot_event, tensor=None)
            logged = True

        return logged

    def _emit_event(self, stats, context, event_type, tensor=None):
        event = {
            'event_type': event_type,
            'timestamp': time.time(),
            'module': stats.get('module'),
            'kind': stats.get('kind'),
            'tensor_index': stats.get('tensor_index'),
            'call_index': stats.get('call_index'),
            'shape': stats.get('shape'),
            'dtype': stats.get('dtype'),
            'numel': stats.get('numel'),
            'min': stats.get('min'),
            'max': stats.get('max'),
            'mean': stats.get('mean'),
            'std': stats.get('std'),
            'abs_max': stats.get('abs_max'),
            'has_nan': stats.get('has_nan'),
            'has_inf': stats.get('has_inf'),
            'device': stats.get('device'),
            'requires_grad': stats.get('requires_grad'),
            'context': context,
        }

        message = (
            f"[NaN Debug][{event_type.upper()}] module={event['module']} kind={event['kind']} "
            f"shape={event['shape']} dtype={event['dtype']} min={event['min']} max={event['max']} "
            f"mean={event['mean']} std={event['std']} abs_max={event['abs_max']} "
            f"context=({self._format_context(context)})"
        )

        if str(event_type).lower().endswith('anomaly'):
            self.logger.error(message)
        else:
            self.logger.info(message)

        if tensor is not None and self.save_on_issue and self.saved_tensor_count < self.max_saved_tensors:
            try:
                file_name = self._build_tensor_filename(event)
                file_path = self.save_dir / file_name
                torch.save(tensor.detach().cpu(), file_path)
                event['tensor_path'] = str(file_path)
                self.saved_tensor_count += 1
            except Exception as exc:
                self.logger.warning("SaveNaN Failed: %s", exc)

        try:
            with open(self.events_path, 'a', encoding='utf-8') as fp:
                fp.write(json.dumps(self._to_serializable(event), ensure_ascii=False) + '\n')
        except Exception as exc:
            self.logger.warning(" NaN Failed: %s", exc)

        if str(event_type).lower().endswith('anomaly') and self.break_on_error:
            raise RuntimeError(message)

    def _format_context(self, context):
        keys = ['phase', 'epoch', 'batch_idx', 'global_step', 'call_index', 'scope', 'module_name']
        parts = []
        for key in keys:
            value = context.get(key)
            if value is not None:
                parts.append(f"{key}={value}")
        return ', '.join(parts)

    def _build_tensor_filename(self, event):
        module = str(event.get('module') or 'tensor').replace('.', '_')
        kind = str(event.get('kind') or 'value')
        call_index = event.get('call_index')
        tensor_index = event.get('tensor_index')
        global_step = event.get('context', {}).get('global_step', 'na')
        phase = event.get('context', {}).get('phase', 'unknown')
        return f"{phase}_step{global_step}_call{call_index}_idx{tensor_index}_{module}_{kind}.pt"

    def _to_serializable(self, event):
        def convert(value):
            if isinstance(value, (int, float, str)) or value is None:
                return value
            if isinstance(value, (np.integer, np.floating)):
                return float(value)
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            try:
                return float(value)
            except Exception:
                return str(value)

        return convert(event)

class EnhancedLoss(nn.Module):
    """Enhanced loss: focal loss + contrastive learning."""
    def __init__(self, alpha=0.25, gamma=2.0, temperature=0.07):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        
    def focal_loss(self, predictions, targets):
        """Focal loss for class imbalance."""
        # Ensure targets are float for BCE
        targets = targets.float()
        
        # Calculate BCE with logits
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Calculate alpha_t
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
    
    def contrastive_loss(self, features, labels):
        """Contrastive loss to improve feature representations."""
        if features is None:
            return torch.tensor(0.0, device=labels.device)
            
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Feature similarity matrix.
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Build positive/negative masks.
        labels = labels.float().view(-1, 1)
        positive_mask = (labels == labels.T).float()
        
        # Mask out self-similarity
        mask_diag = torch.eye(labels.shape[0], device=labels.device)
        positive_mask = positive_mask - mask_diag
        
        negative_mask = 1 - positive_mask - mask_diag
        
        # Contrastive loss.
        # Log-Sum-Exp trick for numerical stability could be used, but following report logic:
        exp_sim = torch.exp(similarity_matrix)
        
        # Mask out self-similarity from exp_sim
        exp_sim = exp_sim * (1 - mask_diag)
        
        positive_sum = (exp_sim * positive_mask).sum(dim=1)
        negative_sum = (exp_sim * negative_mask).sum(dim=1)
        
        # Avoid division by zero
        denominator = positive_sum + negative_sum + 1e-8
        
        # Only compute loss for samples that have positives (other than themselves)
        has_positives = positive_mask.sum(dim=1) > 0
        
        if has_positives.sum() == 0:
             return torch.tensor(0.0, device=features.device)

        loss = -torch.log((positive_sum[has_positives] + 1e-8) / denominator[has_positives])
        return loss.mean()

class AdaptiveWeightScheduler:
    """Adaptive class-weight scheduler."""
    def __init__(self, initial_weights, adaptation_rate=0.01):
        self.weights = initial_weights if isinstance(initial_weights, torch.Tensor) else torch.tensor(initial_weights)
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
    
    def update_weights(self, class_accuracies):
        """Adjust weights dynamically based on per-class accuracy."""
        # Ensure weights are on the same device as before or CPU
        device = self.weights.device
        
        for i, acc in enumerate(class_accuracies):
            if i >= len(self.weights):
                break
                
            if acc < 0.8:  # Increase weight for underperforming classes.
                self.weights[i] *= (1 + self.adaptation_rate)
            elif acc > 0.95:  # Decrease weight for overperforming classes.
                self.weights[i] *= (1 - self.adaptation_rate)
        
        # Normalize weights.
        self.weights = self.weights / self.weights.sum() * len(self.weights)
        self.weights = self.weights.to(device)
        return self.weights

class EnhancedTrainingManager:
    """Enhanced training manager implementing the end-to-end training workflow with verbose logs."""
    
    def __init__(self, model, config, train_loader, val_loader, optimizer, scheduler, device, output_dir):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        
        # Logging.
        self.logger = self._setup_logger()
        
        # TensorBoard.
        self.tb_writer = self._setup_tensorboard()
        
        # Loss configuration.
        self._setup_loss_config()
        
        # Early stopping.
        self._setup_early_stopping()
        
        # Training history.
        self.training_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': [],
            'val_precision': [], 'val_recall': [], 'val_mcc': [], 'val_auprc': []
        }

        # Adaptive dropout scheduler.
        self.dropout_scheduler = None
        if HAS_DROPOUT_SCHEDULER and self.config.get('adaptive_dropout', True):
            try:
                self.dropout_scheduler = DropoutScheduler(self.model, self.config)
                self.logger.info(" Initialize Dropout ")
            except Exception as e:
                self.logger.warning(f"InitializeDropout Failed: {e}")
                self.dropout_scheduler = None

        # Cache standard Dropout layers to support stage-wise regularization adjustments.
        self._dropout_layers = []
        self._dropout_base_rates = {}
        self._cache_dropout_layers()

        # NaN/Inf debugging observer
        self.nan_observer = TensorRangeObserver(
            model=self.model,
            logger=self.logger,
            output_dir=self.output_dir,
            config=self.config,
        )

        # Mixup and random masking.
        self.use_mixup = bool(self.config.get('use_mixup', False))
        self.mixup_alpha = float(self.config.get('mixup_alpha', 0.2))
        self.mixup_prob = float(self.config.get('mixup_prob', 0.0))
        self.random_mask_prob = float(self.config.get('random_mask_prob', 0.0))
        self.random_mask_scale = float(self.config.get('random_mask_scale', 0.0))
        self.random_mask_apply_to_structure = bool(self.config.get('random_mask_apply_to_structure', True))
        self.random_mask_apply_to_local = bool(self.config.get('random_mask_apply_to_local', True))

        # Gradient accumulation and clipping.
        self.grad_accum_steps = max(1, int(self.config.get('grad_accum_steps', 1)))
        
        # Automatic mixed precision (AMP).
        self.use_amp = bool(self.config.get('use_amp', False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.use_amp:
            self.logger.info(" Enable (AMP) Training")

        gradient_clip = self.config.get('gradient_clip_val', None)
        try:
            self.gradient_clip_val = float(gradient_clip) if gradient_clip is not None else None
        except (TypeError, ValueError):
            self.gradient_clip_val = None
        if self.gradient_clip_val is not None and self.gradient_clip_val <= 0:
            self.gradient_clip_val = None

        # Weight decay warmup.
        self.weight_decay_warmup_steps = int(self.config.get('weight_decay_warmup_steps', 0))
        self.weight_decay_warmup_init = float(self.config.get('weight_decay_warmup_init', 0.0))
        self._target_weight_decays = [group.get('weight_decay', self.config.get('weight_decay', 0.0)) for group in self.optimizer.param_groups]
        self._base_target_weight_decays = list(self._target_weight_decays)
        wd_final_cfg = self.config.get('weight_decay_warmup_final')
        self.weight_decay_warmup_final = float(wd_final_cfg) if wd_final_cfg is not None else None
        if self.weight_decay_warmup_final is None and self.config.get('weight_decay') is not None:
            self.weight_decay_warmup_final = float(self.config.get('weight_decay'))
        if self.weight_decay_warmup_final is None and self._target_weight_decays:
            self.weight_decay_warmup_final = float(self._target_weight_decays[0])
        if self.weight_decay_warmup_final is None:
            self.weight_decay_warmup_final = 0.0
        self._base_weight_decay_final = float(self.weight_decay_warmup_final)
        if self.weight_decay_warmup_steps > 0:
            self.logger.info(
                f"Enable Warmup: steps={self.weight_decay_warmup_steps}, "
                f"start={self.weight_decay_warmup_init}, target≈{self.weight_decay_warmup_final}"
            )
            for idx, group in enumerate(self.optimizer.param_groups):
                target = self._target_weight_decays[idx]
                if target == 0.0 and self.weight_decay_warmup_final:
                    target = self.weight_decay_warmup_final
                    self._target_weight_decays[idx] = target
                group['weight_decay'] = self.weight_decay_warmup_init

        # Scheduler info.
        scheduler_config_name = self.config.get('scheduler_normalized', self.config.get('scheduler', 'cosine')) if self.config else 'cosine'
        scheduler_name = str(scheduler_config_name).lower()
        self.scheduler_type = SCHEDULER_ALIASES.get(scheduler_name, scheduler_name)
        if self.scheduler is None:
            self.scheduler_type = None
        self.scheduler_step_on_batch = self.scheduler_type in {'onecycle', 'warmup_cosine'}
        self.global_step = 0

        # Threshold tuning constraints.
        self.threshold_min_precision = float(self.config.get('threshold_min_precision', 0.0)) if self.config else 0.0
        self.threshold_min_specificity = float(self.config.get('threshold_min_specificity', 0.0)) if self.config else 0.0
        self.threshold_min_recall = float(self.config.get('threshold_min_recall', 0.0)) if self.config else 0.0

        # Stage-wise training configuration.
        self.structure_warmup_epochs = int(self.config.get('structure_warmup_epochs', 0)) if self.config else 0
        full_epoch = self.config.get('full_finetune_epoch') if self.config else None
        try:
            self.full_finetune_epoch = int(full_epoch) if full_epoch is not None else None
        except (TypeError, ValueError):
            self.full_finetune_epoch = None
        self._current_stage = None
        
        # Adaptive weight scheduler.
        self.adaptive_weight_scheduler = None
    
    def _setup_logger(self):
        """Create a logger using a format aligned with the original script."""
        logger = logging.getLogger("train_advanced")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers.
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create log directory.
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler.
        file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
        file_formatter = logging.Formatter('%(asctime)s - train_advanced - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler.
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')  # Simplified console formatter
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_tensorboard(self):
        """Set up TensorBoard."""
        # Try importing TensorBoard.
        if _try_import_tensorboard():
            tb_log_dir = os.path.join(self.output_dir, 'tensorboard', 'version_0')
            os.makedirs(tb_log_dir, exist_ok=True)
            return SummaryWriter(log_dir=tb_log_dir)
        else:
            self.logger.warning("TensorBoardUnavailable,SkipTensorBoard ")
            return None
    
    def _setup_loss_config(self):
        """Set up loss configuration."""
        self.loss_config = {
            'use_focal_loss': self.config.get('use_focal_loss', True),
            'use_label_smoothing': self.config.get('use_label_smoothing', True),
            'loss_combination_weight': self.config.get('loss_combination_weight', 0.7),
            'focal_alpha': self.config.get('focal_alpha', 0.25),
            'focal_gamma': self.config.get('focal_gamma', 1.5),
            'label_smoothing': self.config.get('label_smoothing', 0.05),
            'use_class_weights': self.config.get('use_class_weights', True),
            'use_mcc_loss': self.config.get('use_mcc_loss', False),
            'mcc_loss_weight': self.config.get('mcc_loss_weight', 0.5),
            'use_contrastive_loss': self.config.get('use_contrastive_loss', False),
            'contrastive_loss_weight': self.config.get('contrastive_loss_weight', 0.1),
            'use_adaptive_weighting': self.config.get('use_adaptive_weighting', False)
        }

        # Initialize EnhancedLoss
        self.enhanced_loss = EnhancedLoss(
            alpha=self.loss_config['focal_alpha'], 
            gamma=self.loss_config['focal_gamma']
        )
        
        # Initialize AdaptiveWeightScheduler
        if self.loss_config['use_adaptive_weighting']:
            # Assume 2 classes for now, or get from config
            # Using simple [1.0, 1.0] initialization, logic will normalize it
            self.adaptive_weight_scheduler = AdaptiveWeightScheduler(
                initial_weights=[1.0, 1.0],
                adaptation_rate=self.config.get('adaptive_weight_rate', 0.01)
            )

        self.pos_weight = None
        if self.loss_config['use_class_weights']:
            pos_weight_value = self.config.get('pos_weight', 1.0)
            try:
                self.pos_weight = torch.tensor(float(pos_weight_value), dtype=torch.float32, device=self.device)
            except (TypeError, ValueError):
                self.logger.warning(f"pos_weight {pos_weight_value}, ")
                self.pos_weight = None
        self.loss_config['pos_weight'] = self.pos_weight

        # Log final loss-related configuration for debugging parameter wiring.
        pos_weight_display = None
        if isinstance(self.pos_weight, torch.Tensor):
            pos_weight_display = float(self.pos_weight.detach().cpu().item())
        elif self.pos_weight is not None:
            try:
                pos_weight_display = float(self.pos_weight)
            except (TypeError, ValueError):
                pos_weight_display = str(self.pos_weight)

        self.logger.info(
            " Config: "
            f"focal_loss={self.loss_config['use_focal_loss']} (alpha={self.loss_config['focal_alpha']}, gamma={self.loss_config['focal_gamma']}), "
            f"label_smoothing={self.loss_config['use_label_smoothing']} (ε={self.loss_config['label_smoothing']}), "
            f"mcc_loss={self.loss_config['use_mcc_loss']} (weight={self.loss_config['mcc_loss_weight']}), "
            f"loss_mix_weight={self.loss_config['loss_combination_weight']}, "
            f"pos_weight={pos_weight_display if pos_weight_display is not None else 'None'}"
        )
    
    def _setup_early_stopping(self):
        """Set up early stopping configuration."""
        self.early_stopping = {
            'use_early_stopping': self.config.get('use_early_stopping', True),  # Enabled by default
            'patience': self.config.get('early_stopping_patience', 10),  # Patience
            'min_delta': self.config.get('early_stopping_min_delta', 0.001),
            'monitor_metric': self.config.get('early_stopping_monitor', 'val_loss'),  # Metric to monitor
            'mode': self.config.get('early_stopping_mode', 'min'),  # "min" means lower is better
            'best_score': float('inf') if self.config.get('early_stopping_mode', 'min') == 'min' else float('-inf'),
            'counter': 0,
            'restore_best_weights': self.config.get('restore_best_weights', True),  # Restore best weights
            'verbose': self.config.get('early_stopping_verbose', True)
        }
        self.monitor_metric = self.early_stopping['monitor_metric']
    
    def _ensure_micro_env_precomputed(self, loader, name):
        """Ensure micro-environment features are precomputed for a dataset."""
        if not loader or not hasattr(loader, 'dataset'):
            return
            
        dataset = loader.dataset
        # Handle Subset if split was used
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
            
        if hasattr(dataset, 'protein_store') and hasattr(dataset.protein_store, 'precompute_micro_environment'):
            if hasattr(dataset, 'data_frame') and 'uniprot_id' in dataset.data_frame.columns:
                self.logger.info(f" {name}.")
                try:
                    pids = dataset.data_frame['uniprot_id'].unique().tolist()
                    # Trigger pre-computation (it handles skipping if already done/cached)
                    dataset.protein_store.precompute_micro_environment(pids)
                except Exception as e:
                    self.logger.warning(f" {name}: {e}")

    def train(self):
        """Run the full training workflow."""
        self.logger.info("=== Start Enhanced Training ===")
        
        # Ensure micro-environment features are pre-computed
        self._ensure_micro_env_precomputed(self.train_loader, "Training")
        self._ensure_micro_env_precomputed(self.val_loader, "Validation")
        
        # Track best metrics.
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_val_mcc = 0.0
        best_val_auprc = 0.0
        best_epoch = 0
        
        early_stopping_counter = 0
        early_stopping_patience = self.early_stopping['patience']
        use_early_stopping = self.early_stopping['use_early_stopping']
        
        # Check initial GPU status.
        if torch.cuda.is_available():
            self.logger.info("TrainingStart GPU:")
            self.logger.info(check_gpu_memory())
        
        self.logger.info(f": {self.grad_accum_steps}")
        if self.gradient_clip_val is not None:
            self.logger.info(f" Threshold: {self.gradient_clip_val}")
        else:
            self.logger.info(": Enable")
        if self.scheduler_type:
            self.logger.info(f": {self.scheduler_type}")
        else:
            self.logger.info(": Use")

        num_epochs = self.config.get('epochs', 50)
        
        for epoch in range(num_epochs):
            # Epoch header.
            self.logger.info("="*80)
            self.logger.info(f"StartTraining Epoch {epoch+1}/{num_epochs}")
            self.logger.info("="*80)
            
            # GPU memory at epoch start.
            if torch.cuda.is_available():
                self.logger.info(f"Epoch {epoch+1} Start GPU:")
                self.logger.info(check_gpu_memory())
            
            # Stage-wise training schedule.
            self._update_training_stage(epoch)

            # Update center curriculum (if supported by the model).
            if hasattr(self.model, 'update_center_curriculum'):
                self.model.update_center_curriculum(epoch)

            # Train one epoch.
            train_metrics = self._train_epoch(epoch)

            # Validate.
            val_metrics = self._validate_epoch(epoch)

            # Update adaptive weights.
            if self.adaptive_weight_scheduler is not None:
                # Use validation metrics as a proxy for per-class accuracies:
                # - pos_acc: recall
                # - neg_acc: specificity when available; otherwise a fallback heuristic.
                if 'val_specificity' in val_metrics: # check if I added it or if it exists
                     neg_acc = val_metrics['val_specificity']
                else:
                     # Fallback heuristic when specificity is not available.
                     pos_acc = val_metrics.get('val_recall', 0.5)
                     neg_acc = 0.8  # Assume negatives are typically easier
                     
                class_accuracies = [neg_acc, val_metrics.get('val_recall', 0.5)]
                
                new_weights = self.adaptive_weight_scheduler.update_weights(class_accuracies)
                
                # Note: the weights are consumed directly in _compute_basic_loss, so there is no need to
                # additionally update self.pos_weight here unless used for logging.
                self.logger.info(f"Epoch {epoch+1}: {new_weights.tolist()}")

            # Update adaptive dropout schedule.
            if self.dropout_scheduler is not None:
                self.dropout_scheduler.step(epoch, val_metrics['val_loss'])

                # Log dropout rates every 10 epochs.
                if epoch % 10 == 0:
                    dropout_rates = self.dropout_scheduler.get_all_dropout_rates()
                    if dropout_rates:
                        self.logger.info("Current dropout rates:")
                        for layer_name, rate in dropout_rates.items():
                            self.logger.info(f"  {layer_name}: {rate:.4f}")

            # Call model-level dropout scheduling hook when available.
            if hasattr(self.model, 'update_dropout_schedule'):
                self.model.update_dropout_schedule(epoch, val_metrics['val_loss'])

            # Update training history.
            self._update_history(train_metrics, val_metrics)
            
            # Log to TensorBoard.
            self._log_to_tensorboard(epoch, train_metrics, val_metrics)

            # Step epoch-based scheduler.
            self._step_scheduler_epoch(epoch, val_metrics)
            
            # Check early stopping / best model.
            improvement_flag = ""
            should_stop, improvement_flag = self._check_early_stopping(
                val_metrics, epoch, best_val_loss, best_val_acc, best_val_f1,
                best_val_mcc, best_val_auprc, best_epoch
            )

            # Update best metrics (when improved).
            if "New best model" in improvement_flag:
                best_val_loss = val_metrics['val_loss']
                best_val_acc = val_metrics['val_acc']
                best_val_f1 = val_metrics.get('val_f1', 0.0)
                best_val_mcc = val_metrics.get('val_mcc', 0.0)
                best_val_auprc = val_metrics.get('val_auprc', 0.0)
                best_epoch = epoch

            # Stop if early stopping triggers.
            if should_stop:
                self.logger.info("="*80)
                self.logger.info(f", {epoch+1} Training")
                self.logger.info(f"BestModel {best_epoch+1} ")
                self.logger.info("="*80)
                break
            
            # Epoch summary.
            self.logger.info("="*60)
            self.logger.info(f"Epoch {epoch+1} {improvement_flag}")
            self.logger.info(f"Training: Loss={train_metrics.get('loss', 0):.4f}, Acc={train_metrics.get('accuracy', 0):.4f}")
            self.logger.info(f"Validation: Loss={val_metrics['val_loss']:.4f}, Acc={val_metrics['val_acc']:.4f}, "
                           f"F1={val_metrics.get('val_f1', 0):.4f}")
            self.logger.info(f" Best: Epoch {best_epoch+1}, Loss={best_val_loss:.4f}, "
                           f"Acc={best_val_acc:.4f}, F1={best_val_f1:.4f}")
            self.logger.info("="*60)
        
        # Save final model.
        final_model_path = os.path.join(self.output_dir, 'final_model.pt')
        torch.save(self.model.state_dict(), final_model_path)
        
        # Close TensorBoard writer.
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        # Return best metrics.
        return {
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'best_val_f1': best_val_f1,
            'best_val_mcc': best_val_mcc,
            'best_val_auprc': best_val_auprc,
            'best_epoch': best_epoch,
            'history': self.training_history
        }
    
    def _update_training_stage(self, epoch: int) -> None:
        if not hasattr(self.model, 'set_training_stage'):
            return

        if self.structure_warmup_epochs > 0 and epoch < self.structure_warmup_epochs:
            stage = 'structure_warmup'
        elif self.full_finetune_epoch is not None and epoch >= self.full_finetune_epoch:
            stage = 'full_finetune'
        else:
            stage = 'joint'

        if stage == self._current_stage:
            return

        stage_changed = False
        if hasattr(self.model, 'set_training_stage'):
            try:
                self.model.set_training_stage(stage)
                self.logger.info(f" Training -> {stage}")
                stage_changed = True
            except Exception as exc:
                self.logger.warning(f" Training {stage} Failed: {exc}")

        self._apply_stage_regularization(stage)
        if not stage_changed:
            self.logger.info(f" -> {stage}")
        self._current_stage = stage

    def _train_epoch(self, epoch):
        """Train one epoch (verbose logging variant)."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_train_preds = []
        all_train_labels = []
        all_train_probs = []
        
        # Track time.
        epoch_start_time = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        observer = getattr(self, 'nan_observer', None)
        observer_enabled = bool(observer and getattr(observer, 'enabled', False))

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # Ensure tensors are on the correct device.
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
                elif isinstance(batch[key], dict):
                    for subkey in batch[key]:
                        if isinstance(batch[key][subkey], torch.Tensor):
                            batch[key][subkey] = batch[key][subkey].to(self.device)

            # Training-time regularization (random masking, mixup, etc.).
            self._apply_batch_regularization(batch)
            
            # Inject fractional epoch index for center-curriculum scheduling.
            batch['center_curriculum_epoch'] = epoch + (batch_idx / max(1, len(self.train_loader)))
            
            if observer_enabled:
                observer.inspect_batch(
                    batch,
                    phase='train',
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=self.global_step,
                )

            context_pushed = False
            if observer_enabled:
                observer.push_context(
                    phase='train',
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=self.global_step,
                )
                context_pushed = True

            # Forward pass.
            try:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(batch)
                    
                    # Extract auxiliary losses (if provided by the model).
                    batch_aux_loss = 0.0
                    aux_loss_dict = {}
                    if isinstance(outputs, dict) and 'auxiliary_losses' in outputs:
                        for k, v in outputs['auxiliary_losses'].items():
                            if isinstance(v, torch.Tensor):
                                batch_aux_loss += v
                                aux_loss_dict[k] = v.item()
                            else:
                                batch_aux_loss += v
                                aux_loss_dict[k] = v

                    labels = batch['label']

                    # Normalize output formats.
                    importance_scores = None
                    is_multitask = False
                    features = None

                    if isinstance(outputs, tuple):
                        # Multi-task output: (class_logits, strength_logits)
                        # Interpretability output: (predictions, importance_scores)
                        if len(outputs) == 2 and isinstance(outputs[0], torch.Tensor) and isinstance(outputs[1], torch.Tensor):
                            # Detect multi-task format.
                            if outputs[0].dim() == 2 and outputs[0].size(-1) == 2 and outputs[1].dim() in [1, 2]:
                                # Multi-task output.
                                is_multitask = True
                                class_logits, strength_logits = outputs
                                logits = class_logits  # Use class logits as the primary output
                            else:
                                # Interpretability output.
                                logits = outputs[0]
                                importance_scores = outputs[1] if len(outputs) > 1 else None
                        elif len(outputs) == 3:
                             # May contain (logits, features, other) or (logits, strength, aux).
                             logits = outputs[0]
                             if isinstance(outputs[1], torch.Tensor) and outputs[1].dim() > 1:
                                 features = outputs[1] # Assume second element is features if it looks like embeddings
                        else:
                            # Other tuple formats.
                            logits = outputs[0]
                            importance_scores = outputs[1] if len(outputs) > 1 else None
                    elif isinstance(outputs, dict):
                        if 'logits' in outputs:
                            logits = outputs['logits']
                        elif 'predicted_outcome' in outputs:
                            logits = outputs['predicted_outcome']
                        else:
                            logits = outputs[list(outputs.keys())[0]]
                        
                        # Try to extract features.
                        if 'features' in outputs:
                            features = outputs['features']
                        elif 'embedding' in outputs:
                            features = outputs['embedding']
                        elif 'projection' in outputs:
                            features = outputs['projection']
                    else:
                        logits = outputs

                    # Ensure tensor dimensions match.
                    if logits.dim() > 1 and logits.size(-1) == 1:
                        logits = logits.squeeze(-1)
                    if labels.dim() > 1 and labels.size(-1) == 1:
                        labels = labels.squeeze(-1)

                    # Multi-task special-case handling.
                    if is_multitask and logits.dim() == 2 and logits.size(-1) == 2:
                        # Multi-task models may require a specialized loss (e.g. MultiTaskLoss) that consumes
                        # all heads. This path falls back to CrossEntropyLoss when the richer loss is unavailable.
                        if isinstance(outputs, tuple) and len(outputs) == 3:
                             # outputs: (logits, strength, aux_outputs)
                             pass
                             
                        # Convert labels to int64.
                        labels_long = labels.long()
                        # loss = F.cross_entropy(logits, labels_long) # OLD logic
                        
                        # NEW LOGIC: Try to use the model's loss if available or construct it
                        if isinstance(outputs, tuple) and len(outputs) == 3:
                            logits, strength_pred, aux_outputs = outputs
                            # Strength targets may be present in the batch.
                            strengths = batch.get('strengths', torch.zeros_like(labels))
                            
                            # Try a richer criterion signature when supported.
                            try:
                                loss_dict = self.criterion(logits, strength_pred, labels, strengths, aux_outputs)
                                loss = loss_dict['loss']
                                batch_aux_loss = loss_dict.get('loss_aux', 0.0)
                            except TypeError:
                                # Fallback
                                loss = F.cross_entropy(logits, labels_long)
                        else:
                             loss = F.cross_entropy(logits, labels_long)

                    else:
                        # Convert probabilities back to logits when inputs appear to be in (0, 1).
                        if torch.all((logits >= 0) & (logits <= 1)) and torch.all(logits != 0) and torch.all(logits != 1):
                            # Use a numerically stable logit transform.
                            eps = 1e-7
                            logits = torch.clamp(logits, eps, 1-eps)
                            logits = torch.log(logits / (1 - logits))

                        # Compute loss.
                        loss = self._compute_loss(logits, labels)
                        
                        # Add auxiliary loss when present.
                        if isinstance(batch_aux_loss, torch.Tensor) or batch_aux_loss > 0:
                            loss = loss + batch_aux_loss
            finally:
                if observer_enabled and context_pushed:
                    observer.pop_context()

            # [Fix] Guard against NaN loss.
            if torch.isnan(loss):
                self.logger.error(f" Error: Loss NaN (Epoch {epoch}, Batch {batch_idx})")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss_value = float(loss.detach().item())
            total_loss += loss_value

            # Backprop with gradient accumulation (GradScaler).
            self.scaler.scale(loss / self.grad_accum_steps).backward()

            if observer_enabled and isinstance(logits, torch.Tensor):
                observer.observe_tensor(
                    'train_logits',
                    logits.detach(),  # Explicit detach
                    phase='train',
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=self.global_step,
                    reason='logits',
                )

            # Multi-task statistics.
            if is_multitask and logits.dim() == 2 and logits.size(-1) == 2:
                if observer_enabled:
                    observer.observe_scalar(
                        'train_loss',
                        loss.detach(),
                        phase='train',
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=self.global_step,
                    )

                # Multi-class statistics.
                probs = torch.softmax(logits.detach(), dim=-1)
                preds = torch.argmax(probs, dim=-1)
                labels_binary = labels.long()
                preds_binary = preds
                correct += (preds_binary == labels_binary).sum().item()
                total += labels.size(0)

            # Collect predictions and labels for detailed metrics.
                all_train_probs.extend(probs[:, 1].cpu().numpy())  # Positive-class probability
                all_train_preds.extend(preds_binary.cpu().numpy())
                all_train_labels.extend(labels_binary.cpu().numpy())
            else:
                if observer_enabled:
                    observer.observe_scalar(
                        'train_loss',
                        loss.detach(),
                        phase='train',
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=self.global_step,
                    )

                # Statistics.
                # [Fix] Handle NaN in logits before sigmoid
                if torch.isnan(logits).any():
                     self.logger.warning("NaN detected in logits during training statistics. Replacing with zeros.")
                     logits_for_stats = torch.nan_to_num(logits.detach(), nan=0.0)
                else:
                     logits_for_stats = logits.detach()

                probs = torch.sigmoid(logits_for_stats)
                
                # [Fix] Handle NaN in probs
                if torch.isnan(probs).any():
                     self.logger.warning("NaN detected in probabilities. Replacing with 0.")
                     probs = torch.nan_to_num(probs, nan=0.0)

                preds = probs > 0.5
                labels_binary = (labels > 0.5).long()
                preds_binary = preds.long()
                correct += (preds_binary == labels_binary).sum().item()
                total += labels.size(0)

                # Collect predictions and labels for detailed metrics.
                all_train_probs.extend(probs.cpu().numpy())
                all_train_preds.extend(preds_binary.cpu().numpy())
                all_train_labels.extend(labels_binary.cpu().numpy())

            if observer_enabled:
                observer.observe_tensor(
                    'train_probs',
                    probs,
                    phase='train',
                    epoch=epoch,
                    batch_idx=batch_idx,
                    global_step=self.global_step,
                    reason='probabilities',
                )

            should_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or ((batch_idx + 1) == len(self.train_loader))
            if should_step:
                # [Fix] Check for NaN/Inf in gradients before update to prevent weight corruption
                # Use scaler to check for invalid gradients if AMP is enabled
                grad_invalid = False
                
                # Unscale gradients before clipping and checking for NaNs
                self.scaler.unscale_(self.optimizer)
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            grad_invalid = True
                            self.logger.warning(f"Invalid gradients (NaN/Inf) detected in {name}. Skipping optimization step to protect model weights.")
                            break
                
                if grad_invalid:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.update() # Still need to update scaler
                    continue

                if self.gradient_clip_val is not None:
                    try:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_val)
                    except RuntimeError as clip_error:
                        self.logger.warning(f" Failed: {clip_error}")
                
                # scaler.step() first unscales gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                self.scaler.step(self.optimizer)
                
                # Updates the scale for next iteration.
                self.scaler.update()
                
                self.global_step += 1
                self._update_weight_decay()
                if self.scheduler_step_on_batch and self.scheduler is not None:
                    self._step_scheduler_batch()
                self.optimizer.zero_grad(set_to_none=True)

                if observer_enabled:
                    observer.update_global_step(self.global_step)
            
            # Batch processing time.
            batch_time = time.time() - batch_start_time
            
            # Log detailed info every 100 batches.
            if batch_idx % 100 == 0:
                current_lr = self._get_current_lr()
                self.logger.info(f'Epoch {epoch+1}, Batch {batch_idx}/{len(self.train_loader)}, '
                               f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}, '
                               f'Batch Time: {batch_time:.3f}s')
            
            # Log GPU memory every 500 batches.
            if batch_idx % 500 == 0 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                self.logger.info(f'GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB')
        
        # Epoch total time.
        epoch_time = time.time() - epoch_start_time
        
        # Detailed training-set statistics.
        all_train_probs = np.array(all_train_probs)
        all_train_labels = np.array(all_train_labels)
        train_binary_preds = (all_train_probs > 0.5).astype(int)
        
        # Training-set metrics.
        train_precision = precision_score(all_train_labels, train_binary_preds, zero_division=0)
        train_recall = recall_score(all_train_labels, train_binary_preds, zero_division=0)
        train_f1 = f1_score(all_train_labels, train_binary_preds, zero_division=0)
        train_mcc = matthews_corrcoef(all_train_labels, train_binary_preds)
        
        # AUC.
        if len(np.unique(all_train_labels)) > 1:
            # [Fix] Check for NaNs in all_train_probs
            if np.isnan(all_train_probs).any():
                self.logger.warning("NaN detected in all_train_probs before AUC calculation. Replacing with 0.")
                all_train_probs = np.nan_to_num(all_train_probs, nan=0.0)
            
            try:
                train_auc = roc_auc_score(all_train_labels, all_train_probs)
            except ValueError as e:
                self.logger.error(f"Error calculating AUC: {e}")
                train_auc = 0.5
        else:
            train_auc = 0.5
        
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        
        if total > 0:
            accuracy = correct / total
        else:
            self.logger.warning(f"Epoch {epoch+1}: No samples processed successfully (total=0). This usually indicates NaN loss in all batches.")
            accuracy = 0.0
        
        # Detailed training summary.
        self.logger.info("="*60)
        self.logger.info(f"Epoch {epoch+1} training completed (time: {epoch_time:.2f}s)")
        self.logger.info("Training:")
        
        # Label/prediction distribution.
        unique_labels, label_counts = np.unique(all_train_labels, return_counts=True)
        unique_preds, pred_counts = np.unique(train_binary_preds, return_counts=True)
        self.logger.info(f"  Label distribution: {dict(zip(unique_labels.astype(int), label_counts))}")
        self.logger.info(f"  Prediction distribution: {dict(zip(unique_preds.astype(int), pred_counts))}")
        
        # Probability statistics.
        self.logger.info(f"  Probability range: [{all_train_probs.min():.4f}, {all_train_probs.max():.4f}]")
        self.logger.info(f"  Probability mean:  {all_train_probs.mean():.4f}")
        self.logger.info(f"  Probability std:   {all_train_probs.std():.4f}")

        # Confidence distribution.
        high_confidence_count = np.sum(all_train_probs >= 0.8)
        medium_confidence_count = np.sum((all_train_probs >= 0.5) & (all_train_probs < 0.8))
        low_confidence_count = np.sum(all_train_probs < 0.5)
        total_samples = len(all_train_probs)

        self.logger.info("Confidence distribution:")
        self.logger.info(f"  High (>=0.8):   {high_confidence_count}/{total_samples} ({high_confidence_count/total_samples*100:.1f}%)")
        self.logger.info(f"  Medium (0.5-0.8): {medium_confidence_count}/{total_samples} ({medium_confidence_count/total_samples*100:.1f}%)")
        self.logger.info(f"  Low (<0.5):     {low_confidence_count}/{total_samples} ({low_confidence_count/total_samples*100:.1f}%)")

        # Detailed metrics.
        self.logger.info("Training metrics:")
        self.logger.info(f"  Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        self.logger.info(f"  F1: {train_f1:.4f}, MCC: {train_mcc:.4f}")
        self.logger.info(f"  AUC: {train_auc:.4f}")
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1,
            'mcc': train_mcc,
            'auc': train_auc
        }
    
    def _apply_batch_regularization(self, batch):
        """Optionally apply extra regularization to a training batch."""
        if self.random_mask_prob > 0:
            self._apply_random_mask(batch)

        if self.use_mixup and self.mixup_prob > 0 and batch.get('sequence_features') is not None:
            if batch['sequence_features'].size(0) > 1:
                trigger = float(torch.rand(1, device=self.device).item())
                if trigger < self.mixup_prob:
                    self._apply_mixup(batch)

    def _cache_dropout_layers(self):
        """Cache standard Dropout layers for stage-wise tuning."""
        try:
            for module in self.model.modules():
                if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                    if module not in self._dropout_layers and hasattr(module, 'p'):
                        self._dropout_layers.append(module)
                        self._dropout_base_rates[module] = float(module.p)
            if self._dropout_layers:
                self.logger.info(f"Detected {len(self._dropout_layers)} Dropout layers for stage-wise tuning.")
        except Exception as exc:
            self.logger.warning(f"Failed to cache Dropout layers: {exc}")

    def _apply_stage_regularization(self, stage: str) -> None:
        """Adjust dropout and weight decay based on the training stage."""
        if not isinstance(stage, str):
            return

        stage_key = stage.lower().strip()
        dropout_multiplier = float(self.config.get(f"{stage_key}_dropout_multiplier", 1.0) or 1.0)
        weight_decay_multiplier = float(self.config.get(f"{stage_key}_weight_decay_multiplier", 1.0) or 1.0)

        self._set_dropout_multiplier(dropout_multiplier)
        self._set_weight_decay_multiplier(weight_decay_multiplier)

        self.logger.info(
            ": stage=%s, dropout_multiplier=%.3f, weight_decay_multiplier=%.3f",
            stage_key,
            dropout_multiplier,
            weight_decay_multiplier,
        )

    def _set_dropout_multiplier(self, multiplier: float) -> None:
        if not self._dropout_layers:
            return

        try:
            multiplier = float(multiplier)
        except (TypeError, ValueError):
            multiplier = 1.0

        multiplier = max(0.0, multiplier)
        for module in self._dropout_layers:
            base_rate = self._dropout_base_rates.get(module, getattr(module, 'p', 0.0))
            new_rate = max(0.0, min(0.95, base_rate * multiplier))
            module.p = new_rate

    def _set_weight_decay_multiplier(self, multiplier: float) -> None:
        if not self.optimizer or not self.optimizer.param_groups:
            return

        try:
            multiplier = float(multiplier)
        except (TypeError, ValueError):
            multiplier = 1.0

        multiplier = max(0.0, multiplier)

        if not hasattr(self, '_base_target_weight_decays'):
            self._base_target_weight_decays = [group.get('weight_decay', 0.0) for group in self.optimizer.param_groups]

        for idx, group in enumerate(self.optimizer.param_groups):
            base_value = self._base_target_weight_decays[idx] if idx < len(self._base_target_weight_decays) else self.config.get('weight_decay', 0.0)
            target = float(base_value) * multiplier
            if idx < len(self._target_weight_decays):
                self._target_weight_decays[idx] = target
            else:
                self._target_weight_decays.append(target)

        if hasattr(self, '_base_weight_decay_final'):
            self.weight_decay_warmup_final = self._base_weight_decay_final * multiplier

        if self.weight_decay_warmup_steps > 0:
            ratio = 0.0
            if self.weight_decay_warmup_steps > 0:
                ratio = min(1.0, max(0.0, self.global_step / float(self.weight_decay_warmup_steps)))
            for idx, group in enumerate(self.optimizer.param_groups):
                target = self._target_weight_decays[idx] if idx < len(self._target_weight_decays) else self.weight_decay_warmup_final
                adjusted = self.weight_decay_warmup_init + (target - self.weight_decay_warmup_init) * ratio
                group['weight_decay'] = adjusted
        else:
            for idx, group in enumerate(self.optimizer.param_groups):
                target = self._target_weight_decays[idx] if idx < len(self._target_weight_decays) else group.get('weight_decay', 0.0)
                group['weight_decay'] = target

    def _apply_random_mask(self, batch):
        """Apply random masking to input features."""
        seq_features = batch.get('sequence_features')
        if not isinstance(seq_features, torch.Tensor):
            return

        if self.random_mask_prob <= 0:
            return

        batch_size, window_len = seq_features.shape[0], seq_features.shape[1]
        mask = torch.rand(batch_size, window_len, device=seq_features.device) < self.random_mask_prob
        if mask.any():
            seq_features = seq_features.masked_fill(mask.unsqueeze(-1), 0.0)
            if self.random_mask_scale > 0:
                noise = torch.randn_like(seq_features) * self.random_mask_scale
                seq_features = seq_features + noise * mask.unsqueeze(-1).float()
            batch['sequence_features'] = seq_features

        if self.random_mask_apply_to_local and isinstance(batch.get('local_features'), torch.Tensor):
            local_features = batch['local_features']
            local_mask = torch.rand(local_features.shape[0], local_features.shape[1], device=local_features.device) < self.random_mask_prob
            if local_mask.any():
                batch['local_features'] = local_features.masked_fill(local_mask.unsqueeze(-1), 0.0)

        if isinstance(batch.get('global_features'), torch.Tensor):
            global_features = batch['global_features']
            global_mask = (torch.rand(global_features.shape[0], 1, device=global_features.device) < self.random_mask_prob).float()
            if global_mask.any():
                batch['global_features'] = global_features * (1 - global_mask)

        if self.random_mask_apply_to_structure and isinstance(batch.get('structure_features'), torch.Tensor):
            structure_features = batch['structure_features']
            if structure_features.shape[:2] == mask.shape:
                structure_features = structure_features.masked_fill(mask.unsqueeze(-1), 0.0)
            else:
                struct_mask = torch.rand(structure_features.shape[0], structure_features.shape[1], device=structure_features.device) < self.random_mask_prob
                structure_features = structure_features.masked_fill(struct_mask.unsqueeze(-1), 0.0)
            batch['structure_features'] = structure_features

    def _apply_mixup(self, batch):
        """Apply Mixup augmentation on a batch."""
        seq_features = batch.get('sequence_features')
        if not isinstance(seq_features, torch.Tensor) or seq_features.size(0) < 2:
            return

        lam = self._sample_mixup_lambda()
        perm = torch.randperm(seq_features.size(0), device=seq_features.device)

        batch['sequence_features'] = lam * seq_features + (1 - lam) * seq_features[perm]

        if isinstance(batch.get('local_features'), torch.Tensor):
            local_features = batch['local_features']
            batch['local_features'] = lam * local_features + (1 - lam) * local_features[perm]

        if isinstance(batch.get('global_features'), torch.Tensor):
            global_features = batch['global_features']
            batch['global_features'] = lam * global_features + (1 - lam) * global_features[perm]

        if isinstance(batch.get('structure_features'), torch.Tensor):
            structure_features = batch['structure_features']
            batch['structure_features'] = lam * structure_features + (1 - lam) * structure_features[perm]

        if isinstance(batch.get('structure_mask'), torch.Tensor):
            structure_mask = batch['structure_mask']
            batch['structure_mask'] = lam * structure_mask + (1 - lam) * structure_mask[perm]

        labels = batch.get('label')
        if isinstance(labels, torch.Tensor):
            batch['label'] = lam * labels + (1 - lam) * labels[perm]

    def _sample_mixup_lambda(self):
        alpha = max(self.mixup_alpha, 1e-3)
        lam = float(np.random.beta(alpha, alpha))
        return max(lam, 1.0 - lam)

    def _update_weight_decay(self):
        """Update weight decay using the warmup schedule."""
        if self.weight_decay_warmup_steps <= 0:
            return

        step = min(self.global_step, self.weight_decay_warmup_steps)
        if step <= 0:
            return

        ratio = step / float(self.weight_decay_warmup_steps)
        for idx, group in enumerate(self.optimizer.param_groups):
            target = self._target_weight_decays[idx] if idx < len(self._target_weight_decays) else self.weight_decay_warmup_final
            group['weight_decay'] = self.weight_decay_warmup_init + (target - self.weight_decay_warmup_init) * ratio

    def _validate_epoch(self, epoch):
        """Validate one epoch (verbose logging variant)."""
        self.model.eval()
        total_loss = 0
        total_loss_with_aux = 0  # Track total loss including auxiliary losses for comparison with training loss
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        all_aux_seq_probs = []
        all_aux_struct_probs = []
        
        val_start_time = time.time()
        
        observer = getattr(self, 'nan_observer', None)
        observer_enabled = bool(observer and getattr(observer, 'enabled', False))

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Ensure tensors are on the correct device.
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                    elif isinstance(batch[key], dict):
                        for subkey in batch[key]:
                            if isinstance(batch[key][subkey], torch.Tensor):
                                batch[key][subkey] = batch[key][subkey].to(self.device)
                
                if observer_enabled:
                    observer.inspect_batch(
                        batch,
                        phase='val',
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=self.global_step,
                    )

                context_pushed = False
                if observer_enabled:
                    observer.push_context(
                        phase='val',
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=self.global_step,
                    )
                    context_pushed = True

                try:
                    outputs = self.model(batch)
                finally:
                    if observer_enabled and context_pushed:
                        observer.pop_context()
                
                # Sum auxiliary losses (for Total Loss reporting).
                batch_aux_loss_sum = 0.0
                if isinstance(outputs, dict) and 'auxiliary_losses' in outputs:
                    for v in outputs['auxiliary_losses'].values():
                        if isinstance(v, torch.Tensor):
                            batch_aux_loss_sum += v.item()
                        else:
                            batch_aux_loss_sum += v

                # Collect auxiliary-branch probabilities.
                if isinstance(outputs, dict):
                    if 'aux_seq_logits' in outputs and outputs['aux_seq_logits'] is not None:
                        aux_seq = outputs['aux_seq_logits']
                        if aux_seq.dim() > 1 and aux_seq.size(-1) == 1:
                            aux_seq = aux_seq.squeeze(-1)
                        # Check whether a probability-to-logit conversion is needed.
                        if torch.all((aux_seq >= 0) & (aux_seq <= 1)) and torch.all(aux_seq != 0) and torch.all(aux_seq != 1):
                             # Already probabilities? Unlikely; usually these are logits.
                             pass
                        aux_seq_probs = torch.sigmoid(aux_seq).cpu().numpy()
                        all_aux_seq_probs.extend(aux_seq_probs)
                    
                    if 'aux_struct_logits' in outputs and outputs['aux_struct_logits'] is not None:
                        aux_struct = outputs['aux_struct_logits']
                        if aux_struct.dim() > 1 and aux_struct.size(-1) == 1:
                            aux_struct = aux_struct.squeeze(-1)
                        aux_struct_probs = torch.sigmoid(aux_struct).cpu().numpy()
                        all_aux_struct_probs.extend(aux_struct_probs)

                labels = batch['label']

                # Normalize output formats.
                importance_scores = None
                is_multitask = False

                if isinstance(outputs, tuple):
                    # Multi-task output detection (acetylation_transformer).
                    if len(outputs) == 2 and isinstance(outputs[0], torch.Tensor) and isinstance(outputs[1], torch.Tensor):
                        # Detect multi-task output.
                        if outputs[0].dim() == 2 and outputs[0].size(-1) == 2 and outputs[1].dim() in [1, 2]:
                            # Multi-task output.
                            is_multitask = True
                            class_logits, strength_logits = outputs
                            logits = class_logits  # Use class logits as the primary output
                        else:
                            # Interpretability output.
                            logits = outputs[0]
                            importance_scores = outputs[1] if len(outputs) > 1 else None
                    else:
                        # Other tuple formats.
                        logits = outputs[0]
                        importance_scores = outputs[1] if len(outputs) > 1 else None
                elif isinstance(outputs, dict):
                    if 'logits' in outputs:
                        logits = outputs['logits']
                    elif 'predicted_outcome' in outputs:
                        logits = outputs['predicted_outcome']
                    else:
                        logits = outputs[list(outputs.keys())[0]]
                else:
                    logits = outputs

                # Ensure tensor dimensions match.
                if logits.dim() > 1 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                if labels.dim() > 1 and labels.size(-1) == 1:
                    labels = labels.squeeze(-1)

                if observer_enabled and isinstance(logits, torch.Tensor):
                    observer.observe_tensor(
                        'val_logits',
                        logits,
                        phase='val',
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=self.global_step,
                        reason='logits',
                    )

                # Multi-task special-case handling.
                if is_multitask and logits.dim() == 2 and logits.size(-1) == 2:
                    # For multi-task outputs, use CrossEntropyLoss.
                    labels_long = labels.long()
                    loss = F.cross_entropy(logits, labels_long)
                    total_loss += loss.item()
                    total_loss_with_aux += (loss.item() + batch_aux_loss_sum)

                    if observer_enabled:
                        observer.observe_scalar(
                            'val_loss',
                            loss,
                            phase='val',
                            epoch=epoch,
                            batch_idx=batch_idx,
                            global_step=self.global_step,
                        )

                    # Prediction probabilities.
                    probs = torch.softmax(logits, dim=-1)
                    labels_binary = labels_long
                else:
                    # Convert probabilities to logits when needed (to match training behavior).
                    if torch.all((logits >= 0) & (logits <= 1)) and torch.all(logits != 0) and torch.all(logits != 1):
                        eps = 1e-7
                        logits = torch.clamp(logits, eps, 1 - eps)
                        logits = torch.log(logits / (1 - logits))

                    # Compute loss.
                    loss = self._compute_loss(logits, labels)
                    total_loss += loss.item()
                    total_loss_with_aux += (loss.item() + batch_aux_loss_sum)

                    if observer_enabled:
                        observer.observe_scalar(
                            'val_loss',
                            loss,
                            phase='val',
                            epoch=epoch,
                            batch_idx=batch_idx,
                            global_step=self.global_step,
                        )

                    # Prediction probabilities.
                    # [Fix] Handle NaNs in validation logits
                    if torch.isnan(logits).any():
                        logits = torch.nan_to_num(logits, nan=0.0)

                    probs = torch.sigmoid(logits)
                    
                    # [Fix] Handle NaNs in validation probs
                    if torch.isnan(probs).any():
                        probs = torch.nan_to_num(probs, nan=0.0)

                    labels_binary = (labels > 0.5).long()
                
                if observer_enabled:
                    observer.observe_tensor(
                        'val_probs',
                        probs,
                        phase='val',
                        epoch=epoch,
                        batch_idx=batch_idx,
                        global_step=self.global_step,
                        reason='probabilities',
                    )

                if is_multitask:
                    preds_binary = torch.argmax(probs, dim=-1)
                else:
                    preds_binary = (probs > 0.5).long()

                correct += (preds_binary == labels_binary).sum().item()
                total += labels.size(0)

                all_preds.extend(preds_binary.cpu().numpy())
                all_labels.extend(labels_binary.cpu().numpy())
                if is_multitask:
                    all_probs.extend(probs[:, 1].cpu().numpy())  # Positive-class probability
                else:
                    all_probs.extend(probs.cpu().numpy())
        
        val_time = time.time() - val_start_time
        
        # Compute extra metrics.
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Recompute the optimal threshold for final reporting.
        best_threshold = 0.5
        best_score = -np.inf
        metric_name = str(self.config.get('threshold_metric', 'f1')).lower()

        for threshold in np.arange(0.1, 0.9, 0.05):
            temp_binary_preds = (all_probs > threshold).astype(int)
            if len(np.unique(temp_binary_preds)) < 2:
                continue
            tn, fp, fn, tp = confusion_matrix(all_labels, temp_binary_preds, labels=[0, 1]).ravel()
            temp_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            temp_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            temp_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            if temp_precision < self.threshold_min_precision:
                continue
            if temp_recall < self.threshold_min_recall:
                continue
            if temp_specificity < self.threshold_min_specificity:
                continue
            if metric_name == 'mcc':
                temp_score = matthews_corrcoef(all_labels, temp_binary_preds)
            elif metric_name in {'balanced_accuracy', 'balanced_acc', 'balanced'}:
                temp_score = balanced_accuracy_score(all_labels, temp_binary_preds)
            elif metric_name in {'recall', 'sensitivity', 'tpr'}:
                temp_score = recall_score(all_labels, temp_binary_preds, zero_division=0)
            elif metric_name in {'precision', 'ppv'}:
                temp_score = precision_score(all_labels, temp_binary_preds, zero_division=0)
            elif metric_name in {'specificity', 'tnr'}:
                tn, fp, fn, tp = confusion_matrix(all_labels, temp_binary_preds, labels=[0, 1]).ravel()
                temp_score = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            elif metric_name.startswith('f') and metric_name[1:].replace('.', '', 1).isdigit():
                try:
                    beta = float(metric_name[1:])
                    if beta <= 0:
                        raise ValueError
                except ValueError:
                    beta = 1.0
                temp_score = fbeta_score(all_labels, temp_binary_preds, beta=beta, zero_division=0)
            else:
                temp_score = f1_score(all_labels, temp_binary_preds, zero_division=0)

            if temp_score > best_score:
                best_score = temp_score
                best_threshold = threshold

        if best_score == -np.inf:
            self.logger.warning(
                "Threshold tuning: no threshold satisfies precision/recall/specificity constraints; falling back to 0.5"
            )
            best_threshold = 0.5
        
        # Recompute predictions using the optimal threshold.
        binary_preds = (all_probs > best_threshold).astype(int)
        
        precision = precision_score(all_labels, binary_preds, zero_division=0)
        recall = recall_score(all_labels, binary_preds, zero_division=0)
        f1 = f1_score(all_labels, binary_preds, zero_division=0)
        mcc = matthews_corrcoef(all_labels, binary_preds)
        
        if len(np.unique(all_labels)) > 1:
            # [Fix] Handle NaNs in validation all_probs
            if np.isnan(all_probs).any():
                self.logger.warning("NaN detected in validation all_probs. Replacing with 0.")
                all_probs = np.nan_to_num(all_probs, nan=0.0)

            try:
                auc_score = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auc_score = 0.5
            
            try:
                precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
                auprc = auc(recall_curve, precision_curve)
            except ValueError:
                auprc = 0.0
        else:
            auc_score = 0.5
            auprc = 0.0
        
        # Auxiliary branch metrics.
        val_seq_mcc = 0.0
        val_struct_mcc = 0.0
        
        self.logger.info("--- Validation ---")
        
        if all_aux_seq_probs:
            all_aux_seq_probs = np.array(all_aux_seq_probs)
            
            # [Fix] Handle NaNs in auxiliary sequence probs
            if np.isnan(all_aux_seq_probs).any():
                self.logger.warning("NaN detected in validation all_aux_seq_probs. Replacing with 0.")
                all_aux_seq_probs = np.nan_to_num(all_aux_seq_probs, nan=0.0)

            seq_preds = (all_aux_seq_probs > 0.5).astype(int)
            
            val_seq_acc = accuracy_score(all_labels, seq_preds) if 'accuracy_score' in globals() else (seq_preds == all_labels).mean()
            val_seq_mcc = matthews_corrcoef(all_labels, seq_preds)
            val_seq_f1 = f1_score(all_labels, seq_preds, zero_division=0)
            
            if len(np.unique(all_labels)) > 1:
                val_seq_auc = roc_auc_score(all_labels, all_aux_seq_probs)
            else:
                val_seq_auc = 0.5
                
            self.logger.info(f" (Sequence Branch):")
            self.logger.info(f"    MCC: {val_seq_mcc:.4f}")
            self.logger.info(f"    ACC: {val_seq_acc:.4f}")
            self.logger.info(f"    F1 : {val_seq_f1:.4f}")
            self.logger.info(f"    AUC: {val_seq_auc:.4f}")
            
        if all_aux_struct_probs:
            all_aux_struct_probs = np.array(all_aux_struct_probs)
            
            # [Fix] Handle NaNs in auxiliary structure probs
            if np.isnan(all_aux_struct_probs).any():
                self.logger.warning("NaN detected in validation all_aux_struct_probs. Replacing with 0.")
                all_aux_struct_probs = np.nan_to_num(all_aux_struct_probs, nan=0.0)
                
            struct_preds = (all_aux_struct_probs > 0.5).astype(int)
            
            val_struct_acc = accuracy_score(all_labels, struct_preds) if 'accuracy_score' in globals() else (struct_preds == all_labels).mean()
            val_struct_mcc = matthews_corrcoef(all_labels, struct_preds)
            val_struct_f1 = f1_score(all_labels, struct_preds, zero_division=0)
            
            if len(np.unique(all_labels)) > 1:
                val_struct_auc = roc_auc_score(all_labels, all_aux_struct_probs)
            else:
                val_struct_auc = 0.5

            self.logger.info(f" (Structure Branch):")
            self.logger.info(f"    MCC: {val_struct_mcc:.4f}")
            self.logger.info(f"    ACC: {val_struct_acc:.4f}")
            self.logger.info(f"    F1 : {val_struct_f1:.4f}")
            self.logger.info(f"    AUC: {val_struct_auc:.4f}")

        avg_loss = total_loss / len(self.val_loader)
        avg_total_loss = total_loss_with_aux / len(self.val_loader)
        # Compute accuracy using predictions obtained at the globally tuned threshold (consistent with confusion matrix).
        accuracy = (binary_preds == all_labels).mean()
        
        # Detailed validation summary.
        self.logger.info(f"Validation completed (time: {val_time:.2f}s)")
        self.logger.info(f"Using threshold: {best_threshold:.3f}")
        self.logger.info("Validation:")
        
        # Label/prediction distribution.
        unique_labels, label_counts = np.unique(all_labels, return_counts=True)
        unique_preds, pred_counts = np.unique(binary_preds, return_counts=True)
        self.logger.info(f"  Label distribution: {dict(zip(unique_labels.astype(int), label_counts))}")
        self.logger.info(f"  Prediction distribution: {dict(zip(unique_preds.astype(int), pred_counts))}")
        
        # Probability statistics.
        self.logger.info(f"  Probability range: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
        self.logger.info(f"  Probability mean:  {all_probs.mean():.4f}")
        self.logger.info(f"  Probability std:   {all_probs.std():.4f}")

        # Confidence distribution.
        high_confidence_count = np.sum(all_probs >= 0.8)
        medium_confidence_count = np.sum((all_probs >= 0.5) & (all_probs < 0.8))
        low_confidence_count = np.sum(all_probs < 0.5)
        total_samples = len(all_probs)

        self.logger.info("Confidence distribution:")
        self.logger.info(f"  High (>=0.8):   {high_confidence_count}/{total_samples} ({high_confidence_count/total_samples*100:.1f}%)")
        self.logger.info(f"  Medium (0.5-0.8): {medium_confidence_count}/{total_samples} ({medium_confidence_count/total_samples*100:.1f}%)")
        self.logger.info(f"  Low (<0.5):     {low_confidence_count}/{total_samples} ({low_confidence_count/total_samples*100:.1f}%)")

        # Detailed metrics.
        self.logger.info("Validation metrics:")
        self.logger.info(f"  Main Loss: {avg_loss:.4f}")
        self.logger.info(f"  Total Loss (w/ Aux): {avg_total_loss:.4f} (Comparable to Train Loss)")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
        self.logger.info(f"  F1: {f1:.4f}, MCC: {mcc:.4f}")
        self.logger.info(f"  AUC: {auc_score:.4f}, AUPRC: {auprc:.4f}")
        
        # Class-balance analysis.
        val_specificity = 0.0
        if len(unique_labels) == 2:
            positive_ratio = label_counts[1] / sum(label_counts) if len(label_counts) > 1 else 0
            pred_positive_ratio = pred_counts[1] / sum(pred_counts) if len(pred_counts) > 1 and len(pred_counts) == 2 else 0
            self.logger.info(f"  Positive label ratio: {positive_ratio:.3f}")
            self.logger.info(f"  Predicted positive ratio: {pred_positive_ratio:.3f}")
            
            # Confusion matrix components.
            cm = confusion_matrix(all_labels, binary_preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                val_specificity = specificity
                self.logger.info(f"  Specificity: {specificity:.4f}")
                self.logger.info(f"  Sensitivity: {sensitivity:.4f}")
                self.logger.info(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            else:
                self.logger.info(f" Confusion Matrix: {cm.shape}")
                self.logger.info(f"  Confusion Matrix: {cm}")
        
        self.logger.info("="*60)
        
        return {
            'val_loss': avg_loss,
            'val_acc': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_specificity': val_specificity,
            'val_f1': f1,
            'val_auc': auc_score,
            'val_mcc': mcc,
            'val_auprc': auprc,
            'val_seq_mcc': val_seq_mcc,
            'val_struct_mcc': val_struct_mcc,
            'best_threshold': best_threshold
        }
    
    def _compute_loss(self, logits, labels, features=None):
        """Compute loss."""
        if hasattr(self.model, 'compute_enhanced_loss'):
            # TODO: Update model interface to accept features if needed
            return self.model.compute_enhanced_loss(
                logits, labels, 
                use_focal=self.loss_config['use_focal_loss'],
                use_label_smooth=self.loss_config['use_label_smoothing']
            )
        else:
            return self._compute_basic_loss(logits, labels, features)
    
    def _compute_basic_loss(self, logits, labels, features=None):
        """Compute the base loss, with optional OHEM (hard negative mining)."""
        loss = None
        
        # Prepare reduction mode.
        # When hard negative mining is enabled, per-sample loss is required.
        use_ohem = self.loss_config.get('use_hard_negative_mining', False)
        reduction = 'none' if use_ohem else 'mean'
        
        # Focal loss.
        if self.loss_config.get('use_focal_loss', False):
            # Adaptive weighting path.
            if self.loss_config.get('use_adaptive_weighting', False) and self.adaptive_weight_scheduler:
                weights = self.adaptive_weight_scheduler.weights
                sample_weights = torch.where(labels == 1, weights[1], weights[0])
                
                alpha = self.loss_config.get('focal_alpha', 0.25)
                gamma = self.loss_config.get('focal_gamma', 2.0)
                bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
                pt = torch.exp(-bce_loss)
                alpha_t = torch.where(labels == 1, alpha, 1 - alpha)
                focal_loss_per_sample = alpha_t * (1 - pt) ** gamma * bce_loss
                
                # Apply adaptive weights.
                per_sample_loss = focal_loss_per_sample * sample_weights
                
                if use_ohem:
                    loss = self._apply_ohem(per_sample_loss, labels)
                else:
                    loss = per_sample_loss.mean()
            else:
                # Standard focal loss.
                if use_ohem:
                    # EnhancedLoss.focal_loss returns a mean-reduced loss. For OHEM we need per-sample losses,
                    # so re-implement the focal loss here with reduction='none'.
                    alpha = self.loss_config.get('focal_alpha', 0.25)
                    gamma = self.loss_config.get('focal_gamma', 2.0)
                    bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
                    pt = torch.exp(-bce_loss)
                    alpha_t = torch.where(labels == 1, alpha, 1 - alpha)
                    per_sample_loss = alpha_t * (1 - pt) ** gamma * bce_loss
                    loss = self._apply_ohem(per_sample_loss, labels)
                else:
                    loss = self.enhanced_loss.focal_loss(logits, labels)
                
            if self.loss_config.get('use_label_smoothing', False):
                smooth_loss = self._label_smooth_loss(logits, labels)
                weight = self.loss_config.get('loss_combination_weight', 0.5)
                loss = (weight * loss + (1 - weight) * smooth_loss)
        elif self.loss_config.get('use_label_smoothing', False):
            loss = self._label_smooth_loss(logits, labels)
        else:
            # Standard BCE.
            if self.pos_weight is not None:
                # ... (Adaptive weighting check)
                 if self.loss_config.get('use_adaptive_weighting', False) and self.adaptive_weight_scheduler:
                     weights = self.adaptive_weight_scheduler.weights
                     dynamic_pos_weight = weights[1] / (weights[0] + 1e-8)
                     per_sample_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=dynamic_pos_weight, reduction='none')
                 else:
                    per_sample_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=self.pos_weight, reduction='none')
            else:
                per_sample_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
            
            if use_ohem:
                loss = self._apply_ohem(per_sample_loss, labels)
            else:
                loss = per_sample_loss.mean()
        
        # MCC loss (auxiliary; typically does not use OHEM).
        if self.loss_config.get('use_mcc_loss', False):
            mcc_loss = self._soft_mcc_loss(logits, labels)
            loss = loss + self.loss_config.get('mcc_loss_weight', 0.5) * mcc_loss
            
        # Contrastive loss.
        if self.loss_config.get('use_contrastive_loss', False) and features is not None:
            contrastive_loss = self.enhanced_loss.contrastive_loss(features, labels)
            loss = loss + self.loss_config.get('contrastive_loss_weight', 0.1) * contrastive_loss
            
        return loss

    def _apply_ohem(self, per_sample_loss, labels):
        """Apply online hard example mining (OHEM)."""
        # Keep the top ratio of samples by loss.
        ratio = self.loss_config.get('hard_negative_mining_ratio', 0.5)  # Default: keep 50%
        num_samples = per_sample_loss.size(0)
        num_keep = int(num_samples * ratio)
        
        if num_keep < 1:
            return per_sample_loss.mean()
            
        # Keep samples with the highest loss.
        top_k_loss, _ = torch.topk(per_sample_loss, num_keep)
        return top_k_loss.mean()
    
    def _focal_loss(self, inputs, targets):
        """
        Focal Loss implementation.
        Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        """
        alpha = self.loss_config.get('focal_alpha', 0.25)
        gamma = self.loss_config.get('focal_gamma', 2.0)
        
        # Calculate BCE with logits, no reduction to keep shape [batch_size, *]
        # We do NOT use pos_weight here because alpha handles class balancing in Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate pt
        # pt = exp(-bce_loss) is correct because bce_loss = -log(pt)
        pt = torch.exp(-bce_loss)
        
        # Calculate alpha_t
        # alpha for class 1, 1-alpha for class 0
        alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
        
        # Calculate Focal Loss
        focal_loss = alpha_t * (1 - pt) ** gamma * bce_loss
        
        return focal_loss.mean()

    def _soft_mcc_loss(self, inputs, targets):
        """
        Soft Matthews Correlation Coefficient Loss
        Approximates MCC as a differentiable loss function.
        Loss = 1 - MCC
        """
        probs = torch.sigmoid(inputs)
        targets = targets.float()
        
        tp = torch.sum(probs * targets)
        tn = torch.sum((1 - probs) * (1 - targets))
        fp = torch.sum(probs * (1 - targets))
        fn = torch.sum((1 - probs) * targets)
        
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        # Add epsilon to denominator to avoid division by zero
        mcc = numerator / (denominator + 1e-7)
        return 1 - mcc
    
    def _label_smooth_loss(self, inputs, targets):
        """Label smoothing loss."""
        smoothing = self.loss_config['label_smoothing']
        targets_smooth = targets * (1 - smoothing) + 0.5 * smoothing
        if self.pos_weight is not None:
            return F.binary_cross_entropy_with_logits(inputs, targets_smooth, pos_weight=self.pos_weight)
        return F.binary_cross_entropy_with_logits(inputs, targets_smooth)
    
    def _log_to_tensorboard(self, epoch, train_metrics, val_metrics):
        """Log metrics to TensorBoard."""
        if self.tb_writer is not None:
            # Training metrics.
            for key, value in train_metrics.items():
                self.tb_writer.add_scalar(f'Train/{key}', value, epoch)

            # Validation metrics.
            for key, value in val_metrics.items():
                if key != 'best_threshold':
                    self.tb_writer.add_scalar(f'Validation/{key}', value, epoch)

            # Learning rate.
            self.tb_writer.add_scalar('Learning_Rate', self._get_current_lr(), epoch)

    def _get_current_lr(self):
        """Get current learning rate."""
        if self.scheduler is not None and hasattr(self.scheduler, 'get_last_lr'):
            last_lr = self.scheduler.get_last_lr()
            if isinstance(last_lr, (list, tuple)):
                return float(last_lr[0])
            return float(last_lr)
        return float(self.optimizer.param_groups[0]['lr'])

    def _step_scheduler_batch(self):
        """Step scheduler on each batch when configured."""
        if not self.scheduler or not self.scheduler_step_on_batch:
            return
        try:
            self.scheduler.step()
        except TypeError:
            self.scheduler.step(self.global_step)

    def _step_scheduler_epoch(self, epoch, val_metrics):
        """Step scheduler at epoch boundaries."""
        if not self.scheduler or self.scheduler_step_on_batch:
            return

        if self.scheduler_type == 'plateau':
            metric_value = val_metrics.get(self.monitor_metric)
            if metric_value is None:
                metric_value = val_metrics.get('val_loss', 0.0)
            self.scheduler.step(metric_value)
        elif self.scheduler_type == 'cosine_warm_restarts':
            try:
                self.scheduler.step(epoch + 1)
            except TypeError:
                self.scheduler.step()
        else:
            try:
                self.scheduler.step()
            except TypeError:
                self.scheduler.step(epoch + 1)
    
    def _update_history(self, train_metrics, val_metrics):
        """Update training history."""
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['train_acc'].append(train_metrics['accuracy'])
        self.training_history['train_f1'].append(train_metrics['f1'])
        
        self.training_history['val_loss'].append(val_metrics['val_loss'])
        self.training_history['val_acc'].append(val_metrics['val_acc'])
        self.training_history['val_f1'].append(val_metrics['val_f1'])
        self.training_history['val_auc'].append(val_metrics['val_auc'])
        self.training_history['val_precision'].append(val_metrics['val_precision'])
        self.training_history['val_recall'].append(val_metrics['val_recall'])
        self.training_history['val_mcc'].append(val_metrics['val_mcc'])
        self.training_history['val_auprc'].append(val_metrics['val_auprc'])
    
    def evaluate_test_set(self, test_loader):
        """Evaluate on a test set."""
        if test_loader is None:
            return None
        
        self.logger.info("=== Testing ===")
        
        # Load the best checkpoint when available.
        best_model_path = os.path.join(self.output_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Ensure tensors are on the correct device.
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                    elif isinstance(batch[key], dict):
                        for subkey in batch[key]:
                            if isinstance(batch[key][subkey], torch.Tensor):
                                batch[key][subkey] = batch[key][subkey].to(self.device)
                
                outputs = self.model(batch)
                labels = batch['label']

                # Normalize output formats.
                importance_scores = None
                is_multitask = False

                if isinstance(outputs, tuple):
                    # Multi-task output detection (acetylation_transformer).
                    if len(outputs) == 2 and isinstance(outputs[0], torch.Tensor) and isinstance(outputs[1], torch.Tensor):
                        # Detect multi-task output.
                        if outputs[0].dim() == 2 and outputs[0].size(-1) == 2 and outputs[1].dim() in [1, 2]:
                            # Multi-task output.
                            is_multitask = True
                            class_logits, strength_logits = outputs
                            logits = class_logits  # Use class logits as the primary output
                        else:
                            # Interpretability output.
                            logits = outputs[0]
                            importance_scores = outputs[1] if len(outputs) > 1 else None
                    else:
                        # Other tuple formats.
                        logits = outputs[0]
                        importance_scores = outputs[1] if len(outputs) > 1 else None
                elif isinstance(outputs, dict):
                    if 'logits' in outputs:
                        logits = outputs['logits']
                    elif 'predicted_outcome' in outputs:
                        logits = outputs['predicted_outcome']
                    else:
                        logits = outputs[list(outputs.keys())[0]]
                else:
                    logits = outputs

                # Ensure tensor dimensions match.
                if logits.dim() > 1 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                if labels.dim() > 1 and labels.size(-1) == 1:
                    labels = labels.squeeze(-1)

                # Multi-task special-case handling.
                if is_multitask and logits.dim() == 2 and logits.size(-1) == 2:
                    # For multi-task outputs, use softmax probabilities.
                    probs = torch.softmax(logits, dim=-1)
                    # Positive-class probability.
                    all_probs.extend(probs[:, 1].cpu().numpy())
                    all_labels.extend(labels.long().cpu().numpy())
                else:
                    # Probabilities.
                    probs = torch.sigmoid(logits)
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        # Test metrics.
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Find the best threshold.
        best_threshold = 0.5
        best_f1 = 0
        for threshold in np.arange(0.1, 0.9, 0.05):
            temp_preds = (all_probs > threshold).astype(int)
            if len(np.unique(temp_preds)) > 1:
                temp_f1 = f1_score(all_labels, temp_preds, zero_division=0)
                if temp_f1 > best_f1:
                    best_f1 = temp_f1
                    best_threshold = threshold
        
        binary_preds = (all_probs > best_threshold).astype(int)
        
        test_metrics = {
            'test_accuracy': (binary_preds == all_labels).mean(),
            'test_precision': precision_score(all_labels, binary_preds, zero_division=0),
            'test_recall': recall_score(all_labels, binary_preds, zero_division=0),
            'test_f1': f1_score(all_labels, binary_preds, zero_division=0),
            'test_mcc': matthews_corrcoef(all_labels, binary_preds),
            'test_threshold': best_threshold
        }
        
        if len(np.unique(all_labels)) > 1:
            test_metrics['test_auc'] = roc_auc_score(all_labels, all_probs)
            precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
            test_metrics['test_auprc'] = auc(recall_curve, precision_curve)
        
        self.logger.info("Test Results:")
        for key, value in test_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
            
        # Specificity and sensitivity.
        tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        self.logger.info(f"  test_specificity: {specificity:.4f}")
        self.logger.info(f"  test_sensitivity: {sensitivity:.4f}")
        self.logger.info(f"  test_confusion_matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        test_metrics['test_specificity'] = specificity
        test_metrics['test_sensitivity'] = sensitivity
        
        return test_metrics

    def _check_early_stopping(self, val_metrics, epoch, best_val_loss, best_val_acc,
                             best_val_f1, best_val_mcc, best_val_auprc, best_epoch):
        """
        Check early stopping criteria.

        Args:
            val_metrics: Validation metric dict.
            epoch: Current epoch.
            best_val_*: Current best metrics.
            best_epoch: Best epoch so far.

        Returns:
            tuple: (should_stop, improvement_flag)
        """
        if not self.early_stopping['use_early_stopping']:
            # Even without early stopping, save a best checkpoint when improved.
            if val_metrics['val_loss'] < best_val_loss:
                best_model_path = os.path.join(self.output_dir, 'best_model.pt')
                torch.save(self.model.state_dict(), best_model_path)
                self.logger.info(f"Saved best model to: {best_model_path}")
                return False, " *** New best model! ***"
            return False, ""

        monitor_metric = self.early_stopping['monitor_metric']
        mode = self.early_stopping['mode']
        min_delta = self.early_stopping['min_delta']
        patience = self.early_stopping['patience']

        # Current monitored metric value.
        current_score = val_metrics.get(monitor_metric, val_metrics['val_loss'])

        # Determine whether there is an improvement.
        improved = False
        if mode == 'min':
            improved = current_score < (self.early_stopping['best_score'] - min_delta)
        else:  # mode == 'max'
            improved = current_score > (self.early_stopping['best_score'] + min_delta)

        improvement_flag = ""

        if improved:
            # Improvement: update best score and reset counter.
            self.early_stopping['best_score'] = current_score
            self.early_stopping['counter'] = 0
            improvement_flag = " *** New best model! ***"

            # Save best model.
            best_model_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)

            if self.early_stopping['verbose']:
                self.logger.info(f"Saved best model to: {best_model_path}")
                self.logger.info(f"Monitored metric {monitor_metric} improved: {current_score:.6f}")
        else:
            # No improvement: increment counter.
            self.early_stopping['counter'] += 1
            improvement_flag = f" (patience {self.early_stopping['counter']}/{patience})"

            if self.early_stopping['verbose'] and self.early_stopping['counter'] > 0:
                self.logger.info(
                    f"Monitored metric {monitor_metric} did not improve: {current_score:.6f} "
                    f"(best: {self.early_stopping['best_score']:.6f})"
                )

        # Check whether training should stop.
        should_stop = self.early_stopping['counter'] >= patience

        if should_stop and self.early_stopping['restore_best_weights']:
            # Restore best weights.
            best_model_path = os.path.join(self.output_dir, 'best_model.pt')
            if os.path.exists(best_model_path):
                self.model.load_state_dict(torch.load(best_model_path))
                if self.early_stopping['verbose']:
                    self.logger.info("Restored best model weights")

        return should_stop, improvement_flag
