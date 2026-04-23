
import os
import sys
import argparse
import json
import math
import copy
import logging
from typing import Any, Dict, List, Optional, Set
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Limit thread counts to reduce contention during multi-process data loading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Import required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from tqdm import tqdm

# Optional import: plotting dependencies
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
else:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('ggplot')
        except Exception:
            pass

# Import project modules
from fluxsite.utils.config_manager import ConfigManager
from fluxsite.utils.training_utils_enhanced import EnhancedTrainingManager
from fluxsite.utils.rl_training_manager import RLEnhancedTrainingManager
from fluxsite.utils.optimizer_utils import create_optimizer, create_scheduler
from fluxsite.utils.kfold_validator import KFoldValidator

# Import visualization manager only when needed to avoid dependency issues
try:
    from fluxsite.utils.visualization_utils_enhanced import EnhancedVisualizationManager
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    EnhancedVisualizationManager = None
from fluxsite.utils.metrics import calculate_metrics, calculate_optimal_threshold
from fluxsite.utils.common_utils import custom_collate_fn, set_seed
from fluxsite.data.unified_data_processor import UnifiedDataProcessor, UnifiedPTMDataset
from fluxsite.utils.data_loading_optimizer import (
    DataLoadingOptimizer,
    LazyProteinFeatureStore,
    OnDemandPTMDataset,
    normalize_uniprot_id,
    StratifiedBatchSampler,
)
from fluxsite.utils.ptm_residue_info import resolve_ptm_residue_info
from fluxsite.models.acetylation_predictor import AcetylationPredictor, DualBranchFusionPredictor
from fluxsite.models.cnn_model import CNNDualStreamPredictor
from fluxsite.utils.feature_normalization import FeatureNormalizer
from fluxsite.utils.feature_reduction import FeatureReducer


def _resolve_branch_hidden_dim(config: dict) -> int:
    branch_dim = config.get('branch_hidden_dim')
    if branch_dim is None:
        branch_dim = config.get('fusion_hidden_dim')
    if branch_dim is None:
        branch_dim = 256
    return max(1, int(branch_dim))


def _positive_int(config: dict, key: str, default: int) -> int:
    value = config.get(key, default)
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, value_int)


def _collect_divisors(value: int) -> list:
    divisors = set()
    upper = int(math.sqrt(value)) + 1
    for candidate in range(1, upper):
        if value % candidate == 0:
            divisors.add(candidate)
            divisors.add(value // candidate)
    return sorted(divisors)


def adjust_cross_attention_hyperparams(config: dict) -> None:
    """Adjust cross-attention head count and contrastive projection dim for stability."""
    branch_dim = _resolve_branch_hidden_dim(config)
    config['branch_hidden_dim'] = branch_dim

    min_heads = _positive_int(config, 'cross_attention_min_heads', 4)
    max_heads = max(min_heads, _positive_int(config, 'cross_attention_max_heads', min_heads))

    min_per_head = _positive_int(config, 'cross_attention_min_per_head_dim', 32)
    max_per_head = max(min_per_head, _positive_int(config, 'cross_attention_max_per_head_dim', 96))
    target_per_head = int(config.get('cross_attention_target_per_head_dim', 64))
    target_per_head = max(min_per_head, min(max_per_head, target_per_head))

    divisors = _collect_divisors(branch_dim)
    head_candidates = [
        (heads, branch_dim // heads)
        for heads in divisors
        if min_heads <= heads <= max_heads and branch_dim % heads == 0
    ]

    preferred = None
    bounded_candidates = [candidate for candidate in head_candidates if min_per_head <= candidate[1] <= max_per_head]
    if bounded_candidates:
        preferred = min(bounded_candidates, key=lambda pair: abs(pair[1] - target_per_head))
    elif head_candidates:
        preferred = min(head_candidates, key=lambda pair: abs(pair[1] - target_per_head))
    else:
        fallback_heads = min(max_heads, divisors[-1]) if divisors else min_heads
        if fallback_heads < min_heads:
            fallback_heads = min_heads
        while branch_dim % fallback_heads != 0 and fallback_heads > 1:
            fallback_heads -= 1
        if branch_dim % fallback_heads != 0:
            fallback_heads = 1
        preferred = (fallback_heads, branch_dim // max(1, fallback_heads))

    heads, per_head_dim = preferred
    if heads < min_heads and branch_dim % min_heads == 0:
        heads = min_heads
        per_head_dim = branch_dim // heads
    config['cross_attention_heads'] = int(heads)
    config['cross_attention_per_head_dim'] = int(per_head_dim)

    ratio = float(config.get('contrastive_projection_ratio', 1.0))
    min_proj = max(per_head_dim, int(config.get('contrastive_projection_min', per_head_dim)))
    max_proj = int(config.get('contrastive_projection_max', max(branch_dim, min_proj)))
    round_to = _positive_int(config, 'contrastive_projection_round_to', 32)

    proposed = int(branch_dim * ratio)
    if proposed <= 0:
        proposed = branch_dim
    proposed = max(min_proj, proposed)
    proposed = min(max_proj, proposed)

    if round_to > 1:
        proposed = int(math.ceil(proposed / round_to) * round_to)
    proposed = min(max_proj, max(min_proj, proposed))

    config['contrastive_projection_dim'] = int(proposed)
    config['contrastive_projection_ratio_effective'] = proposed / max(1, branch_dim)

class TrainerKFoldDataManager:
    """Data manager adapter providing the interface required by KFoldValidator."""

    def __init__(self, dataset, config, args):
        self.dataset = dataset
        self.config = config
        self.args = args

    def get_full_dataset_for_kfold(self):
        """Return indices, labels, and protein grouping information for K-fold splitting."""
        if hasattr(self.dataset, 'data_frame'):
            df = getattr(self.dataset, 'data_frame')
            indices = np.arange(len(df))
            label_series = pd.to_numeric(df.get('label', pd.Series([0] * len(df))), errors='coerce').fillna(0.0)
            groups_raw = df.get('uniprot_id', pd.Series([None] * len(df), dtype=object))
            groups = []
            for idx, value in enumerate(groups_raw):
                value_str = "" if value is None else str(value).strip()
                groups.append(value_str if value_str else f"sample_{idx}")

            labels_array = label_series.astype(np.float32).to_numpy()
            groups_array = np.asarray(groups, dtype=object)
            return indices, labels_array, groups_array

        labels: List[float] = []
        groups: List[str] = []

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]

            # Extract label
            if isinstance(sample, dict):
                label_value = sample.get('label', sample.get('labels'))
                if label_value is None:
                    label_value = sample.get('is_target_ptm', sample.get('is_phosphorylated', 0))
            elif isinstance(sample, (list, tuple)) and len(sample) > 1:
                label_value = sample[1]
            else:
                label_value = 0.0

            if isinstance(label_value, torch.Tensor):
                labels.append(float(label_value.item()))
            else:
                labels.append(float(label_value))

            # Extract protein grouping information
            group_id = None
            if isinstance(sample, dict):
                for key in ['uniprot_id', 'protein_id', 'protein_accession', 'entry']:
                    value = sample.get(key)
                    if value not in (None, ''):
                        group_id = str(value)
                        break
            elif isinstance(sample, (list, tuple)) and len(sample) > 2 and isinstance(sample[2], dict):
                meta = sample[2]
                for key in ['uniprot_id', 'protein_id', 'protein_accession', 'entry']:
                    value = meta.get(key) if isinstance(meta, dict) else None
                    if value not in (None, ''):
                        group_id = str(value)
                        break

            if not group_id:
                group_id = f"sample_{idx}"
            groups.append(group_id)

        indices = np.arange(len(self.dataset))
        labels_array = np.array(labels, dtype=np.float32)
        groups_array = np.asarray(groups, dtype=object)
        return indices, labels_array, groups_array

    def create_kfold_loaders(self, train_indices, val_indices):
        """Create K-fold training and validation data loaders."""
        train_subset = Subset(self.dataset, list(train_indices))
        val_subset = Subset(self.dataset, list(val_indices))

        loader_kwargs = {
            'batch_size': self.config.get('batch_size', 16),
            'num_workers': self.args.num_workers,
            'pin_memory': torch.cuda.is_available(),
            'collate_fn': custom_collate_fn
        }

        train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)

        return train_loader, val_loader

# ... (EnhancedPTMsTrainer class with modification) ...

class EnhancedPTMsTrainer:
    """
    Enhanced PTM prediction trainer.
    Integrates training logging, visualization, and metric computation.
    """
    
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self._device_warnings: List[str] = []

        raw_gpu_request = self.config.get('gpu_ids', getattr(args, 'gpu_ids', None))
        try:
            self.requested_gpu_selection = self._normalize_gpu_selection(raw_gpu_request)
        except ValueError as exc:
            self._device_warnings.append(f" gpu_ids='{raw_gpu_request}': {exc}")
            self.requested_gpu_selection = None

        self.parallel_gpu_ids: List[int] = []
        self.primary_device_index: Optional[int] = None
        self.device = torch.device('cpu')  # Placeholder initialization
        self._apply_device_selection()

        self.target_ptm_type = str(self.config.get('target_ptm_type', 'phosphorylation')).strip().lower()
        self.config['target_ptm_type'] = self.target_ptm_type
        
        # Set output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logger()

        for message in self._device_warnings:
            self.logger.warning(message)
        self._device_warnings.clear()
        self._log_device_configuration()
        
        # Initialize components
        self.data_manager = None
        self.model_manager = None
        self.training_manager = None
        self.viz_manager = None
        self.full_train_dataset = None
        self.feature_normalizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training history
        self.training_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_precision': [], 'train_recall': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_precision': [], 'val_recall': [],
            'val_auc': [], 'val_mcc': [], 'val_auprc': []
        }

        self.residue_info = None

        # Threshold and calibration
        self.optimize_threshold = self.config.get('optimize_threshold', True)
        self.threshold_metric = self.config.get('threshold_metric', 'f1')
        self.calibration_method = self.config.get('calibration_method', 'none')
        self.decision_threshold = 0.5
        self.temperature = 1.0
        self.platt_model = None
        self.last_test_metrics = None

        self.enable_feature_normalization = self._parse_bool(
            self.config.get('enable_feature_normalization'), True
        )
        self.config['enable_feature_normalization'] = self.enable_feature_normalization

        # DataLoader management state (supports automatic fallback when workers fail)
        self._loader_base_kwargs: Dict[str, object] = {}
        self._loader_configs: List[Dict[str, object]] = []
        self._current_loader_config_index: int = 0
        
    @staticmethod
    def _parse_bool(value, default: bool = True) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            return default
        return bool(value)

    @staticmethod
    def _parse_int(value, default: Optional[int] = None, minimum: Optional[int] = None) -> Optional[int]:
        if value is None:
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        if minimum is not None and parsed < minimum:
            return minimum
        return parsed

    @staticmethod
    def _normalize_gpu_selection(value: Optional[Any]) -> Optional[Any]:
        if value is None:
            return None

        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None

            lowered = normalized.lower()
            if lowered in {"cpu", "none", "off"}:
                return []
            if lowered in {"auto", "all"}:
                return "auto"

            tokens = normalized.replace(';', ',').split(',')
            gpu_ids: List[int] = []
            for token in tokens:
                cleaned = token.strip()
                if not cleaned:
                    continue
                if cleaned.lower().startswith("cuda:"):
                    cleaned = cleaned.split(':', 1)[1].strip()
                gpu_ids.append(int(cleaned))

            if not gpu_ids:
                return None
            return gpu_ids

        if isinstance(value, (list, tuple)):
            gpu_ids: List[int] = []
            for entry in value:
                if entry is None:
                    continue
                gpu_ids.append(int(str(entry).strip()))
            return gpu_ids

        if isinstance(value, int):
            return [value]

        return None

    def _apply_device_selection(self) -> None:
        cuda_available = torch.cuda.is_available()

        if self.requested_gpu_selection == []:
            self.device = torch.device('cpu')
            self.parallel_gpu_ids = []
            self.primary_device_index = None
        elif not cuda_available:
            self.device = torch.device('cpu')
            self.parallel_gpu_ids = []
            self.primary_device_index = None
            if self.requested_gpu_selection not in (None, []):
                self._device_warnings.append(" UseGPU, CUDADevice,Falling back toCPU.")
        else:
            if self.requested_gpu_selection == 'auto':
                gpu_count = torch.cuda.device_count()
                if gpu_count <= 0:
                    self.device = torch.device('cpu')
                    self.parallel_gpu_ids = []
                    self.primary_device_index = None
                    self._device_warnings.append(" Use GPU, CUDADevice,Falling back toCPU.")
                else:
                    self.parallel_gpu_ids = list(range(gpu_count))
                    self.primary_device_index = self.parallel_gpu_ids[0]
                    self.device = torch.device(f'cuda:{self.primary_device_index}')
                    try:
                        torch.cuda.set_device(self.primary_device_index)
                    except Exception as exc:
                        self._device_warnings.append(f" GPUFailed: {exc}")
            elif isinstance(self.requested_gpu_selection, list) and self.requested_gpu_selection:
                available = torch.cuda.device_count()
                if available <= 0:
                    self.device = torch.device('cpu')
                    self.parallel_gpu_ids = []
                    self.primary_device_index = None
                    self._device_warnings.append(" GPUDevice,Falling back toCPU.")
                else:
                    valid_ids: List[int] = []
                    for raw_idx in self.requested_gpu_selection:
                        try:
                            idx = int(raw_idx)
                        except (TypeError, ValueError):
                            self._device_warnings.append(f" GPU '{raw_idx}'")
                            continue

                        if idx < 0:
                            self._device_warnings.append(f" GPU {idx}")
                            continue
                        if idx >= available:
                            self._device_warnings.append(
                                f"GPU {idx} [0, {max(available - 1, 0)}]"  # noqa: E501
                            )
                            continue
                        if idx not in valid_ids:
                            valid_ids.append(idx)

                    if not valid_ids:
                        self.parallel_gpu_ids = []
                        self.primary_device_index = torch.cuda.current_device()
                        self.device = torch.device(f'cuda:{self.primary_device_index}')
                        self._device_warnings.append(" gpu_ids,Falling back to GPU.")
                    else:
                        self.parallel_gpu_ids = valid_ids
                        self.primary_device_index = valid_ids[0]
                        self.device = torch.device(f'cuda:{self.primary_device_index}')
                        try:
                            torch.cuda.set_device(self.primary_device_index)
                        except Exception as exc:
                            self._device_warnings.append(f" GPUFailed: {exc}")
            else:
                self.primary_device_index = torch.cuda.current_device()
                self.parallel_gpu_ids = [int(self.primary_device_index)]
                self.device = torch.device(f'cuda:{self.primary_device_index}')

            if self.device.type == 'cuda' and not self.parallel_gpu_ids:
                index = self.device.index if self.device.index is not None else torch.cuda.current_device()
                self.parallel_gpu_ids = [int(index)]
                self.primary_device_index = int(index)

        if self.device.type != 'cuda':
            self.parallel_gpu_ids = []
            self.primary_device_index = None

        if self.requested_gpu_selection is not None:
            self.config['gpu_ids_normalized'] = self.requested_gpu_selection

        self.config['effective_device'] = str(self.device)
        self.config['effective_gpu_ids'] = list(self.parallel_gpu_ids)
        self.config['primary_gpu_index'] = self.primary_device_index

    def _log_device_configuration(self) -> None:
        if self.logger is None:
            return

        if self.requested_gpu_selection == []:
            self.logger.info("GPU: cpu, UseCPUTraining")
        elif self.requested_gpu_selection == 'auto':
            self.logger.info("GPU: auto (Use GPU)")
        elif isinstance(self.requested_gpu_selection, list) and self.requested_gpu_selection:
            self.logger.info(
                " UseGPU: %s",
                ', '.join(str(idx) for idx in self.requested_gpu_selection),
            )

        if self.device.type == 'cuda':
            device_entries: List[str] = []
            unique_ids = []
            for idx in self.parallel_gpu_ids:
                if idx not in unique_ids:
                    unique_ids.append(idx)
            for idx in unique_ids:
                try:
                    name = torch.cuda.get_device_name(idx)
                except Exception:
                    name = ' CUDADevice'
                device_entries.append(f"cuda:{idx} ({name})")

            if len(unique_ids) > 1:
                self.logger.info(
                    "Use GPUTraining (Device %s): %s",
                    self.device,
                    ', '.join(device_entries),
                )
            else:
                label = device_entries[0] if device_entries else str(self.device)
                self.logger.info("UseDevice %s (%s)", self.device, label)
        else:
            self.logger.info("UseDevice %s", self.device)

    def _setup_logger(self):
        """Set up the logger."""
        logger = logging.getLogger('enhanced_ptms_trainer')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create file handler
        log_file = self.output_dir / 'training.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Set formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _save_training_history(self):
        """Save training history to the output directory."""
        history_path = self.output_dir / 'training_history.json'
        serializable_history = {}

        for key, value in self.training_history.items():
            if isinstance(value, list):
                serializable_history[key] = [
                    float(v.item()) if hasattr(v, 'item') else float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for v in value
                ]
            else:
                if hasattr(value, 'item'):
                    serializable_history[key] = float(value.item())
                elif isinstance(value, (np.floating, np.integer)):
                    serializable_history[key] = float(value)
                else:
                    serializable_history[key] = value

        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)

        self.logger.info(f"Training Save: {history_path}")

    def _filter_missing_feature_samples(
        self,
        df: pd.DataFrame,
        protein_store: LazyProteinFeatureStore,
        dataset_name: str,
    ) -> pd.DataFrame:
        """Remove samples missing precomputed features to prevent downstream dataset construction errors."""
        if df.empty:
            return df

        unique_ids: Set[str] = set()
        for uniprot_id in df['uniprot_id'].values:
            normalized = "" if uniprot_id is None else str(uniprot_id).strip()
            if normalized:
                unique_ids.add(normalized)

        missing_proteins: Set[str] = set()
        for protein_id in unique_ids:
            entry = protein_store.get_entry(protein_id)
            if entry is None:
                missing_proteins.add(protein_id)

        if not missing_proteins:
            return df

        drop_mask = df['uniprot_id'].isin(missing_proteins)
        removed_samples = int(drop_mask.sum())
        filtered_df = df.loc[~drop_mask].reset_index(drop=True)

        examples = sorted(missing_proteins)
        preview = ', '.join(examples[:5])
        if len(examples) > 5:
            preview += ' ...'

        self.logger.warning(
            "%s %d Samples, %d Protein: %s",
            dataset_name,
            removed_samples,
            len(missing_proteins),
            preview if preview else ' ',
        )

        if filtered_df.empty:
            raise RuntimeError(f"{dataset_name}, Training")

        return filtered_df

    def _filter_position_out_of_range_samples(
        self,
        df: pd.DataFrame,
        protein_store: LazyProteinFeatureStore,
        dataset_name: str,
    ) -> pd.DataFrame:
        """Filter samples with out-of-range or invalid positions to avoid window-construction errors."""
        if df.empty:
            return df

        # Normalize UniProt IDs and convert positions to numeric values
        normalized_ids = df['uniprot_id'].astype(str).str.strip()
        positions_numeric = pd.to_numeric(df['position'], errors='coerce')

        # Cache protein sequence lengths to avoid repeated HDF5 access
        seq_length_map: Dict[str, Optional[int]] = {}
        for protein_id in normalized_ids.unique():
            if not protein_id:
                seq_length_map[protein_id] = None
                continue
            try:
                seq_length_map[protein_id] = protein_store.get_sequence_length(protein_id)
            except Exception as exc:
                self.logger.warning(" Protein %s Failed: %s", protein_id, exc)
                seq_length_map[protein_id] = None

        seq_lengths = normalized_ids.map(seq_length_map)

        # Flag non-integer or invalid positions
        fractional_mask = (~positions_numeric.isna()) & (np.floor(positions_numeric) != positions_numeric)
        invalid_mask = positions_numeric.isna() | fractional_mask | (positions_numeric < 1)
        invalid_mask |= seq_lengths.isna()
        invalid_mask |= positions_numeric > seq_lengths

        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            sample_entries = []
            for uid, pos_val, len_val in zip(
                normalized_ids[invalid_mask],
                positions_numeric[invalid_mask],
                seq_lengths[invalid_mask],
            ):
                if len(sample_entries) >= 5:
                    break
                if pd.isna(pos_val):
                    pos_display = 'NaN'
                elif float(pos_val).is_integer():
                    pos_display = f"{int(pos_val)}"
                else:
                    pos_display = f"{pos_val:.3f}"

                len_display = 'NaN' if pd.isna(len_val) else f"{int(len_val)}"
                sample_entries.append(f"{uid}:{pos_display}>{len_display}")

            preview = ', '.join(sample_entries)
            self.logger.warning(
                "%s %d Samples: %s",
                dataset_name,
                invalid_count,
                preview if preview else ' ',
            )

        filtered_df = df.loc[~invalid_mask].copy()
        if filtered_df.empty:
            raise RuntimeError(f"{dataset_name} Samples, Training")

        valid_positions = positions_numeric.loc[~invalid_mask].round().astype(int)
        filtered_df.loc[:, 'uniprot_id'] = normalized_ids.loc[~invalid_mask].values
        filtered_df.loc[:, 'position'] = valid_positions.values
        filtered_df.reset_index(drop=True, inplace=True)

        return filtered_df

    def _filter_samples_with_structure(self, samples, dataset_name, return_indices: bool = False):
        """Filter samples missing structure features."""
        # Skip when structure features are disabled in config
        if not self.config.use_structure:
            if return_indices:
                return samples, list(range(len(samples)))
            return samples

        if not samples:
            return (samples, list(range(len(samples)))) if return_indices else samples

        filtered = []
        indices = []
        for idx, sample in enumerate(samples):
            struct_value = sample.get('structure_features')
            if struct_value is None:
                continue
            if np.asarray(struct_value).size == 0:
                continue
            filtered.append(sample)
            indices.append(idx)
        dropped = len(samples) - len(filtered)

        if dropped > 0:
            self.logger.info(f"{dataset_name} {dropped} Samples ({dropped/len(samples)*100:.1f}%)" if len(samples) > 0 else f"{dataset_name} {dropped} Samples")

        if not filtered:
            self.logger.warning(f"{dataset_name} Samples, Use Data")
            return (samples, list(range(len(samples)))) if return_indices else samples

        if return_indices:
            return filtered, indices
        return filtered

    def _apply_feature_normalization(self, train_samples, extra_datasets=None):
        """Apply feature normalization and reuse the same parameters for additional datasets."""
        if not self.enable_feature_normalization:
            self.logger.info(" Disable,Skip ")
            self.feature_normalizer = None
            return

        if not train_samples:
            self.logger.warning("TrainingSamples,Skip ")
            return

        extra_datasets = extra_datasets or {}
        self.feature_normalizer = FeatureNormalizer()
        self.feature_normalizer.fit(train_samples)
        self.feature_normalizer.transform(train_samples)

        for name, dataset in extra_datasets.items():
            if not dataset:
                self.logger.info(f"{name} Dataset,Skip ")
                continue
            self.feature_normalizer.transform(dataset)
            self.logger.info(f" {name} Dataset (Samples: {len(dataset)})")

        stats_path = self.output_dir / 'normalization_stats.npz'
        self.feature_normalizer.save(stats_path)
        self.logger.info(f" Parameters Save: {stats_path}")

    def _filter_structure_by_index(self, dataset, dataset_name):
        """Streaming filter: only check structure availability and return valid indices without materializing samples."""
        valid_indices = []
        dropped_count = 0
        total_count = len(dataset)
        
        self.logger.info(f"Start {dataset_name}.")
        
        # Batch checks for efficiency
        batch_size = 1000
        for batch_start in range(0, total_count, batch_size):
            batch_end = min(batch_start + batch_size, total_count)
            
            for idx in range(batch_start, batch_end):
                # Read only required metadata; do not load full samples
                row = dataset.data_frame.iloc[idx]
                uniprot_id = str(row['uniprot_id'])
                
                # Check whether the protein has structure features
                has_structure = dataset.protein_store.has_structure(uniprot_id)
                
                if has_structure:
                    valid_indices.append(idx)
                else:
                    dropped_count += 1
            
            # Periodic progress report
            if (batch_end % 10000) == 0 or batch_end == total_count:
                self.logger.info(f" {batch_end}/{total_count} Samples")
        
        if dropped_count > 0:
            self.logger.info(
                f"{dataset_name} {dropped_count} Samples "
                f"({dropped_count/total_count*100:.1f}%)"
            )
        
        if not valid_indices:
            self.logger.warning(f"{dataset_name} Samples, Use Samples")
            return list(range(total_count))
        
        self.logger.info(f"{dataset_name} {len(valid_indices)}/{total_count} Samples")
        return valid_indices

    def _apply_feature_normalization_lazy(self, dataset, valid_indices):
        """Stream computation of normalization statistics over valid samples without loading everything into memory."""
        if not self.enable_feature_normalization:
            self.logger.info(" Disable,Skip ")
            self.feature_normalizer = None
            return

        self.feature_normalizer = FeatureNormalizer()

        total_samples = len(valid_indices)
        if total_samples == 0:
            self.logger.warning(" TrainingSamples,Skip ")
            return

        self.logger.info(f" {total_samples} TrainingSamples Parameters.")

        visited = 0
        processed = 0
        skipped = 0
        log_interval = max(1000, total_samples // 20) if total_samples > 0 else 1000

        def sample_iterator():
            nonlocal visited, processed, skipped
            for idx in valid_indices:
                visited += 1
                try:
                    sample = dataset._build_raw_sample(idx)
                except Exception as exc:
                    skipped += 1
                    if skipped <= 5 or skipped % 500 == 0:
                        self.logger.warning("Skip Samples %d: %s", idx, exc)
                    continue
                processed += 1
                if visited % log_interval == 0 or visited == total_samples:
                    self.logger.info(": %d/%d", visited, total_samples)
                yield sample

        self.feature_normalizer.fit_incremental(sample_iterator())

        # Finalize statistics
        self.feature_normalizer.finalize_stats()

        stats_path = self.output_dir / 'normalization_stats.npz'
        self.feature_normalizer.save(stats_path)
        self.logger.info(
            " Parameters Save: %s (Samples: %d, Skip: %d)",
            stats_path,
            processed,
            skipped,
        )

    def _split_train_val_by_protein_lazy(self, dataset, base_dataset, valid_indices, val_ratio, seed):
        """Split train/val by protein groups (lazy version) to avoid materializing the full dataset."""
        if not 0 < val_ratio < 0.5:
            self.logger.warning(f"Validation {val_ratio} (0, 0.5), 0.2")
            val_ratio = 0.2

        # Stream extraction of group information and labels
        groups = []
        labels = []
        
        self.logger.info(" Protein Info.")
        batch_size = 1000
        total_samples = len(valid_indices)
        
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_indices = valid_indices[batch_start:batch_end]
            
            for idx in batch_indices:
                row = base_dataset.data_frame.iloc[idx]
                uniprot_id = str(row.get('uniprot_id', f'sample_{idx}'))
                groups.append(uniprot_id)
                
                # Extract labels
                label_value = row.get('label')
                if label_value is None or pd.isna(label_value):
                    label_value = 0
                labels.append(int(label_value))
            
            if (batch_end % 10000) == 0 or batch_end == total_samples:
                self.logger.info(f" {batch_end}/{total_samples} Samples Info")
        
        groups = np.array(groups)
        labels = np.array(labels)
        indices = np.array(valid_indices)

        # Split by protein groups
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(splitter.split(indices, groups=groups))

        train_subset = Subset(dataset, list(train_idx))
        val_subset = Subset(dataset, list(val_idx))

        train_proteins = np.unique(groups[train_idx])
        val_proteins = np.unique(groups[val_idx])
        overlap = np.intersect1d(train_proteins, val_proteins)

        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        stats = {
            'train_proteins': len(train_proteins),
            'val_proteins': len(val_proteins),
            'overlap_proteins': len(overlap),
            'train_pos': int(train_labels.sum()),
            'train_neg': int(len(train_labels) - train_labels.sum()),
            'val_pos': int(val_labels.sum()),
            'val_neg': int(len(val_labels) - val_labels.sum())
        }

        return train_subset, val_subset, stats

    def _update_config_with_dataset_metadata(self, dataset):
        """Update model config from dataset metadata to keep explicit-embedding dimensions consistent."""
        if dataset is None:
            return

        if hasattr(dataset, 'ptm_type_to_id'):
            ptm_mapping = dict(dataset.ptm_type_to_id)
            num_ptm_types = max(1, len(ptm_mapping))
            self.config['ptm_type_to_id'] = ptm_mapping
            self.config['num_ptm_types'] = num_ptm_types

        if hasattr(dataset, 'residue_vocab_size'):
            self.config['num_residue_types'] = int(dataset.residue_vocab_size)
        elif hasattr(dataset, 'residue_to_id'):
            self.config['num_residue_types'] = len(getattr(dataset, 'residue_to_id', {}))
        else:
            self.config.setdefault('num_residue_types', 4)

        if hasattr(dataset, 'residue_to_id'):
            self.config['residue_to_id'] = dict(dataset.residue_to_id)

        if hasattr(dataset, 'fixed_window_size'):
            max_positions = int(dataset.fixed_window_size * 2)
            current_max = int(self.config.get('max_position_embeddings', 0) or 0)
            self.config['max_position_embeddings'] = max(current_max, max_positions, dataset.fixed_window_size)

    # ------------------------------------------------------------------
    # DataLoader management helpers
    # ------------------------------------------------------------------

    def _build_loader_base_kwargs(self) -> Dict[str, object]:
        batch_size_cfg = self.config.get('batch_size', getattr(self.args, 'batch_size', 16))
        try:
            batch_size = int(batch_size_cfg)
        except (TypeError, ValueError):
            batch_size = 16
        batch_size = max(1, batch_size)

        base_kwargs: Dict[str, object] = {
            'batch_size': batch_size,
            'collate_fn': custom_collate_fn,
        }

        drop_last = self.config.get('protein_batch_drop_last')
        if drop_last is not None:
            base_kwargs['drop_last'] = bool(drop_last)

        persistent_workers = self.config.get('dataloader_persistent_workers')
        if persistent_workers:
            base_kwargs['persistent_workers'] = True

        return base_kwargs

    def _resolve_initial_num_workers(self) -> int:
        workers_cfg = self.config.get('dataloader_num_workers', getattr(self.args, 'num_workers', 0))
        try:
            workers = int(workers_cfg)
        except (TypeError, ValueError):
            workers = 0
        return max(0, workers)

    def _build_dataloader_configs(self) -> List[Dict[str, object]]:
        configs: List[Dict[str, object]] = []

        initial_workers = self._resolve_initial_num_workers()
        self.args.num_workers = initial_workers

        device_is_cuda = self.device.type == 'cuda'
        pin_memory_cfg = self.config.get('pin_memory')
        if pin_memory_cfg is None:
            pin_memory_default = device_is_cuda
        else:
            pin_memory_default = bool(pin_memory_cfg) and device_is_cuda

        prefetch_cfg = self.config.get('dataloader_prefetch_factor')
        try:
            base_prefetch = max(1, int(prefetch_cfg)) if prefetch_cfg is not None else None
        except (TypeError, ValueError):
            base_prefetch = None

        def make_config(num_workers: int, pin_memory: bool) -> Dict[str, object]:
            num_workers = max(0, int(num_workers))
            pin_enabled = bool(pin_memory) and device_is_cuda

            cfg: Dict[str, object] = {
                'num_workers': num_workers,
                'pin_memory': pin_enabled,
            }

            if num_workers > 0:
                prefetch_val = base_prefetch
                if prefetch_val is None:
                    prefetch_val = 2 if num_workers > 1 else 1
                else:
                    prefetch_val = max(1, int(prefetch_val))
                    if num_workers == 1 and prefetch_val > 2:
                        prefetch_val = 2
                cfg['prefetch_factor'] = prefetch_val

                if self.config.get('dataloader_persistent_workers'):
                    cfg['persistent_workers'] = True

            return cfg

        seen_keys = set()

        def register_config(num_workers: int, pin_memory: bool) -> None:
            key = (max(0, int(num_workers)), bool(pin_memory) and device_is_cuda)
            if key in seen_keys:
                return
            configs.append(make_config(*key))
            seen_keys.add(key)

        register_config(initial_workers, pin_memory_default)

        for workers in range(initial_workers - 1, -1, -1):
            register_config(workers, pin_memory_default)

        if pin_memory_default:
            register_config(0, False)

        if not configs:
            register_config(0, pin_memory_default)

        return configs

    def _get_current_loader_config(self) -> Dict[str, object]:
        if not self._loader_configs:
            return {'num_workers': 0, 'pin_memory': self.device.type == 'cuda'}
        index = min(self._current_loader_config_index, len(self._loader_configs) - 1)
        return self._loader_configs[index]

    def _apply_current_loader_config_metadata(self) -> None:
        cfg = self._get_current_loader_config()
        self.args.num_workers = int(cfg.get('num_workers', 0))
        self.config['effective_num_workers'] = int(cfg.get('num_workers', 0))
        self.config['effective_pin_memory'] = bool(cfg.get('pin_memory', False))
        self.config['effective_prefetch_factor'] = cfg.get('prefetch_factor')

    def _log_loader_config(self, prefix: str = "DataLoad ") -> None:
        cfg = self._get_current_loader_config()
        prefetch_val = cfg.get('prefetch_factor', 'None')
        persistent_workers = cfg.get('persistent_workers', False)
        self.logger.info(
            "%sConfig: batch_size=%s, num_workers=%s, pin_memory=%s, prefetch_factor=%s, persistent_workers=%s",
            prefix,
            self._loader_base_kwargs.get('batch_size'),
            cfg.get('num_workers'),
            cfg.get('pin_memory'),
            prefetch_val,
            persistent_workers,
        )

    def _get_dataset_labels(self, dataset):
        """Recursively obtain dataset labels, supporting nested Subset objects."""
        try:
            current_dataset = dataset
            indices_chain = []
            
            while isinstance(current_dataset, Subset):
                indices_chain.append(current_dataset.indices)
                current_dataset = current_dataset.dataset
            
            if hasattr(current_dataset, 'data_frame'):
                # Ensure labels are numeric
                all_labels = pd.to_numeric(current_dataset.data_frame['label'], errors='coerce').fillna(0).values
                
                if not indices_chain:
                    return all_labels
                
                # Map indices from the outermost Subset down to the base dataset
                # indices_chain = [subset_n.indices, ..., subset_1.indices]
                # Goal: evaluate subset_1[subset_2[...[subset_n]...]]
                
                mapped_indices = np.array(indices_chain[0])
                for parent_indices in indices_chain[1:]:
                    parent_indices = np.array(parent_indices)
                    mapped_indices = parent_indices[mapped_indices]
                
                return all_labels[mapped_indices]
                
        except Exception as e:
            self.logger.warning(f" DatasetLabelsFailed: {e}")
            return None
        return None

    def _create_data_loader(self, dataset, *, shuffle: bool, pos_ratio: float = None) -> DataLoader:
        cfg = self._get_current_loader_config()
        kwargs: Dict[str, object] = dict(self._loader_base_kwargs)
        
        # If pos_ratio is provided and shuffling is needed (typically training), use stratified sampling
        if pos_ratio is not None and shuffle:
            try:
                # Get labels
                labels = self._get_dataset_labels(dataset)
                
                if labels is not None:
                    batch_size = int(kwargs.get('batch_size', 16))
                    
                    if self.config.get('use_deepmvp_sampling', False):
                        from utils.data_loading_optimizer import RotatingBalancedSampler
                        self.logger.info(f"Enable DeepMVP: Samples ={pos_ratio:.4f}")
                        sampler = RotatingBalancedSampler(labels, batch_size, pos_ratio=pos_ratio, shuffle=True)
                    else:
                        from utils.data_loading_optimizer import StratifiedBatchSampler
                        self.logger.info(f"Enable: Samples ={pos_ratio:.4f}")
                        sampler = StratifiedBatchSampler(labels, batch_size, pos_ratio=pos_ratio, shuffle=True)

                    kwargs['batch_sampler'] = sampler
                    
                    # batch_sampler is mutually exclusive with batch_size, shuffle, and drop_last
                    kwargs.pop('batch_size', None)
                    kwargs.pop('shuffle', None)
                    kwargs.pop('drop_last', None)
                else:
                    self.logger.warning(" DatasetLabels,Skip, Shuffle")
                    kwargs['shuffle'] = bool(shuffle)
            except Exception as e:
                self.logger.error(f"Create Failed: {e}, Shuffle")
                kwargs['shuffle'] = bool(shuffle)
        else:
            kwargs['shuffle'] = bool(shuffle)

        num_workers = int(cfg.get('num_workers', 0))
        kwargs['num_workers'] = num_workers
        kwargs['pin_memory'] = bool(cfg.get('pin_memory', False))

        prefetch_val = cfg.get('prefetch_factor')
        if num_workers > 0 and prefetch_val is not None:
            kwargs['prefetch_factor'] = int(prefetch_val)
        else:
            kwargs.pop('prefetch_factor', None)

        persistent_workers = cfg.get('persistent_workers')
        if persistent_workers and num_workers > 0:
            kwargs['persistent_workers'] = True
        else:
            kwargs.pop('persistent_workers', None)

        return DataLoader(dataset, **kwargs)

    def _initialize_loader_management(self) -> None:
        self._loader_base_kwargs = self._build_loader_base_kwargs()
        self._loader_configs = self._build_dataloader_configs()
        self._current_loader_config_index = 0
        self._apply_current_loader_config_metadata()

    def _build_all_data_loaders(self, train_subset, val_subset, test_dataset) -> None:
        self.train_dataset = train_subset
        self.val_dataset = val_subset
        self.test_dataset = test_dataset

        # Compute test-set positive ratio for training-set stratified sampling
        target_pos_ratio = None
        
        # Prefer manual override from config
        manual_pos_ratio = self.config.get('train_pos_ratio')
        if manual_pos_ratio is not None:
            try:
                target_pos_ratio = float(manual_pos_ratio)
                self.logger.info(f"UseConfig Training Samples: {target_pos_ratio:.4f}")
                if not (0.0 < target_pos_ratio < 1.0):
                    self.logger.warning(f" Samples {target_pos_ratio} (0, 1), ")
                    target_pos_ratio = None
            except (TypeError, ValueError):
                self.logger.warning(f" train_pos_ratio='{manual_pos_ratio}'")
                target_pos_ratio = None

        # Default to stratified sampling (or read from config)
        if target_pos_ratio is None and self.config.get('stratified_batch_sampling', True):
            try:
                # Try to compute a baseline from the training set when not manually specified.
                # Note: for highly imbalanced data, we often want to oversample positives, so test/train ratio may be
                # insufficient; keep the existing logic that references the test-set ratio or a reasonable default.
                
                test_labels = self._get_dataset_labels(test_dataset)
                
                if test_labels is not None:
                    target_pos_ratio = float(test_labels.mean())
                    self.logger.info(f" Test Samples: {target_pos_ratio:.4f}")
                    
                    # Avoid extreme ratios that make sampling impossible
                    if target_pos_ratio < 0.01:
                        self.logger.warning(f"Test Samples ({target_pos_ratio:.4f}), 0.01")
                        target_pos_ratio = 0.01
                    elif target_pos_ratio > 0.99:
                        self.logger.warning(f"Test Samples ({target_pos_ratio:.4f}), 0.99")
                        target_pos_ratio = 0.99
            except Exception as e:
                self.logger.warning(f" Test Failed: {e}")

        self.train_loader = self._create_data_loader(train_subset, shuffle=True, pos_ratio=target_pos_ratio)
        self.val_loader = self._create_data_loader(val_subset, shuffle=False)
        self.test_loader = self._create_data_loader(test_dataset, shuffle=False)

        self._log_loader_config("InitializeDataLoad ")

    def _rebuild_data_loaders(self) -> bool:
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            return False

        self.train_loader = self._create_data_loader(self.train_dataset, shuffle=True)
        self.val_loader = self._create_data_loader(self.val_dataset, shuffle=False)
        self.test_loader = self._create_data_loader(self.test_dataset, shuffle=False)

        return True

    def _advance_dataloader_config_for_retry(self) -> bool:
        if not self._loader_configs:
            return False

        if self._current_loader_config_index >= len(self._loader_configs) - 1:
            return False

        self._current_loader_config_index += 1
        self._apply_current_loader_config_metadata()

        if not self._rebuild_data_loaders():
            return False

        cfg = self._get_current_loader_config()
        self.logger.warning(
            "DataLoader: Use num_workers=%s, pin_memory=%s, prefetch_factor=%s",
            cfg.get('num_workers'),
            cfg.get('pin_memory'),
            cfg.get('prefetch_factor', 'None'),
        )

        return True

    @staticmethod
    def _should_retry_dataloader_error(exc: Exception) -> bool:
        if not isinstance(exc, (RuntimeError, OSError)):
            return False

        message = str(exc)
        keywords = (
            'DataLoader worker',
            'dataloader worker',
            'worker exited unexpectedly',
            'killed by signal',
            'Broken pipe',
            'shared memory',
            'bus error',
            'Cannot allocate memory',
            'Errno 12',
            'can\'t synchronously read data',
        )
        return any(keyword in message for keyword in keywords)

    def _cleanup_training_manager(self) -> None:
        if not hasattr(self, 'training_manager') or self.training_manager is None:
            self.training_manager = None
            return

        writer = getattr(self.training_manager, 'tb_writer', None)
        if writer is not None:
            try:
                writer.flush()
                writer.close()
            except Exception:
                pass

        observer = getattr(self.training_manager, 'nan_observer', None)
        if observer is not None:
            try:
                observer.close()
            except Exception:
                pass

        self.training_manager = None

    def prepare_data(self):
        """Prepare data."""
        self.logger.info("="*80)
        self.logger.info("StartData ")
        self.logger.info("="*80)
        self.logger.info(f" PTM: {self.target_ptm_type}")

        # Check whether to use the optimized data loader
        use_optimized_loading = self.config.get('use_optimized_loading', True)

        if use_optimized_loading:
            return self.prepare_data_optimized()
        else:
            return self.prepare_data_original()

    def prepare_data_original(self):
        """Original data loading method using UnifiedDataProcessor as fallback."""
        self.logger.info("Using original data loading method (UnifiedDataProcessor)...")
        
        train_path = self.config.get('train_data_path')
        test_path = self.config.get('test_data_path')
        features_path = self.config.get('features_path')
        
        processor = UnifiedDataProcessor(
            train_path=train_path,
            test_path=test_path,
            esm_features_path=features_path,
            window_size=self.config.get('window_size', 61),
            local_window_size=self.config.get('local_window_size', 31),
            target_ptm_type=self.target_ptm_type
        )
        
        train_data = processor.prepare_train_dataset()
        test_data = processor.prepare_test_dataset()
        
        train_dataset = UnifiedPTMDataset(
            train_data, 
            fixed_window_size=self.config.get('window_size', 61),
            target_ptm_type=self.target_ptm_type
        )
        test_dataset = UnifiedPTMDataset(
            test_data, 
            fixed_window_size=self.config.get('window_size', 61),
            target_ptm_type=self.target_ptm_type
        )
        
        # Split train/val
        val_ratio = self.config.get('val_ratio', 0.2)
        seed = self.config.get('seed', 42)
        
        # Use simple split for fallback (or reuse the protein split logic if possible, but keep it simple here)
        # To reuse _split_train_val_by_protein, we need to adapt inputs.
        # But _split_train_val_by_protein expects dataset and processed_data (list of dicts).
        # train_dataset.data is processed_data.
        
        train_subset, val_subset, split_stats = self._split_train_val_by_protein(
            train_dataset,
            train_data,
            val_ratio,
            seed
        )
        
        self.logger.info("Dataset split (Original):")
        self.logger.info(f"  Train: {len(train_subset)}, Val: {len(val_subset)}")
        
        self._build_all_data_loaders(train_subset, val_subset, test_dataset)
        return True

    def prepare_data_optimized(self):
        """Prepare data using the optimized data loader."""
        self.logger.info("Use DataLoad.")


        # Resolve paths from config
        train_data_path = self.config.get('train_data_path')
        test_data_path = self.config.get('test_data_path')
        features_path = self.config.get('features_path')
        fasta_dir = self.config.get('fasta_dir')
        pdb_dir = self.config.get('pdb_dir')

        # Validate required paths
        if not train_data_path:
            raise ValueError("Config 'train_data_path'")
        if not test_data_path:
            raise ValueError("Config 'test_data_path'")
        if not features_path:
            raise ValueError("Config 'features_path'")
        
        if pdb_dir:
            self.logger.info(f"PDB Directory: {pdb_dir}")
        else:
            self.logger.warning("Config 'pdb_dir', ")

        # Configure cache directory (supports shared cache)
        cache_dir_config = self.config.get('protein_cache_dir') or getattr(self.args, 'protein_cache_dir', None)
        if cache_dir_config:
            cache_dir = Path(cache_dir_config)
        else:
            cache_dir = self.output_dir / 'protein_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"TrainingData Path: {train_data_path}")
        self.logger.info(f"TestData Path: {test_data_path}")
        self.logger.info(f" Path: {features_path}")
        self.logger.info(f" Directory: {cache_dir}")

        optimizer: Optional[DataLoadingOptimizer] = None
        try:
            window_size = self.config.get('window_size', 61)
            local_window_size = self.config.get('local_window_size', 31)

            lazy_loading_flag = self._parse_bool(self.config.get('lazy_loading'), True)
            self.config['lazy_loading'] = lazy_loading_flag

            lazy_cache_size_cfg = self._parse_int(self.config.get('lazy_cache_size'), default=None, minimum=32)
            if lazy_cache_size_cfg is not None:
                self.logger.info(" Load %d Protein", lazy_cache_size_cfg)
                self.config['lazy_cache_size'] = lazy_cache_size_cfg
            else:
                self.logger.info(" Load Use PTM_LAZY_CACHE_MAX_ENTRIES ")

            optimizer = DataLoadingOptimizer(
                features_path,
                num_workers=self.args.num_workers,
                lazy_loading=lazy_loading_flag,
                fasta_dir=fasta_dir,
                pdb_dir=pdb_dir,
            )

            def _clean_dataframe(df: pd.DataFrame, source: str) -> pd.DataFrame:
                raw_count = len(df)
                df = df.copy()
                df['uniprot_id'] = df['uniprot_id'].apply(normalize_uniprot_id)
                invalid_mask = df['uniprot_id'].isna()
                invalid_count = int(invalid_mask.sum())
                if invalid_count > 0:
                    self.logger.warning(
                        f"{source} {invalid_count} UniProt ID ({invalid_count / max(raw_count, 1) * 100:.2f}% ), "
                    )
                    df = df.loc[~invalid_mask].copy()
                try:
                    df['uniprot_id'] = df['uniprot_id'].astype(str)
                except Exception:
                    pass
                df.reset_index(drop=True, inplace=True)
                return df

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            train_df = _clean_dataframe(train_df, "TrainingData")
            test_df = _clean_dataframe(test_df, "TestData")

            def _apply_sample_limit(df: pd.DataFrame, limit: int | None, source: str) -> pd.DataFrame:
                if limit is None:
                    return df
                try:
                    limit_value = int(limit)
                except (TypeError, ValueError):
                    return df
                if limit_value <= 0 or len(df) <= limit_value:
                    return df
                seed_value = int(getattr(self.args, 'seed', 42))
                self.logger.info(f"{source}: subsampling {limit_value} / {len(df)} samples for a smoke run")
                return df.sample(n=limit_value, random_state=seed_value).reset_index(drop=True)

            train_df = _apply_sample_limit(train_df, getattr(self.args, 'max_train_samples', None), "TrainingData")
            test_df = _apply_sample_limit(test_df, getattr(self.args, 'max_test_samples', None), "TestData")

            self.logger.info(f"LoadTrainingData: {len(train_df)} Samples")
            self.logger.info(f"LoadTestData: {len(test_df)} Samples")

            # 1. Precompute micro-environment features for all relevant proteins. This is important for 50k+ samples and
            #    significantly reduces DataLoader overhead.
            all_initial_pids = sorted(list(set(train_df['uniprot_id'].unique()) | set(test_df['uniprot_id'].unique())))
            self.logger.info(f" {len(all_initial_pids)} Protein.")
            
            # In lazy mode, only return the micro-environment feature dict and avoid keeping HDF5 handles open.
            micro_env_cache = optimizer.preload_protein_features(all_initial_pids, cache_dir=cache_dir)
            
            # 2. Initialize the actual feature store and inject the precomputed micro-environment features
            store_settings = optimizer.h5_cache_settings
            protein_store = LazyProteinFeatureStore(
                features_path,
                lazy_loading=optimizer.lazy_loading,
                h5_cache_bytes=store_settings['rdcc_nbytes'],
                h5_cache_slots=store_settings['rdcc_nslots'],
                h5_cache_w0=store_settings['rdcc_w0'],
                max_lazy_cache_entries=lazy_cache_size_cfg,
                fasta_dir=fasta_dir,
                pdb_dir=pdb_dir,
                micro_env_cache=micro_env_cache,
            )

            train_df = self._filter_missing_feature_samples(train_df, protein_store, "TrainingData")
            test_df = self._filter_missing_feature_samples(test_df, protein_store, "TestData")

            train_df = self._filter_position_out_of_range_samples(train_df, protein_store, "TrainingData")
            test_df = self._filter_position_out_of_range_samples(test_df, protein_store, "TestData")

            self.logger.info(f" TrainingData: {len(train_df)} Samples")
            self.logger.info(f" TestData: {len(test_df)} Samples")

            all_protein_ids = sorted(list(set(train_df['uniprot_id'].unique()) | set(test_df['uniprot_id'].unique())))
            protein_store.register_known_proteins(all_protein_ids)
            
            if micro_env_cache:
                self.logger.info(f" {len(micro_env_cache)} Protein ")
            else:
                self.logger.warning(", ")

            train_residue_series = train_df['residue'].dropna().astype(str).str.strip()
            observed_residues = {res for res in train_residue_series if res}
            residue_info = resolve_ptm_residue_info(self.target_ptm_type, observed_residues)
            self.logger.info(
                " LoadUse Residue Training: %s",
                residue_info['residue_to_id'],
            )

            test_residue_series = test_df['residue'].dropna().astype(str).str.strip()
            unseen_test_residues = sorted({res for res in test_residue_series if res} - observed_residues)
            if unseen_test_residues:
                self.logger.info(
                    "Test Training Residue %s, Other",
                    unseen_test_residues,
                )

            optimizer.close()

            # ----------------------------------------------------------------
            # PCA dimensionality reduction (load model)
            # ----------------------------------------------------------------
            feature_reducer = None
            # Try multiple candidate locations for a PCA model
            possible_pca_paths = [
                os.path.join(os.path.dirname(self.args.train_data_path), "pca_models.joblib"),
                os.path.join(os.getcwd(), "sumoylation_data_2", "pca_models.joblib"),
                "sumoylation_data_2/pca_models.joblib"
            ]
            
            pca_model_path = None
            for path in possible_pca_paths:
                if os.path.exists(path):
                    pca_model_path = path
                    break
            
            if pca_model_path:
                self.logger.info(f"[PCA] PCA Model: {os.path.abspath(pca_model_path)}, Load.")
                try:
                    feature_reducer = FeatureReducer()
                    feature_reducer.load(pca_model_path)
                    self.logger.info("[PCA]ModelLoad, DataLoad ")
                    
                    # Update feature dimensions in config
                    if 'sequence_features' in feature_reducer.models:
                        seq_dim = feature_reducer.models['sequence_features'].n_components_
                        self.config['seq_feature_dim'] = int(seq_dim)
                        self.logger.info(f"[PCA]: {seq_dim}")
                        
                    if 'structure_features' in feature_reducer.models:
                        struct_dim = feature_reducer.models['structure_features'].n_components_
                        self.config['struct_feature_dim'] = int(struct_dim)
                        self.logger.info(f"[PCA]: {struct_dim}")
                        
                except Exception as e:
                    self.logger.error(f"[PCA]Load PCA ModelFailed: {e}")
                    feature_reducer = None
            else:
                self.logger.warning(f"[PCA] Path PCA Model: {possible_pca_paths}, Skip!")

            self.logger.info(" Load Dataset.")

            train_base_dataset = OnDemandPTMDataset(
                train_df,
                protein_store,
                window_size=window_size,
                local_window_size=local_window_size,
                target_ptm_type=self.target_ptm_type,
                residue_info=residue_info,
                feature_reducer=feature_reducer,
            )
            test_base_dataset = OnDemandPTMDataset(
                test_df,
                protein_store,
                window_size=window_size,
                local_window_size=local_window_size,
                target_ptm_type=self.target_ptm_type,
                residue_info=residue_info,
                feature_reducer=feature_reducer,
            )

            # Streaming filter: only check structure availability and return valid indices without materializing samples.
            # Skip structure filtering when structure features are disabled.
            use_structure = self.config.get('use_structure', True)
            filter_structure = self.config.get('filter_missing_structure', True) and use_structure
            if filter_structure:
                train_valid_indices = self._filter_structure_by_index(train_base_dataset, "Training")
                test_valid_indices = self._filter_structure_by_index(test_base_dataset, "Test")
            else:
                if not use_structure:
                    self.logger.info(" Disable,Skip,Use Samples")
                train_valid_indices = list(range(len(train_base_dataset)))
                test_valid_indices = list(range(len(test_base_dataset)))

            # Stream computation of normalization statistics over valid training samples without loading everything into memory
            normalization_stats = None
            if self.enable_feature_normalization:
                self.logger.info("Start Parameters.")
                self._apply_feature_normalization_lazy(train_base_dataset, train_valid_indices)
                normalization_stats = self.feature_normalizer.get_stats() if self.feature_normalizer else None
            else:
                self.logger.info(" ConfigDisable,Skip ")

            train_base_dataset.set_normalization_stats(normalization_stats)
            test_base_dataset.set_normalization_stats(normalization_stats)

            train_dataset = train_base_dataset if not filter_structure else Subset(train_base_dataset, train_valid_indices)
            test_dataset = test_base_dataset if not filter_structure else Subset(test_base_dataset, test_valid_indices)

            self.residue_info = residue_info
            self.full_train_dataset = train_dataset
            if self.residue_info:
                self.logger.info(f"Use Residue: {self.residue_info['residue_to_id']}")

            self._update_config_with_dataset_metadata(train_base_dataset)

            val_ratio = self.config.get('val_ratio', 0.2)
            seed = self.config.get('seed', 42)
            train_subset, val_subset, split_stats = self._split_train_val_by_protein_lazy(
                train_dataset,
                train_base_dataset,
                train_valid_indices,
                val_ratio,
                seed,
            )

            self.logger.info("Dataset Completed:")
            self.logger.info(f" Training: {len(train_subset)} Samples,Protein: {split_stats['train_proteins']}, Samples: {split_stats['train_pos']}, Samples: {split_stats['train_neg']}")
            self.logger.info(f" Validation: {len(val_subset)} Samples,Protein: {split_stats['val_proteins']}, Samples: {split_stats['val_pos']}, Samples: {split_stats['val_neg']}")
            self.logger.info(f" Protein: {split_stats['overlap_proteins']}")
            self.logger.info(f" Test: {len(test_dataset)} Samples")

            self._initialize_loader_management()
            self._build_all_data_loaders(train_subset, val_subset, test_dataset)

            self.logger.info(" DataLoad CreateCompleted")
            return True

        except Exception as e:
            self.logger.error(f" DataLoadFailed: {e}")
            self.logger.info(" DataLoad.")
            return self.prepare_data_original()
        finally:
            if optimizer is not None:
                optimizer.close()

   
    
    def _split_train_val_by_protein(self, dataset, processed_data, val_ratio, seed):
        """Split train/val by protein groups to reduce data leakage."""
        if not 0 < val_ratio < 0.5:
            self.logger.warning(f"Validation {val_ratio} (0, 0.5), 0.2")
            val_ratio = 0.2

        groups = np.array([sample.get('uniprot_id', '') for sample in processed_data])
        indices = np.arange(len(processed_data))

        splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(splitter.split(indices, groups=groups))

        train_subset = Subset(dataset, list(train_idx))
        val_subset = Subset(dataset, list(val_idx))

        train_proteins = np.unique(groups[train_idx])
        val_proteins = np.unique(groups[val_idx])
        overlap = np.intersect1d(train_proteins, val_proteins)

        train_labels = np.array([
            processed_data[i].get('is_target_ptm', processed_data[i].get('is_phosphorylated', 0))
            for i in train_idx
        ])
        val_labels = np.array([
            processed_data[i].get('is_target_ptm', processed_data[i].get('is_phosphorylated', 0))
            for i in val_idx
        ])

        stats = {
            'train_proteins': len(train_proteins),
            'val_proteins': len(val_proteins),
            'overlap_proteins': len(overlap),
            'train_pos': int(train_labels.sum()),
            'train_neg': int(len(train_labels) - train_labels.sum()),
            'val_pos': int(val_labels.sum()),
            'val_neg': int(len(val_labels) - val_labels.sum())
        }

        return train_subset, val_subset, stats

    def _move_batch_to_device(self, batch):
        if batch is None:
            return None

        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)

        if isinstance(batch, dict):
            return {key: self._move_batch_to_device(value) for key, value in batch.items()}

        if isinstance(batch, list):
            return [self._move_batch_to_device(item) for item in batch]

        if isinstance(batch, tuple):
            return tuple(self._move_batch_to_device(item) for item in batch)

        return batch

    def _extract_logits(self, outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs

        if isinstance(outputs, dict):
            if 'logits' in outputs and isinstance(outputs['logits'], torch.Tensor):
                return outputs['logits']
            if 'predicted_outcome' in outputs and isinstance(outputs['predicted_outcome'], torch.Tensor):
                return outputs['predicted_outcome']
            if 'class_logits' in outputs and isinstance(outputs['class_logits'], torch.Tensor):
                return outputs['class_logits']
            for value in outputs.values():
                if isinstance(value, torch.Tensor):
                    return value
            raise ValueError("ModelOutput logits ")

        if isinstance(outputs, (list, tuple)):
            tensor_items = [item for item in outputs if isinstance(item, torch.Tensor)]
            if tensor_items:
                primary = tensor_items[0]
                # Special handling for multi-task outputs (class_logits, strength_logits)
                if (
                    len(tensor_items) >= 2
                    and primary.dim() == 2
                    and primary.size(-1) == 2
                    and tensor_items[1].dim() in (1, 2)
                ):
                    return primary
                return primary

            # If the tuple contains dicts, extract recursively
            dict_items = [item for item in outputs if isinstance(item, dict)]
            for item in dict_items:
                try:
                    return self._extract_logits(item)
                except ValueError:
                    continue

        raise ValueError(f" ModelOutput {type(outputs)} logits")

    def _collect_predictions(self, data_loader, apply_calibration=True, desc=None, collect_metadata=False):
        """Collect logits, probabilities, labels, and optional sample metadata for a given DataLoader."""
        self.model.eval()
        all_logits = []
        all_labels = []
        
        # Auxiliary-branch collection
        all_aux_seq_logits = []
        all_aux_struct_logits = []
        metadata_records = [] if collect_metadata else None

        iterator = data_loader
        if desc is not None:
            iterator = tqdm(data_loader, desc=desc)

        with torch.no_grad():
            for batch in iterator:
                batch = self._move_batch_to_device(batch)

                outputs = self.model(batch)
                logits = self._extract_logits(outputs)

                # Multi-task output [batch_size, 2]: use the first column as classification logits
                if logits.dim() == 2 and logits.size(-1) == 2:
                    # Multi-task output: [batch_size, 2] classification logits
                    logits = logits[:, 0]

                logits = logits.detach().cpu().numpy()
                label_tensor = batch['label'].detach().cpu()
                labels = label_tensor.numpy()
                batch_size = label_tensor.shape[0]

                all_logits.extend(logits.flatten())
                all_labels.extend(labels.flatten())
                if collect_metadata:
                    batch_metadata = self._extract_batch_metadata(batch, batch_size)
                    metadata_records.extend(batch_metadata)
                
                # Collect auxiliary-branch logits
                if isinstance(outputs, dict):
                    if 'aux_seq_logits' in outputs and outputs['aux_seq_logits'] is not None:
                        aux_seq = outputs['aux_seq_logits'].squeeze(-1).detach().cpu().numpy()
                        all_aux_seq_logits.extend(aux_seq.flatten())
                    
                    if 'aux_struct_logits' in outputs and outputs['aux_struct_logits'] is not None:
                        aux_struct = outputs['aux_struct_logits'].squeeze(-1).detach().cpu().numpy()
                        all_aux_struct_logits.extend(aux_struct.flatten())

        all_logits = np.array(all_logits)
        all_labels = np.array(all_labels)
        all_probabilities = self._apply_calibration(all_logits, apply_calibration)
        
        # Convert to NumPy arrays
        aux_results = {}
        if all_aux_seq_logits:
            aux_seq_logits = np.array(all_aux_seq_logits)
            aux_results['aux_seq_logits'] = aux_seq_logits
            aux_results['aux_seq_probs'] = self._sigmoid(aux_seq_logits)
            
        if all_aux_struct_logits:
            aux_struct_logits = np.array(all_aux_struct_logits)
            aux_results['aux_struct_logits'] = aux_struct_logits
            aux_results['aux_struct_probs'] = self._sigmoid(aux_struct_logits)

        return all_logits, all_probabilities, all_labels, aux_results, metadata_records

    def _extract_batch_metadata(self, batch: Dict[str, Any], batch_size: int) -> List[Dict[str, Any]]:
        """Extract required sample metadata from a batch for error reporting."""
        metadata = []
        uniprot_ids = self._coerce_batch_value(batch.get('uniprot_ids') or batch.get('uniprot_id'), batch_size)
        positions = self._coerce_batch_value(batch.get('positions') or batch.get('position'), batch_size)
        residues = self._coerce_batch_value(batch.get('residues') or batch.get('residue'), batch_size)
        ptm_types = self._coerce_batch_value(batch.get('ptm_types') or batch.get('ptm_type'), batch_size)

        for idx in range(batch_size):
            metadata.append({
                'uniprot_id': uniprot_ids[idx],
                'position': self._safe_cast_to_int(positions[idx]),
                'residue': residues[idx],
                'ptm_type': ptm_types[idx],
            })

        return metadata

    @staticmethod
    def _coerce_batch_value(value, batch_size: int) -> List[Any]:
        if value is None:
            return [None] * batch_size

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.ndim == 0:
                value = value.view(1)
            else:
                value = value.view(value.shape[0], -1) if value.ndim > 1 else value
            flat = value.view(-1).tolist()
        elif isinstance(value, np.ndarray):
            flat = value.reshape(-1).tolist()
        elif isinstance(value, (list, tuple)):
            flat = list(value)
        else:
            flat = [value] * batch_size

        if len(flat) < batch_size:
            flat.extend([None] * (batch_size - len(flat)))
        elif len(flat) > batch_size:
            flat = flat[:batch_size]

        return flat

    @staticmethod
    def _safe_cast_to_int(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return value

    def _apply_calibration(self, logits, apply_calibration=True):
        """Calibrate logits according to config and return probabilities."""
        logits = np.asarray(logits, dtype=np.float64)

        if apply_calibration and self.calibration_method == 'temperature' and self.temperature is not None:
            temp = max(self.temperature, 1e-6)
            scaled_logits = logits / temp
            probabilities = self._sigmoid(scaled_logits)
        elif apply_calibration and self.calibration_method == 'platt' and self.platt_model is not None:
            probabilities = self.platt_model.predict_proba(logits.reshape(-1, 1))[:, 1]
        else:
            probabilities = self._sigmoid(logits)

        return np.clip(probabilities, 1e-6, 1 - 1e-6)

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))

    def _fit_temperature_scaling(self, logits, labels):
        """Fit a temperature-scaling factor on the validation set."""
        logits_tensor = torch.tensor(logits, dtype=torch.float32, device=self.device).unsqueeze(1)
        labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device).unsqueeze(1)

        log_temperature = torch.zeros(1, requires_grad=True, device=self.device)
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature)
            scaled_logits = logits_tensor / temperature
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = float(torch.exp(log_temperature).detach().cpu().item())
        return max(temperature, 1e-3)

    def _fit_platt_scaling(self, logits, labels):
        """Fit a Platt scaling model."""
        if len(np.unique(labels)) < 2:
            raise ValueError("Platt Validation Samples")
        model = LogisticRegression(solver='lbfgs')
        model.fit(logits.reshape(-1, 1), labels.astype(int))
        return model

    def _tune_threshold_and_calibration(self):
        """Tune calibration parameters and decision threshold on the validation set."""
        need_calibration = self.calibration_method in {'temperature', 'platt'}
        if not (self.optimize_threshold or need_calibration):
            self.logger.info("SkipThreshold Calibration ")
            return

        self.logger.info("Start Validation Threshold CalibrationParameters")
        val_logits, val_probs, val_labels, _, _ = self._collect_predictions(self.val_loader, apply_calibration=False)

        # Calibration
        if self.calibration_method == 'temperature':
            try:
                self.temperature = self._fit_temperature_scaling(val_logits, val_labels)
                self.logger.info(f" Completed,: {self.temperature:.4f}")
                val_probs = self._apply_calibration(val_logits, apply_calibration=True)
            except Exception as e:
                self.logger.warning(f" Failed, CalibrationProbabilities: {e}")
                self.temperature = 1.0
                val_probs = self._apply_calibration(val_logits, apply_calibration=False)
        elif self.calibration_method == 'platt':
            try:
                self.platt_model = self._fit_platt_scaling(val_logits, val_labels)
                self.logger.info("Platt Completed")
                val_probs = self._apply_calibration(val_logits, apply_calibration=True)
            except Exception as e:
                self.logger.warning(f"Platt Failed, CalibrationProbabilities: {e}")
                self.platt_model = None
                val_probs = self._apply_calibration(val_logits, apply_calibration=False)
        else:
            # No calibration
            val_probs = self._apply_calibration(val_logits, apply_calibration=False)

        # Threshold tuning
        if self.optimize_threshold:
            try:
                best_thresh, best_score = calculate_optimal_threshold(
                    val_labels,
                    val_probs,
                    metric=self.threshold_metric,
                    min_precision=self.config.get('threshold_min_precision', 0.0),
                    min_specificity=self.config.get('threshold_min_specificity', 0.0),
                    min_recall=self.config.get('threshold_min_recall', 0.0)
                )
                self.decision_threshold = float(best_thresh)
                self.logger.info(f"Threshold Completed: metric={self.threshold_metric}, threshold={best_thresh:.4f}, score={best_score:.4f}")
            except Exception as e:
                self.logger.warning(f"Threshold Failed,Use 0.5: {e}")
                self.decision_threshold = 0.5

        # Record validation performance
        val_predictions = (val_probs >= self.decision_threshold).astype(int)
        val_metrics = calculate_metrics(
            y_true=val_labels,
            y_pred=val_predictions,
            y_prob=val_probs,
            threshold=self.decision_threshold,
            prefix='val_post_' if not any(k.startswith('val_post_') for k in self.training_history.keys()) else 'val_post_'
        )

        for key, value in val_metrics.items():
            self.training_history.setdefault(key, []).append(float(value))

        self.training_history['decision_threshold'] = self.decision_threshold
        self.training_history['calibration_method'] = self.calibration_method
        self.training_history['temperature'] = self.temperature

        self.logger.info("Validation:")
        for key, value in val_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")

        # Save training history with threshold-tuning results
        self._save_training_history()

    def create_model(self):
        """Create the model."""
        self.logger.info("="*80)
        self.logger.info("StartModelCreate ")
        self.logger.info("="*80)

        # Read advanced-module options from config
        rl_config = self.config.get('reinforcement_learning', {})
        moe_config = self.config.get('mixture_of_experts', {})

        # Debug: print core_model_type from config
        core_model_type = self.config.get('core_model_type', 'bert_bilstm')
        self.logger.info(f"[DEBUG] Config core_model_type: {core_model_type}")
        self.logger.info(f"[DEBUG] Config: {self.config}")

        if core_model_type == 'dual_branch_fusion':
            self.model = DualBranchFusionPredictor(config=self.config)
        elif core_model_type == 'cnn_dual_stream':
            self.model = CNNDualStreamPredictor(config=self.config)
        else:
            # Create model using AcetylationPredictor (which uses models from core_models.py).
            # Use precomputed features and skip ESM encoder loading.
            self.model = AcetylationPredictor(
                config=self.config,
                seq_encoder_type=self.config.get('seq_encoder_type', 'esm2_t33_650M_UR50D'),
                struct_encoder_type=self.config.get('struct_encoder_type', 'esm_if1_gvp4_t16_142M_UR50'),
                core_model_type=core_model_type,
                use_moe=self.config.get('use_moe', False),
                use_reinforcement=self.config.get('use_reinforcement', False),
                use_interpretability=self.config.get('use_interpretability', True),
                use_precomputed_features=True  # Use precomputed features and skip encoder loading
            )

        use_data_parallel = self.device.type == 'cuda' and len(self.parallel_gpu_ids) > 1
        if use_data_parallel:
            gpu_list_str = ', '.join(f"cuda:{idx}" for idx in self.parallel_gpu_ids)
            self.logger.info(f"EnableData,GPU: {gpu_list_str}")
            self.model = nn.DataParallel(
                self.model,
                device_ids=self.parallel_gpu_ids,
                output_device=self.parallel_gpu_ids[0],
            )

        # Move model to device
        self.model = self.model.to(self.device)

        # Log model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"ModelCreateCompleted:")
        self.logger.info(f" Parameters: {total_params:,}")
        self.logger.info(f" TrainingParameters: {trainable_params:,}")
        self.logger.info(f"  Device: {self.device}")

        # Log advanced-module configuration
        self.logger.info(f" Config:")

        self.logger.info(f": {self.config.get('use_reinforcement', False)}")
        if self.config.get('use_reinforcement', False):
            self.logger.info(f" -: {rl_config.get('reward_type', 'performance_based')}")
            self.logger.info(f" - Probabilities: {rl_config.get('exploration_epsilon', 0.1)}")
            self.logger.info(f" -: {rl_config.get('policy_learning_rate', 1e-5)}")
            self.logger.info(f" -: {rl_config.get('discount_factor', 0.95)}")

        self.logger.info(f": {self.config.get('use_moe', False)}")
        if self.config.get('use_moe', False):
            self.logger.info(f" -: {moe_config.get('num_experts', 8)}")
            self.logger.info(f"    - Top-K: {moe_config.get('top_k', 2)}")
            self.logger.info(f" -: {moe_config.get('routing_method', 'top_k')}")

        self.logger.info(f": {self.config.get('integration_method', 'serial')}")

        return True
    
    def setup_training(self):
        """Set up training components."""
        self.logger.info("="*80)
        self.logger.info(" Training ")
        self.logger.info("="*80)
        
        # Create optimizer
        optimizer = create_optimizer(self.model, self.config)

        # Create learning-rate scheduler
        scheduler, scheduler_type = create_scheduler(
            optimizer,
            self.config,
            train_loader=self.train_loader,
            grad_accum_steps=self.config.get('grad_accum_steps', 1)
        )

        if scheduler_type and scheduler is not None:
            self.logger.info(f"Use: {scheduler_type}")
        elif scheduler_type:
            self.logger.info(f" Config {scheduler_type}, Create ")
        else:
            self.logger.info(" Config Use ")
        self.config['scheduler_normalized'] = scheduler_type
        
        # --- MODIFIED: Use RLEnhancedTrainingManager if enabled ---
        if self.config.get('enable_rl_controller', False):
            self.logger.info("Enable Parameters (RL Controller Enabled)")
            self.training_manager = RLEnhancedTrainingManager(
                model=self.model,
                config=self.config,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device,
                output_dir=str(self.output_dir)
            )
        else:
            # Create training manager
            self.training_manager = EnhancedTrainingManager(
                model=self.model,
                config=self.config,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device,
                output_dir=str(self.output_dir)
            )
        # ----------------------------------------------------------
        
        # Create visualization manager
        if HAS_VISUALIZATION:
            self.viz_manager = EnhancedVisualizationManager(str(self.output_dir))
        else:
            self.viz_manager = None
            self.logger.warning(" Unavailable,Skip Create")
        
        loader_cfg = self._get_current_loader_config()
        self.logger.info(
            "TrainingDataLoad: batch_size=%s, num_workers=%s, pin_memory=%s, prefetch_factor=%s, persistent_workers=%s",
            self._loader_base_kwargs.get('batch_size'),
            loader_cfg.get('num_workers'),
            loader_cfg.get('pin_memory'),
            loader_cfg.get('prefetch_factor', 'None'),
            loader_cfg.get('persistent_workers', False),
        )

        self.logger.info("Training Completed")
        return True

    def train(self):
        """Run the training loop."""
        self.logger.info("="*80)
        self.logger.info("StartTraining ")
        self.logger.info("="*80)

        fallback_attempts = 0
        max_fallbacks = max(0, len(self._loader_configs) - 1)

        while True:
            try:
                # Train
                training_results = self.training_manager.train()

                # Update training history
                if 'history' in training_results:
                    self.training_history.update(training_results['history'])
                else:
                    # If no history field exists, use the returned results directly
                    self.training_history.update(training_results)

                # Generate visualizations
                if self.viz_manager is not None:
                    self.viz_manager.create_training_plots(self.training_history)
                else:
                    self.logger.info("Skip (Unavailable)")

                # Save training history
                self._save_training_history()

                return True

            except (RuntimeError, OSError) as e:
                if (
                    fallback_attempts < max_fallbacks
                    and self._should_retry_dataloader_error(e)
                    and self._advance_dataloader_config_for_retry()
                ):
                    fallback_attempts += 1
                    self.logger.warning(
                        " DataLoader, Use Config (%d/%d )",
                        fallback_attempts,
                        max_fallbacks,
                    )

                    self._cleanup_training_manager()

                    if not self.setup_training():
                        self.logger.error("DataLoader Training Failed")
                        break

                    continue

                self.logger.error(f"Training Error: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                return False

            except Exception as e:
                self.logger.error(f"Training Error: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                return False

            break

        self.logger.error(" DataLoader Config,TrainingFailed")
        return False

    def evaluate(self):
        """Evaluate model performance on the configured test split."""
        self.logger.info("=" * 80)
        self.logger.info("Starting model evaluation")
        self.logger.info("=" * 80)

        try:
            # Evaluate on the test set
            test_metrics = self._evaluate_on_dataset(self.test_loader, "Test Set")
            self.last_test_metrics = test_metrics

            # Save evaluation results
            eval_results = {
                'test_metrics': test_metrics,
                'model_config': {
                    'use_reinforcement': self.config.get('use_reinforcement', False),
                    'use_moe': self.config.get('use_moe', False),
                    'integration_method': self.config.get('integration_method', 'serial')
                },
                'post_processing': {
                    'calibration_method': self.calibration_method,
                    'temperature': self.temperature if self.calibration_method == 'temperature' else None,
                    'platt_coefficients': self.platt_model.coef_.tolist() if self.calibration_method == 'platt' and self.platt_model is not None else None,
                    'platt_intercept': self.platt_model.intercept_.tolist() if self.calibration_method == 'platt' and self.platt_model is not None else None,
                    'decision_threshold': self.decision_threshold
                }
            }

            eval_path = self.output_dir / 'evaluation_results.json'
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2)

            self.logger.info(f"Evaluation results saved to: {eval_path}")

            return test_metrics

        except Exception as e:
            self.logger.error(f"Error occurred during evaluation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.last_test_metrics = None
            return None

    @staticmethod
    def _make_ascii_slug(text: str, default: str = 'dataset') -> str:
        """Convert arbitrary text into an ASCII-safe slug."""
        base = str(text or '').strip().lower()
        if not base:
            base = default

        slug = ''.join(ch if ch.isascii() and ch.isalnum() else '_' for ch in base)
        while '__' in slug:
            slug = slug.replace('__', '_')
        slug = slug.strip('_')
        if not slug:
            slug = default
        return slug

    @classmethod
    def _sanitize_metrics_prefix(cls, dataset_name: str) -> str:
        """Return a metrics prefix ending with underscore using ASCII characters only."""
        slug = cls._make_ascii_slug(dataset_name, default='dataset')
        if not slug.endswith('_'):
            slug = f"{slug}_"
        return slug

    def _plot_roc_pr_curves(self, labels, probabilities, dataset_name, aux_seq_probs=None, aux_struct_probs=None):
        """Create polished ROC and PR curve visualizations and save them to disk."""
        if not HAS_MATPLOTLIB:
            self.logger.warning("Matplotlib is not available; skipping ROC/PR plotting")
            return None

        if labels is None or len(labels) == 0:
            self.logger.warning("No labels provided for plotting; skipping ROC/PR plotting")
            return None

        labels_np = np.asarray(labels, dtype=np.int32)
        probs_np = np.asarray(probabilities, dtype=np.float64)

        if len(np.unique(labels_np)) < 2:
            self.logger.warning("ROC/PR plotting skipped because labels contain a single class")
            return None

        # Calculate main metrics
        fpr, tpr, _ = roc_curve(labels_np, probs_np)
        precision, recall, _ = precision_recall_curve(labels_np, probs_np)
        roc_auc = auc(fpr, tpr) if len(fpr) > 1 else float("nan")
        pr_auc = auc(recall, precision) if len(recall) > 1 else float("nan")

        # Calculate sequence branch metrics if available
        seq_fpr, seq_tpr, seq_roc_auc = None, None, None
        seq_precision, seq_recall, seq_pr_auc = None, None, None
        if aux_seq_probs is not None:
            seq_probs_np = np.asarray(aux_seq_probs, dtype=np.float64)
            seq_fpr, seq_tpr, _ = roc_curve(labels_np, seq_probs_np)
            seq_roc_auc = auc(seq_fpr, seq_tpr) if len(seq_fpr) > 1 else float("nan")
            seq_precision, seq_recall, _ = precision_recall_curve(labels_np, seq_probs_np)
            seq_pr_auc = auc(seq_recall, seq_precision) if len(seq_recall) > 1 else float("nan")

        # Calculate structure branch metrics if available
        struct_fpr, struct_tpr, struct_roc_auc = None, None, None
        struct_precision, struct_recall, struct_pr_auc = None, None, None
        if aux_struct_probs is not None:
            struct_probs_np = np.asarray(aux_struct_probs, dtype=np.float64)
            struct_fpr, struct_tpr, _ = roc_curve(labels_np, struct_probs_np)
            struct_roc_auc = auc(struct_fpr, struct_tpr) if len(struct_fpr) > 1 else float("nan")
            struct_precision, struct_recall, _ = precision_recall_curve(labels_np, struct_probs_np)
            struct_pr_auc = auc(struct_recall, struct_precision) if len(struct_recall) > 1 else float("nan")

        slug_ascii = self._make_ascii_slug(dataset_name, default='dataset')
        roc_path = self.output_dir / f"{slug_ascii}_roc_curve.png"
        pr_path = self.output_dir / f"{slug_ascii}_pr_curve.png"

        display_name = dataset_name or 'Dataset'

        # ROC curve
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(fpr, tpr, color='#2a9d8f', lw=2.5, label=f'Combined (AUC = {roc_auc:.4f})')
        
        if seq_fpr is not None:
            ax.plot(seq_fpr, seq_tpr, color='#e76f51', lw=2.0, linestyle='--', label=f'Sequence (AUC = {seq_roc_auc:.4f})')
            
        if struct_fpr is not None:
            ax.plot(struct_fpr, struct_tpr, color='#264653', lw=2.0, linestyle=':', label=f'Structure (AUC = {struct_roc_auc:.4f})')

        ax.plot([0, 1], [0, 1], linestyle='--', color='#8d99ae', lw=1.5, label='Chance Level')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25)
        ax.set_facecolor('#f8f9fb')
        ax.legend(frameon=False, fontsize=11, loc='lower right')
        fig.tight_layout()
        fig.savefig(roc_path, dpi=320)
        plt.close(fig)

        # Precision-Recall curve
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(recall, precision, color='#264653', lw=2.5, label=f'Combined (AUC = {pr_auc:.4f})')
        
        if seq_precision is not None:
            ax.plot(seq_recall, seq_precision, color='#e76f51', lw=2.0, linestyle='--', label=f'Sequence (AUC = {seq_pr_auc:.4f})')
            
        if struct_precision is not None:
            ax.plot(struct_recall, struct_precision, color='#2a9d8f', lw=2.0, linestyle=':', label=f'Structure (AUC = {struct_pr_auc:.4f})')

        baseline = labels_np.mean()
        ax.hlines(baseline, 0, 1, colors='#8d99ae', linestyles='--', lw=1.5, label=f'Baseline = {baseline:.4f}')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25)
        ax.set_facecolor('#f8f9fb')
        ax.legend(frameon=False, fontsize=11, loc='lower left')
        fig.tight_layout()
        fig.savefig(pr_path, dpi=320)
        plt.close(fig)

        self.logger.info(f"ROC curve saved to: {roc_path}")
        self.logger.info(f"Precision-Recall curve saved to: {pr_path}")

        return {
            'roc_curve_path': str(roc_path),
            'pr_curve_path': str(pr_path)
        }

    def _evaluate_on_dataset(self, data_loader, dataset_name):
        """Evaluate the model on a specific data loader and return metric values."""
        dataset_flag = (dataset_name or '').lower()
        collect_metadata = 'test' in dataset_flag
        logits, probabilities, labels, aux_results, sample_metadata = self._collect_predictions(
            data_loader,
            apply_calibration=True,
            desc=f" {dataset_name}",
            collect_metadata=collect_metadata,
        )
        predictions = (probabilities >= self.decision_threshold).astype(int)

        metrics_prefix = self._sanitize_metrics_prefix(dataset_name)
        metrics = calculate_metrics(
            y_true=labels,
            y_pred=predictions,
            y_prob=probabilities,
            threshold=self.decision_threshold,
            prefix=metrics_prefix
        )

        self.logger.info(f"Evaluation metrics for {dataset_name}:")
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"  {metric_name}: {metric_value:.4f}")
            
        # Evaluate auxiliary branches
        if 'aux_seq_probs' in aux_results:
            self.logger.info(f"--- {dataset_name} ---")
            seq_probs = aux_results['aux_seq_probs']
            seq_preds = (seq_probs >= 0.5).astype(int)  # Default threshold: 0.5
            seq_metrics = calculate_metrics(
                y_true=labels,
                y_pred=seq_preds,
                y_prob=seq_probs,
                threshold=0.5,
                prefix=f"{metrics_prefix}seq_"
            )
            for k, v in seq_metrics.items():
                if 'mcc' in k or 'auc' in k or 'acc' in k:
                    self.logger.info(f"  {k}: {v:.4f}")
            metrics.update(seq_metrics)

        if 'aux_struct_probs' in aux_results:
            self.logger.info(f"--- {dataset_name} ---")
            struct_probs = aux_results['aux_struct_probs']
            struct_preds = (struct_probs >= 0.5).astype(int)  # Default threshold: 0.5
            struct_metrics = calculate_metrics(
                y_true=labels,
                y_pred=struct_preds,
                y_prob=struct_probs,
                threshold=0.5,
                prefix=f"{metrics_prefix}struct_"
            )
            for k, v in struct_metrics.items():
                if 'mcc' in k or 'auc' in k or 'acc' in k:
                    self.logger.info(f"  {k}: {v:.4f}")
            metrics.update(struct_metrics)

        if 'test' in dataset_flag:
            # Extract aux probs for plotting
            aux_seq_probs = aux_results.get('aux_seq_probs')
            aux_struct_probs = aux_results.get('aux_struct_probs')
            
            plot_info = self._plot_roc_pr_curves(
                labels, 
                probabilities, 
                dataset_name,
                aux_seq_probs=aux_seq_probs,
                aux_struct_probs=aux_struct_probs
            )
            if plot_info is not None:
                metrics.update({
                    f"{metrics_prefix}roc_curve_path": plot_info['roc_curve_path'],
                    f"{metrics_prefix}pr_curve_path": plot_info['pr_curve_path']
                })

            # Export all predictions
            all_preds_path = self._export_all_predictions(
                dataset_name,
                logits,
                probabilities,
                labels,
                predictions,
                sample_metadata,
                aux_results=aux_results
            )
            if all_preds_path is not None:
                metrics[f"{metrics_prefix}all_predictions_csv"] = all_preds_path

            error_csv_path = self._export_prediction_errors(
                dataset_name,
                logits,
                probabilities,
                labels,
                predictions,
                sample_metadata,
                aux_results=aux_results
            )
            if error_csv_path is not None:
                metrics[f"{metrics_prefix}error_cases_csv"] = error_csv_path

        return metrics

    def _export_all_predictions(self, dataset_name, logits, probabilities, labels, predictions, metadata, aux_results=None):
        """Export all predictions including branch probabilities to CSV."""
        if not metadata:
            return None

        logits_arr = np.asarray(logits).flatten()
        probs_arr = np.asarray(probabilities).flatten()
        labels_arr = np.asarray(labels).flatten()
        preds_arr = np.asarray(predictions).flatten()
        sample_count = len(labels_arr)

        # Extract aux probs
        seq_probs = None
        struct_probs = None
        if aux_results:
            if 'aux_seq_probs' in aux_results:
                seq_probs = np.asarray(aux_results['aux_seq_probs']).flatten()
            if 'aux_struct_probs' in aux_results:
                struct_probs = np.asarray(aux_results['aux_struct_probs']).flatten()

        rows = []
        for idx in range(sample_count):
            sample_meta = metadata[idx] or {}
            row = {
                'dataset': dataset_name,
                'uniprot_id': sample_meta.get('uniprot_id'),
                'position': sample_meta.get('position'),
                'residue': sample_meta.get('residue'),
                'ptm_type': sample_meta.get('ptm_type'),
                'probability': float(probs_arr[idx]),
                'logit': float(logits_arr[idx]),
                'true_label': int(round(labels_arr[idx])),
                'predicted_label': int(round(preds_arr[idx])),
            }
            
            if seq_probs is not None and idx < len(seq_probs):
                row['seq_probability'] = float(seq_probs[idx])
            
            if struct_probs is not None and idx < len(struct_probs):
                row['struct_probability'] = float(struct_probs[idx])
                
            rows.append(row)

        slug_ascii = self._make_ascii_slug(dataset_name, default='dataset')
        output_path = self.output_dir / f"{slug_ascii}_all_predictions.csv"
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        self.logger.info(f"All predictions saved to: {output_path}")
        return str(output_path)

    def _export_prediction_errors(self, dataset_name, logits, probabilities, labels, predictions, metadata, aux_results=None):
        """Export false-positive/false-negative samples from the test set to a CSV file."""
        if not metadata:
            self.logger.warning(" / Samples: DataInfo")
            return None

        logits_arr = np.asarray(logits).flatten()
        probs_arr = np.asarray(probabilities).flatten()
        labels_arr = np.asarray(labels).flatten()
        preds_arr = np.asarray(predictions).flatten()
        sample_count = len(labels_arr)

        # Extract aux probs
        seq_probs = None
        struct_probs = None
        if aux_results:
            if 'aux_seq_probs' in aux_results:
                seq_probs = np.asarray(aux_results['aux_seq_probs']).flatten()
            if 'aux_struct_probs' in aux_results:
                struct_probs = np.asarray(aux_results['aux_struct_probs']).flatten()

        if len(metadata) != sample_count:
            self.logger.warning(
                " / Samples: Data (%s) (%s) ",
                len(metadata),
                sample_count,
            )
            return None

        error_rows = []
        for idx in range(sample_count):
            true_label = int(round(labels_arr[idx]))
            pred_label = int(round(preds_arr[idx]))

            if true_label == pred_label:
                continue

            error_type = 'false_positive' if pred_label == 1 else 'false_negative'
            sample_meta = metadata[idx] or {}

            row = {
                'dataset': dataset_name,
                'error_type': error_type,
                'uniprot_id': sample_meta.get('uniprot_id'),
                'position': sample_meta.get('position'),
                'residue': sample_meta.get('residue'),
                'ptm_type': sample_meta.get('ptm_type'),
                'probability': float(probs_arr[idx]),
                'logit': float(logits_arr[idx]),
                'true_label': true_label,
                'predicted_label': pred_label,
                'decision_threshold': float(self.decision_threshold),
            }

            if seq_probs is not None and idx < len(seq_probs):
                row['seq_probability'] = float(seq_probs[idx])
            
            if struct_probs is not None and idx < len(struct_probs):
                row['struct_probability'] = float(struct_probs[idx])

            error_rows.append(row)

        slug_ascii = self._make_ascii_slug(dataset_name, default='dataset')
        errors_path = self.output_dir / f"{slug_ascii}_prediction_errors.csv"
        # columns = [
        #     'dataset',
        #     'error_type',
        #     'uniprot_id',
        #     'position',
        #     'residue',
        #     'ptm_type',
        #     'probability',
        #     'logit',
        #     'true_label',
        #     'predicted_label',
        #     'decision_threshold',
        # ]
        # error_df = pd.DataFrame(error_rows, columns=columns)
        error_df = pd.DataFrame(error_rows)
        error_df.to_csv(errors_path, index=False)
        self.logger.info(
            "Test / Samples Save: %s (: %s)",
            errors_path,
            len(error_rows),
        )
        return str(errors_path)

    def _run_kfold_validation(self):
        """Run K-fold cross-validation and record variance."""
        if self.full_train_dataset is None or len(self.full_train_dataset) == 0:
            self.logger.warning("K-fold validation requires a training dataset; skipping.")
            return None

        if not self.config.get('use_kfold', False):
            self.logger.info("K-fold validation is disabled in the configuration; skipping.")
            return None

        self.logger.info("=" * 80)
        self.logger.info("K-fold cross-validation")
        self.logger.info("=" * 80)

        try:
            kfold_config = copy.deepcopy(self.config)
            core_for_kfold = kfold_config.get('core_model_type', self.config.get('core_model_type', 'ptm_bert_bilstm'))
            if core_for_kfold == 'dual_branch_fusion':
                kfold_config.setdefault('use_structure', True)
                kfold_config.setdefault('use_precomputed_features', True)
                kfold_config.setdefault('seq_feature_dim', kfold_config.get('seq_dim', 1280))
                kfold_config.setdefault('struct_feature_dim', kfold_config.get('struct_dim', 512))

            data_manager = TrainerKFoldDataManager(self.full_train_dataset, kfold_config, self.args)
            kfold_validator = KFoldValidator(kfold_config, self.device, str(self.output_dir))
            results = kfold_validator.run_kfold_validation(data_manager)

            if results and 'error' not in results:
                f1_mean = results.get('best_val_f1_mean')
                f1_std = results.get('best_val_f1_std')
                acc_mean = results.get('best_val_acc_mean')
                acc_std = results.get('best_val_acc_std')

                self.logger.info("K-fold cross-validation completed:")
                if f1_mean is not None:
                    self.logger.info(f"  Mean validation F1: {f1_mean:.4f} ± {f1_std:.4f}")
                if acc_mean is not None:
                    self.logger.info(f"  Mean validation accuracy: {acc_mean:.4f} ± {acc_std:.4f}")

                self.training_history['kfold_results'] = {
                    'best_val_f1_mean': f1_mean,
                    'best_val_f1_std': f1_std,
                    'best_val_acc_mean': acc_mean,
                    'best_val_acc_std': acc_std,
                    'kfold_config': {
                        'splits': self.config.get('kfold_splits', 5),
                        'stratified': self.config.get('kfold_stratified', True)
                    }
                }

                # Run ensemble evaluation on the held-out test set
                if self.test_loader is not None:
                    self.logger.info("=" * 40)
                    self.logger.info("K-fold ensemble test evaluation")
                    self.logger.info("=" * 40)
                    ensemble_results = kfold_validator.evaluate_ensemble_on_test_set(self.test_loader)
                    if ensemble_results:
                        self.training_history['kfold_ensemble_results'] = ensemble_results
                        # Record key metrics
                        acc = ensemble_results.get('ensemble_test_accuracy') or ensemble_results.get('ensemble_test_acc')
                        f1 = ensemble_results.get('ensemble_test_f1')
                        auc = ensemble_results.get('ensemble_test_roc_auc') or ensemble_results.get('ensemble_test_auc')
                        
                        # Guard against None metrics to avoid formatting errors
                        acc_val = acc if acc is not None else 0.0
                        f1_val = f1 if f1 is not None else 0.0
                        auc_val = auc if auc is not None else 0.0
                        
                        self.logger.info(f"Ensemble test metrics: ACC={acc_val:.4f}, F1={f1_val:.4f}, AUC={auc_val:.4f}")
                else:
                    self.logger.warning("Test dataloader unavailable; skipping ensemble test evaluation.")

            else:
                self.logger.warning(f"K-fold validation produced no valid results: {results}")

            return results

        except Exception as e:
            self.logger.error(f"K-fold validation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def run(self):
        """Run the full training and evaluation workflow."""
        try:
            # 1. Prepare data
            if not self.prepare_data():
                self.logger.error("Data Failed")
                return False

            # 2. Create model
            if not self.create_model():
                self.logger.error("ModelCreateFailed")
                return False

            # 3. Set up training components
            if not self.setup_training():
                self.logger.error("Training Failed")
                return False

            # 4. Run training
            if not self.train():
                self.logger.error("TrainingFailed")
                return False

            # 4.5 Tune validation threshold and calibration
            self._tune_threshold_and_calibration()

            # 5. Evaluate model
            test_metrics = self.evaluate()
            if test_metrics is None:
                self.logger.error(" Failed")
                return False

            # 6. Optional K-Fold validation
            if self.config.get('use_kfold', False):
                self._run_kfold_validation()

            self.logger.info("="*80)
            self.logger.info("Training Completed!")
            self.logger.info("="*80)

            return True

        except Exception as e:
            self.logger.error(f" Error: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False


def parse_arguments():
    """Parse CLI arguments and record explicitly provided options."""
    parser = argparse.ArgumentParser(description='Enhanced PTMs Prediction Training Script')

    # Basic arguments
    parser.add_argument('--data_path', type=str,
                       default='phosphorylation_data_small_batch/acetylation_data_3000.csv',
                       help='Data file path (deprecated; use --train_data_path and --test_data_path).')
    parser.add_argument('--train_data_path', type=str,
                       default='phosphorylation_data_small_batch/train_data.csv',
                       help='Training data CSV path.')
    parser.add_argument('--test_data_path', type=str,
                       default='phosphorylation_data_small_batch/test_data.csv',
                       help='Test data CSV path.')
    parser.add_argument('--features_path', type=str,
                       default='phosphorylation_data_small_batch/features/esm_features.h5',
                       help='Feature H5 path (includes sequence and structure features).')
    parser.add_argument('--target_ptm_type', type=str, default='phosphorylation',
                       help='Target PTM type (e.g., phosphorylation, s-nitrosylation, succinylation, s-palmitoylation, sumoylation, acetylation, methylation, ubiquitination, o_linked_glycosylation, n_linked_glycosylation).')

    # Model architecture
    parser.add_argument('--seq_encoder_type', type=str, default='esm2_t33_650M_UR50D',
                       help='Sequence encoder type.')
    parser.add_argument('--core_model_type', type=str, default='dual_branch_fusion',
                       choices=['transformer', 'bert_bilstm', 'ptm_bert_bilstm', 'acetylation_transformer', 'dual_branch_fusion', 'cnn_dual_stream'],
                       help='Core model type.')
    parser.add_argument('--transformer_layers', type=int, default=None,
                       help='Number of Transformer layers (for bert_bilstm/ptm_bert_bilstm).')
    parser.add_argument('--lstm_hidden_dim', type=int, default=None,
                       help='LSTM hidden dimension (for bert_bilstm/ptm_bert_bilstm).')
    parser.add_argument('--output_dir', type=str, default='',
                       help='Output directory.')
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='GPU IDs for training, e.g. "0" or "0,1"; use "cpu" for CPU-only; use "auto" for all visible GPUs.')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to a config file.')

    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.4, help='Weight decay.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader worker processes.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--protein_cache_dir', type=str, default=None, help='Protein feature cache directory (shared across runs).')
    parser.add_argument('--window_size', type=int, default=61, help='Global window size.')
    parser.add_argument('--local_window_size', type=int, default=31, help='Local window size.')
    parser.add_argument('--window_size_candidates', type=int, nargs='+', help='Candidate global window sizes for grid search.')
    parser.add_argument('--local_window_size_candidates', type=int, nargs='+', help='Candidate local window sizes for grid search.')

    # Debugging and diagnostics
    parser.add_argument('--enable_nan_debugging', action='store_true', help='Enable NaN debugging hooks to record tensor ranges during forward passes.')
    parser.add_argument('--nan_debug_targets', type=str, nargs='*', default=None,
                       help='Module/class names to monitor (substring match supported); use "all" to monitor everything.')
    parser.add_argument('--nan_debug_frequency', type=int, default=50,
                       help='How often to record finite snapshots (effective only when log-finite is enabled).')
    parser.add_argument('--nan_debug_log_finite', action='store_true', help='Record tensor range snapshots even when there is no NaN/Inf.')
    parser.add_argument('--nan_debug_save_dir', type=str, default=None, help='Output directory for NaN-debug artifacts (tensors/logs).')
    parser.add_argument('--nan_debug_max_saved_tensors', type=int, default=10, help='Maximum number of anomalous tensors to save.')
    parser.add_argument('--nan_debug_break_on_error', action='store_true', help='Raise immediately when NaN/Inf is detected.')
    parser.add_argument('--nan_debug_check_batches', action='store_true', help='Check batch inputs for NaN/Inf before forward.')
    parser.add_argument('--nan_debug_batch_frequency', type=int, default=1, help='Batch check frequency (default: every batch).')
    parser.add_argument('--nan_debug_max_tensors_per_call', type=int, default=6,
                       help='Maximum tensors recorded per forward call (prevents log explosion).')

    # Data loading optimization
    parser.add_argument('--use_optimized_loading', action='store_true', default=True,
                       help='Use the optimized DataLoader pipeline (enabled by default).')
    parser.add_argument('--disable_optimized_loading', action='store_true',
                       help='Disable the optimized DataLoader pipeline and use the legacy path.')

    # Advanced module toggles
    parser.add_argument('--use_reinforcement', action='store_true',
                       help='Enable reinforcement learning module.')
    parser.add_argument('--enable_rl_controller', action='store_true',
                       help='Enable the RL controller for hyperparameter control.')
    parser.add_argument('--use_moe', action='store_true',
                       help='Enable mixture-of-experts module.')
    parser.add_argument('--integration_method', type=str, default='serial',
                       choices=['serial', 'ensemble', 'dynamic'],
                       help='Module integration method.')

    # Reinforcement learning
    parser.add_argument('--rl_reward_type', type=str, default='performance_based',
                       choices=['performance_based', 'confidence_based', 'hybrid'],
                       help='Reward type.')
    parser.add_argument('--rl_exploration_epsilon', type=float, default=0.1,
                       help='Exploration probability (epsilon).')
    parser.add_argument('--rl_policy_lr', type=float, default=1e-5,
                       help='Policy network learning rate.')

    # Mixture of experts
    parser.add_argument('--moe_num_experts', type=int, default=8,
                       help='Number of experts.')
    parser.add_argument('--moe_top_k', type=int, default=2,
                       help='Top-K experts to route to.')

    # Early stopping
    parser.add_argument('--use_early_stopping', action='store_true',
                       help='Enable early stopping.')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='Patience.')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0001,
                       help='Minimum improvement threshold.')
    parser.add_argument('--early_stopping_monitor', type=str, default='val_f1',
                       choices=['val_loss', 'val_acc', 'val_f1', 'val_auc'],
                       help='Metric to monitor.')
    parser.add_argument('--early_stopping_mode', type=str, default='max',
                       choices=['min', 'max'],
                       help='Mode: "min" means lower is better; "max" means higher is better.')
    parser.add_argument('--restore_best_weights', action='store_true', default=True,
                       help='Restore best weights when early stopping triggers.')

    # Dropout
    parser.add_argument('--adaptive_dropout', action='store_true', default=True,
                       help='Enable adaptive dropout.')
    parser.add_argument('--dropout_schedule', type=str, default='adaptive',
                       choices=['linear', 'exponential', 'cosine', 'adaptive'],
                       help='Dropout schedule.')
    parser.add_argument('--input_dropout', type=float, default=0.2,
                       help='Input dropout rate.')
    parser.add_argument('--hidden_dropout', type=float, default=0.3,
                       help='Hidden-layer dropout rate.')
    parser.add_argument('--output_dropout', type=float, default=0.1,
                       help='Output dropout rate.')

    # Dual-branch fusion
    parser.add_argument('--branch_hidden_dim', type=int, default=256,
                       help='Hidden dimension for sequence/structure branches.')
    parser.add_argument('--fusion_hidden_dim', type=int, default=384,
                       help='Hidden dimension for the fusion layer.')
    parser.add_argument('--fusion_hidden_dim_candidates', type=int, nargs='+',
                       help='Candidate fusion hidden dimensions to compare.')
    parser.add_argument('--seq_transformer_layers', type=int, default=2,
                       help='Number of Transformer layers in the sequence branch.')
    parser.add_argument('--struct_transformer_layers', type=int, default=2,
                       help='Number of Transformer layers in the structure branch.')
    parser.add_argument('--cross_attention_heads', type=int, default=4,
                       help='Number of cross-modal attention heads.')
    parser.add_argument('--alignment_weight', type=float, default=0.1,
                       help='Sequence-structure alignment loss weight.')
    parser.add_argument('--contrastive_weight', type=float, default=0.05,
                       help='Contrastive loss weight for sequence representations.')
    parser.add_argument('--aux_cls_weight', type=float, default=0.5,
                       help='Auxiliary classifier loss weight.')
    parser.add_argument('--contrastive_temperature', type=float, default=0.07,
                       help='Temperature for contrastive loss.')
    parser.add_argument('--structure_warmup_epochs', type=int, default=3,
                       help='Warmup epochs training the structure branch only.')
    parser.add_argument('--full_finetune_epoch', type=int, default=10,
                       help='Epoch to start full finetuning (inclusive).')
    parser.add_argument('--lr_scale_seq', type=float, default=0.3,
                       help='Learning-rate scale factor for the sequence branch.')
    parser.add_argument('--lr_scale_struct', type=float, default=1.0,
                       help='Learning-rate scale factor for the structure branch.')
    parser.add_argument('--lr_scale_fusion', type=float, default=1.0,
                       help='Learning-rate scale factor for the fusion layer.')
    parser.add_argument('--contrastive_projection_dim', type=int, default=128,
                       help='Projection dimension for contrastive learning.')
    parser.add_argument('--use_attention_pooling', type=str, default=None,
                       help='Whether to enable attention pooling (true/false).')

    # Post-processing
    parser.add_argument('--optimize_threshold', action='store_true',
                       help='Automatically search the best decision threshold on the validation set.')
    parser.add_argument('--no_optimize_threshold', action='store_true',
                       help='Disable automatic threshold tuning.')
    parser.add_argument('--threshold_metric', type=str, default='mcc',
                       choices=['f1', 'mcc', 'balanced_accuracy'],
                       help='Metric used for threshold search.')
    parser.add_argument('--calibration_method', type=str, default='platt',
                       choices=['none', 'temperature', 'platt'],
                       help='Probability calibration method.')
    parser.add_argument('--use_small_demo_dataset', action='store_true',
                       help='Use the small demo dataset under phosphorylation_data_small_batch.')
    parser.add_argument('--max_train_samples', type=int, default=None,
                       help='Optional cap on the number of training samples to load (useful for smoke runs).')
    parser.add_argument('--max_test_samples', type=int, default=None,
                       help='Optional cap on the number of test samples to load (useful for smoke runs).')

    args = parser.parse_args()

    # Record argument defaults
    default_map = {}
    for action in parser._actions:
        if not action.dest or action.dest == argparse.SUPPRESS:
            continue
        default_map[action.dest] = parser.get_default(action.dest)

    # Determine which args were explicitly provided (different from defaults)
    specified_args = set()
    for dest, default_val in default_map.items():
        if not hasattr(args, dest):
            continue
        current_val = getattr(args, dest)
        if current_val != default_val:
            specified_args.add(dest)

    setattr(args, '_specified_args', specified_args)
    setattr(args, '_arg_defaults', default_map)

    return args


def main():
    """Main entry point."""
    # Parse CLI arguments
    args = parse_arguments()

    # Set random seed
    set_seed(args.seed)

    # Create config manager
    config_manager = ConfigManager(config_path=args.config_path, args=args)
    config = config_manager.get_config()

    def _sync_arg_config(attr_name):
        """Helper to keep argparse namespace and config dictionary in sync."""
        if not hasattr(args, attr_name):
            return
        if attr_name in specified_args:
            config[attr_name] = getattr(args, attr_name)
        elif attr_name in config:
            setattr(args, attr_name, config[attr_name])
        else:
            config[attr_name] = getattr(args, attr_name)

    # Override non-explicit CLI args from the config file
    specified_args = getattr(args, '_specified_args', set())
    for key, value in config.items():
        if not hasattr(args, key):
            continue
        if key in specified_args:
            continue
        # Only sync scalar/simple values to avoid overwriting complex structures
        if isinstance(value, (dict, list, tuple, set)):
            continue
        setattr(args, key, value)

    # Ensure CLI arguments properly override config
    # Handle core_model_type specially to ensure CLI has highest priority
    if 'core_model_type' in specified_args:
        config['core_model_type'] = args.core_model_type
        print(f"[INFO] Parameters core_model_type: {args.core_model_type}")

    if 'transformer_layers' in specified_args:
        config['transformer_layers'] = args.transformer_layers
        print(f"[INFO] Parameters transformer_layers: {args.transformer_layers}")

    if 'lstm_hidden_dim' in specified_args:
        config['lstm_hidden_dim'] = args.lstm_hidden_dim
        print(f"[INFO] Parameters lstm_hidden_dim: {args.lstm_hidden_dim}")

    if 'gpu_ids' in specified_args:
        config['gpu_ids'] = args.gpu_ids
    elif 'gpu_ids' in config:
        args.gpu_ids = config['gpu_ids']

    if 'seq_encoder_type' in specified_args:
        config['seq_encoder_type'] = args.seq_encoder_type

    if 'struct_encoder_type' in specified_args:
        config['struct_encoder_type'] = args.struct_encoder_type

    if 'branch_hidden_dim' in specified_args:
        config['branch_hidden_dim'] = args.branch_hidden_dim
    elif 'branch_hidden_dim' in config:
        args.branch_hidden_dim = config['branch_hidden_dim']
    else:
        config.setdefault('branch_hidden_dim', args.branch_hidden_dim)

    if 'fusion_hidden_dim' in specified_args:
        config['fusion_hidden_dim'] = args.fusion_hidden_dim
    elif 'fusion_hidden_dim' in config:
        args.fusion_hidden_dim = config['fusion_hidden_dim']
    else:
        config.setdefault('fusion_hidden_dim', args.fusion_hidden_dim)

    if 'fusion_hidden_dim_candidates' in config and 'fusion_hidden_dim_candidates' not in specified_args:
        setattr(args, 'fusion_hidden_dim_candidates', config['fusion_hidden_dim_candidates'])

    if 'window_size' in specified_args:
        config['window_size'] = args.window_size
    elif 'window_size' in config:
        args.window_size = config['window_size']
    else:
        config.setdefault('window_size', args.window_size)

    if 'local_window_size' in specified_args:
        config['local_window_size'] = args.local_window_size
    elif 'local_window_size' in config:
        args.local_window_size = config['local_window_size']
    else:
        config.setdefault('local_window_size', args.local_window_size)

    if 'window_size_candidates' in config and 'window_size_candidates' not in specified_args:
        setattr(args, 'window_size_candidates', config['window_size_candidates'])

    if 'local_window_size_candidates' in config and 'local_window_size_candidates' not in specified_args:
        setattr(args, 'local_window_size_candidates', config['local_window_size_candidates'])

    if 'protein_cache_dir' in specified_args:
        config['protein_cache_dir'] = args.protein_cache_dir
    elif 'protein_cache_dir' in config:
        args.protein_cache_dir = config['protein_cache_dir']
    elif args.protein_cache_dir is not None:
        config['protein_cache_dir'] = args.protein_cache_dir

    # Sync data paths
    if 'train_data_path' in specified_args:
        config['train_data_path'] = args.train_data_path
    elif 'train_data_path' in config:
        args.train_data_path = config['train_data_path']

    if 'test_data_path' in specified_args:
        config['test_data_path'] = args.test_data_path
    elif 'test_data_path' in config:
        args.test_data_path = config['test_data_path']

    if 'features_path' in specified_args:
        config['features_path'] = args.features_path
    elif 'features_path' in config:
        args.features_path = config['features_path']

    for nan_field in [
        'enable_nan_debugging',
        'nan_debug_targets',
        'nan_debug_frequency',
        'nan_debug_log_finite',
        'nan_debug_save_dir',
        'nan_debug_max_saved_tensors',
        'nan_debug_break_on_error',
        'nan_debug_check_batches',
        'nan_debug_batch_frequency',
        'nan_debug_max_tensors_per_call',
    ]:
        _sync_arg_config(nan_field)

    if isinstance(args.nan_debug_targets, str):
        args.nan_debug_targets = [args.nan_debug_targets]
    if isinstance(args.nan_debug_targets, tuple):
        args.nan_debug_targets = list(args.nan_debug_targets)
    if isinstance(config.get('nan_debug_targets'), str):
        config['nan_debug_targets'] = [config['nan_debug_targets']]
    if isinstance(config.get('nan_debug_targets'), tuple):
        config['nan_debug_targets'] = list(config['nan_debug_targets'])

    # Update advanced-module settings in config
    if args.use_reinforcement:
        config['use_reinforcement'] = True
        config['reinforcement_learning'].update({
            'reward_type': args.rl_reward_type,
            'exploration_epsilon': args.rl_exploration_epsilon,
            'policy_learning_rate': args.rl_policy_lr
        })
    
    # --- MODIFIED: Enable RL Controller ---
    if args.enable_rl_controller:
        config['enable_rl_controller'] = True
    # --------------------------------------

    if args.use_moe:
        config['use_moe'] = True
        if 'mixture_of_experts' not in config or not isinstance(config.get('mixture_of_experts'), dict):
            config['mixture_of_experts'] = {}
        config['mixture_of_experts'].update({
            'num_experts': args.moe_num_experts,
            'top_k': args.moe_top_k
        })

    config['integration_method'] = args.integration_method

    # Update early stopping config
    if args.use_early_stopping:
        config['use_early_stopping'] = True
        config['early_stopping_patience'] = args.early_stopping_patience
        config['early_stopping_min_delta'] = args.early_stopping_min_delta
        config['early_stopping_monitor'] = args.early_stopping_monitor
        config['early_stopping_mode'] = args.early_stopping_mode
        config['restore_best_weights'] = args.restore_best_weights

    # Update dropout config
    if args.adaptive_dropout:
        config['adaptive_dropout'] = True
        for key in ['dropout_schedule', 'input_dropout', 'hidden_dropout', 'output_dropout']:
            if key in specified_args:
                config[key] = getattr(args, key)
            elif key not in config:
                config[key] = getattr(args, key)

    if args.use_attention_pooling is not None:
        config['use_attention_pooling'] = (str(args.use_attention_pooling).lower() == 'true')

    # Threshold and calibration
    if args.no_optimize_threshold:
        config['optimize_threshold'] = False
    elif args.optimize_threshold:
        config['optimize_threshold'] = True
    else:
        config.setdefault('optimize_threshold', True)

    if args.threshold_metric:
        config['threshold_metric'] = args.threshold_metric

    if args.calibration_method:
        config['calibration_method'] = args.calibration_method

    target_ptm = str(getattr(args, 'target_ptm_type', config.get('target_ptm_type', 'phosphorylation'))).strip().lower()
    config['target_ptm_type'] = target_ptm
    setattr(args, 'target_ptm_type', target_ptm)

    # Small demo dataset configuration
    if args.use_small_demo_dataset:
        demo_root = Path(__file__).resolve().parent / 'phosphorylation_data_small_batch'
        config['train_data_path'] = str(demo_root / 'train_data.csv')
        config['test_data_path'] = str(demo_root / 'test_data.csv')
        config['features_path'] = str(demo_root / 'features' / 'esm_features.h5')
        if args.output_dir == './acetylation_data_ptms_output':
            args.output_dir = './phosphorylation_small_output'
        config['use_optimized_loading'] = True
        print("Use Dataset:")
        print(f"  TrainingData: {config['train_data_path']}")
        print(f"  TestData: {config['test_data_path']}")
        print(f": {config['features_path']}")

    # Set data loading optimization options
    if args.disable_optimized_loading:
        config['use_optimized_loading'] = False
    else:
        config['use_optimized_loading'] = args.use_optimized_loading

    # Print configuration
    config_manager.print_config()

    fusion_candidates = getattr(args, 'fusion_hidden_dim_candidates', None)
    dims = []
    if fusion_candidates:
        dims = [int(val) for val in fusion_candidates]
    else:
        base_dim = config.get('fusion_hidden_dim', args.fusion_hidden_dim)
        if base_dim is not None:
            dims = [int(base_dim)]

    if not dims:
        dims = [384]

    unique_dims = []
    for dim in dims:
        if dim not in unique_dims:
            unique_dims.append(dim)
    dims = unique_dims

    window_candidates = getattr(args, 'window_size_candidates', None)
    local_candidates = getattr(args, 'local_window_size_candidates', None)

    if window_candidates:
        window_candidates = [int(v) for v in window_candidates]
    else:
        window_candidates = [int(config.get('window_size', args.window_size))]

    if local_candidates:
        local_candidates = [int(v) for v in local_candidates]
    else:
        local_candidates = [int(config.get('local_window_size', args.local_window_size))]

    def _build_window_pairs(global_list, local_list):
        if not global_list:
            return [(int(config.get('window_size', args.window_size)), int(config.get('local_window_size', args.local_window_size)))]
        if not local_list:
            return [(w, int(config.get('local_window_size', args.local_window_size))) for w in global_list]
        if len(global_list) == len(local_list):
            return list(zip(global_list, local_list))
        if len(local_list) == 1:
            return [(w, local_list[0]) for w in global_list]
        if len(global_list) == 1:
            return [(global_list[0], l) for l in local_list]
        pairs = []
        for w in global_list:
            for l in local_list:
                pairs.append((w, l))
        return pairs

    window_pairs = _build_window_pairs(window_candidates, local_candidates)

    multi_run = len(dims) > 1
    multi_window = len(window_pairs) > 1
    base_output_dir = Path(args.output_dir)
    if multi_run or multi_window:
        base_output_dir.mkdir(parents=True, exist_ok=True)

    shared_cache_dir = None
    if args.protein_cache_dir:
        shared_cache_dir = Path(args.protein_cache_dir)
    elif multi_run or multi_window:
        shared_cache_dir = base_output_dir / 'shared_protein_cache'

    if shared_cache_dir is not None:
        shared_cache_dir.mkdir(parents=True, exist_ok=True)
        shared_cache_dir_str = str(shared_cache_dir)
    else:
        shared_cache_dir_str = None

    print(f" fusion_hidden_dim Config: {dims}")
    print(f" Config: {window_pairs}")
    run_results = []
    for dim in dims:
        for window_size, local_window_size in window_pairs:
            print(f"\n=== Start fusion_hidden_dim={dim}, window_size={window_size}, local_window_size={local_window_size} Training ===")
            config_run = copy.deepcopy(config)
            args_run = copy.deepcopy(args)
            config_run['fusion_hidden_dim'] = dim
            args_run.fusion_hidden_dim = dim
            config_run['window_size'] = window_size
            config_run['local_window_size'] = local_window_size
            args_run.window_size = window_size
            args_run.local_window_size = local_window_size

            if multi_run or multi_window:
                run_output_dir = base_output_dir / f"fusion_dim_{dim}" / f"window_{window_size}_local_{local_window_size}"
                run_output_dir.mkdir(parents=True, exist_ok=True)
                args_run.output_dir = str(run_output_dir)
            config_run['output_dir'] = args_run.output_dir

            if shared_cache_dir_str is not None:
                config_run['protein_cache_dir'] = shared_cache_dir_str
                args_run.protein_cache_dir = shared_cache_dir_str
            else:
                run_cache_dir = Path(args_run.output_dir) / 'protein_cache'
                run_cache_dir.mkdir(parents=True, exist_ok=True)
                config_run['protein_cache_dir'] = str(run_cache_dir)
                args_run.protein_cache_dir = str(run_cache_dir)

            adjust_cross_attention_hyperparams(config_run)
            args_run.cross_attention_heads = config_run.get('cross_attention_heads')
            args_run.contrastive_projection_dim = config_run.get('contrastive_projection_dim')
            args_run.branch_hidden_dim = config_run.get('branch_hidden_dim')

            branch_dim_effective = config_run.get('branch_hidden_dim')
            per_head_dim = config_run.get('cross_attention_per_head_dim')
            per_head_dim_display = per_head_dim if per_head_dim is not None else 'N/A'
            print(
                ": branch_hidden_dim={} -> heads={}, per-head={}".format(
                    branch_dim_effective,
                    config_run.get('cross_attention_heads'),
                    per_head_dim_display,
                )
            )
            print(
                ": contrastive_projection_dim={}".format(
                    config_run.get('contrastive_projection_dim')
                )
            )

            trainer = EnhancedPTMsTrainer(config_run, args_run)
            success = trainer.run()
            metrics = trainer.last_test_metrics if success else None

            if isinstance(metrics, dict):
                metrics_serializable = {}
                for key, value in metrics.items():
                    if isinstance(value, (np.floating, np.integer)):
                        metrics_serializable[key] = float(value)
                    else:
                        metrics_serializable[key] = value
            else:
                metrics_serializable = metrics

            run_results.append({
                'fusion_hidden_dim': dim,
                'window_size': window_size,
                'local_window_size': local_window_size,
                'success': bool(success),
                'output_dir': args_run.output_dir,
                'test_metrics': metrics_serializable
            })

            if success:
                print(
                    f"✅ fusion_hidden_dim={dim}, window_size={window_size}, "
                    f"local_window_size={local_window_size} training completed. Output directory: {args_run.output_dir}"
                )
            else:
                print(
                    f"⚠️ fusion_hidden_dim={dim}, window_size={window_size}, "
                    f"local_window_size={local_window_size} training failed. Output directory: {args_run.output_dir}"
                )

    if multi_run or multi_window:
        summary = {
            'fusion_hidden_dim_candidates': dims,
            'window_size_candidates': window_candidates,
            'local_window_size_candidates': local_candidates,
            'runs': run_results
        }
        summary_path = base_output_dir / 'sweep_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n📄 Saved sweep summary: {summary_path}")

    all_success = all(result['success'] for result in run_results)
    if all_success:
        if multi_run or multi_window:
            print("🎉 Sweep completed!")
        else:
            print("🎉 Training Completed!")
        return 0

    failed_runs = [
        {
            'fusion_hidden_dim': result['fusion_hidden_dim'],
            'window_size': result.get('window_size'),
            'local_window_size': result.get('local_window_size')
        }
        for result in run_results if not result['success']
    ]
    print(f"❌ Failed runs: {failed_runs}")
    return 1


if __name__ == "__main__":
    exit(main())
