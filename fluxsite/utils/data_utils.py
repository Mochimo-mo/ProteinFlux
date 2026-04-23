"""
Data management module.
Responsible for data preprocessing, dataset splitting, and DataLoader creation.
"""

import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import random

from .common_utils import custom_collate_fn


class DataManager:
    """Data manager responsible for preprocessing and DataLoader creation."""
    
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.full_dataset = None  # Full dataset used for K-fold splitting
    
    def prepare_data_loaders(self):
        """Prepare training/validation/test DataLoaders."""
        # 1. Process data
        self._process_data()
        
        # 2. Split dataset
        self._split_dataset()
        
        # 3. Create DataLoaders
        train_loader = self._create_train_loader()
        val_loader = self._create_val_loader()
        test_loader = self._create_test_loader()
        
        return train_loader, val_loader, test_loader
    
    def _process_data(self):
        """Process raw inputs and build datasets."""
        print(" Data.")
        
        from data.unified_data_processor import UnifiedDataProcessor, UnifiedPTMDataset
        
        # Configure data processor
        target_ptm_type = str(self.config.get('target_ptm_type', 'phosphorylation')).strip().lower()
        window_size = self.config.get('window_size', 61)
        local_window_size = self.config.get('local_window_size', 31)
        fasta_dir = self.config.get('fasta_dir')
        
        data_processor = UnifiedDataProcessor(
            data_path=self.config['data_path'],
            esm_features_path=self.config.get('features_path', self.config.get('struct_features_path')),
            window_size=window_size,
            local_window_size=local_window_size,
            target_ptm_type=target_ptm_type,
            fasta_dir=fasta_dir
        )

        # Build dataset
        processed_data = data_processor.prepare_dataset()
        residue_info = getattr(data_processor, 'residue_info', None) # UnifiedDataProcessor might not set this on self, but prepare_dataset might. 
        # Actually UnifiedDataProcessor.prepare_dataset returns list of dicts.
        # residue_info is usually resolved during dataset creation if not passed.
        # But UnifiedPTMDataset calculates it if None.
        
        original_dataset = UnifiedPTMDataset(
            processed_data,
            fixed_window_size=window_size,
            fixed_local_size=local_window_size,
            target_ptm_type=target_ptm_type,
            residue_info=residue_info
        )

        # Keep an independent copy of the original dataset for K-fold
        self.full_dataset = UnifiedPTMDataset(
            processed_data,
            fixed_window_size=window_size,
            fixed_local_size=local_window_size,
            target_ptm_type=target_ptm_type,
            residue_info=residue_info
        )
        
        # Separate instance for normal training
        self.dataset = original_dataset

        if hasattr(original_dataset, 'residue_vocab_size'):
            self.config['num_residue_types'] = int(original_dataset.residue_vocab_size)
        if hasattr(original_dataset, 'residue_to_id'):
            self.config['residue_to_id'] = dict(original_dataset.residue_to_id)
        
        # Apply data augmentation to self.dataset while keeping self.full_dataset unchanged
        self._apply_data_augmentation()

        if hasattr(self.dataset, 'ptm_type_to_id'):
            ptm_mapping = dict(self.dataset.ptm_type_to_id)
            self.config.setdefault('num_ptm_types', len(ptm_mapping))
            self.config['ptm_type_to_id'] = ptm_mapping
            print(f"PTM: {ptm_mapping}")
        
        print(f"Dataset: {len(self.dataset)}")
        print(f" Dataset: {len(self.full_dataset)}")

    def _apply_data_augmentation(self):
        """Apply data augmentation."""
        # Basic augmentation
        if self.config.get('use_data_augmentation', False):
            print("Enable Data.")
            from models.advanced_components import DataAugmentationWrapper
            self.dataset = DataAugmentationWrapper(
                self.dataset, 
                augment_prob=self.config.get('augment_prob', 0.5),
                noise_level=self.config.get('noise_level', 0.1)
            )
        
        # Enhanced augmentation for self.dataset; keep self.full_dataset as the original
        if self.config.get('use_enhanced_augmentation', True):
            print("Enable Data.")
            from models.advanced_components import EnhancedDataAugmentationWrapper
            self.dataset = EnhancedDataAugmentationWrapper(self.dataset, self.config)
            # Note: do not augment self.full_dataset
    
    def _split_dataset(self):
        """Split dataset into train/val/test subsets."""
        val_ratio = self.config.get('val_ratio', 0.2)
        test_ratio = self.config.get('test_ratio', 0.1)
        use_cv = self.config.get('cross_validation_folds', 0) > 1
        stratified_split = self.config.get('stratified_split', True)
        
        print(f"Validation: {val_ratio}, Test: {test_ratio}")
        print(f"Cross-validation: {use_cv},: {stratified_split}")
        
        # Prepare labels
        all_labels = np.array([self.dataset[i]['label'].item() for i in range(len(self.dataset))])
        
        if use_cv:
            self._split_with_cv(all_labels, val_ratio, test_ratio, stratified_split)
        else:
            self._split_without_cv(all_labels, val_ratio, test_ratio, stratified_split)
        
        # Save test indices
        self._save_test_indices()
        
        print(f"Training: {len(self.train_dataset)}, Validation: {len(self.val_dataset)}")
        if self.test_dataset is not None:
            print(f"Test: {len(self.test_dataset)}")
    
    def _split_with_cv(self, all_labels, val_ratio, test_ratio, stratified_split):
        """Split data using cross-validation."""
        cv_folds = self.config.get('cross_validation_folds', 5)
        
        if stratified_split:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.args.seed)
            train_idx, val_idx = next(iter(skf.split(np.zeros(len(self.dataset)), all_labels)))
            
            # If a test set is requested, split it from the training set
            if test_ratio > 0:
                test_size = int(len(train_idx) * test_ratio / (1 - val_ratio))
                train_labels = all_labels[train_idx]
                train_idx_new, test_idx = train_test_split(
                    train_idx, 
                    test_size=test_size, 
                    stratify=train_labels,
                    random_state=self.args.seed
                )
                train_idx = train_idx_new
            else:
                test_idx = None
        else:
            # Simple random split
            indices = list(range(len(self.dataset)))
            np.random.shuffle(indices)
            fold_size = len(indices) // cv_folds
            val_idx = indices[:fold_size]
            remaining = indices[fold_size:]
            
            if test_ratio > 0:
                test_size = int(len(remaining) * test_ratio / (1 - val_ratio))
                test_idx = remaining[:test_size]
                train_idx = remaining[test_size:]
            else:
                train_idx = remaining
                test_idx = None
        
        # Create dataset subsets
        self.train_dataset = Subset(self.dataset, train_idx)
        self.val_dataset = Subset(self.dataset, val_idx)
        self.test_dataset = Subset(self.dataset, test_idx) if test_idx is not None else None
    
    def _split_without_cv(self, all_labels, val_ratio, test_ratio, stratified_split):
        """Split data without cross-validation."""
        # Check whether to use protein-level split
        use_protein_split = self.config.get('use_protein_level_split', True)

        if use_protein_split:
            print("UseProtein Data.")
            train_idx, val_idx, test_idx = protein_level_split(
                self.dataset,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                random_state=self.args.seed
            )

            self.train_dataset = Subset(self.dataset, train_idx)
            self.val_dataset = Subset(self.dataset, val_idx)
            self.test_dataset = Subset(self.dataset, test_idx) if test_idx is not None else None

        elif stratified_split:
            if test_ratio > 0:
                # Three-way split
                train_idx, temp_idx, train_labels, temp_labels = train_test_split(
                    range(len(self.dataset)), all_labels,
                    test_size=val_ratio + test_ratio,
                    stratify=all_labels,
                    random_state=self.args.seed
                )

                # Further split validation and test sets
                relative_test_ratio = test_ratio / (val_ratio + test_ratio)
                val_idx, test_idx, _, _ = train_test_split(
                    temp_idx, temp_labels,
                    test_size=relative_test_ratio,
                    stratify=temp_labels,
                    random_state=self.args.seed
                )

                self.train_dataset = Subset(self.dataset, train_idx)
                self.val_dataset = Subset(self.dataset, val_idx)
                self.test_dataset = Subset(self.dataset, test_idx)
            else:
                # Two-way split
                train_idx, val_idx, _, _ = train_test_split(
                    range(len(self.dataset)), all_labels,
                    test_size=val_ratio,
                    stratify=all_labels,
                    random_state=self.args.seed
                )

                self.train_dataset = Subset(self.dataset, train_idx)
                self.val_dataset = Subset(self.dataset, val_idx)
                self.test_dataset = None
        else:
            # Random split
            if test_ratio > 0:
                dataset_size = len(self.dataset)
                train_size = int((1 - val_ratio - test_ratio) * dataset_size)
                val_size = int(val_ratio * dataset_size)
                test_size = dataset_size - train_size - val_size

                self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                    self.dataset,
                    [train_size, val_size, test_size],
                    generator=torch.Generator().manual_seed(self.args.seed)
                )
            else:
                train_size = int((1 - val_ratio) * len(self.dataset))
                val_size = len(self.dataset) - train_size

                self.train_dataset, self.val_dataset = random_split(
                    self.dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.args.seed)
                )
                self.test_dataset = None
    
    def _save_test_indices(self):
        """Save test-set indices."""
        if self.test_dataset is not None:
            output_dir = self.config['output_dir']
            test_indices_path = os.path.join(output_dir, 'test_indices.json')
            
            if hasattr(self.test_dataset, 'indices'):
                test_indices = self.test_dataset.indices
            else:
                test_indices = list(range(len(self.test_dataset)))
            
            with open(test_indices_path, 'w') as f:
                json.dump(test_indices if isinstance(test_indices, list) else test_indices.tolist(), f)
    
    def _create_train_loader(self):
        """Create the training DataLoader."""
        pin_memory = torch.cuda.is_available()
        balanced_sampling = self.config.get('balanced_sampling', True)
        
        if balanced_sampling:
            print("Use.")
            sampler = self._create_balanced_sampler()
            
            return DataLoader(
                self.train_dataset,
                batch_size=self.config.get('batch_size', 16),
                sampler=sampler,
                num_workers=self.args.num_workers,
                pin_memory=pin_memory,
                collate_fn=custom_collate_fn
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.config.get('batch_size', 16),
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=pin_memory,
                collate_fn=custom_collate_fn
            )
    
    def _create_balanced_sampler(self):
        """Create a balanced sampler."""
        # Get labels from the training data
        all_labels = np.array([self.dataset[i]['label'].item() for i in range(len(self.dataset))])
        
        if isinstance(self.train_dataset, Subset):
            train_labels = [all_labels[i] for i in self.train_dataset.indices]
        else:
            train_labels = [all_labels[i] for i in range(len(self.train_dataset))]
        
        # Ensure labels are integers
        train_labels = np.array(train_labels).astype(int)
        
        # Compute class weights
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        weights = [class_weights[int(label)] for label in train_labels]
        
        print(f": {class_counts}")
        
        return WeightedRandomSampler(weights, len(weights), replacement=True)
    
    def _create_val_loader(self):
        """Create the validation DataLoader."""
        pin_memory = torch.cuda.is_available()
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn
        )
    
    def _create_test_loader(self):
        """Create the test DataLoader."""
        if self.test_dataset is None:
            return None
        
        pin_memory = torch.cuda.is_available()
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn
        )
    
    def get_full_dataset_for_kfold(self):
        """Return the full dataset for K-fold cross-validation."""
        if self.full_dataset is None:
            self._process_data()
        
        # Ensure full_dataset is not None
        if self.full_dataset is None:
            print("Error: full_dataset None,Use Dataset")
            self.full_dataset = self.dataset
            
        print(f" k-foldDataset,: {len(self.full_dataset)}")
        
        # Collect labels directly from the dataset
        all_labels = []
        for i in range(len(self.full_dataset)):
            try:
                sample = self.full_dataset[i]
                if isinstance(sample, dict):
                    label = sample['label']
                else:
                    _, label = sample
                
                if isinstance(label, torch.Tensor):
                    all_labels.append(label.item())
                else:
                    all_labels.append(int(label))
            except Exception as e:
                print(f" Samples {i} Labels: {e}")
                # Fallback: default label
                all_labels.append(0)
        
        # Create simple index array for K-fold splitting
        X_for_split = np.arange(len(self.full_dataset))
        y_numpy = np.array(all_labels)
        
        print(f"k-foldData: X ={X_for_split.shape}, y ={y_numpy.shape}")
        print(f"Labels: Samples={np.sum(y_numpy)}, Samples={len(y_numpy)-np.sum(y_numpy)}")
        
        return X_for_split, y_numpy
    
    def create_kfold_loaders(self, train_indices, val_indices):
        """Create training and validation DataLoaders for K-fold."""
        # Ensure data has been processed
        if self.full_dataset is None:
            print("full_dataset None, Data.")
            self._process_data()
        
        # Check again
        if self.full_dataset is None:
            print("Error: Data full_dataset None")
            raise ValueError(" Dataset")
            
        print(f"Createk-foldLoad: Training ={len(train_indices)}, Validation ={len(val_indices)}")
        print(f" Dataset: {len(self.full_dataset)}")
        
        # Validate index range
        max_train_idx = max(train_indices) if len(train_indices) > 0 else -1
        max_val_idx = max(val_indices) if len(val_indices) > 0 else -1
        max_dataset_idx = len(self.full_dataset) - 1
        
        if max_train_idx > max_dataset_idx or max_val_idx > max_dataset_idx:
            print(f"Warning:! Training ={max_train_idx}, Validation ={max_val_idx}, Dataset ={max_dataset_idx}")
            # Filter invalid indices
            train_indices = [i for i in train_indices if i <= max_dataset_idx]
            val_indices = [i for i in val_indices if i <= max_dataset_idx]
            print(f": Training ={len(train_indices)}, Validation ={len(val_indices)}")
        
        # Create subset datasets
        train_subset = Subset(self.full_dataset, train_indices)
        val_subset = Subset(self.full_dataset, val_indices)
        
        print(f" DatasetCreate: Training ={len(train_subset)}, Validation ={len(val_subset)}")
        
        # Use the original dataset for K-fold; do not apply data augmentation.
        # The augmentation block is intentionally disabled to keep folds comparable.
        # if self.config.get('use_enhanced_augmentation', False):
        #     train_subset = self._apply_augmentation_to_subset(train_subset)
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate_fn
        )
        
        print(f"K-FoldDataLoad CreateCompleted:")
        print(f" Training: {len(train_subset)} Samples")
        print(f" Validation: {len(val_subset)} Samples")
        
        return train_loader, val_loader


def protein_level_split(dataset, val_ratio=0.2, test_ratio=0.1, random_state=42):
    """
    Protein-ID-based splitting that keeps all samples from the same protein in the same split.

    Args:
        dataset: Dataset; each sample should contain a 'uniprot_id' field.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.
        random_state: Random seed.

    Returns:
        train_indices, val_indices, test_indices: Indices for train/val/test splits.
    """
    # Collect protein IDs and their sample indices
    protein_to_indices = defaultdict(list)
    protein_to_labels = defaultdict(list)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        protein_id = sample['uniprot_id']
        label = sample['label'].item() if isinstance(sample['label'], torch.Tensor) else sample['label']

        protein_to_indices[protein_id].append(idx)
        protein_to_labels[protein_id].append(label)

    # Compute per-protein label distribution (for stratification)
    protein_label_stats = {}
    for protein_id, labels in protein_to_labels.items():
        positive_count = sum(labels)
        total_count = len(labels)
        protein_label_stats[protein_id] = {
            'positive_ratio': positive_count / total_count,
            'total_samples': total_count,
            'positive_count': positive_count
        }

    # List all protein IDs
    protein_ids = list(protein_to_indices.keys())

    # Create protein-level labels for stratified splitting.
    # Use each protein's positive ratio as the stratification signal.
    protein_labels = []
    for protein_id in protein_ids:
        ratio = protein_label_stats[protein_id]['positive_ratio']
        # Map continuous ratios to discrete buckets
        if ratio == 0:
            protein_labels.append(0)  # All-negative protein
        elif ratio == 1:
            protein_labels.append(2)  # All-positive protein
        else:
            protein_labels.append(1)  # Mixed protein

    print(f"Protein:")
    print(f" Protein: {len(protein_ids)}")
    print(f" SamplesProtein: {protein_labels.count(0)}")
    print(f" SamplesProtein: {protein_labels.count(1)}")
    print(f" SamplesProtein: {protein_labels.count(2)}")

    # Set random seeds
    random.seed(random_state)
    np.random.seed(random_state)

    # Split protein IDs
    if test_ratio > 0:
        # Three-way split: train/val/test
        try:
            train_proteins, temp_proteins, _, _ = train_test_split(
                protein_ids, protein_labels,
                test_size=val_ratio + test_ratio,
                stratify=protein_labels,
                random_state=random_state
            )

            # Recompute labels for the temporary set
            temp_labels = [protein_labels[protein_ids.index(pid)] for pid in temp_proteins]
            relative_test_ratio = test_ratio / (val_ratio + test_ratio)

            val_proteins, test_proteins, _, _ = train_test_split(
                temp_proteins, temp_labels,
                test_size=relative_test_ratio,
                stratify=temp_labels,
                random_state=random_state
            )
        except ValueError as e:
            print(f" Failed,Use: {e}")
            # Fallback to random split
            shuffled_proteins = protein_ids.copy()
            random.shuffle(shuffled_proteins)

            n_proteins = len(shuffled_proteins)
            n_test = int(n_proteins * test_ratio)
            n_val = int(n_proteins * val_ratio)

            test_proteins = shuffled_proteins[:n_test]
            val_proteins = shuffled_proteins[n_test:n_test + n_val]
            train_proteins = shuffled_proteins[n_test + n_val:]
    else:
        # Two-way split: train/val
        try:
            train_proteins, val_proteins, _, _ = train_test_split(
                protein_ids, protein_labels,
                test_size=val_ratio,
                stratify=protein_labels,
                random_state=random_state
            )
            test_proteins = []
        except ValueError as e:
            print(f" Failed,Use: {e}")
            # Fallback to random split
            shuffled_proteins = protein_ids.copy()
            random.shuffle(shuffled_proteins)

            n_proteins = len(shuffled_proteins)
            n_val = int(n_proteins * val_ratio)

            val_proteins = shuffled_proteins[:n_val]
            train_proteins = shuffled_proteins[n_val:]
            test_proteins = []

    # Collect sample indices for each split
    train_indices = []
    val_indices = []
    test_indices = []

    for protein_id in train_proteins:
        train_indices.extend(protein_to_indices[protein_id])

    for protein_id in val_proteins:
        val_indices.extend(protein_to_indices[protein_id])

    for protein_id in test_proteins:
        test_indices.extend(protein_to_indices[protein_id])

    # Summary statistics
    print(f"Protein Results:")
    print(f" Training: {len(train_proteins)} Protein, {len(train_indices)} Samples")
    print(f" Validation: {len(val_proteins)} Protein, {len(val_indices)} Samples")
    if test_proteins:
        print(f" Test: {len(test_proteins)} Protein, {len(test_indices)} Samples")

    # Validate no protein overlap
    all_train_proteins = set(train_proteins)
    all_val_proteins = set(val_proteins)
    all_test_proteins = set(test_proteins)

    assert len(all_train_proteins & all_val_proteins) == 0, "Training Validation Protein "
    assert len(all_train_proteins & all_test_proteins) == 0, "Training Test Protein "
    assert len(all_val_proteins & all_test_proteins) == 0, "Validation Test Protein "

    print("✓ Protein Validation, ")

    return train_indices, val_indices, test_indices if test_indices else None
