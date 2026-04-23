

import random
import numpy as np
import torch
import logging

# Set up logging
logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    Set random seeds to improve reproducibility.
    
    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f": {seed}")


def custom_collate_fn(batch):
    """
    Custom collate function for batches with pre-extracted features.
    
    Handles batches that may include sequence features, structure features, labels, and other metadata,
    and organizes them into batched tensors.
    
    Args:
        batch (list): List of samples; each sample is typically a dict.
        
    Returns:
        dict: Batched data dict that may include:
            - sequence_features: [batch_size, seq_len, feature_dim]
            - local_features: [batch_size, local_len, feature_dim]
            - global_features: [batch_size, feature_dim]
            - structure_features: [batch_size, seq_len, struct_dim] (optional)
            - residue_types: [batch_size, seq_len]
            - position_ids: [batch_size, seq_len]
            - ptm_type_ids: [batch_size]
            - label: [batch_size]
            - uniprot_ids: list of UniProt IDs
            - positions: list of positions
    """
    try:
        # Validate batch
        if not batch:
            raise ValueError(" Data ")
        
        # Use the first sample to infer structure
        first_sample = batch[0]
        
        # Sequence features
        if 'sequence' in first_sample and isinstance(first_sample['sequence'], dict):
            # New format: sequence is a dict containing multiple feature types
            sequence_features = []
            local_features = []
            global_features = []
            
            for item in batch:
                seq_data = item['sequence']
                
                # Window features
                if 'window_features' in seq_data:
                    sequence_features.append(seq_data['window_features'])
                elif 'features' in seq_data:
                    sequence_features.append(seq_data['features'])
                else:
                    raise KeyError(" Data Info")
                
                # Local features
                if 'local_features' in seq_data:
                    local_features.append(seq_data['local_features'])
                elif 'local' in seq_data:
                    local_features.append(seq_data['local'])
                else:
                    # Fallback: reuse window features
                    local_features.append(sequence_features[-1])
                
                # Global features
                if 'global_features' in seq_data:
                    global_features.append(seq_data['global_features'])
                elif 'global' in seq_data:
                    global_features.append(seq_data['global'])
                else:
                    # Fallback: mean-pool window features
                    global_features.append(torch.mean(sequence_features[-1], dim=0))
            
            # Convert to tensors
            sequence_features = torch.stack(sequence_features)
            local_features = torch.stack(local_features)
            global_features = torch.stack(global_features)
            
        elif 'sequence_features' in first_sample:
            # Flat format (DataLoadingOptimizer output)
            sequence_features = torch.stack([
                torch.from_numpy(item['sequence_features']) if isinstance(item['sequence_features'], np.ndarray) else item['sequence_features'] 
                for item in batch
            ])
            
            if 'local_features' in first_sample:
                local_features = torch.stack([
                    torch.from_numpy(item['local_features']) if isinstance(item['local_features'], np.ndarray) else item['local_features']
                    for item in batch
                ])
            else:
                local_features = sequence_features

            if 'global_features' in first_sample:
                global_features = torch.stack([
                    torch.from_numpy(item['global_features']) if isinstance(item['global_features'], np.ndarray) else item['global_features']
                    for item in batch
                ])
            else:
                global_features = torch.mean(sequence_features, dim=1)

        else:
            # Legacy format: use sequence directly as features
            sequence_features = torch.stack([item['sequence'] for item in batch])
            local_features = sequence_features  # Use the same features
            global_features = torch.mean(sequence_features, dim=1)  # Global feature: mean pooling
        
        # Structure features (optional; allow missing values)
        structure_features = None
        structure_mask = None
        if 'structure' in first_sample:
            structure_values = [item.get('structure') for item in batch]
            tensor_structures = [value for value in structure_values if isinstance(value, torch.Tensor)]

            if tensor_structures:
                reference_tensor = tensor_structures[0]
                padded_structures = []
                mask_values = []

                for value in structure_values:
                    if isinstance(value, torch.Tensor):
                        padded_structures.append(value)
                        mask_values.append(1.0)
                    else:
                        padded_structures.append(torch.zeros_like(reference_tensor))
                        mask_values.append(0.0)

                if tensor_structures and len(tensor_structures) != len(structure_values):
                    missing_count = len(structure_values) - len(tensor_structures)
                    if not getattr(custom_collate_fn, "_structure_warning_issued", False):
                        logger.warning(
                            "custom_collate_fn: %d Samples,Use structure_mask", missing_count
                        )
                        setattr(custom_collate_fn, "_structure_warning_issued", True)

                structure_features = torch.stack(padded_structures)
                structure_mask = torch.tensor(mask_values, dtype=reference_tensor.dtype)
            else:
                structure_features = None
                structure_mask = None
        elif 'structure_features' in first_sample:
            # Handle flat format from DataLoadingOptimizer
            structure_values = [item.get('structure_features') for item in batch]
            valid_structures = [s for s in structure_values if s is not None]
            
            if valid_structures:
                # Get reference tensor/array for shape
                ref = valid_structures[0]
                if isinstance(ref, np.ndarray):
                    ref_tensor = torch.from_numpy(ref)
                else:
                    ref_tensor = ref
                
                padded_structures = []
                mask_values = []
                
                for s in structure_values:
                    if s is not None:
                        if isinstance(s, np.ndarray):
                            padded_structures.append(torch.from_numpy(s))
                        else:
                            padded_structures.append(s)
                        mask_values.append(1.0)
                    else:
                        padded_structures.append(torch.zeros_like(ref_tensor))
                        mask_values.append(0.0)
                
                structure_features = torch.stack(padded_structures)
                structure_mask = torch.tensor(mask_values, dtype=torch.float32)
        
        # Residue types
        residue_types = None
        if 'residue_types' in first_sample:
            residue_types = torch.stack([item['residue_types'] for item in batch])
        elif 'residue_type' in first_sample:
            residue_types = torch.stack([item['residue_type'] for item in batch])

        # Target-site indicator
        site_indicator = None
        if 'site_indicator' in first_sample:
            site_indicator = torch.stack([item['site_indicator'] for item in batch])

        # Position information
        position_ids = None
        relative_position_offsets = None
        if 'position_ids' in first_sample:
            position_ids = torch.stack([item['position_ids'] for item in batch])
        if 'relative_position_offsets' in first_sample:
            relative_position_offsets = torch.stack([item['relative_position_offsets'] for item in batch])

        # PTM type
        ptm_type_ids = None
        if 'ptm_type_id' in first_sample:
            ptm_type_ids = torch.tensor([
                int(item['ptm_type_id'].item()) if isinstance(item['ptm_type_id'], torch.Tensor) else int(item['ptm_type_id'])
                for item in batch
            ], dtype=torch.long)
        elif 'ptm_type_ids' in first_sample:
            ptm_type_ids = torch.tensor([
                int(item['ptm_type_ids']) for item in batch
            ], dtype=torch.long)
        
        # Micro-environment features
        micro_env_features = None
        if 'micro_env_features' in first_sample:
            # Normalize feature dimensions to ensure stacking succeeds
            target_dim = 6
            processed_features = []
            for item in batch:
                raw_feat = item.get('micro_env_features')
                if raw_feat is None:
                     # Default to zeros if missing
                     feat = torch.zeros(target_dim, dtype=torch.float32)
                else:
                    feat = torch.as_tensor(raw_feat, dtype=torch.float32)
                    if feat.shape[0] > target_dim:
                        feat = feat[:target_dim]
                    elif feat.shape[0] < target_dim:
                        padding = torch.zeros(target_dim - feat.shape[0], dtype=feat.dtype)
                        feat = torch.cat([feat, padding])
                processed_features.append(feat)
            
            micro_env_features = torch.stack(processed_features)

        # Labels
        labels = []
        for item in batch:
            lbl = item.get('label')
            if lbl is None:
                labels.append(torch.tensor(-1.0, dtype=torch.float32)) # Dummy label
            elif isinstance(lbl, torch.Tensor):
                labels.append(lbl)
            else:
                labels.append(torch.tensor(lbl, dtype=torch.float32))
        labels = torch.stack(labels)
        
        # Metadata
        uniprot_ids = [item.get('uniprot_id', '') for item in batch]
        positions = [item.get('position', -1) for item in batch]
        residues = [item.get('residue', '') for item in batch]
        ptm_types = [item.get('ptm_type', '') for item in batch]
        
        # Build output dict
        result = {
            'sequence_features': sequence_features,
            'local_features': local_features,
            'global_features': global_features,
            'label': labels,
            'uniprot_ids': uniprot_ids,
            'positions': positions,
            'residues': residues,
            'ptm_types': ptm_types
        }

        if micro_env_features is not None:
            result['micro_env_features'] = micro_env_features
        
        # Add optional fields
        if structure_features is not None:
            result['structure_features'] = structure_features
            if structure_mask is not None:
                result['structure_mask'] = structure_mask
        
        if residue_types is not None:
            result['residue_types'] = residue_types

        if position_ids is not None:
            result['position_ids'] = position_ids
        if relative_position_offsets is not None:
            result['relative_position_offsets'] = relative_position_offsets
        if site_indicator is not None:
            result['site_indicator'] = site_indicator

        if ptm_type_ids is not None:
            result['ptm_type_ids'] = ptm_type_ids

        # Provide nested \"sequence\"/\"structure\" keys for models that expect them
        result['sequence'] = {
            'window_features': sequence_features,
            'local_features': local_features,
            'global_features': global_features
        }

        if structure_features is not None:
            result['structure'] = structure_features
            # Some models also check structure_mask under the sequence dict
            if structure_mask is not None:
                result['sequence']['structure_mask'] = structure_mask

        if residue_types is not None:
            result['sequence']['residue_types'] = residue_types
        if position_ids is not None:
            result['sequence']['position_ids'] = position_ids
        if relative_position_offsets is not None:
            result['sequence']['relative_position_offsets'] = relative_position_offsets
        if site_indicator is not None:
            result['sequence']['site_indicator'] = site_indicator
        if ptm_type_ids is not None:
            result['sequence']['ptm_type_ids'] = ptm_type_ids
        
        return result
        
    except Exception as e:
        logger.error(f": {str(e)}")
        logger.error(f": {len(batch) if batch else 0}")
        if batch:
            logger.error(f" Samples: {list(first_sample.keys())}")
        raise


def check_gpu_memory():
    """
    Check GPU memory usage.
    
    Returns:
        str: A human-readable description of GPU memory usage.
    """
    if not torch.cuda.is_available():
        return "CUDAUnavailable"
    
    try:
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        
        total_gb = total_memory / (1024**3)
        allocated_gb = allocated_memory / (1024**3)
        cached_gb = cached_memory / (1024**3)
        free_gb = total_gb - allocated_gb
        
        return (f"GPU {device}: {allocated_gb:.2f}GB/{total_gb:.2f}GB Use, "
                f"{cached_gb:.2f}GB, {free_gb:.2f}GB ")
    except Exception as e:
        return f" GPU Info: {str(e)}"


def get_device(gpu_id=None):
    """
    Get an available compute device.
    
    Args:
        gpu_id (int, optional): Specific GPU ID to use.
        
    Returns:
        torch.device: Compute device.
    """
    if gpu_id is not None and torch.cuda.is_available():
        if gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
            logger.info(f"Use GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            return device
        else:
            logger.warning(f" GPU {gpu_id},Use Device")
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"UseGPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = torch.device("cpu")
        logger.info("UseCPU")
        return device


def format_time(seconds):
    """
    Format a duration for display.
    
    Args:
        seconds (float): Duration in seconds.
        
    Returns:
        str: Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.2f} "
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} {secs:.2f} "
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} {minutes} {secs:.2f} "


def count_parameters(model):
    """
    Count model parameters.
    
    Args:
        model (torch.nn.Module): PyTorch model.
        
    Returns:
        dict: Dictionary with total, trainable, and non-trainable parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """
    Save a model checkpoint.
    
    Args:
        model: Model.
        optimizer: Optimizer.
        scheduler: Learning-rate scheduler.
        epoch: Current epoch.
        loss: Current loss value.
        filepath: Output path.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    logger.info(f" Save: {filepath}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Load a model checkpoint.
    
    Args:
        filepath: Path to the checkpoint file.
        model: Model.
        optimizer: Optimizer (optional).
        scheduler: Learning-rate scheduler (optional).
        
    Returns:
        dict: Dictionary containing epoch and loss information.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f" {filepath} Load")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf'))
    }
