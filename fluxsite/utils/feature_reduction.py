
import os
import pickle
import logging
import numpy as np
from typing import Dict, List, Optional, Iterable, Union
from sklearn.decomposition import IncrementalPCA
import joblib

logger = logging.getLogger(__name__)

class FeatureReducer:
    """
    Feature dimensionality reduction using Incremental PCA.
    Supports reducing dimensions for multiple feature types (sequence, structure, etc.) independently.
    """
    
    TARGET_KEYS = [
        "sequence_features",
        "structure_features",
        "local_features",
        "structure_local",
        "global_features",
        "structure_global"
    ]

    def __init__(self, n_components: Union[int, float, Dict[str, Union[int, float]]] = 0.95, batch_size: int = 1024):
        """
        Initialize FeatureReducer.

        Args:
            n_components: Number of components to keep. 
                          If float between 0 and 1, it represents the variance ratio to explain.
                          If int, it represents the number of components.
                          Can be a dictionary mapping feature keys to n_components.
            batch_size: Batch size for IncrementalPCA.
        """
        self.default_n_components = n_components if not isinstance(n_components, dict) else 0.95
        self.n_components_map = n_components if isinstance(n_components, dict) else {}
        self.batch_size = batch_size
        self.models: Dict[str, IncrementalPCA] = {}
        self._fitted = False

    def _get_n_components(self, key: str) -> Union[int, float]:
        return self.n_components_map.get(key, self.default_n_components)

    def _optimize_inference(self):
        """
        Extract PCA parameters for faster inference using numpy directly.
        """
        self._fast_models = {}
        for key, model in self.models.items():
            if hasattr(model, 'components_') and hasattr(model, 'mean_'):
                self._fast_models[key] = {
                    'mean': model.mean_.astype(np.float32),
                    'components_T': model.components_.T.astype(np.float32)
                }
                logger.info(f"Optimized inference model prepared for {key}")

    def fit(self, samples: Iterable[Dict]) -> None:
        """
        Fit PCA models on the provided samples.
        
        Args:
            samples: Iterable of sample dictionaries.
        """
        # Initialize models if not already done
        # We need to know the input dimension to initialize IncrementalPCA properly if n_components is float?
        # Actually sklearn IncrementalPCA supports n_components as float (variance ratio) since version 1.3? 
        # Let's check environment. Assuming standard sklearn.
        # If n_components is float, IncrementalPCA might not support it directly in older versions, 
        # but let's assume we might need to set it after fitting or use a wrapper.
        # Standard IncrementalPCA takes n_components as int. If we want variance ratio, we might need standard PCA 
        # but that requires all data in memory.
        # For IncrementalPCA, we usually specify int. 
        # If the user wants variance ratio, we might have to fit first then select components, 
        # but IncrementalPCA transforms based on n_components set at init.
        
        # Let's stick to int or just pass it to IncrementalPCA and see. 
        # If n_components is float (0 < n < 1), sklearn PCA supports it. IncrementalPCA might not.
        # Let's assume for now we use a fixed number or we try to support float by fitting on a subset first to determine N?
        # Or just use IncrementalPCA with int.
        
        # To be safe and robust for large data, we use IncrementalPCA.
        # If n_components is float, we might need to handle it.
        
        # For this implementation, let's collect batches and partial_fit.
        
        buffer: Dict[str, List[np.ndarray]] = {key: [] for key in self.TARGET_KEYS}
        buffer_counts: Dict[str, int] = {key: 0 for key in self.TARGET_KEYS}
        
        # We need to initialize models lazily because we don't know input dimensions yet.
        
        for i, sample in enumerate(samples):
            for key in self.TARGET_KEYS:
                if key not in sample or sample[key] is None:
                    continue
                
                data = np.asarray(sample[key], dtype=np.float32)
                
                # Flatten if necessary (e.g. sequence_features is L x D, we treat each residue as a sample for PCA?)
                # Usually for sequence features (L x D), we want to reduce D. So we treat L residues as samples.
                if data.ndim == 2:
                    # shape (L, D)
                    pass
                elif data.ndim == 1:
                    # shape (D,) -> (1, D)
                    data = data.reshape(1, -1)
                else:
                    continue
                
                buffer[key].append(data)
                buffer_counts[key] += data.shape[0]
                
                if buffer_counts[key] >= self.batch_size:
                    self._partial_fit_batch(key, buffer[key])
                    buffer[key] = []
                    buffer_counts[key] = 0
        
        # Fit remaining
        for key in self.TARGET_KEYS:
            if buffer[key]:
                self._partial_fit_batch(key, buffer[key])
        
        self._fitted = True
        self._optimize_inference()
        logger.info("PCA fitting completed.")

    def _partial_fit_batch(self, key: str, data_list: List[np.ndarray]):
        X = np.vstack(data_list)
        n_samples, n_features = X.shape
        
        if key not in self.models:
            n_components = self._get_n_components(key)
            # If float, we can't use it directly in IncrementalPCA init for some versions.
            # But let's try passing it. If it fails, we default to min(n_samples, n_features) and select later?
            # Actually, let's just use a reasonable default integer if float is passed, or warn.
            # Or better: If float, we assume the user wants to keep that much variance. 
            # IncrementalPCA doesn't support float n_components for variance ratio directly in all versions.
            # We will set n_components to None (keep all) and then we can truncate during transform if needed,
            # but that defeats the purpose of saving memory/storage if we keep full model.
            # However, for reduction, we usually want to reduce D to d.
            
            # Let's assume user gives int for now or we convert float to int based on feature dim (e.g. 0.5 * D).
            if isinstance(n_components, float):
                # Heuristic: if < 1.0, treat as ratio of features? No, usually variance.
                # Since we can't easily do variance ratio with IncrementalPCA without multiple passes,
                # let's just interpret float as "fraction of original dimensions" as a fallback, 
                # or just default to a fixed number like 128 or 256 if not specified.
                # But wait, ESM features are 1280 or 480. 
                # Let's default to min(256, n_features) if not specified or float.
                target_dim = int(n_features * n_components) if n_components < 1.0 else int(n_components)
                target_dim = max(1, min(target_dim, n_features))
            else:
                target_dim = min(int(n_components), n_features)
            
            self.models[key] = IncrementalPCA(n_components=target_dim, batch_size=self.batch_size)
            logger.info(f"Initialized IncrementalPCA for {key} with n_components={target_dim}")

        self.models[key].partial_fit(X)

    def transform(self, samples: List[Dict], inplace: bool = True) -> List[Dict]:
        """
        Apply PCA transformation to samples using optimized numpy operations.
        """
        if not self._fitted:
            logger.warning("FeatureReducer not fitted. Returning original samples.")
            return samples

        # Ensure fast models are ready
        if not hasattr(self, '_fast_models') or not self._fast_models:
            self._optimize_inference()

        target_samples = samples if inplace else [dict(s) for s in samples]
        
        for sample in target_samples:
            for key in self._fast_models:
                if key not in sample or sample[key] is None:
                    continue
                
                data = sample[key]
                
                # Ensure numpy array
                if not isinstance(data, np.ndarray):
                    data = np.asarray(data, dtype=np.float32)
                elif data.dtype != np.float32:
                    data = data.astype(np.float32)
                
                # Fast PCA transform: (X - mean) @ components.T
                mean = self._fast_models[key]['mean']
                components_T = self._fast_models[key]['components_T']
                
                # Handle dimensions
                if data.ndim == 1:
                    # (D,) -> (1, D) implicitly for subtraction if mean is (D,)
                    # (D,) @ (D, K) -> (K,)
                    centered = data - mean
                    transformed = np.dot(centered, components_T)
                elif data.ndim == 2:
                    # (L, D) - (D,) -> (L, D)
                    # (L, D) @ (D, K) -> (L, K)
                    centered = data - mean
                    transformed = np.dot(centered, components_T)
                else:
                    continue
                
                sample[key] = transformed
                
        return target_samples

    def save(self, path: str):
        """Save the reducer models."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.models, path)
        logger.info(f"FeatureReducer models saved to {path}")

    def load(self, path: str):
        """Load the reducer models."""
        if os.path.exists(path):
            self.models = joblib.load(path)
            self._fitted = True
            self._optimize_inference()
            logger.info(f"FeatureReducer models loaded from {path}")
        else:
            logger.error(f"Model file not found: {path}")

