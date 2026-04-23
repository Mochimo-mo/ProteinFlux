import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import pickle
import logging
import re
from typing import Dict, Optional, Set, List, Tuple, Union

try:
    from Bio import SeqIO
    import Bio.PDB as bpdb
except ImportError:
    SeqIO = None
    bpdb = None

from fluxsite.utils.ptm_residue_info import PTM_RESIDUE_INFO, resolve_ptm_residue_info

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('unified_data_processor')

def _collect_observed_residues(*series_list):
    observed = set()
    for series in series_list:
        if series is None:
            continue
        try:
            iterator = series.dropna().tolist()
        except AttributeError:
            iterator = [value for value in series if value is not None]

        for value in iterator:
            value_str = str(value).strip()
            if not value_str:
                continue
            observed.add(value_str)
    return observed

def _resolve_and_log_residue_info(target_ptm_type: str, observed_residues, context: str):
    info = resolve_ptm_residue_info(target_ptm_type, observed_residues)
    base_expected = PTM_RESIDUE_INFO.get(info['ptm_type'], {}).get('expected_residues', [])
    base_expected_set = {res.upper() for res in base_expected}
    extras = sorted(info['expected_residues'] - base_expected_set)
    if extras:
        logger.info(f"{context} Residue {extras}, ")
    logger.info(f"{context} Use Residue: {info['residue_to_id']}")
    return info

class UnifiedDataProcessor:
    """
    Unified Data Processor for PTM Prediction.
    Supports both single dataset processing and separated train/test processing.
    Handles sequence (ESM-2) and structure (ESM-IF1) features from HDF5 or raw files.
    """

    def __init__(self, 
                 data_path: Optional[str] = None, 
                 train_path: Optional[str] = None,
                 test_path: Optional[str] = None,
                 esm_features_path: Optional[str] = None, 
                 pdb_dir: Optional[str] = None, 
                 window_size: int = 61, 
                 local_window_size: int = 31, 
                 target_ptm_type: str = 'acetylation', 
                 fasta_dir: Optional[str] = None, 
                 cache_dir: Optional[str] = None):
        """
        Initialize the unified data processor.

        Args:
            data_path (str, optional): Path to single CSV data file.
            train_path (str, optional): Path to training CSV data file.
            test_path (str, optional): Path to testing CSV data file.
            esm_features_path (str, optional): Path to ESM features HDF5 file.
            pdb_dir (str, optional): Directory containing PDB structure files.
            window_size (int): Size of sequence window around PTM sites.
            local_window_size (int): Size of local window for focused analysis.
            target_ptm_type (str): Target PTM type (e.g., 'acetylation', 'phosphorylation').
            fasta_dir (str, optional): Directory containing FASTA files.
            cache_dir (str, optional): Directory for caching processed data.
        """
        self.data_path = data_path
        self.train_path = train_path
        self.test_path = test_path
        self.esm_features_path = esm_features_path
        self.pdb_dir = pdb_dir
        self.target_ptm_label = str(target_ptm_type).strip()
        self.target_ptm_type = self.target_ptm_label.lower()
        self.window_size = window_size
        self.half_window = window_size // 2
        self.local_window_size = local_window_size
        self.local_half_window = local_window_size // 2
        self.fasta_dir = fasta_dir
        self.cache_dir = cache_dir
        
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        self._ptm_override_logged: Set[str] = set()
        self._fasta_cache: Dict[str, str] = {}
        
        # Determine mode
        self.is_separated = bool(train_path and test_path)
        
        if self.is_separated:
            self.train_data = pd.read_csv(train_path)
            self.test_data = pd.read_csv(test_path)
            logger.info(f"Loaded separated data: Train {len(self.train_data)}, Test {len(self.test_data)}")
            self._validate_separated_data()
        elif self.data_path:
            self.data = pd.read_csv(data_path)
            logger.info(f"Loaded single dataset: {len(self.data)} samples")
            self._validate_data()
        else:
            logger.warning("No data paths provided. Processor initialized in empty state.")

        # Initialize ESM features access
        if self.esm_features_path:
            self._init_esm_features()

    def _validate_data(self):
        """Validate single dataset format."""
        required_columns = ['uniprot_id', 'position', 'residue', 'ptm_type']
        missing_columns = [col for col in required_columns if col not in self.data.columns]

        if missing_columns:
            raise ValueError(f"Data missing required columns: {missing_columns}")

        if 'label' in self.data.columns:
            logger.info("Label column detected, using predefined labels.")
        else:
            logger.info("No label column, inferring labels from ptm_type.")

        # Check Residues
        residue_series = self.data['residue'] if 'residue' in self.data else None
        observed_residues = _collect_observed_residues(residue_series)
        residue_info = _resolve_and_log_residue_info(self.target_ptm_type, observed_residues, "Validation Phase")
        self.residue_info = residue_info
        self.residue_to_id = residue_info['residue_to_id']
        self.expected_residues = residue_info['expected_residues']
        self.other_residue_id = residue_info['other_id']

    def _validate_separated_data(self):
        """Validate separated train/test data format."""
        required_columns = ['uniprot_id', 'position', 'residue', 'ptm_type', 'label']

        for name, df in [("Train", self.train_data), ("Test", self.test_data)]:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"{name} data missing required columns: {missing}")

        # Check Residues (Combined)
        train_residues = _collect_observed_residues(self.train_data['residue'])
        residue_info = _resolve_and_log_residue_info(self.target_ptm_type, train_residues, "Separated Validation Phase")
        self.residue_info = residue_info
        self.residue_to_id = residue_info['residue_to_id']
        self.expected_residues = residue_info['expected_residues']
        self.other_residue_id = residue_info['other_id']

        test_residues = _collect_observed_residues(self.test_data['residue'])
        only_in_test = sorted(test_residues - train_residues)
        if only_in_test:
            logger.info(f"Test data contains residues not in training: {only_in_test}, mapping to Other")

    def _init_esm_features(self):
        """Initialize ESM feature file access."""
        if not os.path.exists(self.esm_features_path):
            raise FileNotFoundError(f"ESM features file not found: {self.esm_features_path}")
        
        try:
            with h5py.File(self.esm_features_path, 'r') as f:
                if 'proteins' in f:
                    protein_ids = list(f['proteins'].keys())
                    logger.info(f"ESM features file contains {len(protein_ids)} proteins")
                else:
                    raise ValueError("Invalid ESM features file format: missing 'proteins' group")
        except Exception as e:
            logger.error(f"Failed to read ESM features file: {e}")
            raise

    def _sanitize_ptm_value(self, raw_value, dataset_name: str) -> str:
        """Normalize PTM type value."""
        normalized_target = self.target_ptm_type
        canonical_value = self.target_ptm_label

        if raw_value is None:
            raw_str = ""
        else:
            raw_str = str(raw_value).strip()
        raw_lower = raw_str.lower()

        if raw_lower and raw_lower != normalized_target:
            reason = "Unknown PTM type"
            if raw_lower.startswith("non_") and raw_lower.endswith(normalized_target):
                reason = "Detected non-target PTM marker"
            key = f"{dataset_name}:{raw_lower}"
            if key not in self._ptm_override_logged:
                logger.warning(
                    f"{dataset_name} sample contains ptm_type='{raw_str}', mapping to '{canonical_value}' to avoid leakage ({reason})"
                )
                self._ptm_override_logged.add(key)

        return canonical_value

    def _extract_window_features(self, features, center_pos, window_size, half_window):
        """Extract window features with padding."""
        seq_len, feature_dim = features.shape
        
        start = max(0, center_pos - half_window)
        end = min(seq_len, center_pos + half_window + 1)
        
        window_features = features[start:end]
        
        left_pad_size = max(0, half_window - center_pos)
        right_pad_size = max(0, center_pos + half_window + 1 - seq_len)
        
        if left_pad_size > 0:
            left_pad = np.zeros((left_pad_size, feature_dim), dtype=features.dtype)
            window_features = np.vstack([left_pad, window_features])
        
        if right_pad_size > 0:
            right_pad = np.zeros((right_pad_size, feature_dim), dtype=features.dtype)
            window_features = np.vstack([window_features, right_pad])
        
        if window_features.shape[0] != window_size:
            if window_features.shape[0] < window_size:
                extra_pad = np.zeros((window_size - window_features.shape[0], feature_dim), dtype=features.dtype)
                window_features = np.vstack([window_features, extra_pad])
            else:
                window_features = window_features[:window_size]
        
        return window_features

    def _load_sequence_from_fasta(self, uniprot_id: str) -> Optional[str]:
        """Load sequence from FASTA."""
        if not self.fasta_dir:
            return None
            
        if uniprot_id in self._fasta_cache:
            return self._fasta_cache[uniprot_id]
            
        fasta_path = os.path.join(self.fasta_dir, f"{uniprot_id}.fasta")
        if not os.path.exists(fasta_path):
            # Try alt paths
            for ext in ['', '.fa', '.txt']:
                alt = os.path.join(self.fasta_dir, f"{uniprot_id}{ext}")
                if os.path.exists(alt):
                    fasta_path = alt
                    break
            else:
                return None
        
        try:
            if SeqIO is None:
                return None
            with open(fasta_path, "r") as f:
                for record in SeqIO.parse(f, "fasta"):
                    seq = str(record.seq)
                    self._fasta_cache[uniprot_id] = seq
                    return seq
        except Exception as e:
            logger.warning(f"Failed to read FASTA for {uniprot_id}: {e}")
            return None
        return None

    def _load_structure(self, uniprot_id):
        """Load PDB structure and extract CA coords."""
        if not self.pdb_dir or not bpdb:
            return None
            
        pdb_path = os.path.join(self.pdb_dir, f"{uniprot_id}.pdb")
        if not os.path.exists(pdb_path):
            return None
            
        try:
            parser = bpdb.PDBParser(QUIET=True)
            structure = parser.get_structure(uniprot_id, pdb_path)
            
            coords = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            coords.append(residue['CA'].get_coord())
                break 
            
            if not coords:
                return None
                
            return np.array(coords, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to load structure {uniprot_id}: {e}")
            return None

    def _extract_features_from_cache(self, cached_features, position):
        """Extract features from cached protein data."""
        try:
            sequence_features = cached_features['sequence_features']
            sequence_mean = cached_features['sequence_mean']
            sequence_cls = cached_features.get('sequence_cls')
            structure_features = cached_features.get('structure_features')
            structure_mean = cached_features.get('structure_mean')
            structure_cls = cached_features.get('structure_cls')

            seq_len = len(sequence_features)
            pos_idx = position - 1

            if pos_idx < 0 or pos_idx >= seq_len:
                return None

            seq_window_features = self._extract_window_features(
                sequence_features, pos_idx, self.window_size, self.half_window
            )
            
            seq_local_features = self._extract_window_features(
                sequence_features, pos_idx, self.local_window_size, self.local_half_window
            )

            struct_window_features = None
            struct_local_features = None
            if structure_features is not None:
                struct_window_features = self._extract_window_features(
                    structure_features, pos_idx, self.window_size, self.half_window
                )
                struct_local_features = self._extract_window_features(
                    structure_features, pos_idx, self.local_window_size, self.local_half_window
                )

            # Sequence string window
            seq_window_str = None
            sequence_str = cached_features.get('sequence_str')
            if sequence_str:
                start = pos_idx - self.half_window
                end = pos_idx + self.half_window + 1
                pad_left = max(0, -start) if start < 0 else 0
                pad_right = max(0, end - len(sequence_str)) if end > len(sequence_str) else 0
                
                safe_start = max(0, start)
                safe_end = min(len(sequence_str), end)
                
                seq_window_str = "X" * pad_left + sequence_str[safe_start:safe_end] + "X" * pad_right

            return {
                'sequence_features': seq_window_features,
                'local_features': seq_local_features,
                'global_features': sequence_mean,
                'sequence_cls': sequence_cls,
                'structure_features': struct_window_features,
                'structure_local': struct_local_features,
                'structure_global': structure_mean,
                'structure_cls': structure_cls,
                'sequence_window_str': seq_window_str,
                'sequence_length': seq_len
            }
        except Exception as e:
            logger.error(f"Error extracting features from cache: {e}")
            return None

    def process_dataset(self, data: pd.DataFrame, dataset_name: str) -> List[Dict]:
        """Process a dataframe of samples."""
        # Try loading from cache
        if self.cache_dir:
            base_name = "dataset"
            if dataset_name == "Train" and self.train_path:
                base_name = os.path.basename(self.train_path)
            elif dataset_name == "Test" and self.test_path:
                base_name = os.path.basename(self.test_path)
            elif self.data_path:
                base_name = os.path.basename(self.data_path)
            
            cache_key = f"{base_name}_{dataset_name}_{self.window_size}_{self.target_ptm_type}.pkl"
            cache_key = re.sub(r'[^\w\-\.]', '_', cache_key)
            cache_path = os.path.join(self.cache_dir, cache_key)
            
            if os.path.exists(cache_path):
                logger.info(f"Loading cached data from {cache_path}")
                try:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")

        processed_data = []
        logger.info(f"Processing {len(data)} {dataset_name} samples...")
        
        unique_proteins = data['uniprot_id'].unique()
        logger.info(f"Loading features for {len(unique_proteins)} proteins")
        
        protein_features_cache = {}
        
        # Load all needed proteins into memory
        if self.esm_features_path:
            try:
                with h5py.File(self.esm_features_path, 'r') as f:
                    for protein_id in unique_proteins:
                        if protein_id in f['proteins']:
                            p_grp = f['proteins'][protein_id]
                            
                            seq_str = None
                            if 'sequence' in p_grp.attrs:
                                try:
                                    sb = p_grp.attrs['sequence']
                                    seq_str = sb.decode('utf-8') if isinstance(sb, bytes) else str(sb)
                                except: pass
                            
                            if seq_str is None and self.fasta_dir:
                                seq_str = self._load_sequence_from_fasta(protein_id)
                            
                            # Load structure coords if needed
                            # Note: compute_micro_environment_features is not implemented here for brevity, 
                            # as it was complex and maybe not strictly needed for all models.
                            # If needed, it can be added.
                            
                            protein_features_cache[protein_id] = {
                                'sequence_features': p_grp['sequence_features'][()],
                                'sequence_mean': p_grp['sequence_mean'][()],
                                'sequence_cls': p_grp['sequence_cls'][()] if 'sequence_cls' in p_grp else None,
                                'structure_features': p_grp['structure_features'][()] if 'structure_features' in p_grp else None,
                                'structure_mean': p_grp['structure_mean'][()] if 'structure_mean' in p_grp else None,
                                'structure_cls': p_grp['structure_cls'][()] if 'structure_cls' in p_grp else None,
                                'sequence_str': seq_str
                            }
            except Exception as e:
                logger.error(f"Error loading HDF5 features: {e}")
                raise

        # Process samples
        for idx, row in data.iterrows():
            uniprot_id = row['uniprot_id']
            position = row['position']
            residue = row['residue']
            
            ptm_type_raw = row.get('ptm_type', self.target_ptm_label)
            ptm_type = self._sanitize_ptm_value(ptm_type_raw, dataset_name)
            
            label_val = row.get('label')
            has_label = pd.notna(label_val)
            label = int(label_val) if has_label else None
            
            if uniprot_id not in protein_features_cache:
                continue
                
            features = self._extract_features_from_cache(protein_features_cache[uniprot_id], position)
            if not features:
                continue
                
            processed_data.append({
                'uniprot_id': uniprot_id,
                'position': position,
                'residue': residue,
                'ptm_type': ptm_type,
                'target_ptm_type': self.target_ptm_label,
                'is_target_ptm': label if label is not None else 0, # Assuming binary classification on label
                'label': label,
                'has_label': has_label,
                **features
            })
            
            if len(processed_data) % 1000 == 0:
                logger.info(f"Processed {len(processed_data)} samples")

        logger.info(f"Finished processing {len(processed_data)} samples")
        
        # Save cache
        if self.cache_dir:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(processed_data, f)
                logger.info(f"Saved processed data to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
                
        return processed_data

    def get_train_data(self) -> pd.DataFrame:
        """Get training data DataFrame."""
        if not self.is_separated:
            raise ValueError("Processor not initialized for separated data.")
        return self.train_data

    def get_test_data(self) -> pd.DataFrame:
        """Get testing data DataFrame."""
        if not self.is_separated:
            raise ValueError("Processor not initialized for separated data.")
        return self.test_data

    def prepare_dataset(self):
        """Prepare single dataset."""
        if self.is_separated:
            raise ValueError("Processor initialized for separated data. Use prepare_train_dataset/prepare_test_dataset.")
        return self.process_dataset(self.data, "All")

    def prepare_train_dataset(self):
        """Prepare training dataset."""
        if not self.is_separated:
            raise ValueError("Processor not initialized for separated data.")
        return self.process_dataset(self.train_data, "Train")

    def prepare_test_dataset(self):
        """Prepare testing dataset."""
        if not self.is_separated:
            raise ValueError("Processor not initialized for separated data.")
        return self.process_dataset(self.test_data, "Test")


class UnifiedPTMDataset(Dataset):
    """Unified PTM Dataset compatible with various PTM types."""

    def __init__(
        self,
        processed_data,
        fixed_window_size: int = 61,
        fixed_local_size: int = 31,
        target_ptm_type: str = 'phosphorylation',
        residue_info: Optional[Dict[str, object]] = None,
    ) -> None:
        """Initialize dataset."""
        self.data = processed_data
        self.target_ptm_label = str(target_ptm_type).strip()
        self.target_ptm_type = self.target_ptm_label.lower()
        self._ptm_override_logged: Set[str] = set()

        # Normalize ptm_type
        for item in self.data:
            ptm_value = item.get('ptm_type')
            item['ptm_type'] = self._normalize_ptm_value(ptm_value)

        self.fixed_window_size = fixed_window_size
        self.fixed_local_size = fixed_local_size
        self.center_index = self.fixed_window_size // 2

        if residue_info is None:
            observed_residues = {item.get('residue') for item in processed_data}
            residue_info = resolve_ptm_residue_info(self.target_ptm_type, observed_residues)

        self.residue_info = residue_info
        self.residue_to_id = dict(residue_info['residue_to_id'])
        self.other_residue_id = int(residue_info['other_id'])
        self.residue_vocab_size = int(residue_info['vocab_size'])
        self.ptm_type_to_id = {self.target_ptm_label: 0}

    def _normalize_ptm_value(self, value) -> str:
        if value is None:
            return self.target_ptm_label

        value_str = str(value).strip()
        if not value_str:
            return self.target_ptm_label

        value_lower = value_str.lower()
        if value_lower != self.target_ptm_type:
            if value_lower not in self._ptm_override_logged:
                # Only log once per type
                self._ptm_override_logged.add(value_lower)
            return self.target_ptm_label

        return self.target_ptm_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        seq_features = item['sequence_features']
        feature_dim = seq_features.shape[1]

        if seq_features.shape[0] != self.fixed_window_size:
            if seq_features.shape[0] < self.fixed_window_size:
                padding = np.zeros((self.fixed_window_size - seq_features.shape[0], feature_dim))
                seq_features = np.vstack([seq_features, padding])
            else:
                seq_features = seq_features[:self.fixed_window_size]

        local_features = item['local_features']
        if local_features.shape[0] != self.fixed_local_size:
            if local_features.shape[0] < self.fixed_local_size:
                padding = np.zeros((self.fixed_local_size - local_features.shape[0], feature_dim))
                local_features = np.vstack([local_features, padding])
            else:
                local_features = local_features[:self.fixed_local_size]

        seq_features = torch.tensor(seq_features, dtype=torch.float32)
        local_features = torch.tensor(local_features, dtype=torch.float32)
        global_features = torch.tensor(item['global_features'], dtype=torch.float32)

        struct_features = None
        if item['structure_features'] is not None:
            struct_features = item['structure_features']
            if struct_features.shape[0] != self.fixed_window_size:
                struct_dim = struct_features.shape[1]
                if struct_features.shape[0] < self.fixed_window_size:
                    padding = np.zeros((self.fixed_window_size - struct_features.shape[0], struct_dim))
                    struct_features = np.vstack([struct_features, padding])
                else:
                    struct_features = struct_features[:self.fixed_window_size]
            struct_features = torch.tensor(struct_features, dtype=torch.float32)
        
        # Micro-environment features
        micro_env = item.get('micro_env_features')
        if micro_env is None:
            micro_env = np.zeros(6, dtype=np.float32)
        else:
            micro_env = np.asarray(micro_env, dtype=np.float32)
            if micro_env.shape[0] > 6:
                micro_env = micro_env[:6]
            elif micro_env.shape[0] < 6:
                micro_env = np.pad(micro_env, (0, 6 - micro_env.shape[0]), 'constant')
        micro_env = torch.tensor(micro_env, dtype=torch.float32)

        residue_value = item.get('residue')
        if residue_value is None or (isinstance(residue_value, float) and np.isnan(residue_value)):
            residue_key = None
        else:
            residue_key = str(residue_value).strip().upper()
        residue_type_id = self.residue_to_id.get(residue_key, self.other_residue_id)

        residue_types = torch.full((self.fixed_window_size,), self.other_residue_id, dtype=torch.long)
        
        # Try to populate full window residues
        seq_window_str = item.get('sequence_window_str')
        if seq_window_str and len(seq_window_str) == self.fixed_window_size:
            for i, char in enumerate(seq_window_str):
                # Skip padding/unknown
                if char == 'X': 
                    continue
                rid = self.residue_to_id.get(char.upper(), self.other_residue_id)
                residue_types[i] = rid
        else:
            # Fallback: only set center residue
            residue_types[self.center_index] = residue_type_id

        relative_position_offsets = torch.arange(self.fixed_window_size, dtype=torch.long) - self.center_index
        # position_ids should be absolute (0-indexed) for embedding layers to avoid negative indices
        position_ids = torch.arange(self.fixed_window_size, dtype=torch.long)

        site_indicator = torch.zeros(self.fixed_window_size, dtype=torch.float32)
        site_indicator[self.center_index] = 1.0

        ptm_type_value = item.get('ptm_type', self.target_ptm_label)
        ptm_type_id = self.ptm_type_to_id.get(ptm_type_value, 0)

        label_value = item.get('is_target_ptm', item.get('is_phosphorylated', 0))
        label_tensor = torch.tensor(float(label_value), dtype=torch.float32)

        return {
            'uniprot_id': item['uniprot_id'],
            'position': item['position'],
            'residue': item['residue'],
            'residue_type_id': residue_type_id,
            'residue_types': residue_types,
            'position_ids': position_ids,
            'relative_position_offsets': relative_position_offsets,
            'site_indicator': site_indicator,
            'sequence': {
                'window_features': seq_features,
                'local_features': local_features,
                'global_features': global_features,
                'position_ids': position_ids,
                'relative_position_offsets': relative_position_offsets,
                'site_indicator': site_indicator,
                'residue_types': residue_types,
            },
            'structure': struct_features,
            'micro_environment': micro_env,
            'label': label_tensor,
            'ptm_type': ptm_type_value,
            'ptm_type_id': ptm_type_id,
        }
