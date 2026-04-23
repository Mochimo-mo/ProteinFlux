import os
import sys

# Set HDF5 environment variable to avoid file-locking issues.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import pandas as pd
import logging
import atexit
import threading
from collections import OrderedDict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager, cpu_count
import pickle
import time
from typing import Dict, List, Tuple, Any, Optional, Set, Iterable, Iterator, Generator

import torch
from torch.utils.data import Dataset, Sampler

try:
    from Bio import SeqIO
    import Bio.PDB as bpdb
except ImportError:
    SeqIO = None
    bpdb = None

try:
    import lmdb  # type: ignore
    _LMDB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    lmdb = None  # type: ignore
    _LMDB_AVAILABLE = False

from fluxsite.utils.ptm_residue_info import resolve_ptm_residue_info

LMDB_CACHE_FILENAME = "protein_features.lmdb"
DEFAULT_LMDB_MAP_SIZE_BYTES = 8 * 1024 * 1024 * 1024  # 8 GB default capacity


def _resolve_lmdb_map_size() -> int:
    """Resolve LMDB map size from environment variables with sane fallback."""

    env_value = os.environ.get("PTM_LMDB_MAP_SIZE")
    if env_value:
        try:
            map_size = int(env_value)
            if map_size > 0:
                return map_size
            logger.warning("PTM_LMDB_MAP_SIZE, %s ", env_value)
        except ValueError:
            logger.warning(" PTM_LMDB_MAP_SIZE='%s', Use ", env_value)
    return DEFAULT_LMDB_MAP_SIZE_BYTES

logger = logging.getLogger(__name__)


def _sanitize_ptm_type(raw_value: Any, target_ptm_type: str, override_cache: Optional[Set[str]] = None, context: str = "") -> str:
    """Normalize ptm_type values to avoid leaking label information."""
    canonical_value = str(target_ptm_type).strip()
    normalized_target = canonical_value.lower()

    if raw_value is None:
        raw_str = ""
    else:
        raw_str = str(raw_value).strip()

    raw_lower = raw_str.lower()
    if raw_lower and raw_lower != normalized_target:
        reason = " PTM "
        if raw_lower.startswith("non_") and raw_lower.endswith(normalized_target):
            reason = " PTM "
        cache_key = f"{context}:{raw_lower}" if context else raw_lower
        if override_cache is not None and cache_key not in override_cache:
            logger.warning(
                "DataLoadingOptimizer: Samples ptm_type='%s' '%s' (%s)",
                raw_str,
                canonical_value,
                reason,
            )
            override_cache.add(cache_key)
        return canonical_value

    return canonical_value


def _normalize_uniprot_id(value: Any) -> Optional[str]:
    """Normalize a raw UniProt ID to a valid string; return None for invalid values."""
    if value is None:
        return None

    # Handle missing values/NaN
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except TypeError:
        # Some custom objects do not support pd.isna; ignore this error.
        pass

    # Convert to string and strip surrounding whitespace.
    normalized = str(value).strip()
    if not normalized:
        return None

    lower = normalized.lower()
    if lower in {"nan", "none", "null"}:
        return None

    return normalized


def normalize_uniprot_id(value: Any) -> Optional[str]:
    """Public API wrapping the internal UniProt ID normalization logic."""
    return _normalize_uniprot_id(value)


def _parse_env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_env_int(key: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
        return max(value, minimum)
    except ValueError:
        logger.warning(" %s='%s', Use %s", key, raw, default)
        return default


def _parse_env_float(key: str, default: float, minimum: float = 0.0) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        value = float(raw)
        if minimum is not None:
            value = max(value, minimum)
        return value
    except ValueError:
        logger.warning(" %s='%s', Use %.3f", key, raw, default)
        return default


def _extract_ca_coords_fast(pdb_path: str) -> Optional[np.ndarray]:
    """Quickly extract CA atom coordinates from a PDB file, avoiding Bio.PDB overhead."""
    coords = []
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
                    except (ValueError, IndexError):
                        continue
        if coords:
            return np.array(coords, dtype=np.float32)
    except Exception:
        pass
    return None


def load_sequence_from_fasta(protein_id: str, fasta_dir: str) -> Optional[str]:
    """Load a sequence from a FASTA file."""
    if not fasta_dir or not SeqIO:
        return None
    fasta_path = os.path.join(fasta_dir, f"{protein_id}.fasta")
    if not os.path.exists(fasta_path):
        return None
    try:
        record = SeqIO.read(fasta_path, "fasta")
        return str(record.seq)
    except Exception:
        return None


def _compute_single_protein_micro_env(protein_id: str, pdb_dir: str) -> Tuple[str, Optional[np.ndarray]]:
    """
    Compute micro-environment features for a single protein.

    This must be a top-level function so it can be pickled by ProcessPoolExecutor.
    """
    if not pdb_dir:
        return protein_id, None
    
    pdb_path = os.path.join(pdb_dir, f"{protein_id}.pdb")
    if not os.path.exists(pdb_path):
        return protein_id, None
        
    # Prefer the fast parser.
    coords = _extract_ca_coords_fast(pdb_path)
    
    # Fallback to Bio.PDB.
    if coords is None and bpdb:
        try:
            parser = bpdb.PDBParser(QUIET=True)
            structure = parser.get_structure(protein_id, pdb_path)
            coords_list = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            coords_list.append(residue['CA'].get_coord())
                break
            if coords_list:
                coords = np.array(coords_list, dtype=np.float32)
        except Exception:
            pass
            
    if coords is None or len(coords) == 0:
        return protein_id, None
        
    # Compute features.
    try:
        num_residues = len(coords)
        features = np.zeros((num_residues, 6), dtype=np.float32)
        
        # Vectorized distance-matrix computation (memory-heavy; huge proteins may need chunking).
        if num_residues < 2000:
            # Small proteins: use broadcasting for speed.
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dists_matrix = np.linalg.norm(diff, axis=2)
            
            for i in range(num_residues):
                dists = dists_matrix[i]
                
                # 1. Local density (number of neighbors within 10 Å)
                radius_10 = 10.0
                mask_10 = (dists < radius_10) & (dists > 1e-6)
                local_density_10 = np.sum(mask_10)
                
                # 2. Local concavity (distance to neighbor centroid)
                if local_density_10 > 0:
                    neighbor_coords = coords[mask_10]
                    centroid = np.mean(neighbor_coords, axis=0)
                    dist_to_centroid = np.linalg.norm(coords[i] - centroid)
                    
                    # 3. Mean neighbor distance (10 Å)
                    avg_dist_10 = np.mean(dists[mask_10])
                else:
                    dist_to_centroid = 0.0
                    avg_dist_10 = 0.0
                
                # 4. Tight packing (number of neighbors within 5 Å)
                radius_5 = 5.0
                mask_5 = (dists < radius_5) & (dists > 1e-6)
                local_density_5 = np.sum(mask_5)
                
                # 5. Mean tight-neighbor distance (5 Å)
                if local_density_5 > 0:
                    avg_dist_5 = np.mean(dists[mask_5])
                else:
                    avg_dist_5 = 0.0
                    
                # 6. Nearest-neighbor distance
                dists_no_self = dists[dists > 1e-6]
                if len(dists_no_self) > 0:
                    min_dist = np.min(dists_no_self)
                else:
                    min_dist = 10.0 # Default large value
                    
                features[i] = [
                    local_density_10 / 20.0,
                    dist_to_centroid / 5.0,
                    avg_dist_10 / 10.0,
                    local_density_5 / 10.0,
                    avg_dist_5 / 5.0,
                    min_dist / 5.0
                ]
        else:
            # Large proteins: compute in a loop to save memory.
            for i in range(num_residues):
                center_coord = coords[i]
                dists = np.linalg.norm(coords - center_coord, axis=1)
                
                # 1. Local density (number of neighbors within 10 Å)
                radius_10 = 10.0
                mask_10 = (dists < radius_10) & (dists > 1e-6)
                neighbor_indices_10 = np.where(mask_10)[0]
                local_density_10 = len(neighbor_indices_10)
                
                # 2. Local concavity (distance to neighbor centroid)
                if local_density_10 > 0:
                    neighbor_coords = coords[neighbor_indices_10]
                    centroid = np.mean(neighbor_coords, axis=0)
                    dist_to_centroid = np.linalg.norm(center_coord - centroid)
                    
                    # 3. Mean neighbor distance (10 Å)
                    avg_dist_10 = np.mean(dists[neighbor_indices_10])
                else:
                    dist_to_centroid = 0.0
                    avg_dist_10 = 0.0
                
                # 4. Tight packing (number of neighbors within 5 Å)
                radius_5 = 5.0
                mask_5 = (dists < radius_5) & (dists > 1e-6)
                neighbor_indices_5 = np.where(mask_5)[0]
                local_density_5 = len(neighbor_indices_5)
                
                # 5. Mean tight-neighbor distance (5 Å)
                if local_density_5 > 0:
                    avg_dist_5 = np.mean(dists[neighbor_indices_5])
                else:
                    avg_dist_5 = 0.0
                    
                # 6. Nearest-neighbor distance
                dists_no_self = dists[dists > 1e-6]
                if len(dists_no_self) > 0:
                    min_dist = np.min(dists_no_self)
                else:
                    min_dist = 10.0 # Default large value
                
                features[i] = [
                    local_density_10 / 20.0,
                    dist_to_centroid / 5.0,
                    avg_dist_10 / 10.0,
                    local_density_5 / 10.0,
                    avg_dist_5 / 5.0,
                    min_dist / 5.0
                ]
        
        return protein_id, features
    except Exception:
        return protein_id, None


class LazyProteinEntry:
    """Read window features from HDF5 on demand to avoid loading entire proteins into memory."""

    __slots__ = (
        "protein_id",
        "_group",
        "_sequence_dataset",
        "_sequence_cls_ds",
        "_sequence_mean_ds",
        "_structure_dataset",
        "_structure_cls_ds",
        "_structure_mean_ds",
        "_sequence_cls_cache",
        "_sequence_mean_cache",
        "_structure_cls_cache",
        "_structure_mean_cache",
        "_structure_coords_cache",
        "_micro_env_cache",
        "_sequence_str",
        "pdb_dir",
    )

    def __init__(self, protein_id: str, protein_group: "h5py.Group", fasta_dir: Optional[str] = None, pdb_dir: Optional[str] = None) -> None:
        self.protein_id = protein_id
        self.pdb_dir = pdb_dir
        self._group = protein_group
        
        self._sequence_dataset = None
        self._sequence_cls_ds = None
        self._sequence_mean_ds = None
        self._structure_dataset = None
        self._structure_cls_ds = None
        self._structure_mean_ds = None
        
        # Try to load sequence string from attributes
        try:
            seq_bytes = self._group.attrs.get("sequence")
            if isinstance(seq_bytes, bytes):
                self._sequence_str = seq_bytes.decode('utf-8')
            else:
                self._sequence_str = str(seq_bytes) if seq_bytes is not None else None
        except Exception:
            self._sequence_str = None

        # Fallback to FASTA if sequence is missing
        if not self._sequence_str and fasta_dir:
            self._sequence_str = self._load_sequence_from_fasta(protein_id, fasta_dir)

        self._sequence_cls_cache: Optional[np.ndarray] = None
        self._sequence_mean_cache: Optional[np.ndarray] = None
        self._structure_cls_cache: Optional[np.ndarray] = None
        self._structure_mean_cache: Optional[np.ndarray] = None
        self._structure_coords_cache: Optional[np.ndarray] = None
        self._micro_env_cache: Optional[np.ndarray] = None

    def _load_sequence_from_fasta(self, protein_id: str, fasta_dir: str) -> Optional[str]:
        if not SeqIO:
            return None
        fasta_path = os.path.join(fasta_dir, f"{protein_id}.fasta")
        if not os.path.exists(fasta_path):
            return None
        try:
            record = SeqIO.read(fasta_path, "fasta")
            return str(record.seq)
        except Exception:
            return None

    @property
    def sequence_dataset(self) -> Optional["h5py.Dataset"]:
        if self._sequence_dataset is None and self._group:
            self._sequence_dataset = self._group.get("sequence_features")
        return self._sequence_dataset

    @property
    def structure_dataset(self) -> Optional["h5py.Dataset"]:
        if self._structure_dataset is None and self._group:
            self._structure_dataset = self._group.get("structure_features")
        return self._structure_dataset

    def _cached_array(self, dataset: Optional["h5py.Dataset"], cache_name: str) -> Optional[np.ndarray]:
        if dataset is None:
            return None
        cached = getattr(self, cache_name)
        if cached is None:
            data = dataset[()]
            array = np.asarray(data, dtype=np.float32)
            setattr(self, cache_name, array)
            return array
        return cached

    @property
    def sequence_cls(self) -> Optional[np.ndarray]:
        if self._sequence_cls_ds is None and self._group:
            self._sequence_cls_ds = self._group.get("sequence_cls")
        return self._cached_array(self._sequence_cls_ds, "_sequence_cls_cache")

    @property
    def sequence_mean(self) -> Optional[np.ndarray]:
        if self._sequence_mean_ds is None and self._group:
            self._sequence_mean_ds = self._group.get("sequence_mean")
        return self._cached_array(self._sequence_mean_ds, "_sequence_mean_cache")

    @property
    def structure_cls(self) -> Optional[np.ndarray]:
        if self._structure_cls_ds is None and self.has_structure:
            self._structure_cls_ds = self._group.get("structure_cls")
        return self._cached_array(self._structure_cls_ds, "_structure_cls_cache")

    @property
    def structure_mean(self) -> Optional[np.ndarray]:
        if self._structure_mean_ds is None and self.has_structure:
            self._structure_mean_ds = self._group.get("structure_mean")
        return self._cached_array(self._structure_mean_ds, "_structure_mean_cache")

    @property
    def sequence_length(self) -> int:
        if self.sequence_dataset is None:
            return 0
        return int(self.sequence_dataset.shape[0])

    @property
    def has_structure(self) -> bool:
        return self.structure_dataset is not None

    def clear_cached_vectors(self) -> None:
        """Release cached global vectors to reduce peak memory for long-running jobs."""
        self._sequence_cls_cache = None
        self._sequence_mean_cache = None
        self._structure_cls_cache = None
        self._structure_mean_cache = None

    def close(self) -> None:
        """Detach from underlying HDF5 objects, used during cache eviction."""
        self.clear_cached_vectors()
        self._group = None
        self._sequence_dataset = None
        self._sequence_cls_ds = None
        self._sequence_mean_ds = None
        self._structure_dataset = None
        self._structure_cls_ds = None
        self._structure_mean_ds = None

    @property
    def sequence(self) -> Optional[str]:
        return self._sequence_str

    @property
    def micro_env_features(self) -> Optional[np.ndarray]:
        """Get or compute whole-protein micro-environment features."""
        if self._micro_env_cache is not None:
            return self._micro_env_cache

        # If not cached, try to load from HDF5.
        if self._group and "micro_env_features" in self._group:
            try:
                self._micro_env_cache = self._group["micro_env_features"][()]
                return self._micro_env_cache
            except Exception:
                pass
            
        # If structure coordinates exist, try to compute on the fly.
        coords = self.structure_coords
        if coords is not None:
            try:
                # Compute micro-environment features for the whole protein (per residue).
                # This can be expensive, so it is computed only when needed.
                # To avoid circular imports, keep the core logic here or call a static method.
                features_list = []
                for i in range(len(coords)):
                    feat = DataLoadingOptimizer._compute_single_residue_micro_env(coords, i)
                    features_list.append(feat)
                
                self._micro_env_cache = np.array(features_list, dtype=np.float32)
                return self._micro_env_cache
            except Exception as e:
                # logger.warning(f"Failed to compute micro-environment features on the fly: {e}")
                pass
                
        return None

    @micro_env_features.setter
    def micro_env_features(self, value: np.ndarray):
        self._micro_env_cache = value

    @property
    def structure_coords(self) -> Optional[np.ndarray]:
        if self._structure_coords_cache is not None:
            return self._structure_coords_cache
            
        if not self.pdb_dir:
            return None
            
        pdb_path = os.path.join(self.pdb_dir, f"{self.protein_id}.pdb")
        if not os.path.exists(pdb_path):
            return None
            
        # Prefer the fast parser.
        coords = _extract_ca_coords_fast(pdb_path)
        if coords is not None:
            self._structure_coords_cache = coords
            return coords
            
        # Fallback to Bio.PDB.
        if bpdb:
            try:
                parser = bpdb.PDBParser(QUIET=True)
                structure = parser.get_structure(self.protein_id, pdb_path)
                coords_list = []
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if 'CA' in residue:
                                coords_list.append(residue['CA'].get_coord())
                    break
                
                if coords_list:
                    self._structure_coords_cache = np.array(coords_list, dtype=np.float32)
                    return self._structure_coords_cache
            except Exception:
                pass
            
        return None


class DataLoadingOptimizer:
    """Data loading optimizer."""
    
    def __init__(self, esm_features_path: str, num_workers: int = None, *, lazy_loading: Optional[bool] = None, fasta_dir: Optional[str] = None, pdb_dir: Optional[str] = None):
        """
        Initialize the data loading optimizer.
        
        Args:
            esm_features_path: Path to the ESM feature file.
            num_workers: Number of worker processes (defaults to CPU cores with an upper bound).
            fasta_dir: FASTA directory used to supplement missing sequences.
            pdb_dir: PDB directory used to compute micro-environment geometry features.
        """
        self.esm_features_path = esm_features_path
        self.num_workers = num_workers or min(cpu_count(), 8)  # Cap the maximum number of processes
        self.fasta_dir = fasta_dir
        self.pdb_dir = pdb_dir
        self._ptm_override_cache: Set[str] = set()
        self._missing_micro_env_warning_count = 0
        self._lmdb_map_size = _resolve_lmdb_map_size()
        self.lazy_loading = lazy_loading if lazy_loading is not None else _parse_env_bool("PTM_LAZY_LOADING", True)
        self._h5_cache_bytes = _parse_env_int("PTM_H5_CACHE_BYTES", 128 * 1024 * 1024, minimum=8 * 1024 * 1024)
        self._h5_cache_slots = _parse_env_int("PTM_H5_CACHE_SLOTS", 1_048_575, minimum=1024)
        self._h5_cache_w0 = _parse_env_float("PTM_H5_CACHE_W0", 0.75, minimum=0.01)
        self._h5_file: Optional[h5py.File] = None
        self._proteins_group: Optional[h5py.Group] = None
        self._lazy_cache: Dict[str, Any] = {}

        if self.lazy_loading:
            logger.info(
                " Load Enable, %s (HDF5 cache: %.2f MB, slots=%d)",
                self.esm_features_path,
                self._h5_cache_bytes / 1024 / 1024,
                self._h5_cache_slots,
            )
        else:
            logger.info(" Load Disable, Use Load ")

        atexit.register(self.close)
        
        logger.info(f"InitializeDataLoad,Use {self.num_workers} ")

    def _load_sequence_from_fasta(self, protein_id: str) -> Optional[str]:
        if not self.fasta_dir or not SeqIO:
            return None
        fasta_path = os.path.join(self.fasta_dir, f"{protein_id}.fasta")
        if not os.path.exists(fasta_path):
            return None
        try:
            record = SeqIO.read(fasta_path, "fasta")
            return str(record.seq)
        except Exception:
            return None

    @property
    def h5_cache_settings(self) -> Dict[str, float]:
        """Return current HDF5 cache settings for reuse by external lazy loaders."""
        return {
            "rdcc_nbytes": self._h5_cache_bytes,
            "rdcc_nslots": self._h5_cache_slots,
            "rdcc_w0": self._h5_cache_w0,
        }

    def _get_h5_file(self, mode: str = "r") -> h5py.File:
        # Check if we need to reopen (either file is None, or mode mismatch)
        need_reopen = self._h5_file is None
        if not need_reopen:
            # If current mode is 'r' but we need 'r+', we must reopen
            if mode == "r+" and self._h5_file.mode == "r":
                need_reopen = True
        
        if need_reopen:
            if self._h5_file is not None:
                try:
                    # CRITICAL: When closing/reopening, all existing LazyProteinEntry objects 
                    # holding references to datasets in the old file handle become invalid.
                    # We must clear the cache to force reloading from the new handle.
                    for entry in self._lazy_cache.values():
                        try:
                            if isinstance(entry, LazyProteinEntry):
                                entry.close()
                        except Exception:
                            pass
                    self._lazy_cache.clear()
                    
                    self._h5_file.close()
                except Exception:
                    pass
                self._h5_file = None
                # CRITICAL: Invalidate the group cache when file is closed
                self._proteins_group = None
            
            try:
                self._h5_file = h5py.File(
                    self.esm_features_path,
                    mode,
                    rdcc_nbytes=self._h5_cache_bytes,
                    rdcc_nslots=self._h5_cache_slots,
                    rdcc_w0=self._h5_cache_w0,
                )
                logger.info(
                    " HDF5 (mode=%s),Use rdcc_nbytes=%.2f MB, rdcc_nslots=%d, rdcc_w0=%.2f",
                    mode,
                    self._h5_cache_bytes / 1024 / 1024,
                    self._h5_cache_slots,
                    self._h5_cache_w0,
                )
            except Exception as exc:
                # Ensure state is clean on failure
                self._h5_file = None
                self._proteins_group = None
                raise RuntimeError(f" {self.esm_features_path} (mode={mode}): {exc}") from exc
                
        return self._h5_file

    def _get_proteins_group(self, mode: str = "r") -> "h5py.Group":
        # Check conditions to refresh group:
        # 1. Group is None
        # 2. File is None (should imply 1, but safety first)
        # 3. Mode upgrade needed (r -> r+)
        refresh = (self._proteins_group is None) or (self._h5_file is None)
        
        if not refresh and mode == "r+" and self._h5_file is not None:
             if hasattr(self._h5_file, 'mode') and self._h5_file.mode == "r":
                 refresh = True

        if refresh:
            h5_file = self._get_h5_file(mode)
            if "proteins" not in h5_file:
                if mode == "r+":
                    self._proteins_group = h5_file.create_group("proteins")
                else:
                    raise KeyError(f" {self.esm_features_path} 'proteins' ")
            else:
                self._proteins_group = h5_file["proteins"]
        return self._proteins_group

    def close(self) -> None:
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception as exc:
                logger.warning(": %s", exc)
            finally:
                self._h5_file = None
                self._proteins_group = None
        
    def create_protein_index(self, output_path: str = None) -> Dict[str, Dict]:
        """
        Create a protein index for fast lookup.
        
        Args:
            output_path: Output path to save the index file.
            
        Returns:
            Dict: Protein index dictionary.
        """
        logger.info("CreateProtein.")
        
        protein_index = {}
        
        proteins_group = self._get_proteins_group()

        for protein_id in proteins_group.keys():
            protein_group = proteins_group[protein_id]

            # Record basic protein information.
            seq_features = protein_group['sequence_features']
            seq_len = seq_features.shape[0]
            seq_dim = seq_features.shape[1]

            protein_info = {
                'sequence_length': seq_len,
                'sequence_dim': seq_dim,
                'has_structure': 'structure_features' in protein_group
            }

            if protein_info['has_structure']:
                struct_features = protein_group['structure_features']
                protein_info['structure_dim'] = struct_features.shape[1]

            protein_index[protein_id] = protein_info
                
        logger.info(f"Create {len(protein_index)} Protein ")
        
        # Save index.
        if output_path:
            with open(output_path, 'wb') as f:
                pickle.dump(protein_index, f)
            logger.info(f"Protein Save: {output_path}")
            
        return protein_index
    
    def precompute_micro_environment(self, protein_ids: List[str], cache_dir: str = None, use_lmdb: bool = True, save_to_h5: bool = True) -> Dict[str, np.ndarray]:
        """Batch precompute micro-environment features for proteins."""
        if not self.pdb_dir:
            logger.warning(" pdb_dir,Skip ")
            return {}
            
        results = {}
        micro_env_cache_path = None
        lmdb_env = None
        
        # 0. Prefer loading from LMDB/Pickle (LMDB first).
        remaining_ids = list(protein_ids) # Start with all IDs
        lmdb_loaded_ids = set()
        
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            if use_lmdb and _LMDB_AVAILABLE:
                lmdb_path = cache_path / "micro_env_features.lmdb"
                try:
                    # Use map_size (auto-growth strategy via configured size).
                    lmdb_env = lmdb.open(
                        str(lmdb_path),
                        map_size=self._lmdb_map_size,
                        subdir=False,
                        create=True,
                        lock=True, 
                        readahead=False, 
                        meminit=False
                    )
                    # Load existing entries from LMDB.
                    with lmdb_env.begin(write=False) as txn:
                        cursor = txn.cursor()
                        for pid in list(remaining_ids):
                            payload = cursor.get(pid.encode("utf-8"))
                            if payload is not None:
                                try:
                                    feat = pickle.loads(payload)
                                    # [Fix] Validate feature dimension (must be 6)
                                    if len(feat.shape) == 2 and feat.shape[1] == 6:
                                        results[pid] = feat
                                        lmdb_loaded_ids.add(pid)
                                        remaining_ids.remove(pid)
                                except Exception:
                                    continue
                    if results:
                        logger.info(f" LMDB Load, {len(results)} ")
                except Exception as e:
                    logger.warning(f" micro_env LMDB Failed: {e}, Use ")
                    if lmdb_env:
                        lmdb_env.close()
                        lmdb_env = None

            if not lmdb_env and remaining_ids:
                micro_env_cache_path = cache_path / "micro_env_cache.pkl"
                if micro_env_cache_path.exists():
                    try:
                        with open(micro_env_cache_path, 'rb') as f:
                            pickle_results = pickle.load(f)
                            for pid in list(remaining_ids):
                                if pid in pickle_results:
                                    feat = pickle_results[pid]
                                    if len(feat.shape) == 2 and feat.shape[1] == 6:
                                        results[pid] = feat
                                        lmdb_loaded_ids.add(pid)
                                        remaining_ids.remove(pid)
                        logger.info(f" Pickle Load, {len(results)} ")
                    except Exception as e:
                        logger.warning(f"Load Failed: {e}")
        
        # 0.5 If save_to_h5 is enabled, ensure LMDB-loaded entries also exist in HDF5.
        # Otherwise, LazyProteinEntry may not be able to load them.
        if save_to_h5 and lmdb_loaded_ids:
            try:
                # Check whether entries exist in HDF5.
                proteins_group = self._get_proteins_group("r")
                missing_in_h5 = {}
                for pid in lmdb_loaded_ids:
                    # Check both the group and the dataset.
                    in_h5 = False
                    if pid in proteins_group:
                        pg = proteins_group[pid]
                        if "micro_env_features" in pg:
                            in_h5 = True
                    
                    if not in_h5:
                        missing_in_h5[pid] = results[pid]
                
                if missing_in_h5:
                    logger.info(f" {len(missing_in_h5)} LMDB HDF5,.")
                    # Retry mechanism for HDF5 writing
                    max_retries = 1 # [Optimization] Reduce retries to avoid blocking
                    for attempt in range(max_retries):
                        try:
                            self.save_micro_environment_to_h5(missing_in_h5)
                            break
                        except Exception as e:
                            if "file is already open" in str(e):
                                logger.warning(f"HDF5,Skip (Data LMDB ): {e}")
                                break
                            
                            if attempt < max_retries - 1:
                                logger.warning(f" LMDB HDF5 Failed ({attempt+1}/{max_retries}): {e},.")
                                time.sleep(1)
                            else:
                                logger.warning(f" LMDB HDF5 Failed (Error): {e}")
            except Exception as e:
                logger.error(f" LMDB HDF5: {e}")

        # 1. Next, try loading from HDF5 (fallback; best-effort).
        if remaining_ids and save_to_h5:
            try:
                # Try reading in "r" mode; ignore failures.
                # Note: if a previous HDF5 write failed, data may be unavailable here.
                proteins_group = self._get_proteins_group("r")
                h5_loaded_count = 0
                for pid in list(remaining_ids):
                    if pid in proteins_group and "micro_env_features" in proteins_group[pid]:
                        try:
                            feat = proteins_group[pid]["micro_env_features"][()]
                            if feat is not None and feat.size > 0 and len(feat.shape) == 2 and feat.shape[1] == 6:
                                results[pid] = feat
                                remaining_ids.remove(pid)
                                h5_loaded_count += 1
                        except Exception:
                            pass  # Failed to read one entry; ignore.
                
                if h5_loaded_count > 0:
                    logger.info(f" HDF5 Load {h5_loaded_count} Protein ")
            except Exception as e:
                logger.warning(f" HDF5 Failed (): {e}")

        # 3. Compute the remaining entries.
        if not remaining_ids:
            if lmdb_env:
                lmdb_env.close()
            return results
            
        logger.info(f"Start {len(remaining_ids)} Protein (Use {self.num_workers} ).")
        
        start_time = time.time()
        
        # Parallel computation with a process pool.
        newly_computed = {}
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit tasks in batches to avoid memory issues.
            batch_size = 500
            for i in range(0, len(remaining_ids), batch_size):
                batch_ids = remaining_ids[i:i+batch_size]
                futures = [executor.submit(_compute_single_protein_micro_env, pid, self.pdb_dir) for pid in batch_ids]
                
                current_batch_results = {}
                for future in futures:
                    try:
                        pid, features = future.result()
                        if features is not None:
                            results[pid] = features
                            current_batch_results[pid] = features
                            newly_computed[pid] = features
                    except Exception as e:
                        logger.error(f" Protein Failed: {e}")
                
                # Flush to LMDB promptly.
                lmdb_success = False
                if lmdb_env and current_batch_results:
                    try:
                        with lmdb_env.begin(write=True) as txn:
                            for pid, feat in current_batch_results.items():
                                txn.put(pid.encode("utf-8"), pickle.dumps(feat, protocol=pickle.HIGHEST_PROTOCOL))
                        lmdb_success = True
                    except Exception as e:
                        logger.warning(f" LMDB Failed: {e}")

                # Flush to HDF5 promptly (enables resumability).
                # [Optimization] If LMDB write succeeds, HDF5 write becomes optional (best effort).
                # Avoid training stalls caused by HDF5 file locks.
                if save_to_h5 and current_batch_results:
                    try:
                        self.save_micro_environment_to_h5(current_batch_results)
                    except Exception as e:
                        if lmdb_success:
                            logger.debug(f"HDF5 Skip (Save LMDB): {e}")
                        else:
                            logger.warning(f"HDF5 Failed LMDB Unavailable: {e}")
                
                if (i + batch_size) % 2000 == 0 or (i + batch_size) >= len(remaining_ids):
                    logger.info(f" {min(i + batch_size, len(remaining_ids))}/{len(remaining_ids)} Protein.")
                    
        duration = time.time() - start_time
        logger.info(f" Completed, {duration:.2f}, {len(newly_computed)} Protein")
        
        # 5. If using Pickle cache and new entries were computed, persist them.
        if not lmdb_env and micro_env_cache_path and newly_computed:
            try:
                # Rewriting the whole pickle can be slow, but it is a fallback when LMDB is unavailable.
                with open(micro_env_cache_path, 'wb') as f:
                    pickle.dump(results, f)
                logger.info(f" Save Pickle: {micro_env_cache_path}")
            except Exception as e:
                logger.warning(f"Save Failed: {e}")
        
        if lmdb_env:
            lmdb_env.close()
                
        return results

    def save_micro_environment_to_h5(self, micro_env_dict: Dict[str, np.ndarray]) -> None:
        """Save precomputed micro-environment features back into the HDF5 file."""
        # [User Request] Disabled HDF5 saving to avoid file locking issues and errors
        return
        
        if not micro_env_dict:
            return
            
        try:
            proteins_group = self._get_proteins_group("r+")
            count = 0
            missing_pids = []
            for pid, features in micro_env_dict.items():
                if pid in proteins_group:
                    pg = proteins_group[pid]
                    # If pg is a dataset rather than a group, the HDF5 structure is invalid.
                    if not isinstance(pg, h5py.Group):
                        logger.warning(f"HDF5 Protein {pid},Skip ")
                        continue
                        
                    if "micro_env_features" in pg:
                        del pg["micro_env_features"]
                    pg.create_dataset(
                        "micro_env_features", 
                        data=features, 
                        compression="gzip", 
                        compression_opts=4
                    )
                    count += 1
                else:
                    # If the protein does not exist in HDF5 at all, do not create a group because missing
                    # sequence_features makes it unusable for downstream loading.
                    missing_pids.append(pid)
                    # logger.debug(f"Protein {pid} is not present in HDF5; skip writing micro-env features")
            
            if count > 0:
                logger.info(f" /Create {count} Protein HDF5")
            if missing_pids:
                logger.warning(f" {len(missing_pids)} Protein HDF5 CreateFailed")
        except Exception as e:
            if "file is already open" in str(e) or "BlockingIOError" in str(e):
                logger.warning(f"Save HDF5 (, Skip): {e}")
            else:
                logger.error(f"Save HDF5 Failed: {e}")

    def save_micro_environment_features(self, micro_env_dict: Dict[str, np.ndarray], cache_dir: str = None) -> None:
        """
        Save micro-environment features into LMDB (if available) and/or HDF5.

        This is used by OnDemandPTMDataset to persist results after fallback computation.
        """
        if not micro_env_dict:
            return

        # 1. Update in-memory cache (so subsequent calls get it fast)
        for pid, feat in micro_env_dict.items():
            self._micro_env_cache_external[pid] = feat

        # 2. Save to HDF5 (Persistent storage)
        # [User Request] Disabled HDF5 saving to avoid file locking issues and errors
        # self.save_micro_environment_to_h5(micro_env_dict)
        
        # 3. Save to LMDB (Fast cache) if available
        # We try to use the cache_dir if provided, or fallback to inferring from internal state if possible
        # However, DataLoadingOptimizer doesn't store cache_dir. 
        # But we can try to look for 'micro_env_features.lmdb' in standard locations if needed.
        # For now, we rely on HDF5 persistence which is robust enough.
        # If the user provided cache_dir to precompute_micro_environment, we might have stored it? No.
        
        # Optimization: If we have an open LMDB env (unlikely here as it's closed after preload), we could use it.
        # Given the complexity of reopening LMDB here without a known path, we skip LMDB write for on-demand fallback.
        # HDF5 write is sufficient for persistence across epochs.
        pass

    def preload_protein_features(self, protein_ids: List[str], 
                               cache_dir: str = None) -> Dict[str, Any]:
        """
        Preload features for the specified proteins into memory or cache files.
        
        Args:
            protein_ids: List of protein IDs to preload.
            cache_dir: Cache directory path.
            
        Returns:
            Dict: In lazy-loading mode, returns prepared LazyProteinEntry objects (with precomputed micro-env
            features injected when available). In eager mode, returns a full protein feature dictionary.
        """
        logger.info(f" Load {len(protein_ids)} Protein.")

        start_time = time.time()
        
        # Precompute micro-environment features (single pass; significantly improves downstream training speed).
        micro_env_data = {}
        if self.pdb_dir:
            micro_env_data = self.precompute_micro_environment(protein_ids, cache_dir=cache_dir)

        if self.lazy_loading:
            # In lazy-loading mode, build and return LazyProteinEntry objects so batch_extract_features can read
            # features via the entry.
            proteins_group = self._get_proteins_group("r")
            protein_features = {}
            
            # Use self._lazy_cache to avoid creating duplicate objects.
            for pid in protein_ids:
                if pid not in proteins_group:
                    continue
                
                pg = proteins_group[pid]
                if "sequence_features" not in pg:
                    continue

                if pid in self._lazy_cache:
                    entry = self._lazy_cache[pid]
                else:
                    entry = LazyProteinEntry(pid, pg, self.fasta_dir, self.pdb_dir)
                    self._lazy_cache[pid] = entry
                
                # Inject precomputed micro-environment features.
                if pid in micro_env_data:
                    entry.micro_env_features = micro_env_data[pid]
                
                protein_features[pid] = entry

            load_time = time.time() - start_time
            logger.info(
                " Load Completed, %d Protein LazyEntry (%d ), %.2f ",
                len(protein_features),
                len(micro_env_data),
                load_time
            )
            return protein_features

        # Eager mode
        protein_features: Dict[str, Any] = {}

        cache_path: Optional[Path] = None
        lmdb_env: Optional["lmdb.Environment"] = None
        use_lmdb_cache = False

        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

            if _LMDB_AVAILABLE:
                lmdb_path = cache_path / LMDB_CACHE_FILENAME
                try:
                    lmdb_env = lmdb.open(
                        str(lmdb_path),
                        map_size=self._lmdb_map_size,
                        subdir=False,
                        create=True,
                        lock=True,
                        readahead=True,
                        max_readers=max(64, self.num_workers * 4),
                    )
                    use_lmdb_cache = True
                    logger.debug("Use LMDB: %s", lmdb_path)
                except Exception as exc:  # pragma: no cover - unexpected environment issues
                    logger.warning(" LMDB Failed, Pickle: %s", exc)
                    lmdb_env = None
            else:
                logger.warning(" lmdb, Pickle ")

        def _write_batch_to_lmdb(env: Optional["lmdb.Environment"], items: Dict[str, Dict[str, Any]]) -> None:
            if env is None or not items:
                return

            serialized = {pid: pickle.dumps(features, protocol=pickle.HIGHEST_PROTOCOL) for pid, features in items.items()}
            attempts = 0
            while attempts < 4:
                try:
                    with env.begin(write=True) as txn:
                        for pid, payload in serialized.items():
                            txn.put(pid.encode("utf-8"), payload)
                    env.sync()
                    return
                except lmdb.MapFullError as exc:  # pragma: no cover - requires full map
                    attempts += 1
                    current_size = env.info().get("map_size", self._lmdb_map_size)
                    new_size = max(current_size * 2, current_size + sum(len(v) for v in serialized.values()))
                    logger.warning(
                        "LMDB (%.2f GB), %.2f GB (%d )",
                        current_size / 1024 ** 3,
                        new_size / 1024 ** 3,
                        attempts,
                    )
                    env.set_mapsize(new_size)
                except Exception as exc:  # pragma: no cover - unexpected write error
                    logger.warning(" LMDB Failed: %s", exc)
                    return

        cached_count = 0
        remaining_ids = set(protein_ids)

        try:
            if use_lmdb_cache and lmdb_env is not None and remaining_ids:
                with lmdb_env.begin(write=False) as txn:
                    for protein_id in list(remaining_ids):
                        payload = txn.get(protein_id.encode("utf-8"))
                        if payload is None:
                            continue
                        try:
                            protein_features[protein_id] = pickle.loads(payload)
                            remaining_ids.discard(protein_id)
                            cached_count += 1
                        except Exception as exc:
                            logger.warning(" LMDB Protein %s Failed, Load: %s", protein_id, exc)

                # Migrate legacy Pickle cache files.
                legacy_loaded: Dict[str, Dict[str, Any]] = {}
                if cache_path is not None and remaining_ids:
                    for protein_id in list(remaining_ids):
                        legacy_file = cache_path / f"{protein_id}.pkl"
                        if not legacy_file.exists():
                            continue
                        try:
                            with open(legacy_file, "rb") as fh:
                                features = pickle.load(fh)
                            protein_features[protein_id] = features
                            remaining_ids.discard(protein_id)
                            legacy_loaded[protein_id] = features
                            cached_count += 1
                        except Exception as exc:
                            logger.warning(" %s Failed: %s", legacy_file, exc)

                    if legacy_loaded:
                        _write_batch_to_lmdb(lmdb_env, legacy_loaded)
                        for protein_id in legacy_loaded:
                            legacy_file = cache_path / f"{protein_id}.pkl"
                            try:
                                legacy_file.unlink()
                            except OSError as exc:
                                logger.debug(" %s Failed: %s", legacy_file, exc)

            elif cache_path is not None and remaining_ids:
                # Pickle cache fallback logic.
                for protein_id in list(remaining_ids):
                    cache_file = cache_path / f"{protein_id}.pkl"
                    if not cache_file.exists():
                        continue
                    try:
                        with open(cache_file, "rb") as fh:
                            protein_features[protein_id] = pickle.load(fh)
                        remaining_ids.discard(protein_id)
                        cached_count += 1
                    except Exception as exc:
                        logger.warning("Load %s Failed: %s", cache_file, exc)

            if cache_path is not None:
                logger.info(" Load %d Protein ", cached_count)

            remaining_to_load = list(remaining_ids)
            newly_loaded: Dict[str, Dict[str, Any]] = {}

            if remaining_to_load:
                with h5py.File(self.esm_features_path, "r") as f:
                    proteins_group = f["proteins"]

                    for protein_id in remaining_to_load:
                        if protein_id not in proteins_group:
                            logger.warning("Protein %s ", protein_id)
                            continue

                        protein_group = proteins_group[protein_id]
                        
                        if "sequence_features" not in protein_group:
                            # logger.warning(f"Protein {protein_id} is missing sequence_features; skip")
                            continue

                        features = {
                            "sequence_features": protein_group["sequence_features"][()],
                            "sequence_cls": protein_group["sequence_cls"][()],
                            "sequence_mean": protein_group["sequence_mean"][()],
                        }
                        
                        # Try to load sequence string
                        try:
                            seq_bytes = protein_group.attrs.get("sequence")
                            if isinstance(seq_bytes, bytes):
                                features["sequence_str"] = seq_bytes.decode('utf-8')
                            else:
                                features["sequence_str"] = str(seq_bytes) if seq_bytes is not None else None
                        except Exception:
                            features["sequence_str"] = None

                        if not features["sequence_str"]:
                            features["sequence_str"] = self._load_sequence_from_fasta(protein_id)

                        # Load structure coords if pdb_dir provided
                        features["structure_coords"] = None
                        if self.pdb_dir and bpdb:
                             pdb_path = os.path.join(self.pdb_dir, f"{protein_id}.pdb")
                             if os.path.exists(pdb_path):
                                 try:
                                     parser = bpdb.PDBParser(QUIET=True)
                                     structure = parser.get_structure(protein_id, pdb_path)
                                     coords = []
                                     for model in structure:
                                         for chain in model:
                                             for residue in chain:
                                                 if 'CA' in residue:
                                                     coords.append(residue['CA'].get_coord())
                                         break
                                     if coords:
                                         features["structure_coords"] = np.array(coords, dtype=np.float32)
                                 except Exception:
                                     pass

                        if "structure_features" in protein_group:
                            features.update(
                                {
                                    "structure_features": protein_group["structure_features"][()],
                                    "structure_cls": protein_group["structure_cls"][()],
                                    "structure_mean": protein_group["structure_mean"][()],
                                }
                            )
                        else:
                            features.update(
                                {
                                    "structure_features": None,
                                    "structure_cls": None,
                                    "structure_mean": None,
                                }
                            )

                        protein_features[protein_id] = features
                        newly_loaded[protein_id] = features

                if newly_loaded:
                    if use_lmdb_cache and lmdb_env is not None:
                        _write_batch_to_lmdb(lmdb_env, newly_loaded)
                    elif cache_path is not None:
                        for protein_id, features in newly_loaded.items():
                            cache_file = cache_path / f"{protein_id}.pkl"
                            try:
                                with open(cache_file, "wb") as fh:
                                    pickle.dump(features, fh)
                            except Exception as exc:
                                logger.warning("Save %s Failed: %s", cache_file, exc)

        finally:
            if lmdb_env is not None:
                lmdb_env.close()

        # Ensure all loaded features include precomputed micro-environment features (eager mode).
        if micro_env_data:
            for pid, feat in protein_features.items():
                if pid in micro_env_data:
                    if isinstance(feat, dict):
                        if "micro_env_features" not in feat or feat["micro_env_features"] is None:
                            feat["micro_env_features"] = micro_env_data[pid]

        load_time = time.time() - start_time
        if protein_features:
            average_time = load_time / len(protein_features)
            logger.info(" LoadCompleted, %.2f, Protein %.3f ", load_time, average_time)
        else:
            logger.warning(" Load Protein, %.2f ", load_time)

        return protein_features
    
    def batch_extract_features_generator(
        self,
        data: pd.DataFrame,
        protein_features: Dict[str, Any],
        window_size: int = 61,
        local_window_size: int = 31,
        target_ptm_type: str = 'phosphorylation',
    ) -> Generator[Dict, None, None]:
        """
        Generator version of batch feature extraction, yielding samples one by one to reduce memory usage.
        """
        logger.info(f"Start {len(data)} Samples.")
        
        half_window = window_size // 2
        local_half_window = local_window_size // 2
        target_ptm_label = str(target_ptm_type).strip()

        count = 0
        for idx, row in data.iterrows():
            uniprot_id = row['uniprot_id']
            position = row['position']

            entry = protein_features.get(uniprot_id)
            if entry is None:
                logger.warning(f"SkipSamples {uniprot_id}:{position},Protein Load")
                continue

            try:
                if isinstance(entry, LazyProteinEntry):
                    seq_dataset = entry.sequence_dataset
                    if seq_dataset is None:
                        logger.warning("Protein %s Load Unavailable,SkipSamples", uniprot_id)
                        continue
                    seq_len = entry.sequence_length
                    structure_dataset = entry.structure_dataset
                    struct_mean_raw = entry.structure_mean
                    structure_mean = None if struct_mean_raw is None else np.asarray(struct_mean_raw, dtype=np.float32)
                    struct_cls_raw = entry.structure_cls
                    structure_cls = None if struct_cls_raw is None else np.asarray(struct_cls_raw, dtype=np.float32)
                    structure_coords = entry.structure_coords
                    seq_mean_raw = entry.sequence_mean
                    if seq_mean_raw is None:
                        feature_dim = int(seq_dataset.shape[1]) if seq_dataset.ndim == 2 else 0
                        sequence_mean = np.zeros((feature_dim,), dtype=np.float32)
                    else:
                        sequence_mean = np.asarray(seq_mean_raw, dtype=np.float32)
                else:
                    features = entry
                    seq_dataset = features['sequence_features']
                    seq_len = len(seq_dataset)
                    structure_dataset = features.get('structure_features')
                    structure_mean = features.get('structure_mean')
                    structure_cls = features.get('structure_cls')
                    sequence_mean = features['sequence_mean']

                pos_idx = position - 1  # Convert to 0-indexed

                if pos_idx < 0 or pos_idx >= seq_len:
                    logger.warning(f" {position} Protein {uniprot_id} ")
                    continue

                # Extract sequence window features.
                seq_window = self._extract_window_features(
                    seq_dataset, pos_idx, window_size, half_window
                )

                seq_local = self._extract_window_features(
                    seq_dataset, pos_idx, local_window_size, local_half_window
                )

                # Extract structure window features.
                struct_window = None
                struct_local = None
                micro_env_features = None

                if structure_dataset is not None:
                    struct_window = self._extract_window_features(
                        structure_dataset, pos_idx, window_size, half_window
                    )

                    struct_local = self._extract_window_features(
                        structure_dataset, pos_idx, local_window_size, local_half_window
                    )
                    
                    # Try to obtain micro-environment features.
                    micro_env_all = None
                    if isinstance(entry, LazyProteinEntry):
                        micro_env_all = entry.micro_env_features
                    elif isinstance(entry, dict):
                        micro_env_all = entry.get("micro_env_features")
                    
                    if micro_env_all is not None and pos_idx < len(micro_env_all):
                        micro_env_features = micro_env_all[pos_idx]
                    elif structure_coords is not None:
                        # Compute on demand.
                        if self._missing_micro_env_warning_count < 5:
                            logger.warning(f"Protein {uniprot_id}, (Training ).")
                        elif self._missing_micro_env_warning_count == 5:
                            logger.warning(" 5 Protein, Warning.")
                        self._missing_micro_env_warning_count += 1
                        
                        micro_env_features = self._compute_single_residue_micro_env(structure_coords, pos_idx)

                ptm_type_raw = row['ptm_type'] if 'ptm_type' in row else target_ptm_label
                ptm_type = _sanitize_ptm_type(ptm_type_raw, target_ptm_type, self._ptm_override_cache, "batch_extract_features")

                label_value = row['label'] if 'label' in row else None
                has_label = label_value is not None and not pd.isna(label_value)
                target_label = int(label_value) if has_label else 0

                sample = {
                    'uniprot_id': uniprot_id,
                    'position': position,
                    'residue': row['residue'],
                    'ptm_type': ptm_type,
                    'ptm_type_raw': ptm_type_raw,
                    'target_ptm_type': target_ptm_label,
                    'sequence_features': seq_window,
                    'local_features': seq_local,
                    'global_features': sequence_mean,
                    'structure_features': struct_window,
                    'structure_local': struct_local,
                    'structure_global': structure_mean,
                    'structure_cls_vector': structure_cls,
                    'micro_env_features': micro_env_features,
                    'is_target_ptm': target_label,
                    'is_phosphorylated': target_label,
                    'has_label': has_label,
                    'label': target_label if has_label else None,
                    'sequence_length': seq_len
                }

                yield sample
                count += 1

                # Progress report.
                if count % 1000 == 0:
                    logger.info(f" {count}/{len(data)} Samples")

            except Exception as e:
                logger.error(f" Samples {uniprot_id}:{position}: {e}")
                continue

    def batch_extract_features(
        self,
        data: pd.DataFrame,
    protein_features: Dict[str, Any],
        window_size: int = 61,
        local_window_size: int = 31,
        target_ptm_type: str = 'phosphorylation',
    ) -> List[Dict]:
        """
        Batch extract features.
        
        Args:
            data: DataFrame containing sample metadata.
            protein_features: Preloaded protein features.
            window_size: Window size.
            local_window_size: Local window size.
            
        Returns:
            List: Processed sample list.
        """
        logger.info(f" {len(data)} Samples.")
        
        start_time = time.time()
        processed_samples = []
        
        half_window = window_size // 2
        local_half_window = local_window_size // 2
        target_ptm_label = str(target_ptm_type).strip()

        for idx, row in data.iterrows():
            uniprot_id = row['uniprot_id']
            position = row['position']

            entry = protein_features.get(uniprot_id)
            if entry is None:
                logger.warning(f"SkipSamples {uniprot_id}:{position},Protein Load")
                continue

            try:
                if isinstance(entry, LazyProteinEntry):
                    seq_dataset = entry.sequence_dataset
                    if seq_dataset is None:
                        logger.warning("Protein %s Load Unavailable,SkipSamples", uniprot_id)
                        continue
                    seq_len = entry.sequence_length
                    structure_dataset = entry.structure_dataset
                    struct_mean_raw = entry.structure_mean
                    structure_mean = None if struct_mean_raw is None else np.asarray(struct_mean_raw, dtype=np.float32)
                    struct_cls_raw = entry.structure_cls
                    structure_cls = None if struct_cls_raw is None else np.asarray(struct_cls_raw, dtype=np.float32)
                    structure_coords = entry.structure_coords
                    seq_mean_raw = entry.sequence_mean
                    if seq_mean_raw is None:
                        feature_dim = int(seq_dataset.shape[1]) if seq_dataset.ndim == 2 else 0
                        sequence_mean = np.zeros((feature_dim,), dtype=np.float32)
                    else:
                        sequence_mean = np.asarray(seq_mean_raw, dtype=np.float32)
                else:
                    features = entry
                    seq_dataset = features['sequence_features']
                    seq_len = len(seq_dataset)
                    structure_dataset = features.get('structure_features')
                    structure_mean = features.get('structure_mean')
                    structure_cls = features.get('structure_cls')
                    sequence_mean = features['sequence_mean']

                pos_idx = position - 1  # Convert to 0-indexed

                if pos_idx < 0 or pos_idx >= seq_len:
                    logger.warning(f" {position} Protein {uniprot_id} ")
                    continue

                # Extract sequence window features.
                seq_window = self._extract_window_features(
                    seq_dataset, pos_idx, window_size, half_window
                )

                seq_local = self._extract_window_features(
                    seq_dataset, pos_idx, local_window_size, local_half_window
                )

                # Extract structure window features.
                struct_window = None
                struct_local = None
                micro_env_features = None

                if structure_dataset is not None:
                    struct_window = self._extract_window_features(
                        structure_dataset, pos_idx, window_size, half_window
                    )

                    struct_local = self._extract_window_features(
                        structure_dataset, pos_idx, local_window_size, local_half_window
                    )
                    
                    # Try to obtain micro-environment features.
                    micro_env_all = None
                    if isinstance(entry, LazyProteinEntry):
                        micro_env_all = entry.micro_env_features
                    elif isinstance(entry, dict):
                        micro_env_all = entry.get("micro_env_features")
                    
                    if micro_env_all is not None and pos_idx < len(micro_env_all):
                        micro_env_features = micro_env_all[pos_idx]
                    elif structure_coords is not None:
                        # Compute on demand.
                        if self._missing_micro_env_warning_count < 5:
                            logger.warning(f"Protein {uniprot_id}, (Training ).")
                        elif self._missing_micro_env_warning_count == 5:
                            logger.warning(" 5 Protein, Warning.")
                        self._missing_micro_env_warning_count += 1
                        
                        micro_env_features = self._compute_single_residue_micro_env(structure_coords, pos_idx)

                ptm_type_raw = row['ptm_type'] if 'ptm_type' in row else target_ptm_label
                ptm_type = _sanitize_ptm_type(ptm_type_raw, target_ptm_type, self._ptm_override_cache, "batch_extract_features")

                label_value = row['label'] if 'label' in row else None
                has_label = label_value is not None and not pd.isna(label_value)
                target_label = int(label_value) if has_label else 0

                sample = {
                    'uniprot_id': uniprot_id,
                    'position': position,
                    'residue': row['residue'],
                    'ptm_type': ptm_type,
                    'ptm_type_raw': ptm_type_raw,
                    'target_ptm_type': target_ptm_label,
                    'sequence_features': seq_window,
                    'local_features': seq_local,
                    'global_features': sequence_mean,
                    'structure_features': struct_window,
                    'structure_local': struct_local,
                    'structure_global': structure_mean,
                    'structure_cls_vector': structure_cls,
                    'micro_env_features': micro_env_features,
                    'is_target_ptm': target_label,
                    'is_phosphorylated': target_label,
                    'has_label': has_label,
                    'label': target_label if has_label else None,
                    'sequence_length': seq_len
                }

                processed_samples.append(sample)

                # Progress report.
                if len(processed_samples) % 1000 == 0:
                    logger.info(f" {len(processed_samples)}/{len(data)} Samples")

            except Exception as e:
                logger.error(f" Samples {uniprot_id}:{position}: {e}")
                continue
        
        process_time = time.time() - start_time
        logger.info(f" Completed, {process_time:.2f}, {len(processed_samples)} Samples")
        
        return processed_samples
    
    @staticmethod
    def _extract_window_features(features: np.ndarray, center_pos: int,
                                window_size: int, half_window: int) -> np.ndarray:
        """Extract window features while avoiding repeated stacking and temporary arrays."""
        seq_len, feature_dim = features.shape

        start = center_pos - half_window
        end = start + window_size

        src_start = max(start, 0)
        src_end = min(end, seq_len)
        left_pad = max(0, -start)
        right_pad = max(0, end - seq_len)

        window = np.zeros((window_size, feature_dim), dtype=features.dtype)

        if src_start < src_end:
            dest_start = left_pad
            valid_length = src_end - src_start
            dest_end = dest_start + valid_length
            # Assign by slice to avoid repeated copies.
            window[dest_start:dest_end] = features[src_start:src_end]

        return window


class LazyProteinFeatureStore:
    """Lightweight store for on-demand protein feature access, compatible with multi-process loading."""

    def __init__(
        self,
        features_path: str,
        *,
        lazy_loading: bool = True,
        h5_cache_bytes: int = 512 * 1024 * 1024,
        h5_cache_slots: int = 1_048_575,
        h5_cache_w0: float = 0.75,
        known_proteins: Optional[Iterable[str]] = None,
        structure_map: Optional[Dict[str, bool]] = None,
        sequence_lengths: Optional[Dict[str, int]] = None,
        max_lazy_cache_entries: Optional[int] = None,
        fasta_dir: Optional[str] = None,
        pdb_dir: Optional[str] = None,
        micro_env_cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.features_path = str(features_path)
        self.lazy_loading = bool(lazy_loading)
        self._h5_cache_bytes = int(h5_cache_bytes)
        self._h5_cache_slots = int(h5_cache_slots)
        self._h5_cache_w0 = float(h5_cache_w0)
        self._known_proteins = set(known_proteins) if known_proteins is not None else None
        self._structure_map = dict(structure_map) if structure_map else {}
        self._sequence_lengths = dict(sequence_lengths) if sequence_lengths else {}
        self.fasta_dir = fasta_dir
        self.pdb_dir = pdb_dir
        self._micro_env_cache_external = micro_env_cache or {}
        self._h5_file: Optional[h5py.File] = None
        self._proteins_group: Optional[h5py.Group] = None
        if max_lazy_cache_entries is None:
            max_lazy_cache_entries = _parse_env_int("PTM_LAZY_CACHE_MAX_ENTRIES", 512, minimum=32)
        self._max_lazy_cache_entries: Optional[int] = int(max_lazy_cache_entries) if max_lazy_cache_entries else None
        self._lazy_cache: "OrderedDict[str, LazyProteinEntry]" = OrderedDict()
        self._eager_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        atexit.register(self.close)

    def _get_h5_file(self, mode: str = "r") -> h5py.File:
        # Check if we need to reopen (either file is None, or mode mismatch)
        need_reopen = self._h5_file is None
        if not need_reopen:
            # If current mode is 'r' but we need 'r+', we must reopen
            if mode == "r+" and self._h5_file.mode == "r":
                need_reopen = True
        
        if need_reopen:
            if self._h5_file is not None:
                try:
                    # CRITICAL: Clear stale LazyProteinEntry references when closing file
                    for entry in self._lazy_cache.values():
                        try:
                            if isinstance(entry, LazyProteinEntry):
                                entry.close()
                        except Exception:
                            pass
                    self._lazy_cache.clear()

                    self._h5_file.close()
                except Exception:
                    pass
                self._h5_file = None
                # CRITICAL: Invalidate the group cache when file is closed
                self._proteins_group = None
            
            try:
                self._h5_file = h5py.File(
                    self.features_path,
                    mode,
                    rdcc_nbytes=self._h5_cache_bytes,
                    rdcc_nslots=self._h5_cache_slots,
                    rdcc_w0=self._h5_cache_w0,
                )
            except Exception as exc:  # pragma: no cover - rare hardware/IO exception
                # Ensure state is clean on failure
                self._h5_file = None
                self._proteins_group = None
                raise RuntimeError(f" {self.features_path} (mode={mode}): {exc}") from exc
        return self._h5_file

    def _get_proteins_group(self, mode: str = "r") -> "h5py.Group":
        # Check conditions to refresh group:
        # 1. Group is None
        # 2. File is None (should imply 1, but safety first)
        # 3. Mode upgrade needed (r -> r+)
        refresh = (self._proteins_group is None) or (self._h5_file is None)
        
        if not refresh and mode == "r+" and self._h5_file is not None:
             if hasattr(self._h5_file, 'mode') and self._h5_file.mode == "r":
                 refresh = True

        if refresh:
            h5_file = self._get_h5_file(mode)
            if "proteins" not in h5_file:
                if mode == "r+":
                    self._proteins_group = h5_file.create_group("proteins")
                else:
                    raise KeyError(f" {self.features_path} 'proteins' ")
            else:
                self._proteins_group = h5_file["proteins"]
        return self._proteins_group

    def _get_protein_group(self, protein_id: str) -> Optional["h5py.Group"]:
        group = self._get_proteins_group()
        if protein_id not in group:
            return None
        return group[protein_id]

    def _prune_lazy_cache(self) -> None:
        if self._max_lazy_cache_entries is None:
            return
        while len(self._lazy_cache) > self._max_lazy_cache_entries:
            old_pid, old_entry = self._lazy_cache.popitem(last=False)
            try:
                if isinstance(old_entry, LazyProteinEntry):
                    old_entry.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.debug(" LoadProtein %s: %s", old_pid, exc)

    def has_protein(self, protein_id: str) -> bool:
        if self._known_proteins is not None:
            return protein_id in self._known_proteins
        group = self._get_proteins_group()
        return protein_id in group

    def register_known_proteins(self, protein_ids: Iterable[str]) -> None:
        if protein_ids is None:
            return
        if self._known_proteins is None:
            self._known_proteins = set()
        self._known_proteins.update(str(pid) for pid in protein_ids)

    def has_structure(self, protein_id: str) -> bool:
        if protein_id in self._structure_map:
            return bool(self._structure_map[protein_id])
        protein_group = self._get_protein_group(protein_id)
        if protein_group is None:
            self._structure_map[protein_id] = False
            return False
        has_struct = "structure_features" in protein_group
        self._structure_map[protein_id] = has_struct
        return has_struct

    def get_sequence_length(self, protein_id: str) -> Optional[int]:
        if protein_id in self._sequence_lengths:
            return self._sequence_lengths[protein_id]
        protein_group = self._get_protein_group(protein_id)
        if protein_group is None:
            return None
        seq_dataset = protein_group.get("sequence_features")
        if seq_dataset is None:
            return None
        seq_len = int(seq_dataset.shape[0])
        self._sequence_lengths[protein_id] = seq_len
        return seq_len

    def get_entry(self, protein_id: str) -> Optional[Any]:
        if self.lazy_loading:
            entry = self._lazy_cache.get(protein_id)
            if entry is not None:
                self._lazy_cache.move_to_end(protein_id)
                return entry

            with self._lock:
                entry = self._lazy_cache.get(protein_id)
                if entry is None:
                    protein_group = self._get_protein_group(protein_id)
                    if protein_group is None:
                        return None
                
                    if "sequence_features" not in protein_group:
                        return None
                    
                    entry = LazyProteinEntry(protein_id, protein_group, fasta_dir=self.fasta_dir, pdb_dir=self.pdb_dir)
                    
                    # Inject precomputed micro-environment features.
                    if protein_id in self._micro_env_cache_external:
                        entry.micro_env_features = self._micro_env_cache_external[protein_id]
                        
                    self._lazy_cache[protein_id] = entry
                    self._lazy_cache.move_to_end(protein_id)
                    self._prune_lazy_cache()
                else:
                    self._lazy_cache.move_to_end(protein_id)
                return entry

        entry = self._eager_cache.get(protein_id)
        if entry is not None:
            return entry

        with self._lock:
            entry = self._eager_cache.get(protein_id)
            if entry is None:
                protein_group = self._get_protein_group(protein_id)
                if protein_group is None:
                    return None
                
                if "sequence_features" not in protein_group:
                    return None
                    
                entry = {
                    "sequence_features": protein_group["sequence_features"][()],
                    "sequence_cls": protein_group["sequence_cls"][()],
                    "sequence_mean": protein_group["sequence_mean"][()],
                }
                
                # Try to load sequence string
                try:
                    seq_bytes = protein_group.attrs.get("sequence")
                    if isinstance(seq_bytes, bytes):
                        entry["sequence_str"] = seq_bytes.decode('utf-8')
                    else:
                        entry["sequence_str"] = str(seq_bytes) if seq_bytes is not None else None
                except Exception:
                    entry["sequence_str"] = None

                if not entry["sequence_str"] and self.fasta_dir:
                    entry["sequence_str"] = load_sequence_from_fasta(protein_id, self.fasta_dir)

                if "structure_features" in protein_group:
                    entry.update(
                        {
                            "structure_features": protein_group["structure_features"][()],
                            "structure_cls": protein_group["structure_cls"][()],
                            "structure_mean": protein_group["structure_mean"][()],
                        }
                    )
                    self._structure_map.setdefault(protein_id, True)
                else:
                    entry.update(
                        {
                            "structure_features": None,
                            "structure_cls": None,
                            "structure_mean": None,
                        }
                    )
                    self._structure_map.setdefault(protein_id, False)
                
                # Inject precomputed micro-environment features.
                if protein_id in self._micro_env_cache_external:
                    entry["micro_env_features"] = self._micro_env_cache_external[protein_id]
                else:
                    entry["micro_env_features"] = None
            
        self._eager_cache[protein_id] = entry
        return entry

    def save_micro_environment_to_h5(self, micro_env_dict: Dict[str, np.ndarray]) -> None:
        """Save precomputed micro-environment features back into the HDF5 file."""
        # [User Request] Disabled HDF5 saving to avoid file locking issues and errors
        return
        
        if not micro_env_dict:
            return
            
        try:
            # Use r+ mode to obtain the group.
            proteins_group = self._get_proteins_group("r+")
            count = 0
            missing_pids = []
            for pid, features in micro_env_dict.items():
                if pid in proteins_group:
                    pg = proteins_group[pid]
                    # If pg is a dataset rather than a group, the HDF5 structure is invalid.
                    if not isinstance(pg, h5py.Group):
                        logger.warning(f"HDF5 Protein {pid},Skip ")
                        continue
                        
                    if "micro_env_features" in pg:
                        del pg["micro_env_features"]
                    pg.create_dataset(
                        "micro_env_features", 
                        data=features, 
                        compression="gzip", 
                        compression_opts=4
                    )
                    count += 1
                else:
                    # If the protein does not exist in HDF5 at all, do not create a group.
                    missing_pids.append(pid)
            
            if count > 0:
                logger.debug(f" /Create {count} Protein HDF5")
            if missing_pids:
                logger.debug(f" {len(missing_pids)} Protein HDF5 CreateFailed")
        except Exception as e:
            logger.error(f"Save HDF5 Failed: {e}")

    def save_micro_environment_features(self, micro_env_dict: Dict[str, np.ndarray], cache_dir: str = None) -> None:
        """
        Save micro-environment features into LMDB (if available) and/or HDF5.
        """
        if not micro_env_dict:
            return

        # 1. Update in-memory cache
        for pid, feat in micro_env_dict.items():
             self.update_micro_env_cache(pid, feat)

        # 2. Save to HDF5 (Persistent storage)
        self.save_micro_environment_to_h5(micro_env_dict)

    def update_micro_env_cache(self, protein_id: str, features: np.ndarray) -> None:
        """Update the in-memory micro-environment cache."""
        if features is None:
            return
        self._micro_env_cache_external[protein_id] = features
        
        # Try to update active lazy entry if exists
        if self.lazy_loading:
            if protein_id in self._lazy_cache:
                entry = self._lazy_cache[protein_id]
                if isinstance(entry, LazyProteinEntry):
                    entry.micro_env_features = features
        else:
            if protein_id in self._eager_cache:
                entry = self._eager_cache[protein_id]
                if isinstance(entry, dict):
                    entry["micro_env_features"] = features

    def close(self) -> None:
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception as exc:  # pragma: no cover - rare IO exception
                logger.warning(": %s", exc)
            finally:
                self._h5_file = None
                self._proteins_group = None
                for entry in self._lazy_cache.values():
                    try:
                        if isinstance(entry, LazyProteinEntry):
                            entry.close()
                    except Exception as exc:  # pragma: no cover - defensive cleanup
                        logger.debug(" Load: %s", exc)
                self._lazy_cache.clear()
                self._eager_cache.clear()

    def __del__(self) -> None:  # pragma: no cover - destructor is inherently unpredictable
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state['_h5_file'] = None
        state['_proteins_group'] = None
        state['_lazy_cache'] = OrderedDict()
        state['_lock'] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lazy_cache = OrderedDict()
        self._lock = threading.Lock()
        atexit.register(self.close)


class OnDemandPTMDataset(Dataset):
    """On-demand PTM dataset that extracts window features from an HDF5 feature file."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        protein_store: LazyProteinFeatureStore,
        *,
        window_size: int = 61,
        local_window_size: int = 31,
        target_ptm_type: str = "phosphorylation",
        residue_info: Optional[Dict[str, Any]] = None,
        feature_reducer: Optional[Any] = None,
        use_structure: bool = True,
    ) -> None:
        self.data_frame = data_frame.reset_index(drop=True)
        self.protein_store = protein_store
        self.fixed_window_size = int(window_size)
        self.fixed_local_size = int(local_window_size)
        self.center_index = self.fixed_window_size // 2
        self.half_window = self.fixed_window_size // 2
        self.local_half_window = self.fixed_local_size // 2
        self.target_ptm_label = str(target_ptm_type).strip()
        self.target_ptm_type = self.target_ptm_label.lower()
        self._ptm_override_cache: Set[str] = set()
        self._normalization_stats: Dict[str, Dict[str, np.ndarray]] = {}
        self.feature_reducer = feature_reducer
        self.use_structure = use_structure

        if residue_info is None:
            observed_residues = {
                str(res).strip().upper()
                for res in self.data_frame.get("residue", [])
                if isinstance(res, str) and res.strip()
            }
            residue_info = resolve_ptm_residue_info(self.target_ptm_type, observed_residues)

        self.residue_info = residue_info
        self.residue_to_id = dict(residue_info['residue_to_id'])
        self.other_residue_id = int(residue_info['other_id'])
        self.residue_vocab_size = int(residue_info['vocab_size'])
        self.ptm_type_to_id = {self.target_ptm_label: 0}

    def _compute_micro_environment_features(self, coords: np.ndarray, center_idx: int) -> np.ndarray:
        """Compute micro-environment features (density + concavity + mean distance)."""
        if coords is None or len(coords) == 0:
            return np.zeros(3, dtype=np.float32)
        
        if center_idx < 0 or center_idx >= len(coords):
             return np.zeros(3, dtype=np.float32)

        center_coord = coords[center_idx]
        # Compute distances from all points to the center point.
        dists = np.linalg.norm(coords - center_coord, axis=1)
        
        # 1. Local density (number of neighbors within 10 Å)
        radius = 10.0
        mask = (dists < radius) & (dists > 1e-6)
        neighbor_indices = np.where(mask)[0]
        local_density = len(neighbor_indices)
        
        # 2. Local concavity (distance to neighbor centroid)
        if local_density > 0:
            neighbor_coords = coords[neighbor_indices]
            centroid = np.mean(neighbor_coords, axis=0)
            dist_to_centroid = np.linalg.norm(center_coord - centroid)
            
            # 3. Mean neighbor distance
            avg_dist = np.mean(dists[neighbor_indices])
        else:
            dist_to_centroid = 0.0
            avg_dist = 0.0
            
        # Normalize (see encoders.py for the scaling factors).
        return np.array([
            local_density / 20.0,
            dist_to_centroid / 5.0,
            avg_dist / 10.0
        ], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._build_raw_sample(idx)
        
        if self.feature_reducer:
            self.feature_reducer.transform([sample], inplace=True)
            
        if self._normalization_stats:
            self._apply_normalization(sample)
        return self._to_torch(sample)

    def set_normalization_stats(self, stats: Optional[Dict[str, Dict[str, np.ndarray]]]) -> None:
        if not stats:
            self._normalization_stats = {}
            return

        normalized_stats: Dict[str, Dict[str, np.ndarray]] = {}
        for key, value in stats.items():
            mean = np.array(value.get("mean"), dtype=np.float32, copy=True)
            std = np.array(value.get("std"), dtype=np.float32, copy=True)
            normalized_stats[key] = {"mean": mean, "std": std}
        self._normalization_stats = normalized_stats

    def iter_raw_samples(self) -> Iterator[Dict[str, Any]]:
        """Iterate over raw samples one by one to avoid loading the full dataset into memory.

        Warning: do not call list(iter_raw_samples()) on large datasets, as it can exhaust memory.
        Prefer streaming or batch-wise processing.
        """
        for idx in range(len(self.data_frame)):
            yield self._build_raw_sample(idx)
    
    def iter_raw_samples_batch(self, batch_size: int = 1000) -> Iterator[List[Dict[str, Any]]]:
        """Iterate over raw samples in batches to balance memory usage and throughput.

        Args:
            batch_size: Number of samples per batch.

        Yields:
            List[Dict]: A batch of raw samples.
        """
        total = len(self.data_frame)
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch = []
            for idx in range(start_idx, end_idx):
                try:
                    sample = self._build_raw_sample(idx)
                    batch.append(sample)
                except Exception as e:
                    logger.warning(f"SkipSamples {idx}: {e}")
                    continue
            if batch:
                yield batch

    def _unpack_entry(self, entry: Any, uniprot_id: str):
        if isinstance(entry, LazyProteinEntry):
            seq_dataset = entry.sequence_dataset
            if seq_dataset is None:
                raise RuntimeError(f"Protein {uniprot_id} Unavailable (Load )")
            seq_len = entry.sequence_length
            structure_dataset = entry.structure_dataset

            struct_mean_raw = entry.structure_mean
            structure_mean = None if struct_mean_raw is None else np.asarray(struct_mean_raw, dtype=np.float32).copy()

            struct_cls_raw = entry.structure_cls
            structure_cls = None if struct_cls_raw is None else np.asarray(struct_cls_raw, dtype=np.float32).copy()

            structure_coords = entry.structure_coords

            seq_mean_raw = entry.sequence_mean
            if seq_mean_raw is None:
                feature_dim = int(seq_dataset.shape[1]) if seq_dataset.ndim == 2 else 0
                if feature_dim == 0:
                    logger.warning("Protein %s sequence_mean,Use ", uniprot_id)
                    sequence_mean = np.zeros((0,), dtype=np.float32)
                else:
                    logger.debug("Protein %s,Use ", uniprot_id)
                    sequence_mean = np.zeros((feature_dim,), dtype=np.float32)
            else:
                sequence_mean = np.asarray(seq_mean_raw, dtype=np.float32).copy()
            sequence_str = entry.sequence
        else:
            seq_dataset = entry['sequence_features']
            seq_len = seq_dataset.shape[0]
            structure_dataset = entry.get('structure_features')
            structure_mean = entry.get('structure_mean')
            if structure_mean is not None:
                structure_mean = np.array(structure_mean, copy=True)
            structure_cls = entry.get('structure_cls')
            if structure_cls is not None:
                structure_cls = np.array(structure_cls, copy=True)
            structure_coords = entry.get('structure_coords')
            sequence_mean = np.array(entry['sequence_mean'], copy=True)
            sequence_str = entry.get('sequence_str')
            
        return seq_dataset, seq_len, structure_dataset, structure_mean, structure_cls, structure_coords, sequence_mean, sequence_str

    def _build_raw_sample(self, idx: int) -> Dict[str, Any]:
        row = self.data_frame.iloc[idx]
        uniprot_id = row['uniprot_id']
        if not isinstance(uniprot_id, str):
            uniprot_id = str(uniprot_id)

        entry = self.protein_store.get_entry(uniprot_id)
        if entry is None:
            raise KeyError(f"Protein {uniprot_id} ")

        # Initial unpack
        (seq_dataset, seq_len, structure_dataset, structure_mean, 
         structure_cls, structure_coords, sequence_mean, sequence_str) = self._unpack_entry(entry, uniprot_id)

        position_value = row['position']
        try:
            position = int(position_value)
        except (TypeError, ValueError):
            raise ValueError(f"Samples {uniprot_id} position='{position_value}' ")

        pos_idx = position - 1
        if pos_idx < 0 or pos_idx >= seq_len:
            raise IndexError(f" {position} Protein {uniprot_id} ({seq_len})")

        # Calculate micro-environment features
        micro_env_features = None
        if isinstance(entry, LazyProteinEntry):
            if entry.micro_env_features is not None:
                all_micro_env = entry.micro_env_features
                if hasattr(all_micro_env, '__len__') and pos_idx < len(all_micro_env):
                    micro_env_features = all_micro_env[pos_idx]
        elif isinstance(entry, dict):
            # Eager mode
            all_micro_env = entry.get('micro_env_features')
            if all_micro_env is not None and pos_idx < len(all_micro_env):
                micro_env_features = all_micro_env[pos_idx]
        
        if micro_env_features is None:
            # Fallback logic
            reloaded = False
            
            if self.protein_store.pdb_dir:
                try:
                    _, full_features = _compute_single_protein_micro_env(uniprot_id, self.protein_store.pdb_dir)
                    if full_features is not None:
                        self.protein_store.save_micro_environment_features({uniprot_id: full_features})
                        if pos_idx < len(full_features):
                            micro_env_features = full_features[pos_idx]
                        reloaded = True
                except Exception as e:
                    logger.warning(f"Failed to compute full micro-env features for {uniprot_id}: {e}")

            if micro_env_features is None:
                if structure_coords is not None and len(structure_coords) > 0:
                    try:
                        full_features = _compute_micro_env_from_coords(structure_coords)
                        if full_features is not None:
                            self.protein_store.save_micro_environment_features({uniprot_id: full_features})
                            if pos_idx < len(full_features):
                                micro_env_features = full_features[pos_idx]
                            reloaded = True
                    except Exception as e:
                         logger.debug(f"Failed to compute full micro-env features from coords for {uniprot_id}: {e}")

            if micro_env_features is None:
                 logger.debug(f"Protein {uniprot_id} missing micro-env features, computing on-demand (slow)...")
                 micro_env_features = self._compute_micro_environment_features(structure_coords, pos_idx)
            
            if reloaded:
                # Reload entry and datasets because file handle might have changed
                entry = self.protein_store.get_entry(uniprot_id)
                if entry is None:
                     raise KeyError(f"Protein {uniprot_id} Load ")
                (seq_dataset, seq_len, structure_dataset, structure_mean, 
                 structure_cls, structure_coords, sequence_mean, sequence_str) = self._unpack_entry(entry, uniprot_id)

        seq_window = DataLoadingOptimizer._extract_window_features(seq_dataset, pos_idx, self.fixed_window_size, self.half_window)
        seq_local = DataLoadingOptimizer._extract_window_features(seq_dataset, pos_idx, self.fixed_local_size, self.local_half_window)

        struct_window = None
        struct_local = None
        if self.use_structure and structure_dataset is not None:
            struct_window = DataLoadingOptimizer._extract_window_features(structure_dataset, pos_idx, self.fixed_window_size, self.half_window)
            struct_local = DataLoadingOptimizer._extract_window_features(structure_dataset, pos_idx, self.fixed_local_size, self.local_half_window)

        ptm_type_raw = row['ptm_type'] if 'ptm_type' in row else self.target_ptm_label
        if pd.isna(ptm_type_raw):
            ptm_type_raw = self.target_ptm_label
        ptm_type = _sanitize_ptm_type(ptm_type_raw, self.target_ptm_label, self._ptm_override_cache, "OnDemandPTMDataset")

        label_raw = row['label'] if 'label' in row else None
        has_label = label_raw is not None and not pd.isna(label_raw)
        if has_label:
            try:
                target_label = int(label_raw)
            except (TypeError, ValueError):
                target_label = int(float(label_raw))
        else:
            target_label = 0

        residue_value = row['residue'] if 'residue' in row else None

        return {
            'uniprot_id': uniprot_id,
            'position': position,
            'residue': residue_value,
            'ptm_type': ptm_type,
            'ptm_type_raw': ptm_type_raw,
            'target_ptm_type': self.target_ptm_label,
            'sequence_features': np.array(seq_window, copy=False),
            'local_features': np.array(seq_local, copy=False),
            'global_features': np.array(sequence_mean, copy=True),
            'structure_features': None if struct_window is None else np.array(struct_window, copy=False),
            'structure_local': None if struct_local is None else np.array(struct_local, copy=False),
            'structure_global': None if structure_mean is None else np.array(structure_mean, copy=True),
            'structure_cls_vector': None if structure_cls is None else np.array(structure_cls, copy=True),
            'micro_env_features': micro_env_features,
            'is_target_ptm': target_label,
            'is_phosphorylated': target_label,
            'has_label': has_label,
            'label': target_label if has_label else None,
            'sequence_length': seq_len,
            'sequence': sequence_str,
        }

    def _apply_normalization(self, sample: Dict[str, Any]) -> None:
        for key, stat in self._normalization_stats.items():
            value = sample.get(key)
            if value is None:
                continue
            array = np.array(value, dtype=np.float32, copy=True)
            mean = stat['mean']
            std = stat['std']
            normalized = (array - mean) / std
            sample[key] = normalized.astype(np.float32, copy=False)

    def _to_torch(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        seq_features = torch.from_numpy(np.asarray(sample['sequence_features'], dtype=np.float32))
        local_features = torch.from_numpy(np.asarray(sample['local_features'], dtype=np.float32))
        global_features = torch.from_numpy(np.asarray(sample['global_features'], dtype=np.float32))
        
        micro_env = sample.get('micro_env_features')
        if micro_env is not None:
             micro_env_arr = np.asarray(micro_env, dtype=np.float32)
             # Ensure dimension is 3
             if micro_env_arr.shape[0] > 3:
                 micro_env_arr = micro_env_arr[:3]
             elif micro_env_arr.shape[0] < 3:
                 micro_env_arr = np.pad(micro_env_arr, (0, 3 - micro_env_arr.shape[0]), 'constant')
             micro_env_tensor = torch.from_numpy(micro_env_arr)
        else:
             micro_env_tensor = torch.zeros(3, dtype=torch.float32)

        struct_tensor = None
        if sample['structure_features'] is not None:
            struct_tensor = torch.from_numpy(np.asarray(sample['structure_features'], dtype=np.float32))

        residue_value = sample.get('residue')
        if residue_value is None or (isinstance(residue_value, float) and np.isnan(residue_value)):
            residue_key = None
        else:
            residue_key = str(residue_value).strip().upper()
        residue_type_id = self.residue_to_id.get(residue_key, self.other_residue_id)

        residue_types = torch.full((self.fixed_window_size,), self.other_residue_id, dtype=torch.long)
        
        sequence_str = sample.get('sequence')
        position = int(sample['position'])
        pos_idx = position - 1
        
        if sequence_str:
            seq_len = len(sequence_str)
            start = pos_idx - self.half_window
            end = pos_idx + self.half_window + 1
            
            for i, seq_i in enumerate(range(start, end)):
                if 0 <= seq_i < seq_len:
                    res = sequence_str[seq_i].upper()
                    residue_types[i] = self.residue_to_id.get(res, self.other_residue_id)
        else:
            residue_types[self.center_index] = residue_type_id

        position_ids = torch.arange(self.fixed_window_size, dtype=torch.long) - self.center_index
        relative_offsets = position_ids.clone()

        site_indicator = torch.zeros(self.fixed_window_size, dtype=torch.float32)
        site_indicator[self.center_index] = 1.0

        label_tensor = torch.tensor(float(sample['is_target_ptm']), dtype=torch.float32)

        ptm_type_value = sample['ptm_type']
        ptm_type_id = self.ptm_type_to_id.get(ptm_type_value, 0)

        return {
            'uniprot_id': sample['uniprot_id'],
            'position': sample['position'],
            'residue': residue_value,
            'residue_type_id': residue_type_id,
            'residue_types': residue_types,
            'position_ids': position_ids,
            'relative_position_offsets': relative_offsets,
            'site_indicator': site_indicator,
            'sequence': {
                'window_features': seq_features,
                'local_features': local_features,
                'global_features': global_features,
                'position_ids': position_ids,
                'relative_position_offsets': relative_offsets,
                'site_indicator': site_indicator,
                'residue_types': residue_types,
            },
            'structure': struct_tensor,
            'micro_env_features': micro_env_tensor,
            'label': label_tensor,
            'ptm_type': ptm_type_value,
            'ptm_type_id': ptm_type_id,
        }


def _compute_micro_env_from_coords(coords: np.ndarray) -> np.ndarray:
    """Compute micro-environment features from a coordinate array (no I/O)."""
    seq_len = len(coords)
    features = np.zeros((seq_len, 3), dtype=np.float32)
    
    # Compute distances per residue.
    for i in range(seq_len):
        center_coord = coords[i]
        dists = np.linalg.norm(coords - center_coord, axis=1)
        
        # 1. Local density (number of neighbors within 10 Å)
        radius = 10.0
        mask = (dists < radius) & (dists > 1e-6)
        neighbor_indices = np.where(mask)[0]
        local_density = len(neighbor_indices)
        
        if local_density > 0:
            neighbor_coords = coords[neighbor_indices]
            centroid = np.mean(neighbor_coords, axis=0)
            dist_to_centroid = np.linalg.norm(center_coord - centroid)
            avg_dist = np.mean(dists[neighbor_indices])
        else:
            dist_to_centroid = 0.0
            avg_dist = 0.0
            
        features[i] = [
            local_density / 20.0,
            dist_to_centroid / 5.0,
            avg_dist / 10.0
        ]
    return features

def _compute_single_protein_micro_env(protein_id: str, pdb_dir: str) -> Tuple[str, Optional[np.ndarray]]:
    """Compute micro-environment features for all residues of a single protein."""
    pdb_path = os.path.join(pdb_dir, f"{protein_id}.pdb")
    if not os.path.exists(pdb_path):
        return protein_id, None
        
    coords = _extract_ca_coords_fast(pdb_path)
    if coords is None:
        return protein_id, None
        
    features = _compute_micro_env_from_coords(coords)
    return protein_id, features


def optimize_data_loading(
    train_data_path: str,
    test_data_path: str,
    esm_features_path: str,
    cache_dir: str = None,
    num_workers: int = None,
    target_ptm_type: str = 'phosphorylation',
    window_size: int = 61,
    local_window_size: int = 31,
) -> Tuple[List[Dict], List[Dict], Dict[str, object]]:
    """
    Optimized data loading function.
    
    Args:
        train_data_path: Training data path.
        test_data_path: Test data path.
        esm_features_path: ESM feature file path.
        cache_dir: Cache directory.
        num_workers: Number of worker processes.
        target_ptm_type: Target PTM type.
        window_size: Sequence window size.
        local_window_size: Local window size.
        
    Returns:
        Tuple: (training data, test data, residue mapping info)
    """
    logger.info("Start DataLoad.")

    # Create optimizer.
    optimizer = DataLoadingOptimizer(esm_features_path, num_workers)

    try:
        # Load data.
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)

        def _clean_dataframe(df: pd.DataFrame, source: str) -> pd.DataFrame:
            raw_count = len(df)
            df = df.copy()
            df['uniprot_id'] = df['uniprot_id'].apply(_normalize_uniprot_id)
            invalid_mask = df['uniprot_id'].isna()
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                logger.warning(
                    f"{source} {invalid_count} UniProt ID ({invalid_count / max(raw_count, 1) * 100:.2f}% ), "
                )
                df = df.loc[~invalid_mask].copy()
            coerced_count = int((df['uniprot_id'].apply(lambda v: not isinstance(v, str))).sum())
            if coerced_count > 0:
                # Should not happen, but cast again for safety.
                df['uniprot_id'] = df['uniprot_id'].astype(str)
                logger.debug(f"{source} {coerced_count} UniProt ID ")
            return df

        train_data = _clean_dataframe(train_data, "TrainingData")
        test_data = _clean_dataframe(test_data, "TestData")

        logger.info(f"LoadTrainingData: {len(train_data)} Samples")
        logger.info(f"LoadTestData: {len(test_data)} Samples")

        # Collect all required protein IDs.
        all_protein_ids = set(train_data['uniprot_id'].unique()) | set(test_data['uniprot_id'].unique())
        logger.info(f" Load {len(all_protein_ids)} Protein ")

        train_residue_series = train_data['residue'].dropna().astype(str).str.strip()
        observed_residues = {res for res in train_residue_series if res}
        residue_info = resolve_ptm_residue_info(target_ptm_type, observed_residues)
        logger.info(
            " LoadUse Residue Training: %s",
            residue_info['residue_to_id'],
        )

        test_residue_series = test_data['residue'].dropna().astype(str).str.strip()
        unseen_test_residues = sorted({res for res in test_residue_series if res} - observed_residues)
        if unseen_test_residues:
            logger.info(
                "Test Training Residue %s, Other",
                unseen_test_residues,
            )

        # Preload protein features.
        protein_features = optimizer.preload_protein_features(list(all_protein_ids), cache_dir)

        # Batch extract features.
        train_processed = optimizer.batch_extract_features(
            train_data,
            protein_features,
            window_size=window_size,
            local_window_size=local_window_size,
            target_ptm_type=target_ptm_type,
        )
        test_processed = optimizer.batch_extract_features(
            test_data,
            protein_features,
            window_size=window_size,
            local_window_size=local_window_size,
            target_ptm_type=target_ptm_type,
        )

        logger.info(" DataLoadCompleted")

        return train_processed, test_processed, residue_info
    finally:
        optimizer.close()


class StratifiedBatchSampler(Sampler):
    """
    Stratified batch sampler.

    Ensures the positive/negative ratio in each batch matches the target ratio.
    """
    def __init__(self, labels: np.ndarray, batch_size: int, pos_ratio: float, shuffle: bool = True):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.shuffle = shuffle
        
        self.pos_indices = np.where(self.labels == 1)[0]
        self.neg_indices = np.where(self.labels == 0)[0]
        
        # Compute expected positive/negative counts per batch.
        self.n_pos = int(batch_size * pos_ratio)
        # Ensure at least one positive if positives exist.
        if self.n_pos == 0 and len(self.pos_indices) > 0:
            self.n_pos = 1
        
        self.n_neg = batch_size - self.n_pos
        # Ensure at least one negative if negatives exist.
        if self.n_neg == 0 and len(self.neg_indices) > 0:
            self.n_neg = 1
            self.n_pos = batch_size - self.n_neg
            
        # Fix: compute num_batches based on covering the majority class.
        # Old logic: self.num_batches = int(np.ceil(len(labels) / batch_size))
        # This could drop some negative samples when truncating negatives to satisfy pos_ratio.
        
        batches_needed_pos = 0
        if self.n_pos > 0:
            batches_needed_pos = int(np.ceil(len(self.pos_indices) / self.n_pos))
            
        batches_needed_neg = 0
        if self.n_neg > 0:
            batches_needed_neg = int(np.ceil(len(self.neg_indices) / self.n_neg))
            
        # Take the max so all samples (positive or negative) are covered at least once per epoch.
        self.num_batches = max(batches_needed_pos, batches_needed_neg)
        
        # Safety check: avoid empty datasets.
        if self.num_batches == 0:
             self.num_batches = int(np.ceil(len(labels) / batch_size)) if len(labels) > 0 else 0

        logger.info(f"Initialize: ={batch_size}, Samples ={pos_ratio:.4f}")
        logger.info(f": Samples={self.n_pos}, Samples={self.n_neg}")
        logger.info(f"Epoch: {self.num_batches} Batches Samples (: {int(np.ceil(len(labels) / batch_size))})")
        
    def __iter__(self):
        pos_indices = self.pos_indices.copy()
        neg_indices = self.neg_indices.copy()
        
        if self.shuffle:
            np.random.shuffle(pos_indices)
            np.random.shuffle(neg_indices)
            
        pos_ptr = 0
        neg_ptr = 0
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Sample positives.
            for _ in range(self.n_pos):
                if len(pos_indices) > 0:
                    batch_indices.append(pos_indices[pos_ptr % len(pos_indices)])
                    pos_ptr += 1
                
            # Sample negatives.
            for _ in range(self.n_neg):
                if len(neg_indices) > 0:
                    batch_indices.append(neg_indices[neg_ptr % len(neg_indices)])
                    neg_ptr += 1
            
            # Shuffle within the batch.
            if self.shuffle:
                np.random.shuffle(batch_indices)
                
            yield batch_indices

    def __len__(self):
        return self.num_batches


class RotatingBalancedSampler(Sampler):
    """
    DeepMVP-style rotating balanced sampler.

    Properties:
    1. Uses only a subset of negatives per epoch to keep a balanced ratio (1:1 or a specified ratio).
    2. Rotates the negative subset across epochs to cover all negatives over time.
    3. Avoids excessive repetition of positives within an epoch, reducing overfitting risk.
    """
    def __init__(self, labels: np.ndarray, batch_size: int, pos_ratio: float = 0.5, shuffle: bool = True):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.shuffle = shuffle
        
        self.pos_indices = np.where(self.labels == 1)[0]
        self.neg_indices = np.where(self.labels == 0)[0]
        
        # Ensure both classes exist.
        if len(self.pos_indices) == 0 or len(self.neg_indices) == 0:
            raise ValueError(" Samples Samples 0, ")
            
        # Compute per-batch positive/negative counts.
        self.n_pos_per_batch = int(batch_size * pos_ratio)
        if self.n_pos_per_batch == 0: self.n_pos_per_batch = 1
        self.n_neg_per_batch = batch_size - self.n_pos_per_batch
        if self.n_neg_per_batch == 0: self.n_neg_per_batch = 1
        
        # Compute number of batches per epoch.
        # Strategy: use positives as the base so each epoch traverses all positives once,
        # and sample negatives proportionally.
        self.num_batches = int(np.ceil(len(self.pos_indices) / self.n_pos_per_batch))
        
        # Negative pointer used to rotate across epochs.
        self.neg_ptr = 0
        # Pre-shuffle negatives for randomness.
        if self.shuffle:
            np.random.shuffle(self.neg_indices)
            
        logger.info(f"Initialize (DeepMVP Strategy):")
        logger.info(f" Samples: {len(self.pos_indices)}, Samples: {len(self.neg_indices)}")
        logger.info(f": Samples={self.n_pos_per_batch}, Samples={self.n_neg_per_batch}")
        logger.info(f" Epoch Batch: {self.num_batches} ({len(self.pos_indices)} Samples, {self.num_batches * self.n_neg_per_batch} Samples)")
        
    def __iter__(self):
        # Positives: use all each epoch and shuffle.
        batch_pos_indices = self.pos_indices.copy()
        if self.shuffle:
            np.random.shuffle(batch_pos_indices)
            
        # Negatives: slice starting from where the previous epoch ended.
        batch_neg_indices = []
        needed_neg = self.num_batches * self.n_neg_per_batch
        
        current_ptr = self.neg_ptr
        for _ in range(needed_neg):
            batch_neg_indices.append(self.neg_indices[current_ptr % len(self.neg_indices)])
            current_ptr += 1
            
        # Update global pointer for the next epoch.
        self.neg_ptr = current_ptr % len(self.neg_indices)
        
        # Build batches.
        pos_ptr = 0
        neg_ptr_local = 0
        
        for _ in range(self.num_batches):
            batch = []
            
            # Add positives (wrap around if needed).
            for _ in range(self.n_pos_per_batch):
                batch.append(batch_pos_indices[pos_ptr % len(batch_pos_indices)])
                pos_ptr += 1
                
            # Add negatives.
            for _ in range(self.n_neg_per_batch):
                batch.append(batch_neg_indices[neg_ptr_local])
                neg_ptr_local += 1
                
            # Shuffle within the batch.
            if self.shuffle:
                np.random.shuffle(batch)
                
            yield batch
            
    def __len__(self):
        return self.num_batches
