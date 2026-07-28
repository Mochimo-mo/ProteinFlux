import torch
import os
import numpy as np
import pandas as pd
import h5py
import re

from core.rigid_utils import Rigid
from core.residue_constants import restype_order
from core.geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames
from core.ptm_utils import map_ptm_to_parent
from core.split_utils import filter_short_sequences


def get_restype_idx(char):
    """Map a one-letter amino acid code to its integer index, including PTM residues."""
    if char in restype_order:
        return restype_order[char]
    if char == 'B': return 21  # SEP
    if char == 'J': return 22  # TPO
    if char == 'Z': return 23  # PTR
    return restype_order.get('X', 20)


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, repeat=1):
        super().__init__()
        self.args = args
        self.repeat = repeat

        print(f"Loading split from {split}")
        self.df = pd.read_csv(split, index_col='name')
        self.df = filter_short_sequences(
            self.df, source=split, report=True,
        )
        if self.df.empty:
            raise ValueError(f"No valid protein sequences remain in split: {split}")

        print(f"Loading H5 from {self.args.data_dir}")
        self.h5_file = h5py.File(f'{self.args.data_dir}', 'r')
        self.valid_keys = set(self.h5_file.keys())

        self.total_len = self._calculate_len()
        print(f"Dataset initialized | {os.path.basename(split)} | length: {self.total_len}")

    def _calculate_len(self):
        if getattr(self.args, 'overfit_peptide', None):
            return 1000
        return int(self.repeat * len(self.df))

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Resample on bad samples (too-short trajectory, missing H5 key, length
        # mismatch) instead of crashing — mirrors MultiProteinDataset behaviour.
        max_tries = max(1, getattr(self.args, 'max_resample', 100))
        for attempt in range(max_tries):
            try:
                return self._load_one(idx)
            except (ValueError, KeyError) as e:
                if attempt == max_tries - 1:
                    raise
                idx = np.random.randint(0, len(self.df))

    def _load_one(self, idx):
        idx = idx % len(self.df)
        if self.args.overfit:
            idx = 0

        if self.args.overfit_peptide is None:
            name = self.df.index[idx]
            seqres_str = self.df.seqres[name] if 'seqres' in self.df.columns else ''
        else:
            name = self.args.overfit_peptide
            seqres_str = name

        target_key = name
        if name not in self.valid_keys:
            candidates = [f"{name}_R{i}" for i in range(1, 4)]
            existing_candidates = [c for c in candidates if c in self.valid_keys]
            if existing_candidates:
                target_key = np.random.choice(existing_candidates)
            else:
                raise KeyError(f"Key '{name}' nor its replicas found in H5 file.")

        # PTM-format entries store a GROUP {coords, seqres, token_type_id, ...};
        # base entries store the coords DATASET directly. Auto-detect so base data
        # is untouched. For PTM: keep only residue tokens (token_type_id==0), drop
        # the 4 phosphate-atom tokens (we do not model phosphate atoms), and take
        # seqres from the H5 (it carries the PTM tokens 21/22/23 at the modified site).
        node = self.h5_file[target_key]
        ptm_seqres_idx = None
        if isinstance(node, h5py.Group):
            tt = node['token_type_id'][:]
            res_sel = (tt == 0)
            arr = node['coords'][:][:, res_sel]                  # [F, L_res, 14, 3]
            ptm_seqres_idx = node['seqres'][:].astype(np.int64)  # [L_res], carries 21/22/23
        else:
            arr = node[:]

        frame_interval = getattr(self.args, 'frame_interval', None)
        if frame_interval:
            arr = arr[::frame_interval]

        max_train_frames = getattr(self.args, 'max_train_frames', None)
        if max_train_frames is not None:
            arr = arr[:max_train_frames]

        total_frames = arr.shape[0]
        if total_frames < self.args.num_frames:
            raise ValueError(
                f"Trajectory too short: n_frames={total_frames} < num_frames={self.args.num_frames}"
            )

        frame_start = 0
        arr = np.copy(arr[:self.args.num_frames]).astype(np.float32)

        if self.args.copy_frames:
            arr[1:] = arr[0]

        if ptm_seqres_idx is not None:
            seqres_idx = ptm_seqres_idx                          # H5 seqres (with PTM tokens)
        else:
            seqres_idx = np.array([get_restype_idx(c) for c in seqres_str])

        h5_res_len = arr.shape[1]
        seq_len = len(seqres_idx)
        if seq_len != h5_res_len:
            min_len = min(seq_len, h5_res_len)
            seqres_idx = seqres_idx[:min_len]
            arr = arr[:, :min_len, :, :]

        frames = atom14_to_frames(torch.from_numpy(arr))
        aatype = torch.from_numpy(seqres_idx)[None].expand(self.args.num_frames, -1)

        # Geometry uses parent residues; the PTM channel keeps modification identity.
        aatype_geom = map_ptm_to_parent(aatype)

        atom37 = torch.from_numpy(atom14_to_atom37(arr, aatype_geom)).float()

        L = frames.shape[1]
        mask = np.ones(L, dtype=np.float32)

        torsions, torsion_mask = atom37_to_torsions(atom37, aatype_geom)
        torsion_mask = torsion_mask[0]

        def get_ptm_info_from_name(name):
            match = re.search(r'_(\d+)([STY])_', name)
            if match:
                return int(match.group(1)), match.group(2)
            return None, None

        ptm_idx, _ = get_ptm_info_from_name(name)
        # Prefer the exact PTM site from seqres tokens (21/22/23) — robust to the
        # name format and gives correct 0-based centering for fixed_ptm_crop.
        _ptm_pos = np.where(np.asarray(seqres_idx) >= 21)[0]
        if len(_ptm_pos) > 0:
            ptm_idx = int(_ptm_pos[0])

        if L > self.args.crop:
            if getattr(self.args, 'fixed_ptm_crop', False) and ptm_idx is not None:
                start = max(0, min(ptm_idx - self.args.crop // 2, L - self.args.crop))
            else:
                start = np.random.randint(0, L - self.args.crop + 1)
            torsions = torsions[:, start:start + self.args.crop]
            frames = frames[:, start:start + self.args.crop]
            seqres_idx = seqres_idx[start:start + self.args.crop]
            mask = mask[start:start + self.args.crop]
            torsion_mask = torsion_mask[start:start + self.args.crop]

        elif L < self.args.crop:
            pad = self.args.crop - L
            frames = Rigid.cat([
                frames,
                Rigid.identity((self.args.num_frames, pad), requires_grad=False, fmt='rot_mat')
            ], 1)
            mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
            seqres_idx = np.concatenate([seqres_idx, np.zeros(pad, dtype=int)])
            torsions = torch.cat([torsions, torch.zeros((torsions.shape[0], pad, 7, 2), dtype=torch.float32)], 1)
            torsion_mask = torch.cat([torsion_mask, torch.zeros((pad, 7), dtype=torch.float32)])

        return {
            'name': target_key,
            'frame_start': frame_start,
            'torsions': torsions,
            'torsion_mask': torsion_mask,
            'trans': frames._trans,
            'rots': frames._rots._rot_mats,
            'seqres': seqres_idx,
            'mask': mask,
            'crop_start': start if L > self.args.crop else 0,
            'ptm_orig_idx': ptm_idx if ptm_idx is not None else -1,
        }


class ESM2Dataset(ProteinDataset):
    """Extends ProteinDataset with per-residue ESM2 embeddings loaded from a pkl file."""

    def __init__(self, args, split_csv, repeat=1):
        super().__init__(args, split_csv, repeat=repeat)

        self.esm2_dim = getattr(args, 'esm2_dim', 1280)
        emb_path = getattr(args, 'esm2_emb_path', None)

        if not emb_path:
            raise ValueError("ESM2Dataset requires --esm2_emb_path.")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"ESM2 embeddings file not found: {emb_path}")

        import pickle
        print(f"Loading ESM2 embeddings from: {emb_path}")
        with open(emb_path, 'rb') as f:
            self.esm2_embeddings = pickle.load(f)
        print(f"Loaded ESM2 embeddings for {len(self.esm2_embeddings)} proteins.")

    def __getitem__(self, idx):
        # Resample on bad samples (missing/mismatched ESM2, or trajectory errors
        # raised by super) instead of crashing.
        max_tries = max(1, getattr(self.args, 'max_resample', 100))
        for attempt in range(max_tries):
            try:
                return self._esm2_getitem(idx)
            except (ValueError, KeyError):
                if attempt == max_tries - 1:
                    raise
                idx = np.random.randint(0, len(self.df))

    def _esm2_getitem(self, idx):
        data = super().__getitem__(idx)

        raw_name  = data.get('name', '')
        # Strip replica suffix (_R1, _R2, …) so "6idx_A_R2" → "6idx_A"
        base_name = re.sub(r'_R\d+$', '', raw_name)
        candidates = [
            raw_name,
            base_name,
            raw_name.upper(),
            base_name.upper(),
            raw_name.lower(),
            base_name.lower(),
            raw_name.replace('.pdb', ''),
            base_name.replace('.pdb', ''),
        ]

        emb_raw = None
        for key in candidates:
            if key in self.esm2_embeddings:
                emb_raw = self.esm2_embeddings[key]
                break

        if emb_raw is None:
            raise KeyError(
                f"ESM2 embedding not found for protein '{raw_name}'. "
                f"Tried keys: {candidates}"
            )

        emb = torch.from_numpy(emb_raw).float() if isinstance(emb_raw, np.ndarray) else emb_raw

        seq_len    = len(data['seqres'])          # may be padded to crop size
        crop_start = data.get('crop_start', 0)
        # mask has 1s for real residues and 0s for zero-padding (short proteins)
        actual_len = int(np.array(data['mask']).sum())

        emb_actual = emb[crop_start: crop_start + actual_len]
        if emb_actual.shape[0] != actual_len:
            raise ValueError(
                f"ESM2 embedding length mismatch for '{raw_name}': "
                f"actual protein len={actual_len}, got emb len={emb_actual.shape[0]} "
                f"(crop_start={crop_start}, total_emb_len={emb.shape[0]})"
            )

        # Pad to seq_len if the protein was shorter than crop size
        if actual_len < seq_len:
            pad = seq_len - actual_len
            emb_actual = torch.cat([emb_actual,
                                    torch.zeros(pad, emb.shape[-1])], dim=0)

        data['esm2_emb'] = emb_actual
        return data


class MultiProteinDataset(torch.utils.data.Dataset):
    """Mixed multi-dataset training (ATLAS + MoDEL ...). Supports:
      - Sampling across datasets by weight (dataset_weights)
      - Per-dataset downsampling stride (dataset_frame_intervals)
      - Per-dataset ESM2 embedding pkl (esm_paths)
      - Resampling on too-short trajectories (no padding, to avoid a fake
        "static second half" signal)

    Adapted to ProteinFlux conventions:
      - Output ESM key = 'esm2_emb' (matches trainer/module.py, not 'esm_emb')
      - Sequence mapping via get_restype_idx (PTM-safe)
      - Supports max_train_frames (crop the window first, then take num_frames)
      - time-aligned forward: always take the first num_frames frames from frame0

    args needs (optional): atlas_data_dir / model_data_dir, num_frames, crop,
      dataset_weights(json), dataset_frame_intervals(json), esm_paths(json),
      esm2_dim, max_train_frames, max_resample.
    """

    def __init__(self, args, split_files, repeat=1):
        super().__init__()
        self.args = args
        self.repeat = repeat
        self.max_resample = int(getattr(args, 'max_resample', 20))

        def _json(v, default):
            if v is None: return default
            if isinstance(v, dict): return v
            try:
                import json
                return json.loads(v)
            except Exception:
                return default

        self.dataset_weights         = _json(getattr(args, 'dataset_weights', None), {})
        self.dataset_frame_intervals = _json(getattr(args, 'dataset_frame_intervals', None), {})
        self.esm_paths               = _json(getattr(args, 'esm_paths', None), {})
        self.default_frame_interval  = getattr(args, 'frame_interval', None)
        self.use_esm = bool(self.esm_paths)

        dataset_configs = {
            'atlas': {'h5_path': getattr(args, 'atlas_data_dir', None), 'has_replicas': True},
            'model': {'h5_path': getattr(args, 'model_data_dir', None), 'has_replicas': False},
        }

        self.datasets, self.h5_files = [], {}
        self.dataset_names, self.dataset_lengths = [], []
        for name, split_file in split_files.items():
            if not split_file or not os.path.exists(split_file):
                print(f"Warning: split not found for {name}: {split_file}"); continue
            cfg = dataset_configs.get(name, {})
            h5_path = cfg.get('h5_path')
            if not h5_path or not os.path.exists(h5_path):
                print(f"Warning: H5 not found for {name}: {h5_path}"); continue
            df = pd.read_csv(split_file)
            df = filter_short_sequences(
                df, source=split_file, report=True,
            )
            if df.empty:
                print(f"Warning: no valid sequences in {name}: {split_file}"); continue
            self.datasets.append({'name': name, 'df': df, 'config': cfg})
            self.dataset_names.append(name)
            self.dataset_lengths.append(len(df))
            self.h5_files[name] = h5py.File(h5_path, 'r')
            print(f"Loaded {name}: {len(df)} samples from {split_file}")
        if not self.datasets:
            raise ValueError("No valid datasets loaded!")

        # Sampling probabilities (per dataset, not per concatenated index)
        if not self.dataset_weights:
            self.dataset_weights = {n: 1.0 for n in self.dataset_names}
        self.dataset_weights = {k: float(v) for k, v in self.dataset_weights.items()
                                if k in self.dataset_names} or {n: 1.0 for n in self.dataset_names}
        w = np.array([self.dataset_weights.get(n, 0.0) for n in self.dataset_names], np.float64)
        if (w <= 0).all(): w = np.ones_like(w)
        self._dataset_cdf = np.cumsum(w / w.sum())

        self.total_length = int(self.repeat * sum(self.dataset_lengths))
        print(f"Multi-dataset: {dict(zip(self.dataset_names, self.dataset_lengths))} "
              f"| probs {[round(float(p),3) for p in np.diff(np.r_[0,self._dataset_cdf])]}")

        # Per-dataset ESM2 pkl
        self.esm_dicts = {}
        if self.use_esm:
            import pickle
            for dname in self.dataset_names:
                path = self.esm_paths.get(dname)
                if path and os.path.exists(path):
                    with open(path, 'rb') as f:
                        self.esm_dicts[dname] = pickle.load(f)
                    print(f"Loaded ESM2 for {dname}: {len(self.esm_dicts[dname])} entries")
                else:
                    print(f"Warning: ESM path not found for {dname}: {path}")

    def __del__(self):
        for h in getattr(self, 'h5_files', {}).values():
            try: h.close()
            except Exception: pass

    def __len__(self):
        if getattr(self.args, 'overfit_peptide', None): return 1000
        return self.total_length

    def _sample_dataset(self):
        i = int(np.searchsorted(self._dataset_cdf, np.random.rand(), side='right'))
        return self.dataset_names[min(max(i, 0), len(self.dataset_names) - 1)]

    def _frame_interval(self, name):
        if name in self.dataset_frame_intervals:
            return int(self.dataset_frame_intervals[name])
        if name in ('atlas', 'model'):
            return 1
        return int(self.default_frame_interval) if self.default_frame_interval else None

    def _clean_seqres(self, seq):
        s = str(seq).strip().upper()
        for ch in [',', ' ', '\t', '\n', '\r']:
            s = s.replace(ch, '')
        return s

    def _row(self, name_in, df):
        """Return (name, seqres); handles both atlas(name/seqres) and model(pdbids/protseq)."""
        if 'name' in df.columns and 'seqres' in df.columns:
            r = df.iloc[name_in]; return str(r['name']), str(r['seqres'])
        if 'pdbids' in df.columns and 'protseq' in df.columns:
            r = df.iloc[name_in]; return str(r['pdbids']), str(r['protseq'])
        raise ValueError(f"Unknown split schema: {list(df.columns)}")

    def __getitem__(self, idx):
        for attempt in range(self.max_resample + 1):
            dname = self._sample_dataset()
            ds = next(d for d in self.datasets if d['name'] == dname)
            df, h5 = ds['df'], self.h5_files[dname]

            if getattr(self.args, 'overfit_peptide', None) is None:
                name, seqres = self._row(np.random.randint(0, len(df)), df)
            else:
                name = seqres = self.args.overfit_peptide
            seqres = self._clean_seqres(seqres)
            if len(seqres) == 0:
                if attempt < self.max_resample: continue
                raise ValueError(f"Empty seqres: {dname}/{name}")

            # ESM2 embedding (resample on lookup miss)
            esm_emb = None
            if self.esm_dicts:
                d = self.esm_dicts.get(dname)
                if d is not None:
                    if name in d:
                        esm_emb = torch.from_numpy(np.array(d[name])).float()
                    else:
                        if attempt < self.max_resample: continue
                        raise KeyError(f"ESM2 emb not found: {dname}/{name}")

            # Resolve h5 key (atlas supports _R1-3)
            full = name
            if dname == 'atlas' and not name.upper().endswith(('_R1', '_R2', '_R3')):
                cands = [f"{name}_R{i}" for i in (1, 2, 3) if f"{name}_R{i}" in h5]
                full = np.random.choice(cands) if cands else name
            if full not in h5:
                if attempt < self.max_resample: continue
                raise KeyError(f"h5 key not found: {dname}/{name} (tried {full})")

            arr = h5[full][:]
            fi = self._frame_interval(dname)
            if fi and fi > 1: arr = arr[::fi]
            mtf = getattr(self.args, 'max_train_frames', None)
            if mtf is not None: arr = arr[:mtf]
            if arr.shape[0] < self.args.num_frames:
                if attempt < self.max_resample: continue
                raise ValueError(f"Too short: {dname}/{full} {arr.shape[0]}<{self.args.num_frames}")
            arr = np.copy(arr[:self.args.num_frames]).astype(np.float32)
            break

        if getattr(self.args, 'copy_frames', False): arr[1:] = arr[0]

        seqres_idx = np.array([get_restype_idx(c) for c in seqres])
        h5_len, seq_len = arr.shape[1], len(seqres_idx)
        if seq_len != h5_len:
            m = min(seq_len, h5_len); seqres_idx = seqres_idx[:m]; arr = arr[:, :m]

        frames = atom14_to_frames(torch.from_numpy(arr))
        aatype = torch.from_numpy(seqres_idx)[None].expand(self.args.num_frames, -1)
        aatype_geom = map_ptm_to_parent(aatype)
        atom37 = torch.from_numpy(atom14_to_atom37(arr, aatype_geom)).float()

        L = frames.shape[1]
        mask = np.ones(L, dtype=np.float32)
        torsions, torsion_mask = atom37_to_torsions(atom37, aatype_geom)
        torsion_mask = torsion_mask[0]

        # Align ESM2 length to L (fault-tolerant)
        if esm_emb is not None and esm_emb.shape[0] != L:
            if esm_emb.shape[0] > L:
                esm_emb = esm_emb[:L]
            else:
                esm_emb = torch.cat([esm_emb, torch.zeros(L - esm_emb.shape[0], esm_emb.shape[-1])], 0)

        crop = self.args.crop
        if L > crop:
            start = np.random.randint(0, L - crop + 1)
            torsions = torsions[:, start:start + crop]
            frames = frames[:, start:start + crop]
            seqres_idx = seqres_idx[start:start + crop]
            mask = mask[start:start + crop]
            torsion_mask = torsion_mask[start:start + crop]
            if esm_emb is not None: esm_emb = esm_emb[start:start + crop]
        elif L < crop:
            pad = crop - L
            frames = Rigid.cat([frames, Rigid.identity((self.args.num_frames, pad),
                                requires_grad=False, fmt='rot_mat')], 1)
            mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
            seqres_idx = np.concatenate([seqres_idx, np.zeros(pad, dtype=int)])
            torsions = torch.cat([torsions, torch.zeros((torsions.shape[0], pad, 7, 2))], 1)
            torsion_mask = torch.cat([torsion_mask, torch.zeros((pad, 7))])
            if esm_emb is not None:
                esm_emb = torch.cat([esm_emb, torch.zeros((pad, esm_emb.shape[-1]))], 0)

        out = {
            'name': full, 'dataset': dname, 'frame_start': 0,
            'torsions': torsions, 'torsion_mask': torsion_mask,
            'trans': frames._trans, 'rots': frames._rots._rot_mats,
            'seqres': seqres_idx, 'mask': mask,
            'crop_start': start if L > crop else 0,
        }
        if esm_emb is not None:
            out['esm2_emb'] = esm_emb     # <- key per ProteinFlux convention
        return out


class PTMDataset(ProteinDataset):
    """Attach full-length PTM features using the trajectory's exact crop.

    If ESM2 embeddings are also configured, both conditions are loaded. This is
    important for FluxSite fine-tuning: PTM features augment rather than replace
    the ESM2 condition used by the pretrained ProteinFlux checkpoint.
    """

    def __init__(self, args, split_csv, repeat=1):
        super().__init__(args, split_csv, repeat=repeat)

        self.feat_path = getattr(args, 'ptm_feat_path', None)
        self.ptm_feat_dim = int(getattr(args, 'ptm_feat_dim', 256))
        self.ptm_features = self._load_pickle(self.feat_path, "PTM features")

        self.esm2_dim = int(getattr(args, 'esm2_dim', 1280))
        esm_path = getattr(args, 'esm2_emb_path', None)
        self.esm2_embeddings = self._load_pickle(esm_path, "ESM2 embeddings")

    @staticmethod
    def _load_pickle(path, label):
        if not path:
            return None
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")
        import pickle
        print(f"Loading {label} from: {path}")
        with open(path, 'rb') as f:
            features = pickle.load(f)
        print(f"Loaded {label} for {len(features)} proteins.")
        return features

    def __getitem__(self, idx):
        max_tries = max(1, getattr(self.args, 'max_resample', 100))
        for attempt in range(max_tries):
            try:
                return self._ptm_getitem(idx)
            except (ValueError, KeyError):
                if attempt == max_tries - 1:
                    raise
                idx = np.random.randint(0, len(self.df))

    @staticmethod
    def _feature_candidates(raw_name):
        clean_name = raw_name.replace('.pdb', '')
        base_name = re.sub(r'_R\d+$', '', clean_name, flags=re.IGNORECASE)
        return [
            clean_name, base_name,
            clean_name.upper(), base_name.upper(),
            clean_name.lower(), base_name.lower(),
        ]

    @classmethod
    def _lookup_feature(cls, feature_dict, raw_name, label):
        for key in cls._feature_candidates(raw_name):
            if key in feature_dict:
                return feature_dict[key]
        raise KeyError(
            f"{label} not found for '{raw_name}'. "
            f"Tried keys: {cls._feature_candidates(raw_name)}"
        )

    @staticmethod
    def _crop_and_pad_feature(feature, data, raw_name, label, expected_dim):
        tensor = (
            torch.from_numpy(feature).float()
            if isinstance(feature, np.ndarray)
            else torch.as_tensor(feature).float()
        )
        if tensor.ndim != 2 or tensor.shape[1] != expected_dim:
            raise ValueError(
                f"{label} shape mismatch for '{raw_name}': expected [L,{expected_dim}], "
                f"got {tuple(tensor.shape)}"
            )

        seq_len = len(data['seqres'])
        crop_start = int(data.get('crop_start', 0))
        actual_len = int(np.asarray(data['mask']).sum())
        cropped = tensor[crop_start:crop_start + actual_len]
        if cropped.shape[0] != actual_len:
            raise ValueError(
                f"{label} length mismatch for '{raw_name}': actual_len={actual_len}, "
                f"crop_start={crop_start}, full_feature_len={tensor.shape[0]}"
            )
        if actual_len < seq_len:
            cropped = torch.cat(
                [cropped, torch.zeros(seq_len - actual_len, expected_dim)],
                dim=0,
            )
        return cropped

    def _ptm_getitem(self, idx):
        data = super().__getitem__(idx)
        raw_name = data.get('name', '')

        if self.ptm_features is None:
            raise ValueError("PTM feature dictionary was not loaded")
        ptm_raw = self._lookup_feature(self.ptm_features, raw_name, "PTM feature")
        data['ptm_feat'] = self._crop_and_pad_feature(
            ptm_raw, data, raw_name, "PTM feature", self.ptm_feat_dim
        )

        if self.esm2_embeddings is not None:
            esm_raw = self._lookup_feature(
                self.esm2_embeddings, raw_name, "ESM2 embedding"
            )
            data['esm2_emb'] = self._crop_and_pad_feature(
                esm_raw, data, raw_name, "ESM2 embedding", self.esm2_dim
            )

        return data
