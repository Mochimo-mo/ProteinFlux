import torch
import os
import numpy as np
import pandas as pd
import h5py
import re

from core.rigid_utils import Rigid
from core.residue_constants import restype_order
from core.geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames


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
        idx = idx % len(self.df)
        if self.args.overfit:
            idx = 0

        if self.args.overfit_peptide is None:
            name = self.df.index[idx]
            seqres_str = self.df.seqres[name]
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

        arr = self.h5_file[target_key][:]

        frame_interval = getattr(self.args, 'frame_interval', None)
        if frame_interval:
            arr = arr[::frame_interval]

        total_frames = arr.shape[0]
        if total_frames < self.args.num_frames:
            pad_len = self.args.num_frames - total_frames
            arr = np.pad(arr, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='edge')
            frame_start = 0
        else:
            frame_start = np.random.choice(np.arange(total_frames - self.args.num_frames + 1))

        if self.args.overfit_frame:
            frame_start = 0

        end = frame_start + self.args.num_frames
        arr = np.copy(arr[frame_start:end]).astype(np.float32)

        if self.args.copy_frames:
            arr[1:] = arr[0]

        seqres_idx = np.array([get_restype_idx(c) for c in seqres_str])

        h5_res_len = arr.shape[1]
        seq_len = len(seqres_idx)
        if seq_len != h5_res_len:
            min_len = min(seq_len, h5_res_len)
            seqres_idx = seqres_idx[:min_len]
            arr = arr[:, :min_len, :, :]

        frames = atom14_to_frames(torch.from_numpy(arr))
        aatype = torch.from_numpy(seqres_idx)[None].expand(self.args.num_frames, -1)

        # Map PTM indices to parent amino acids for geometry lookups (SEP->SER, TPO->THR, PTR->TYR)
        aatype_geom = aatype.clone()
        aatype_geom[aatype_geom == 21] = 15
        aatype_geom[aatype_geom == 22] = 16
        aatype_geom[aatype_geom == 23] = 19

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


class PTMDataset(ProteinDataset):
    """Extends ProteinDataset with residue-level PTM features loaded from a pkl file."""

    def __init__(self, args, split_csv, repeat=1):
        super().__init__(args, split_csv, repeat=repeat)

        self.ptm_features = None
        self.feat_path = getattr(args, 'ptm_feat_path', None)

        if self.feat_path and os.path.exists(self.feat_path):
            import pickle
            print(f"Loading PTM features from: {self.feat_path}")
            try:
                with open(self.feat_path, 'rb') as f:
                    self.ptm_features = pickle.load(f)
                print(f"Loaded features for {len(self.ptm_features)} proteins.")
            except Exception as e:
                print(f"Failed to load PTM feature pkl: {e}")

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        seq_len = len(data['seqres'])
        feat_dim = 256
        ptm_tensor = torch.zeros(seq_len, feat_dim)

        if self.ptm_features is not None:
            pdb_name = data.get('name', 'UNKNOWN').replace('.pdb', '').upper()

            if pdb_name in self.ptm_features:
                try:
                    inner_dict = self.ptm_features[pdb_name]
                    sorted_keys = sorted(
                        [k for k in inner_dict.keys() if str(k).isdigit()],
                        key=lambda x: int(x)
                    )
                    if sorted_keys:
                        arrays = [inner_dict[k] for k in sorted_keys]
                        extracted_feat = torch.from_numpy(np.stack(arrays)).float()
                        if extracted_feat.shape[0] == seq_len:
                            ptm_tensor = extracted_feat
                except Exception:
                    pass

        data['ptm_feat'] = ptm_tensor
        return data
