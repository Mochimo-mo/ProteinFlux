"""Precompute per-residue secondary structure labels from H5 trajectory data.

Uses pydssp (pure PyTorch DSSP implementation) on backbone atoms from frame 0.
Output: pickle file mapping protein name -> np.int8 array (L,), 0=H 1=E 2=C.

Usage:
    python scripts/precompute_ss.py --h5_path data/proteins.h5 --output_pkl data/ss_labels.pkl
"""
import argparse
import pickle
import numpy as np
import h5py
import torch
from tqdm import tqdm

try:
    import pydssp
except ImportError:
    raise ImportError("Run: pip install pydssp")

# DSSP 8-class -> 3-class mapping
# H=alpha-helix, G=3-10 helix, I=pi-helix -> 0 (helix)
# E=beta-strand, B=beta-bridge            -> 1 (sheet)
# T=turn, S=bend, C/space=coil            -> 2 (coil)
DSSP8_TO_SS3 = {
    'H': 0, 'G': 0, 'I': 0,
    'E': 1, 'B': 1,
    'T': 2, 'S': 2, 'C': 2, ' ': 2,
}


def assign_ss(atom14_frame: np.ndarray) -> np.ndarray:
    """
    Args:
        atom14_frame: (L, 14, 3) float32, atom14 coords for one frame
    Returns:
        ss: (L,) int8, 0=H 1=E 2=C
    """
    # Backbone atoms: N=0, CA=1, C=2, O=4 in atom14 ordering
    coords = torch.from_numpy(atom14_frame[:, [0, 1, 2, 4], :]).float()
    coords = coords.unsqueeze(0)  # (1, L, 4, 3)

    dssp_out = pydssp.assign(coords)  # list of 1 string, length L, 8-class codes
    ss_str = dssp_out[0]

    ss = np.array([DSSP8_TO_SS3.get(c, 2) for c in ss_str], dtype=np.int8)
    return ss


def main():
    parser = argparse.ArgumentParser(description="Precompute SS labels from H5 data.")
    parser.add_argument('--h5_path', type=str, required=True, help='Path to H5 trajectory file')
    parser.add_argument('--output_pkl', type=str, required=True, help='Output pickle path')
    parser.add_argument('--frame', type=int, default=0, help='Which frame to use (default: 0)')
    args = parser.parse_args()

    print(f"Reading H5: {args.h5_path}")
    h5 = h5py.File(args.h5_path, 'r')
    keys = list(h5.keys())
    print(f"Total entries: {len(keys)}")

    ss_dict = {}
    errors = []

    for name in tqdm(keys):
        try:
            arr = h5[name][:]                                  # (T, L, 14, 3)
            frame_idx = min(args.frame, arr.shape[0] - 1)
            frame = arr[frame_idx].astype(np.float32)          # (L, 14, 3)
            ss = assign_ss(frame)
            ss_dict[name] = ss
        except Exception as e:
            errors.append((name, str(e)))

    h5.close()

    with open(args.output_pkl, 'wb') as f:
        pickle.dump(ss_dict, f)

    print(f"\nSaved {len(ss_dict)} entries to {args.output_pkl}")

    # --- Statistics ---
    all_ss = np.concatenate(list(ss_dict.values()))
    total = len(all_ss)
    h_pct = (all_ss == 0).mean() * 100
    e_pct = (all_ss == 1).mean() * 100
    c_pct = (all_ss == 2).mean() * 100
    print(f"\nSS distribution over {total} residues:")
    print(f"  Helix (H): {h_pct:.1f}%")
    print(f"  Sheet (E): {e_pct:.1f}%")
    print(f"  Coil  (C): {c_pct:.1f}%")

    if errors:
        print(f"\n{len(errors)} errors (first 10):")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")

    # --- Spot-check ---
    print("\nSpot-check (first 5 entries):")
    label_chars = ['H', 'E', 'C']
    for name in list(ss_dict.keys())[:5]:
        ss = ss_dict[name]
        ss_str = ''.join(label_chars[v] for v in ss)
        h = (ss == 0).sum()
        e = (ss == 1).sum()
        c = (ss == 2).sum()
        print(f"  {name}  L={len(ss)}  H={h} E={e} C={c}")
        print(f"    {ss_str[:80]}")


if __name__ == '__main__':
    main()
