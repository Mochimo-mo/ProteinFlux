"""Shared H5 structure reader: handles both plain-dataset and PTM-group formats.

- Plain format: key -> dataset [F, L, 14, 3].
- PTM format: key -> group {coords[F,T,14,3], seqres[L], token_type_id[T], ...},
  where T = L residue tokens + phosphate-atom tokens. Select residue tokens by
  token_type_id==0 (drop phosphate-atom tokens, i.e. "do not model phosphate
  atoms"), and return the H5 seqres (carrying PTM tokens 21/22/23, which drive
  the model's ptm_channel).

inference.py and eval scripts all read H5 through this function, avoiding
duplication and staying correct for PTM data.
"""
import numpy as np
import h5py


def read_structure(node):
    """Return (coords[F, L, 14, 3] float32, seqres_idx[L] int64 or None).

    PTM group -> filter to residue tokens + return H5 seqres; plain dataset ->
    coords as-is, seqres_idx=None (caller uses its own seqres).
    """
    if isinstance(node, h5py.Group):
        tt = np.asarray(node['token_type_id'][:])
        res = (tt == 0)
        coords = np.asarray(node['coords'][:])[:, res].astype(np.float32)
        seqres_idx = np.asarray(node['seqres'][:]).astype(np.int64)
        return coords, seqres_idx
    return np.asarray(node[:]).astype(np.float32), None
