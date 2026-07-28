import os
import tempfile

import numpy as np
from . import protein
from .geometry import atom14_to_atom37
from .ptm_utils import map_ptm_to_parent


def atom14_to_pdb(atom14, aatype, path):
    """Atomically write an atom14 trajectory as a multi-model PDB."""
    aatype_geom = map_ptm_to_parent(aatype)
    prots = []
    for i, pos in enumerate(atom14):
        pos = atom14_to_atom37(pos, aatype_geom)
        prots.append(create_full_prot(pos, aatype=aatype_geom))
    pdb_text = prots_to_pdb(prots)
    if "\nATOM " not in pdb_text and not pdb_text.startswith("ATOM "):
        raise ValueError(f"Refusing to write a PDB without ATOM records: {path}")

    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".pdb_tmp_", suffix=".pdb", dir=out_dir)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(pdb_text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def create_full_prot(atom37: np.ndarray, aatype=None, b_factors=None):
    """Create a Protein object from atom37 coordinates."""
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    residue_index = np.arange(n)
    atom37_mask = np.sum(np.abs(atom37), axis=-1) > 1e-7
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    chain_index = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=chain_index
    )


def prots_to_pdb(prots):
    """Convert a list of Protein objects to a multi-model PDB string."""
    ss = ''
    for i, prot in enumerate(prots):
        ss += f'MODEL {i}\n'
        prot = protein.to_pdb(prot)
        ss += '\n'.join(prot.split('\n')[2:-3])
        ss += '\nENDMDL\n'
    return ss
