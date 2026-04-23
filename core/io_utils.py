import numpy as np
from . import protein
from .geometry import atom14_to_atom37


def atom14_to_pdb(atom14, aatype, path):
    """Write atom14 trajectory to a multi-model PDB file."""
    prots = []
    for i, pos in enumerate(atom14):
        pos = atom14_to_atom37(pos, aatype)
        prots.append(create_full_prot(pos, aatype=aatype))
    with open(path, 'w') as f:
        f.write(prots_to_pdb(prots))


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
