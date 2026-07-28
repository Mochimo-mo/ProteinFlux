"""Shared residue-index helpers for phosphorylated amino acids."""

PTM_TO_PARENT = {
    21: 15,  # SEP / pSer -> SER
    22: 16,  # TPO / pThr -> THR
    23: 18,  # PTR / pTyr -> TYR
}


def map_ptm_to_parent(aatype):
    """Return a copy with PTM tokens replaced by standard parent residues."""
    mapped = aatype.clone() if hasattr(aatype, "clone") else aatype.copy()
    for ptm_idx, parent_idx in PTM_TO_PARENT.items():
        mapped[mapped == ptm_idx] = parent_idx
    return mapped
