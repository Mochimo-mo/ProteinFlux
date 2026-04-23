"""PTM residue mapping utilities.

Provides reusable residue expectations and ID mappings for common post-translational
modifications (PTMs). Falls back to data-driven mappings when the PTM type is not
in the predefined catalogue.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

# Predefined common PTM residue information (at least 10 PTM types)
PTM_RESIDUE_INFO: Dict[str, Dict[str, List[str]]] = {
    'phosphorylation': {
        'expected_residues': ['S', 'T', 'Y'],
    },
    'methylation': {
        'expected_residues': ['K', 'R'],
    },
    'acetylation': {
        'expected_residues': ['K'],
    },
    'ubiquitination': {
        'expected_residues': ['K'],
    },
    'sumoylation': {
        'expected_residues': ['K'],
    },
    'n_linked_glycosylation': {
        'expected_residues': ['N'],
    },
    'o_linked_glycosylation': {
        'expected_residues': ['S', 'T'],
    },
    'glycation': {
        'expected_residues': ['K', 'R'],
    },
    'palmitoylation': {
        'expected_residues': ['C'],
    },
    'nitrosylation': {
        'expected_residues': ['C'],
    },
    'malonylation': {
        'expected_residues': ['K'],
    },
    's-nitrosylation': {
        'expected_residues': ['C'],
    },
    'hydroxylation':{
        'expected_residues': ['P','K','W','Y'],
    },
    'amidation':{
        'expected_residues': ['G','F','Y','R','V','L'],
    },
    "glutathionylation": {
        'expected_residues': ['C'],
    },
    'Crotonylation':{
        'expected_residues': ['K'],
    },
}

# Handle common aliases and normalize case
_PTM_ALIASES = {
    'n-linked glycosylation': 'n_linked_glycosylation',
    'o-linked glycosylation': 'o_linked_glycosylation',
    'n-linked_glycosylation': 'n_linked_glycosylation',
    'o-linked_glycosylation': 'o_linked_glycosylation',
    'smalonylation': 'malonylation',  # legacy typo
}


def _normalize_ptm_type(ptm_type: Optional[str]) -> str:
    if not ptm_type:
        return 'unknown'
    normalized = str(ptm_type).strip().lower()
    normalized = normalized.replace(' ', '_')
    return _PTM_ALIASES.get(normalized, normalized)


def _normalize_residue(residue: Optional[str]) -> Optional[str]:
    if residue is None:
        return None
    res = str(residue).strip()
    if not res:
        return None
    # Only accept single-letter amino acid residues; map everything else to \"Other\"
    res = res.upper()
    if len(res) == 1:
        return res
    return None


def resolve_ptm_residue_info(
    ptm_type: Optional[str],
    observed_residues: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Return residue-mapping information for a given PTM type.

    Args:
        ptm_type: Target PTM type.
        observed_residues: Residues observed in data, used to extend or dynamically build the mapping.

    Returns:
        A dictionary with:
            - ptm_type: Normalized PTM type string.
            - expected_residues: set[str] of expected residues.
            - residue_list: Ordered list[str] used to build indices.
            - residue_to_id: dict[str, int] mapping residues to integer IDs, including \"Other\".
            - other_id: int ID for \"Other\".
            - vocab_size: int size of the residue vocabulary.
    """
    normalized_ptm = _normalize_ptm_type(ptm_type)
    base_info = PTM_RESIDUE_INFO.get(normalized_ptm, {})
    base_residues = [res.upper() for res in base_info.get('expected_residues', [])]

    residue_list: List[str] = []
    seen: Set[str] = set()

    for residue in base_residues:
        if residue and residue not in seen:
            residue_list.append(residue)
            seen.add(residue)

    # [FIX] Always include all standard amino acids to support motif learning (e.g. K-x-E)
    STANDARD_AAS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for aa in STANDARD_AAS:
        if aa not in seen:
            residue_list.append(aa)
            seen.add(aa)

    if observed_residues:
        for residue in observed_residues:
            normalized_residue = _normalize_residue(residue)
            if normalized_residue and normalized_residue not in seen:
                residue_list.append(normalized_residue)
                seen.add(normalized_residue)

    if not residue_list:
        residue_list.append('X')  # Placeholder indicating residue information is missing
        seen.add('X')

    residue_to_id = {residue: idx for idx, residue in enumerate(residue_list)}

    # Ensure an \"Other\" mapping exists and is placed at the end
    if 'Other' not in residue_to_id:
        residue_to_id['Other'] = len(residue_to_id)
    other_id = residue_to_id['Other']

    expected_residue_set = set(residue_list)

    return {
        'ptm_type': normalized_ptm,
        'expected_residues': expected_residue_set,
        'residue_list': residue_list,
        'residue_to_id': residue_to_id,
        'other_id': other_id,
        'vocab_size': len(residue_to_id),
    }


__all__ = ['PTM_RESIDUE_INFO', 'resolve_ptm_residue_info']
