"""PTM-conditioning controls used for inference ablations."""

from __future__ import annotations

import numpy as np


PTM_TOKEN_TO_ID = {21: 1, 22: 2, 23: 3}
PTM_TOKEN_TO_PARENT = {21: 15, 22: 16, 23: 18}
PHOSPHORYLATABLE_TOKENS = frozenset(PTM_TOKEN_TO_PARENT.values())


def ptm_condition_ids(seqres, mode="none"):
    """Build explicit PTM condition IDs without modifying sequence geometry.

    Args:
        seqres: Residue tokens where 21/22/23 denote pSer/pThr/pTyr.
        mode:
          - ``none``: PTM IDs at their true positions.
          - ``zero``: remove all explicit PTM IDs.
          - ``wrong_site``: move each PTM ID to the farthest unmodified
            residue of the same parent type (pSer->Ser, pThr->Thr,
            pTyr->Tyr). If none exists, use the farthest other S/T/Y.

    Returns:
        ``(ids, moves)`` where IDs use 0=no PTM and 1/2/3=pSer/pThr/pTyr.
        ``moves`` contains
        ``(source_index, destination_index, ptm_id, selection_kind)``.
    """
    tokens = np.asarray(seqres, dtype=np.int64)
    if tokens.ndim != 1:
        raise ValueError(f"Expected 1D seqres tokens, got shape {tokens.shape}")
    if mode not in {"none", "zero", "wrong_site"}:
        raise ValueError(f"Unknown PTM ablation mode: {mode}")

    true_ids = np.zeros_like(tokens)
    for token, ptm_id in PTM_TOKEN_TO_ID.items():
        true_ids[tokens == token] = ptm_id

    if mode == "none":
        return true_ids, []
    if mode == "zero":
        return np.zeros_like(true_ids), []
    shifted = np.zeros_like(true_ids)
    moves = []
    occupied = set()
    for source in np.flatnonzero(true_ids):
        source = int(source)
        ptm_id = int(true_ids[source])
        ptm_token = int(tokens[source])
        parent_token = PTM_TOKEN_TO_PARENT[ptm_token]

        same_type = [
            int(index)
            for index in np.flatnonzero(tokens == parent_token)
            if int(index) not in occupied
        ]
        if same_type:
            candidates = same_type
            selection_kind = "same_parent"
        else:
            candidates = [
                index
                for index, token in enumerate(tokens.tolist())
                if token in PHOSPHORYLATABLE_TOKENS and index not in occupied
            ]
            selection_kind = "other_sty"

        if not candidates:
            raise ValueError(
                "wrong_site ablation found no alternative unmodified S/T/Y residue"
            )

        # Farthest in primary-sequence distance; lower index breaks ties.
        destination = max(candidates, key=lambda index: (abs(index - source), -index))
        occupied.add(destination)
        shifted[destination] = ptm_id
        moves.append((source, destination, ptm_id, selection_kind))
    return shifted, moves


def ptm_condition_aatype(seqres, mode="none"):
    """Ablate the PTM signal carried by the aa-embedding TOKENS (21/22/23), for
    AA-base models where phospho enters via aatype_to_emb rather than ptm_channel.

    Returns a copy of the seqres tokens to use for CONDITIONING (geometry / the
    initial structure must keep the original tokens). Reuses the same wrong_site
    destination logic as ptm_condition_ids.

      none       -> unchanged
      zero       -> 21/22/23 -> parent S/T/Y (15/16/18)
      wrong_site -> move each PTM token to the farthest same-type unmodified
                    residue (source -> parent, destination -> PTM token)
    """
    tokens = np.asarray(seqres, dtype=np.int64).copy()
    if mode == "none":
        return tokens
    if mode == "zero":
        for token, parent in PTM_TOKEN_TO_PARENT.items():
            tokens[tokens == token] = parent
        return tokens
    if mode != "wrong_site":
        raise ValueError(f"Unknown PTM ablation mode: {mode}")

    out = tokens.copy()
    occupied = set()
    for source in np.flatnonzero(np.isin(tokens, list(PTM_TOKEN_TO_PARENT))):
        source = int(source)
        ptm_token = int(tokens[source])
        parent_token = PTM_TOKEN_TO_PARENT[ptm_token]
        same_type = [int(i) for i in np.flatnonzero(tokens == parent_token)
                     if int(i) not in occupied]
        if same_type:
            candidates = same_type
        else:
            candidates = [i for i, t in enumerate(tokens.tolist())
                          if t in PHOSPHORYLATABLE_TOKENS and i not in occupied]
        if not candidates:
            raise ValueError("wrong_site (aatype) found no alternative S/T/Y residue")
        destination = max(candidates, key=lambda i: (abs(i - source), -i))
        occupied.add(destination)
        out[source] = parent_token    # source back to unmodified
        out[destination] = ptm_token  # phospho token at the wrong site
    return out
