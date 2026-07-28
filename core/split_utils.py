"""Utilities for validating protein split tables before model execution."""

from __future__ import annotations

from typing import Any


MIN_SEQUENCE_LENGTH = 2


def filter_short_sequences(
    df: Any,
    *,
    seq_col: str = "seqres",
    id_col: str = "name",
    min_length: int = MIN_SEQUENCE_LENGTH,
    source: str = "split",
    report: bool = True,
):
    """Return a copy excluding sequences too short for geometry/model code.

    ProteinFlux builds pairwise/rigid geometry tensors that require at least
    two residues. Empty, missing, and one-residue sequences are therefore
    invalid inference/evaluation samples.
    """
    if seq_col not in df.columns:
        return df

    sequences = df[seq_col].fillna("").astype(str).str.strip()
    lengths = sequences.str.len()
    invalid = lengths < min_length
    if not invalid.any():
        return df

    if report:
        bad_rows = df.loc[invalid]
        bad_lengths = lengths.loc[invalid]
        if id_col in bad_rows.columns:
            identifiers = bad_rows[id_col].astype(str).tolist()
        else:
            identifiers = [str(value) for value in bad_rows.index]
        details = ", ".join(
            f"{identifier}(L={length})"
            for identifier, length in zip(identifiers, bad_lengths.tolist())
        )
        print(
            f"⚠️  Excluding {int(invalid.sum())} invalid sequence(s) from "
            f"{source}; minimum length is {min_length}: {details}"
        )

    return df.loc[~invalid].copy()
