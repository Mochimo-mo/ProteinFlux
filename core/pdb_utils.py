"""Lightweight PDB file validation helpers."""

import os


def is_valid_pdb(path):
    """Quickly check that a PDB exists and contains atoms in a closed model."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    has_atom = False
    has_endmdl = False
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if line.startswith(("ATOM  ", "HETATM")):
                    has_atom = True
                elif line.startswith("ENDMDL"):
                    has_endmdl = True
                if has_atom and has_endmdl:
                    return True
    except OSError:
        return False
    return False
