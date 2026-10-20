"""Convert the smiles column of the constants tables to the anchor format, in place.

The tables are amino_acid.tsv and modification.tsv; the anchor format is the one read by
alphabase.smiles.smiles.

Placeholder format: a residue marks the alpha-amine hydrogens with [Fl] and the
carboxylic hydroxyl with [Ts]. An N-terminal modification either contains [Lv]
standing for the amine nitrogen, or bonds its first atom to that nitrogen.
A C-terminal modification replaces the hydroxyl with its first atom.

Usage::

    python scripts/convert_legacy_smiles.py [CONST_FILES_DIR]
"""

import sys
from pathlib import Path

from rdkit import Chem

from alphabase.smiles.smiles import (
    C_TERM_ANCHOR,
    N_TERM_ANCHOR,
    SmilesKind,
    parse_smiles,
)

DEFAULT_CONST_FILES_DIR = (
    Path(__file__).parents[1] / "alphabase" / "constants" / "const_files"
)


def convert_legacy_smiles(smiles: str, kind: SmilesKind) -> str:
    """Convert a SMILES written with placeholder atoms into the anchor format.

    Parameters
    ----------
    smiles : str
        SMILES in the placeholder format.
    kind : {"residue", "n_term", "c_term"}
        What the SMILES describes.

    Returns
    -------
    str
        Equivalent SMILES in the anchor format, written starting from an anchor.

    Raises
    ------
    ValueError
        If the SMILES is invalid or already carries atom maps.

    """
    legacy_mol = Chem.MolFromSmiles(smiles)
    if legacy_mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    # Without [Lv], an N-terminal modification would otherwise silently gain a nitrogen.
    if any(atom.GetAtomMapNum() for atom in legacy_mol.GetAtoms()):
        raise ValueError(f"SMILES {smiles} is already in the anchor format")
    mol = Chem.RWMol(legacy_mol)

    if kind == "residue":
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "Fl":
                atom.GetNeighbors()[0].SetAtomMapNum(N_TERM_ANCHOR)
                atom.SetAtomicNum(1)
            elif atom.GetSymbol() == "Ts":
                atom.SetAtomicNum(8)
                atom.SetAtomMapNum(C_TERM_ANCHOR)
                atom.SetNoImplicit(False)  # noqa: FBT003
    elif kind == "n_term":
        placeholders = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "Lv"]
        for atom in placeholders:
            atom.SetAtomicNum(7)
            atom.SetAtomMapNum(N_TERM_ANCHOR)
            atom.SetNoImplicit(False)  # noqa: FBT003
        if not placeholders:
            n_idx = mol.AddAtom(Chem.Atom(7))
            mol.GetAtomWithIdx(n_idx).SetAtomMapNum(N_TERM_ANCHOR)
            mol.AddBond(0, n_idx, Chem.BondType.SINGLE)
    else:
        mol.GetAtomWithIdx(0).SetAtomMapNum(C_TERM_ANCHOR)

    converted = mol.GetMol()
    Chem.SanitizeMol(converted)
    # Drops the [Fl] turned into hydrogens as well as redundant explicit hydrogens.
    converted = Chem.RemoveHs(converted)
    root_anchor = C_TERM_ANCHOR if kind == "c_term" else N_TERM_ANCHOR
    root_idx = next(
        atom.GetIdx()
        for atom in converted.GetAtoms()
        if atom.GetAtomMapNum() == root_anchor
    )
    converted_smiles = Chem.MolToSmiles(converted, rootedAtAtom=root_idx)
    parse_smiles(converted_smiles, kind)
    return converted_smiles


def _kind_of_modification(mod_name: str) -> SmilesKind:
    """Classify a modification name as residue-level, N-terminal or C-terminal."""
    # Same classification as AminoAcidModifier; "^Any_N-term" PTMs are residue-level.
    if "@Protein_N-term" in mod_name or "@Any_N-term" in mod_name:
        return "n_term"
    if "@Protein_C-term" in mod_name or "@Any_C-term" in mod_name:
        return "c_term"
    return "residue"


def convert_table(path: Path, *, is_modification_table: bool) -> int:
    """Rewrite the smiles column of a tab-separated constants table in place.

    Only the smiles fields change, so the diff of the table stays minimal.

    Parameters
    ----------
    path : Path
        Path to amino_acid.tsv or modification.tsv.
    is_modification_table : bool
        Whether the kind of each SMILES must be derived from the modification name;
        all rows of amino_acid.tsv are residues.

    Returns
    -------
    int
        Number of converted SMILES.

    """
    lines = path.read_text().split("\n")
    smiles_col = lines[0].split("\t").index("smiles")
    n_converted = 0
    for line_idx, line in enumerate(lines[1:], start=1):
        fields = line.split("\t")
        # Rows without SMILES may omit the trailing empty field.
        if smiles_col >= len(fields) or not fields[smiles_col]:
            continue
        kind = _kind_of_modification(fields[0]) if is_modification_table else "residue"
        fields[smiles_col] = convert_legacy_smiles(fields[smiles_col], kind)
        lines[line_idx] = "\t".join(fields)
        n_converted += 1
    path.write_text("\n".join(lines))
    return n_converted


if __name__ == "__main__":
    const_files_dir = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONST_FILES_DIR
    )
    for file_name, is_modification_table in [
        ("amino_acid.tsv", False),
        ("modification.tsv", True),
    ]:
        n_converted = convert_table(
            const_files_dir / file_name, is_modification_table=is_modification_table
        )
        print(  # noqa: T201
            f"{file_name}: converted {n_converted} SMILES"
        )
