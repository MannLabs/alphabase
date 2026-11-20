"""Constants of the SMILES encoding."""

from enum import IntEnum


class Anchor(IntEnum):
    """Atom-map number that marks the atom at which two molecules are joined.

    Residues and terminal modifications are complete molecules, naming follows
    the functional group in the amino acid.

    Example
    -------
    - Alanine ``[NH2:1][C@@H](C)C(=O)[OH:2]``. The N-Terminus is marked with :1 and the C-terminus with :2
    """

    N_TERM = 1
    """The alpha-amine nitrogen."""

    C_TERM = 2
    """The hydroxyl oxygen of the alpha-carboxylic acid."""
