from typing import Literal, Optional

import pandas as pd

try:
    from rdkit import Chem
except ImportError as e:
    raise ImportError(
        "rdkit is required for the smiles module. "
        "Install it through pip or install alphabase with the 'rdkit' or 'full' extra (or their '-stable' variants)."
    ) from e

from alphabase.constants.aa import aa_formula
from alphabase.constants.modification import MOD_DF

# Residues and terminal modifications are complete molecules. The atoms at which they
# are joined carry these atom-map numbers, e.g. alanine is [NH2:1][C@@H](C)C(=O)[OH:2].
N_TERM_ANCHOR = 1  # the alpha-amine nitrogen
C_TERM_ANCHOR = 2  # the hydroxyl oxygen of the alpha-carboxylic acid

SmilesKind = Literal["residue", "n_term", "c_term"]

_ANCHORS_BY_KIND: dict[str, tuple[int, ...]] = {
    "residue": (N_TERM_ANCHOR, C_TERM_ANCHOR),
    "n_term": (N_TERM_ANCHOR,),
    "c_term": (C_TERM_ANCHOR,),
}
_TETRAHEDRAL_TAGS = (
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
)


def parse_smiles(smiles: str, kind: SmilesKind) -> Chem.Mol:
    """
    Parse the SMILES of a residue or a terminal modification and validate its anchors.

    Parameters
    ----------
    smiles : str
        SMILES with atom-mapped anchors: a residue carries both :1 (alpha-amine nitrogen)
        and :2 (carboxylic hydroxyl oxygen), an N-terminal modification only :1 and a
        C-terminal modification only :2.
    kind : {"residue", "n_term", "c_term"}
        What the SMILES describes.

    Returns
    -------
    Chem.Mol
        The sanitized molecule, anchors still mapped.

    Raises
    ------
    ValueError
        If the SMILES is invalid or does not carry exactly the anchors expected for `kind`.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    expected = _ANCHORS_BY_KIND[kind]
    found = tuple(
        sorted(atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum())
    )
    if found != expected:
        raise ValueError(
            f"A {kind} SMILES must carry exactly the atom maps {list(expected)}, "
            f"found {list(found)} in {smiles}"
        )
    return mol


def merge_at_anchors(
    target: Chem.Mol,
    target_anchor: int,
    fragment: Chem.Mol,
    fragment_anchor: int,
) -> Chem.Mol:
    """
    Fuse the anchor atom of `fragment` into the anchor atom of `target`.

    The fused atom takes element, formal charge and isotope from the fragment's anchor
    and is bonded to the neighbours of both anchors, with the fragment's bond orders.
    Hydrogens are recomputed, and tetrahedral stereo of all neighbours is retained.
    The fused atom keeps the anchor label if both labels are equal, otherwise it is no
    longer an anchor (as for the nitrogen of a peptide bond).

    Parameters
    ----------
    target : Chem.Mol
        Molecule that keeps its atom indices, e.g. a residue or a growing peptide.
    target_anchor : int
        Atom-map number of the anchor in `target`.
    fragment : Chem.Mol
        Molecule fused onto `target`, e.g. a terminal modification or the next residue.
    fragment_anchor : int
        Atom-map number of the anchor in `fragment`.

    Returns
    -------
    Chem.Mol
        The sanitized, fused molecule.

    Raises
    ------
    ValueError
        If either anchor is missing or ambiguous, or if the fused atom exceeds its valence.
    """
    target_idx = _find_anchor(target, target_anchor)
    fragment_idx = _find_anchor(fragment, fragment_anchor) + target.GetNumAtoms()

    mol = Chem.RWMol(Chem.CombineMols(target, fragment))
    # An aromatic bond cannot be moved onto an atom that is not flagged aromatic yet;
    # sanitization re-perceives aromaticity of the fused molecule.
    Chem.Kekulize(mol, clearAromaticFlags=True)

    fragment_neighbors = [
        neighbor.GetIdx()
        for neighbor in mol.GetAtomWithIdx(fragment_idx).GetNeighbors()
    ]
    for neighbor_idx in fragment_neighbors:
        _move_bond(mol, neighbor_idx, fragment_idx, target_idx)

    fragment_atom = mol.GetAtomWithIdx(fragment_idx)
    fused_atom = mol.GetAtomWithIdx(target_idx)
    fused_atom.SetAtomicNum(fragment_atom.GetAtomicNum())
    fused_atom.SetFormalCharge(fragment_atom.GetFormalCharge())
    fused_atom.SetIsotope(fragment_atom.GetIsotope())
    # Hydrogens written on either anchor described its open valences before fusing.
    fused_atom.SetNumExplicitHs(0)
    fused_atom.SetNumRadicalElectrons(0)
    fused_atom.SetNoImplicit(False)
    fused_atom.SetAtomMapNum(target_anchor if target_anchor == fragment_anchor else 0)
    mol.RemoveAtom(fragment_idx)

    fused = mol.GetMol()
    try:
        Chem.SanitizeMol(fused)
    except Chem.rdchem.MolSanitizeException as e:
        raise ValueError(
            f"Cannot fuse anchor :{fragment_anchor} of {_to_smiles(fragment)} "
            f"into anchor :{target_anchor} of {_to_smiles(target)}: {e}"
        ) from e
    return fused


def _find_anchor(mol: Chem.Mol, anchor: int) -> int:
    """Return the index of the single atom carrying the given anchor label."""
    matches = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == anchor
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one atom with atom map :{anchor}, found {len(matches)}"
        )
    return matches[0]


def _move_bond(
    mol: Chem.RWMol, neighbor_idx: int, old_end_idx: int, new_end_idx: int
) -> None:
    """Reattach a bond from one anchor to the other, keeping the neighbour's stereo."""
    neighbor_bonds = [
        bond.GetOtherAtomIdx(neighbor_idx)
        for bond in mol.GetAtomWithIdx(neighbor_idx).GetBonds()
    ]
    bond_type = mol.GetBondBetweenAtoms(neighbor_idx, old_end_idx).GetBondType()
    mol.RemoveBond(neighbor_idx, old_end_idx)
    mol.AddBond(neighbor_idx, new_end_idx, bond_type)

    # A chiral tag is relative to the order of the atom's bonds. The re-added bond is
    # appended last, so skipping an odd number of bonds would invert the configuration.
    skipped_bonds = len(neighbor_bonds) - 1 - neighbor_bonds.index(old_end_idx)
    neighbor = mol.GetAtomWithIdx(neighbor_idx)
    if skipped_bonds % 2 and neighbor.GetChiralTag() in _TETRAHEDRAL_TAGS:
        neighbor.InvertChirality()


def _without_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy without anchor labels, as handed out to users."""
    unmapped = Chem.Mol(mol)
    for atom in unmapped.GetAtoms():
        atom.SetAtomMapNum(0)
    return unmapped


def _to_smiles(mol: Chem.Mol) -> str:
    """Return the SMILES without anchor labels, as handed out to users."""
    return Chem.MolToSmiles(_without_atom_maps(mol))


class AminoAcidModifier:
    """
    A class for modifying amino acids with N-terminal and C-terminal modifications.
    """

    def __init__(self):
        """Defer loading the SMILES tables until they are first accessed."""
        self._aa_smiles = None
        self._aa_mols = None
        self._n_term_modifications = None
        self._c_term_modifications = None
        self._ptm_dict = None

    @property
    def aa_smiles(self) -> dict[str, str]:
        """
        Dictionary of amino acid SMILES.

        Returns
        -------
        Dict[str, str]
            A dictionary with amino acid codes as keys and their SMILES as values.
        """
        if self._aa_smiles is None:
            self._aa_smiles = {
                aa: aa_formula.loc[aa]["smiles"]
                for aa in aa_formula.index
                if not pd.isna(aa_formula.loc[aa]["smiles"])
            }
        return self._aa_smiles

    @property
    def aa_mols(self) -> dict[str, Chem.Mol]:
        """
        Dictionary of amino acid RDKit molecules.

        Returns
        -------
        Dict[str, Chem.Mol]
            A dictionary with amino acid codes as keys and their RDKit molecules
            (anchors mapped) as values.
        """
        if self._aa_mols is None:
            self._aa_mols = {
                aa: parse_smiles(smiles, "residue")
                for aa, smiles in self.aa_smiles.items()
            }
        return self._aa_mols

    @property
    def n_term_modifications(self) -> dict[str, str]:
        """
        Dictionary of N-terminal modifications.

        Returns
        -------
        Dict[str, str]
            A dictionary with modification names as keys and their SMILES as values.
        """
        if self._n_term_modifications is None:
            self._n_term_modifications = self._get_modifications(
                "@Protein_N-term", "@Any_N-term"
            )
        return self._n_term_modifications

    @property
    def c_term_modifications(self) -> dict[str, str]:
        """
        Dictionary of C-terminal modifications.

        Returns
        -------
        Dict[str, str]
            A dictionary with modification names as keys and their SMILES as values.
        """
        if self._c_term_modifications is None:
            self._c_term_modifications = self._get_modifications(
                "@Protein_C-term", "@Any_C-term"
            )
        return self._c_term_modifications

    @property
    def ptm_dict(self) -> dict[str, str]:
        """
        Dictionary of post-translational modifications (PTMs).

        Returns
        -------
        Dict[str, str]
            A dictionary with modification names as keys and their SMILES as values.
        """
        if self._ptm_dict is None:
            self._ptm_dict = {
                name: smiles
                for name, smiles in self._get_modifications("@").items()
                if name not in self.n_term_modifications
                and name not in self.c_term_modifications
            }
        return self._ptm_dict

    def _get_modifications(self, *terms: str) -> dict[str, str]:
        """
        Helper method to get modifications based on specific terms.

        Parameters
        ----------
        *terms : str
            Terms to filter modifications by.

        Returns
        -------
        Dict[str, str]
            A dictionary of modifications with names as keys and SMILES as values.
        """
        mod_df_smiles = MOD_DF[MOD_DF["smiles"] != ""]
        return {
            name: smiles
            for name, smiles in zip(mod_df_smiles["mod_name"], mod_df_smiles["smiles"])
            if any(term in name for term in terms)
        }

    def validate_correct_args(
        self,
        aa_smiles: str,
        n_term_mod: Optional[str] = None,
        c_term_mod: Optional[str] = None,
    ) -> None:
        """
        Validate the amino acid, N-terminal modification, and C-terminal modification.

        Parameters
        ----------
        aa_smiles : str
            SMILES of an amino acid with or without PTMs, with its alpha-amine nitrogen
            mapped as :1 and its carboxylic hydroxyl oxygen mapped as :2.
        n_term_mod : Optional[str], optional
            N-terminal modification name. Options are the keys in the n_term_modifications dict.
        c_term_mod : Optional[str], optional
            C-terminal modification name. Options are the keys in the c_term_modifications dict.

        Raises
        ------
        ValueError
            If the amino acid is invalid, or if N-terminal / C-terminal modification is not recognized.
        """
        errors = []

        try:
            parse_smiles(aa_smiles, "residue")
        except ValueError as e:
            errors.append(str(e))

        if n_term_mod is not None and n_term_mod not in self.n_term_modifications:
            errors.append(f"Unrecognized N-terminal modification: {n_term_mod}")

        if c_term_mod is not None and c_term_mod not in self.c_term_modifications:
            errors.append(f"Unrecognized C-terminal modification: {c_term_mod}")

        if errors:
            raise ValueError("\n".join(errors))

    def modify_amino_acid(
        self,
        aa_smiles: str,
        n_term_mod: Optional[str] = None,
        c_term_mod: Optional[str] = None,
    ) -> str:
        """
        Modify an amino acid with N-terminal and C-terminal modifications.

        Parameters
        ----------
        aa_smiles : str
            SMILES of an amino acid with or without PTMs, with its alpha-amine nitrogen
            mapped as :1 and its carboxylic hydroxyl oxygen mapped as :2.
        n_term_mod : Optional[str], optional
            N-terminal modification name. Options are the keys in the n_term_modifications dict.
            If None, the amino acid keeps its free amine.
        c_term_mod : Optional[str], optional
            C-terminal modification name. Options are the keys in the c_term_modifications dict.
            If None, the amino acid keeps its free carboxylic acid.

        Returns
        -------
        str
            SMILES string of the modified amino acid, without atom maps.
        """
        self.validate_correct_args(aa_smiles, n_term_mod, c_term_mod)
        mol = self._apply_terminal_modifications(
            parse_smiles(aa_smiles, "residue"), n_term_mod, c_term_mod
        )
        return _to_smiles(mol)

    def _apply_terminal_modifications(
        self,
        mol: Chem.Mol,
        n_term_mod: Optional[str],
        c_term_mod: Optional[str],
    ) -> Chem.Mol:
        """
        Fuse terminal modifications onto the anchors of a residue or peptide.

        Parameters
        ----------
        mol : Chem.Mol
            Residue or peptide with mapped anchors.
        n_term_mod : Optional[str]
            N-terminal modification name, or None to keep the free amine.
        c_term_mod : Optional[str]
            C-terminal modification name, or None to keep the free carboxylic acid.

        Returns
        -------
        Chem.Mol
            The modified molecule, anchors still mapped.
        """
        if n_term_mod:
            n_mod_mol = parse_smiles(self.n_term_modifications[n_term_mod], "n_term")
            mol = merge_at_anchors(mol, N_TERM_ANCHOR, n_mod_mol, N_TERM_ANCHOR)
        if c_term_mod:
            c_mod_mol = parse_smiles(self.c_term_modifications[c_term_mod], "c_term")
            mol = merge_at_anchors(mol, C_TERM_ANCHOR, c_mod_mol, C_TERM_ANCHOR)
        return mol
