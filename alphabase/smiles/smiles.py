from typing import Dict, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdmolops import ReplaceSubstructs, SanitizeMol

from alphabase.constants.aa import aa_formula
from alphabase.constants.modification import MOD_DF

N_TERM_PLACEHOLDER_ATOM = "Fl"
C_TERM_PLACEHOLDER_ATOM = "Ts"
# Placeholder in N-terminal modification SMILES representing the N-terminal amine
MOD_N_TERM_PLACEHOLDER_ATOM = "Lv"


def _find_mod_placeholder(n_mod_mol: Chem.Mol) -> tuple[int, list[int]]:
    """
    Find the modification placeholder atom and its neighbor indices.

    Parameters
    ----------
    n_mod_mol : Chem.Mol
        The modification molecule with placeholder.

    Returns
    -------
    tuple[int, list[int]]
        Tuple of (placeholder atom index, list of neighbor atom indices).

    Raises
    ------
    ValueError
        If no placeholder is found or multiple placeholders exist.
    """
    mod_placeholder_idx = None
    mod_placeholder_neighbor_indices = []
    mod_placeholder_count = 0

    for atom in n_mod_mol.GetAtoms():
        if atom.GetSymbol() == MOD_N_TERM_PLACEHOLDER_ATOM:
            mod_placeholder_count += 1
            if mod_placeholder_idx is None:
                mod_placeholder_idx = atom.GetIdx()
                mod_placeholder_neighbor_indices = [
                    n.GetIdx() for n in atom.GetNeighbors()
                ]

    if mod_placeholder_idx is None:
        raise ValueError(
            f"Modification does not contain placeholder [{MOD_N_TERM_PLACEHOLDER_ATOM}]"
        )
    if mod_placeholder_count > 1:
        raise ValueError(
            f"Modification contains {mod_placeholder_count} placeholders "
            f"[{MOD_N_TERM_PLACEHOLDER_ATOM}], expected exactly 1"
        )

    return mod_placeholder_idx, mod_placeholder_neighbor_indices


def _find_n_term_site(aa_mol: Chem.Mol) -> tuple[int, list[int]]:
    """
    Find the N-terminal nitrogen and placeholder indices in the amino acid.

    Parameters
    ----------
    aa_mol : Chem.Mol
        The amino acid molecule with N-terminal placeholders.

    Returns
    -------
    tuple[int, list[int]]
        Tuple of (nitrogen atom index, list of placeholder indices).

    Raises
    ------
    ValueError
        If no N-terminal placeholder is found or placeholder has invalid neighbors.
    """
    n_idx = None
    fl_indices = []

    for atom in aa_mol.GetAtoms():
        if atom.GetSymbol() == N_TERM_PLACEHOLDER_ATOM:
            neighbors = atom.GetNeighbors()
            if len(neighbors) != 1:
                raise ValueError(
                    f"N-terminal placeholder [{N_TERM_PLACEHOLDER_ATOM}] should have "
                    f"exactly one neighbor, found {len(neighbors)}"
                )
            n_idx = neighbors[0].GetIdx()
            fl_indices.append(atom.GetIdx())

    if n_idx is None:
        raise ValueError(
            f"Amino acid does not contain N-terminal placeholder [{N_TERM_PLACEHOLDER_ATOM}]"
        )

    return n_idx, fl_indices


def _replace_remaining_n_term_placeholders(mol: Chem.Mol) -> Chem.Mol:
    """
    Replace any remaining N-terminal placeholders with hydrogen atoms.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule potentially containing placeholders.

    Returns
    -------
    Chem.Mol
        The molecule with placeholders replaced by hydrogen.
    """
    n_term_placeholder_mol = Chem.MolFromSmiles(
        f"[{N_TERM_PLACEHOLDER_ATOM}]", sanitize=False
    )
    return ReplaceSubstructs(
        mol,
        n_term_placeholder_mol,
        Chem.MolFromSmiles("[H]", sanitize=False),
        replaceAll=True,
    )[0]


class AminoAcidModifier:
    """
    A class for modifying amino acids with N-terminal and C-terminal modifications.
    """

    def __init__(self):
        self._aa_smiles = None
        self._aa_mols = None
        self._n_term_modifications = None
        self._c_term_modifications = None
        self._ptm_dict = None

    @property
    def aa_smiles(self) -> Dict[str, str]:
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
    def aa_mols(self) -> Dict[str, Chem.Mol]:
        """
        Dictionary of amino acid RDKit molecules.

        Returns
        -------
        Dict[str, Chem.Mol]
            A dictionary with amino acid codes as keys and their RDKit molecules as values.
        """
        if self._aa_mols is None:
            self._aa_mols = {
                aa: Chem.MolFromSmiles(smiles) for aa, smiles in self.aa_smiles.items()
            }
        return self._aa_mols

    @property
    def n_term_modifications(self) -> Dict[str, str]:
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
    def c_term_modifications(self) -> Dict[str, str]:
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
    def ptm_dict(self) -> Dict[str, str]:
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

    def _get_modifications(self, *terms: str) -> Dict[str, str]:
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
            SMILES of an amino acid with or without PTMs, must have the placeholder atoms [Fl] for N-term and [Ts] for C-term.
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

        n_term_placeholder = f"[{N_TERM_PLACEHOLDER_ATOM}]"
        c_term_placeholder = f"[{C_TERM_PLACEHOLDER_ATOM}]"

        if n_term_placeholder not in aa_smiles:
            errors.append(
                f"The amino acid {aa_smiles} must contain the N-terminal placeholder {n_term_placeholder}."
            )
        if c_term_placeholder not in aa_smiles:
            errors.append(
                f"The amino acid {aa_smiles} must contain the C-terminal placeholder {c_term_placeholder}."
            )

        if n_term_mod is not None and n_term_mod not in self.n_term_modifications:
            errors.append(f"Unrecognized N-terminal modification: {n_term_mod}")

        if c_term_mod is not None and c_term_mod not in self.c_term_modifications:
            errors.append(f"Unrecognized C-terminal modification: {c_term_mod}")

        mol = Chem.MolFromSmiles(aa_smiles)
        if mol is None:
            errors.append(f"Invalid amino acid SMILES: {aa_smiles}")

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
            SMILES of an amino acid with or without PTMs, must have the placeholder atoms [Fl] for N-term and [Ts] for C-term.
        n_term_mod : Optional[str], optional
            N-terminal modification name. Options are the keys in the n_term_modifications dict.
            If None, the amino acid is not modified, so using hydrogen atoms.
        c_term_mod : Optional[str], optional
            C-terminal modification name. Options are the keys in the c_term_modifications dict.
            If None, the amino acid is not modified, so using the -OH group.

        Returns
        -------
        str
            SMILES string of the modified amino acid.
        """
        self.validate_correct_args(aa_smiles, n_term_mod, c_term_mod)
        mol = Chem.MolFromSmiles(aa_smiles)
        n_term_placeholder_mol = Chem.MolFromSmiles(
            f"[{N_TERM_PLACEHOLDER_ATOM}]", sanitize=False
        )
        c_term_placeholder_mol = Chem.MolFromSmiles(
            f"[{C_TERM_PLACEHOLDER_ATOM}]", sanitize=False
        )

        mol = self._apply_n_terminal_modification(
            mol, n_term_placeholder_mol, n_term_mod
        )
        mol = self._apply_c_terminal_modification(
            mol, c_term_placeholder_mol, c_term_mod
        )

        SanitizeMol(mol)
        return Chem.MolToSmiles(mol)

    def _has_n_term_mod_placeholder(self, mod_smiles: str) -> bool:
        """Check if the modification SMILES contains the N-term mod placeholder."""
        return f"[{MOD_N_TERM_PLACEHOLDER_ATOM}]" in mod_smiles

    def _apply_n_term_mod_with_placeholder(
        self, aa_mol: Chem.Mol, n_mod_mol: Chem.Mol
    ) -> Chem.Mol:
        """
        Apply N-terminal modification where [Lv] represents the N-terminal amine.

        The modification SMILES contains a [Lv] placeholder that represents the
        N-terminal nitrogen itself. All neighbors of [Lv] in the modification
        become bonded to the actual nitrogen, replacing the [Fl] placeholders.

        This allows modifications like Dimethyl where [Lv] has multiple neighbors:
        - Acetyl: CC(=O)[Lv] - one neighbor (carbonyl C) bonds to N
        - Dimethyl: C[Lv]C - two neighbors (both methyl C) bond to N

        Parameters
        ----------
        aa_mol : Chem.Mol
            The amino acid molecule with [Fl] N-terminal placeholders.
        n_mod_mol : Chem.Mol
            The modification molecule with [Lv] representing the N-terminal amine.

        Returns
        -------
        Chem.Mol
            The modified molecule with [Fl] placeholders replaced.
        """
        mod_placeholder_idx, mod_neighbor_indices = _find_mod_placeholder(n_mod_mol)
        n_idx, fl_indices = _find_n_term_site(aa_mol)

        # Combine molecules and adjust indices
        combined = Chem.CombineMols(aa_mol, n_mod_mol)
        rw_mol = Chem.RWMol(combined)
        aa_mol_num_atoms = aa_mol.GetNumAtoms()
        mod_placeholder_idx_combined = mod_placeholder_idx + aa_mol_num_atoms
        mod_neighbor_indices_combined = [
            i + aa_mol_num_atoms for i in mod_neighbor_indices
        ]

        # Create bonds from each modification neighbor to the nitrogen
        for neighbor_idx in mod_neighbor_indices_combined:
            rw_mol.AddBond(n_idx, neighbor_idx, Chem.rdchem.BondType.SINGLE)

        # Remove [Lv] and corresponding [Fl] atoms (one [Fl] per new bond)
        fl_to_remove = fl_indices[: len(mod_neighbor_indices)]
        to_remove = sorted([mod_placeholder_idx_combined] + fl_to_remove, reverse=True)
        for idx in to_remove:
            rw_mol.RemoveAtom(idx)

        return _replace_remaining_n_term_placeholders(rw_mol.GetMol())

    def _apply_n_terminal_modification(
        self,
        aa_mol: Chem.Mol,
        n_term_placeholder_mol: Chem.Mol,
        n_term_mod: Optional[str],
    ) -> Chem.Mol:
        """
        Apply N-terminal modification to the molecule.

        Parameters
        ----------
        aa_mol : Chem.Mol
            The amino acid molecule to modify.
        n_term_placeholder_mol : Chem.Mol
            The N-terminal placeholder molecule.
        n_term_mod : Optional[str]
            The N-terminal modification to apply.

        Returns
        -------
        Chem.Mol
            The modified molecule.
        """
        if n_term_mod:
            n_mod_smiles = self.n_term_modifications[n_term_mod]
            n_mod_mol = Chem.MolFromSmiles(n_mod_smiles)

            # Branch: use placeholder-based or index-based modification
            if self._has_n_term_mod_placeholder(n_mod_smiles):
                return self._apply_n_term_mod_with_placeholder(aa_mol, n_mod_mol)
            else:
                # Legacy: index-based connection (first atom connects to N)
                aa_mol = ReplaceSubstructs(aa_mol, n_term_placeholder_mol, n_mod_mol)[0]

        # replacing all leftover N-terminal placeholders with hydrogen atoms
        return ReplaceSubstructs(
            aa_mol,
            n_term_placeholder_mol,
            Chem.MolFromSmiles("[H]", sanitize=False),
            replaceAll=True,
        )[0]

    def _apply_c_terminal_modification(
        self,
        aa_mol: Chem.Mol,
        c_term_placeholder_mol: Chem.Mol,
        c_term_mod: Optional[str],
    ) -> Chem.Mol:
        """
        Apply C-terminal modification to the molecule.

        Parameters
        ----------
        aa_mol : Chem.Mol
            The amino acid molecule to modify.
        c_term_placeholder_mol : Chem.Mol
            The C-terminal placeholder molecule.
        c_term_mod : Optional[str]
            The C-terminal modification to apply.

        Returns
        -------
        Chem.Mol
            The modified molecule.
        """
        if c_term_mod:
            c_mod_smiles = self.c_term_modifications[c_term_mod]
            c_mod_mol = Chem.MolFromSmiles(c_mod_smiles)
            return ReplaceSubstructs(aa_mol, c_term_placeholder_mol, c_mod_mol)[0]
        return ReplaceSubstructs(
            aa_mol, c_term_placeholder_mol, Chem.MolFromSmiles("O", sanitize=False)
        )[0]
