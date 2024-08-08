from typing import Dict, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdmolops import ReplaceSubstructs, SanitizeMol

from alphabase.constants.aa import aa_formula
from alphabase.constants.modification import MOD_DF


class AminoAcidModifier:
    """
    A class for modifying amino acids with N-terminal and C-terminal modifications.
    """

    N_TERM_PLACEHOLDER = "[Xe]"
    C_TERM_PLACEHOLDER = "[Rn]"

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
            SMILES of an amino acid with or without PTMs, must have the placeholder atoms [Xe] for N-term and [Rn] for C-term.
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

        if self.N_TERM_PLACEHOLDER not in aa_smiles:
            errors.append(
                f"The amino acid {aa_smiles} must contain the N-terminal placeholder [{self.N_TERM_PLACEHOLDER}]."
            )
        if self.C_TERM_PLACEHOLDER not in aa_smiles:
            errors.append(
                f"The amino acid {aa_smiles} must contain the C-terminal placeholder [{self.C_TERM_PLACEHOLDER}]."
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
            SMILES of an amino acid with or without PTMs, must have the placeholder atoms [Xe] for N-term and [Rn] for C-term.
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
            self.N_TERM_PLACEHOLDER, sanitize=False
        )
        c_term_placeholder_mol = Chem.MolFromSmiles(
            self.C_TERM_PLACEHOLDER, sanitize=False
        )

        mol = self._apply_n_terminal_modification(
            mol, n_term_placeholder_mol, n_term_mod
        )
        mol = self._apply_c_terminal_modification(
            mol, c_term_placeholder_mol, c_term_mod
        )

        SanitizeMol(mol)
        return Chem.MolToSmiles(mol)

    def _apply_n_terminal_modification(
        self, mol: Chem.Mol, n_term_placeholder_mol: Chem.Mol, n_term_mod: Optional[str]
    ) -> Chem.Mol:
        """
        Apply N-terminal modification to the molecule.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule to modify.
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
            mol = ReplaceSubstructs(mol, n_term_placeholder_mol, n_mod_mol)[0]

            if "Dimethyl" in n_term_mod:
                return ReplaceSubstructs(mol, n_term_placeholder_mol, n_mod_mol)[0]

        # replacing all leftover N-terminal placeholders with hydrogen atoms
        return ReplaceSubstructs(
            mol,
            n_term_placeholder_mol,
            Chem.MolFromSmiles("[H]", sanitize=False),
            replaceAll=True,
        )[0]

    def _apply_c_terminal_modification(
        self, mol: Chem.Mol, c_term_placeholder_mol: Chem.Mol, c_term_mod: Optional[str]
    ) -> Chem.Mol:
        """
        Apply C-terminal modification to the molecule.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule to modify.
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
            return ReplaceSubstructs(mol, c_term_placeholder_mol, c_mod_mol)[0]
        return ReplaceSubstructs(
            mol, c_term_placeholder_mol, Chem.MolFromSmiles("O", sanitize=False)
        )[0]
