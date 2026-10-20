from typing import Optional

try:
    from rdkit import Chem
except ImportError as e:
    raise ImportError(
        "rdkit is required for the smiles module. "
        "Install it through pip or install alphabase with the 'rdkit' or 'full' extra (or their '-stable' variants)."
    ) from e

from alphabase.constants.modification import ModificationKeys
from alphabase.smiles.smiles import (
    C_TERM_ANCHOR,
    N_TERM_ANCHOR,
    AminoAcidModifier,
    _to_smiles,
    _without_atom_maps,
    merge_at_anchors,
    parse_smiles,
)


class PeptideSmilesEncoder:
    """
    A class to encode peptide sequences into SMILES strings or RDKit molecule objects.
    """

    def __init__(self):
        """Use an AminoAcidModifier for residue SMILES and terminal modifications."""
        self.amino_acid_modifier = AminoAcidModifier()

    def _parse_modifications(
        self,
        mods: Optional[str] = "",
        mod_sites: Optional[str] = "",
    ) -> dict:
        """
        Parse the modifications and sites into a dictionary.

        Parameters
        ----------
        mods : Optional[str]
            Modifications in the format "GG@K;Oxidation@M;Carbamidomethyl@C".
        mod_sites : Optional[str]
            Corresponding modification sites in the format "4;5;6".

        Returns
        -------
        dict
            Dictionary mapping position indices to modification strings, empty dict if no mods and mod_sites provided.
        """
        mod_dict = {}
        if mods and mod_sites:
            for site, mod in zip(
                mod_sites.split(ModificationKeys.SEPARATOR),
                mods.split(ModificationKeys.SEPARATOR),
            ):
                # if the modification is technically at the N-terminal, but changes the first amino acid
                if ModificationKeys.TERM_SEPARATOR in mod and site == "0":
                    site = "1"
                mod_dict[int(site)] = mod
        return mod_dict

    def _get_terminal_mods(self, mods: dict) -> tuple:
        """
        Get the N-terminal and C-terminal modifications.

        Parameters
        ----------
        mods : dict
            Dictionary of modifications.

        Returns
        -------
        tuple
            (n_term_mod, c_term_mod)
        """
        n_term_mod = mods.get(0)
        c_term_mod = mods.get(-1, None)
        return n_term_mod, c_term_mod

    def _get_amino_acid_smiles(self, aa: str, mod_idx: int, mods: dict) -> str:
        """
        Get the SMILES string for an amino acid, with or without modification.

        Parameters
        ----------
        aa : str
            The amino acid code.
        mod_idx : int
            The modification index.
        mods : dict
            Dictionary of modifications.

        Returns
        -------
        str
            The SMILES string for the amino acid.

        Raises
        ------
        ValueError
            If the amino acid or modification is unknown.
        """
        if mod_idx in mods:
            aa_smiles = self.amino_acid_modifier.ptm_dict.get(mods[mod_idx], None)
        else:
            aa_smiles = self.amino_acid_modifier.aa_smiles.get(aa, None)

        if not aa_smiles:
            raise ValueError(
                f"Unknown amino acid code: {aa} or modification: {mods.get(mod_idx, 'no mod')} (SMILES for it might be missing)"
            )
        return aa_smiles

    def _get_amino_acid_mol(self, aa: str, mod_idx: int, mods: dict) -> Chem.Mol:
        """
        Get the RDKit molecule for an amino acid, with or without modification.

        Parameters
        ----------
        aa : str
            The amino acid code.
        mod_idx : int
            The modification index.
        mods : dict
            Dictionary of modifications.

        Returns
        -------
        Chem.Mol
            The RDKit molecule for the amino acid.

        Raises
        ------
        ValueError
            If the SMILES string is invalid.
        """
        aa_smiles = self._get_amino_acid_smiles(aa, mod_idx, mods)
        try:
            return parse_smiles(aa_smiles, "residue")
        except ValueError as e:
            raise ValueError(
                f"Invalid SMILES for amino acid {aa}, mod {mods.get(mod_idx, 'no mod')}: {e}"
            ) from e

    def peptide_to_smiles_per_amino_acid(
        self,
        sequence: str,
        mods: Optional[str] = "",
        mod_site: Optional[str] = "",
    ) -> list:
        """
        Encode a peptide sequence into a list of SMILES strings for each amino acid.

        Parameters
        ----------
        sequence : str
            Peptide sequence, e.g., "AFVKMCK".
        mods : Optional[str]
            Modifications in the format "GG@K;Oxidation@M;Carbamidomethyl@C".
        mod_site : Optional[str]
            Corresponding modification sites in the format "4;5;6".

        Returns
        -------
        list
            The list of SMILES strings for each amino acid in the peptide.
        """
        mods = self._parse_modifications(mods, mod_site)
        n_term_mod, c_term_mod = self._get_terminal_mods(mods)
        last_idx = len(sequence) - 1

        smiles_list = []
        for aa_idx, aa in enumerate(sequence):
            aa_mol = self.amino_acid_modifier._apply_terminal_modifications(
                self._get_amino_acid_mol(aa, aa_idx + 1, mods),
                n_term_mod=n_term_mod if aa_idx == 0 else None,
                c_term_mod=c_term_mod if aa_idx == last_idx else None,
            )
            smiles_list.append(_to_smiles(aa_mol))

        return smiles_list

    def peptide_to_smiles(
        self,
        sequence: str,
        mods: Optional[str] = "",
        mod_site: Optional[str] = "",
    ) -> str:
        """
        Encode a peptide sequence into a SMILES string.

        Parameters
        ----------
        sequence : str
            Peptide sequence, e.g., "AFVKMCK".
        mods : Optional[str]
            Modifications in the format "GG@K;Oxidation@M;Carbamidomethyl@C".
        mod_site : Optional[str]
            Corresponding modification sites in the format "4;5;6".

        Returns
        -------
        str
            The SMILES string of the peptide molecule.
        """
        peptide_mol = self._build_peptide(sequence, mods, mod_site)
        return Chem.MolToSmiles(peptide_mol)

    def peptide_to_mol(
        self,
        sequence: str,
        mods: Optional[str] = "",
        mod_site: Optional[str] = "",
    ) -> Chem.Mol:
        """
        Encode a peptide sequence into an RDKit molecule object.

        Parameters
        ----------
        sequence : str
            Peptide sequence, e.g., "AFVKMCK".
        mods : Optional[str]
            Modifications in the format "GG@K;Oxidation@M;Carbamidomethyl@C".
        mod_site : Optional[str]
            Corresponding modification sites in the format "4;5;6".

        Returns
        -------
        Chem.Mol
            The peptide molecule.
        """
        return self._build_peptide(sequence, mods, mod_site)

    def _build_peptide(
        self,
        sequence: str,
        mods: Optional[str] = "",
        mod_site: Optional[str] = "",
    ) -> Chem.Mol:
        """
        Build the peptide molecule from the sequence and modifications.

        Parameters
        ----------
        sequence : str
            Peptide sequence.
        mods : Optional[str]
            Modifications in the format "GG@K;Oxidation@M;Carbamidomethyl@C".
        mod_site : Optional[str]
            Corresponding modification sites in the format "4;5;6".

        Returns
        -------
        Chem.Mol
            The peptide molecule.
        """
        mods = self._parse_modifications(mods, mod_site)

        peptide_mol = self._get_amino_acid_mol(sequence[0], 1, mods)
        for idx, aa in enumerate(sequence[1:], start=2):
            # The chain's carboxylic hydroxyl becomes the amine nitrogen of the next residue.
            peptide_mol = merge_at_anchors(
                peptide_mol,
                C_TERM_ANCHOR,
                self._get_amino_acid_mol(aa, idx, mods),
                N_TERM_ANCHOR,
            )

        n_term_mod, c_term_mod = self._get_terminal_mods(mods)
        peptide_mol = self.amino_acid_modifier._apply_terminal_modifications(
            peptide_mol, n_term_mod, c_term_mod
        )
        return _without_atom_maps(peptide_mol)
