from typing import Optional

from rdkit import Chem

from alphabase.smiles.smiles import AminoAcidModifier


class PeptideSmilesEncoder:
    """
    A class to encode peptide sequences into SMILES strings or RDKit molecule objects.
    """

    def __init__(self):
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
            for site, mod in zip(mod_sites.split(";"), mods.split(";")):
                # if the modification is techinically at the N-terminal, but changes the first amino acid
                if "^" in mod and site == "0":
                    site = "1"
                mod_dict[int(site)] = mod
        return mod_dict

    def _get_terminal_placeholders(self):
        """
        Create the N-terminal and C-terminal placeholder molecules.

        Returns
        -------
        tuple
            (n_term_placeholder_mol, c_term_placeholder_mol)
        """
        n_term_placeholder_mol = Chem.MolFromSmiles(
            self.amino_acid_modifier.N_TERM_PLACEHOLDER, sanitize=False
        )
        c_term_placeholder_mol = Chem.MolFromSmiles(
            self.amino_acid_modifier.C_TERM_PLACEHOLDER, sanitize=False
        )
        return n_term_placeholder_mol, c_term_placeholder_mol

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
        aa_mol = Chem.MolFromSmiles(aa_smiles)
        if aa_mol is None:
            raise ValueError(
                f"Invalid SMILES for amino acid {aa}, mod {mods.get(mod_idx, 'no mod')}: {aa_smiles}"
            )
        return aa_mol

    def _process_amino_acid(
        self,
        aa: str,
        aa_idx: int,
        sequence_length: int,
        mods: dict,
        n_term_placeholder_mol: Chem.Mol,
        c_term_placeholder_mol: Chem.Mol,
        n_term_mod: Optional[str] = None,
        c_term_mod: Optional[str] = None,
    ) -> Chem.Mol:
        """
        Process a single amino acid, applying appropriate modifications.

        Parameters
        ----------
        aa : str
            The amino acid code.
        aa_idx : int
            The index of the amino acid in the sequence.
        sequence_length : int
            The total length of the peptide sequence.
        mods : dict
            Dictionary of modifications.
        n_term_placeholder_mol : Chem.Mol
            N-terminal placeholder molecule.
        c_term_placeholder_mol : Chem.Mol
            C-terminal placeholder molecule.
        n_term_mod : Optional[str]
            N-terminal modification.
        c_term_mod : Optional[str]
            C-terminal modification.

        Returns
        -------
        Chem.Mol
            The processed amino acid molecule.
        """
        mod_idx = aa_idx + 1
        aa_mol = self._get_amino_acid_mol(aa, mod_idx, mods)

        aa_mol = self.amino_acid_modifier._apply_n_terminal_modification(
            aa_mol,
            n_term_placeholder_mol=n_term_placeholder_mol,
            n_term_mod=n_term_mod if aa_idx == 0 else None,
        )

        aa_mol = self.amino_acid_modifier._apply_c_terminal_modification(
            aa_mol,
            c_term_placeholder_mol=c_term_placeholder_mol,
            c_term_mod=c_term_mod if aa_idx == sequence_length - 1 else None,
        )

        return aa_mol

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
        n_term_placeholder_mol, c_term_placeholder_mol = (
            self._get_terminal_placeholders()
        )
        n_term_mod, c_term_mod = self._get_terminal_mods(mods)

        # Process each amino acid in the sequence
        smiles_list = []
        sequence_length = len(sequence)

        for aa_idx, aa in enumerate(sequence):
            aa_mol = self._process_amino_acid(
                aa,
                aa_idx,
                sequence_length,
                mods,
                n_term_placeholder_mol,
                c_term_placeholder_mol,
                n_term_mod,
                c_term_mod,
            )

            smiles_list.append(Chem.MolToSmiles(aa_mol))

        return smiles_list

    def peptide_to_smiles(
        self,
        sequence: str,
        mods: Optional[str] = "",
        mod_site: Optional[str] = "",
    ) -> str:
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
        amino_acid_mols = []

        n_term_placeholder_mol, c_term_placeholder_mol = (
            self._get_terminal_placeholders()
        )

        # Get molecules for each amino acid in the sequence
        for idx, aa in enumerate(sequence):
            mod_idx = idx + 1
            aa_mol = self._get_amino_acid_mol(aa, mod_idx, mods)
            amino_acid_mols.append(aa_mol)

        # Now, assemble the peptide
        n_term_mod, c_term_mod = self._get_terminal_mods(mods)
        peptide_mol = amino_acid_mols[0]
        peptide_mol = self.amino_acid_modifier._apply_n_terminal_modification(
            peptide_mol,
            n_term_placeholder_mol=n_term_placeholder_mol,
            n_term_mod=n_term_mod,
        )
        for idx in range(1, len(amino_acid_mols)):
            peptide_mol = self._connect_amino_acids(peptide_mol, amino_acid_mols[idx])
            peptide_mol = self.amino_acid_modifier._apply_n_terminal_modification(
                peptide_mol,
                n_term_placeholder_mol=n_term_placeholder_mol,
                n_term_mod=None,
            )
        peptide_mol = self.amino_acid_modifier._apply_c_terminal_modification(
            peptide_mol,
            c_term_placeholder_mol=c_term_placeholder_mol,
            c_term_mod=c_term_mod,
        )
        Chem.SanitizeMol(peptide_mol)
        return peptide_mol

    def _connect_amino_acids(self, mol1: Chem.Mol, mol2: Chem.Mol) -> Chem.Mol:
        """
        Connect two amino acids via a peptide bond.

        Parameters
        ----------
        mol1 : Chem.Mol
            The first amino acid molecule (C-terminal).
        mol2 : Chem.Mol
            The second amino acid molecule (N-terminal).

        Returns
        -------
        Chem.Mol
            The combined molecule.
        """
        # Combine the two molecules
        combined = Chem.CombineMols(mol1, mol2)
        rw_mol = Chem.RWMol(combined)
        mol1_num_atoms = mol1.GetNumAtoms()

        # Find the placeholder hydrogens and their neighboring atoms
        c_term_placeholder = self.amino_acid_modifier.C_TERM_PLACEHOLDER.strip("[]")
        n_term_placeholder = self.amino_acid_modifier.N_TERM_PLACEHOLDER.strip("[]")

        c_term_h_idx = None
        c_atom_idx = None
        n_term_h_idx = None
        n_atom_idx = None

        # Find C-terminal placeholder hydrogen in mol1
        for atom in mol1.GetAtoms():
            if atom.GetSymbol() == c_term_placeholder:
                c_term_h_idx = atom.GetIdx()
                neighbors = atom.GetNeighbors()
                if len(neighbors) != 1:
                    raise ValueError(
                        "C-terminal placeholder hydrogen should have exactly one neighbor."
                    )
                c_atom_idx = neighbors[0].GetIdx()
                break

        # Find N-terminal placeholder hydrogen in mol2
        for atom in mol2.GetAtoms():
            if atom.GetSymbol() == n_term_placeholder:
                n_term_h_idx = atom.GetIdx() + mol1_num_atoms
                neighbors = atom.GetNeighbors()
                if len(neighbors) != 1:
                    raise ValueError(
                        "N-terminal placeholder hydrogen should have exactly one neighbor."
                    )
                n_atom_idx = neighbors[0].GetIdx() + mol1_num_atoms
                break

        if c_term_h_idx is None or n_term_h_idx is None:
            raise ValueError(
                "Failed to find terminal placeholders for peptide bond formation."
            )

        # Remove placeholder hydrogens, higher index first to avoid index shifting
        indices_to_remove = sorted([c_term_h_idx, n_term_h_idx], reverse=True)
        for idx in indices_to_remove:
            rw_mol.RemoveAtom(idx)

        # Adjust atom indices after removal
        if n_term_h_idx > c_term_h_idx:
            n_atom_idx -= 1

        # Add peptide bond between the carbon and nitrogen atoms
        rw_mol.AddBond(c_atom_idx, n_atom_idx, Chem.rdchem.BondType.SINGLE)

        # Fix valencies if necessary
        Chem.SanitizeMol(rw_mol)

        return rw_mol
