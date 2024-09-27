from typing import Optional

from rdkit import Chem

from alphabase.smiles.smiles import AminoAcidModifier


class PeptideSmilesEncoder:
    """
    A class to encode peptide sequences into SMILES strings or RDKit molecule objects.
    """

    def __init__(self):
        self.amino_acid_modifier = AminoAcidModifier()

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
        if mods and mod_site:
            mods = {int(m): mod for m, mod in zip(mod_site.split(";"), mods.split(";"))}
        else:
            mods = {}

        # List to hold the amino acid molecules
        amino_acid_mols = []

        n_term_placeholder_mol = Chem.MolFromSmiles(
            self.amino_acid_modifier.N_TERM_PLACEHOLDER, sanitize=False
        )
        c_term_placeholder_mol = Chem.MolFromSmiles(
            self.amino_acid_modifier.C_TERM_PLACEHOLDER, sanitize=False
        )
        n_term_mod = None
        c_term_mod = None

        if 0 in mods:
            n_term_mod = mods[0]
        if -1 in mods:
            c_term_mod = mods[-1]
        # Process each amino acid in the sequence
        for idx, aa in enumerate(sequence):
            if idx + 1 in mods:
                aa_smiles = self.amino_acid_modifier.ptm_dict.get(mods[idx + 1], None)
            else:
                aa_smiles = self.amino_acid_modifier.aa_smiles.get(aa)
            if not aa_smiles:
                raise ValueError(
                    f"Unknown amino acid code: {aa} or modification: {mods.get(idx + 1, 'no mod')} (SMILES for it might be missing)"
                )
            # Convert the amino acid SMILES to a molecule
            aa_mol = Chem.MolFromSmiles(aa_smiles)
            if aa_mol is None:
                raise ValueError(
                    f"Invalid SMILES for amino acid {aa}, mod {mods.get(idx + 1, 'no mod')}: {aa_smiles}"
                )
            amino_acid_mols.append(aa_mol)

        # Now, assemble the peptide
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
