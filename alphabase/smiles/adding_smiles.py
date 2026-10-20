"""
This script defines and processes peptide modifications using SMILES representations.

It initializes a set of common post-translational modifications (PTMs),
N-terminal modifications, and C-terminal modifications with their corresponding SMILES strings.
The script then validates the chemical composition of these modifications by comparing
the SMILES representation of the modified amino acid to the unmodified one.

The main functionalities include:
- Defining modification dictionaries for PTMs, N-terminal, and C-terminal mods.
- A generic `process_modifications` function to calculate chemical compositions from SMILES.
- A `main` function that orchestrates the processing of all modification types and
  updates the global modification DataFrame (`MOD_DF`).
- Saving the updated modifications list to `modification.tsv`, effectively registering
  these modifications for use in the AlphaBase library.

This script is intended to be run to update the modification definitions when new
modifications are added or existing ones are changed.
"""

import os
from typing import Literal

import pandas as pd

from alphabase.constants._const import CONST_FILE_FOLDER
from alphabase.constants.aa import aa_formula
from alphabase.constants.atom import ChemicalCompositonFormula
from alphabase.constants.modification import (
    MOD_DF,
    ModificationKeys,
    add_new_modifications,
)
from alphabase.smiles.smiles import AminoAcidModifier

aa_modifier = AminoAcidModifier()


def process_modifications(
    mods_dict: dict[str, str], mod_type: Literal["ptm", "n_term", "c_term"] = "ptm"
) -> dict[str, dict[str, str]]:
    """
    Process various types of modifications (PTM, N-term, C-term) with a unified approach.

    This function handles the common workflow for processing modifications:
    calculating SMILES, determining formulas, and computing composition differences.

    Parameters
    ----------
    mods_dict : dict
        Dictionary mapping modification names to SMILES strings
    mod_type : str, default="ptm"
        Type of modification: "ptm", "n_term", or "c_term"

    Returns
    -------
    dict
        Dictionary of dictionaries with the following structure:
        {
            "mod_name": {
                "composition": str,
                "smiles": str
            }
        }

    Examples
    --------
    >>> ptms = process_modifications(ptm_dict, mod_type="ptm")
    >>> nterms = process_modifications(n_term_modifications, mod_type="n_term")
    """
    to_add = {}

    for ptm, smiles in mods_dict.items():
        if mod_type == "ptm":
            original_aa = ptm.split(ModificationKeys.SITE_SEPARATOR)[1].split(
                ModificationKeys.TERM_SEPARATOR
            )[0]
            if original_aa.startswith("Any"):
                original_aa = "A"
        else:
            original_aa = "A"

        if mod_type == "ptm":
            smi = aa_modifier.modify_amino_acid(smiles)
        elif mod_type == "n_term":
            smi = aa_modifier.modify_amino_acid(
                aa_formula.loc["A", "smiles"], n_term_mod=ptm
            )
        elif mod_type == "c_term":
            smi = aa_modifier.modify_amino_acid(
                aa_formula.loc["A", "smiles"], c_term_mod=ptm
            )
        else:
            raise ValueError(f"Invalid modification type: {mod_type}")

        ptm_formula = ChemicalCompositonFormula.from_smiles(smi)
        original_aa_brutto_formula = aa_formula.loc[original_aa, "formula"]
        composition = str(
            ptm_formula
            - ChemicalCompositonFormula(original_aa_brutto_formula)
            - ChemicalCompositonFormula("H(2)O(1)")
        )

        to_add[ptm] = {
            "composition": composition,
            "smiles": smiles,
        }

    return to_add


n_term_modifications = {
    "mTRAQ@Any_N-term": "[NH2:1]C(=O)CN1CCN(C)CC1",
    "mTRAQ:13C(3)15N(1)@Any_N-term": "[NH2:1]C(=O)[13CH2][15N]1CCN(C)[13CH2][13CH2]1",
    "mTRAQ:13C(6)15N(2)@Any_N-term": "[NH2:1]C(=O)[13CH2][15N]1[13CH2][13CH2][15N]([13CH3])[13CH2][13CH2]1",
    "Acetyl@Any_N-term": "[NH2:1]C(C)=O",
    "Acetyl@Protein_N-term": "[NH2:1]C(C)=O",
    "Propionyl@Any_N-term": "[NH2:1]C(=O)CC",
    "Biotin@Any_N-term": "[NH2:1]C(=O)CCCCC1SCC2NC(=O)NC21",
    "Carbamidomethyl@Any_N-term": "[NH2:1]C(=O)NC",
    "Carbamyl@Any_N-term": "[NH2:1]C(N)=O",
    "Propionamide@Any_N-term": "[NH2:1]CCC(N)=O",
    "Pyridylacetyl@Any_N-term": "[NH2:1]C(=O)Cc1ccccn1",
    "Methyl@Any_N-term": "[NH2:1]C",
    "Dimethyl@Any_N-term": "[NH:1](C)C",
    "Dimethyl:2H(6)13C(2)@Any_N-term": "[NH:1]([13C]([2H])([2H])[2H])[13C]([2H])([2H])[2H]",
    "Dimethyl:2H(4)@Any_N-term": "[NH:1](C([2H])[2H])C([2H])[2H]",
    "Dimethyl:2H(4)13C(2)@Any_N-term": "[NH:1]([13C]([1H])([2H])[2H])[13C]([1H])([2H])[2H]",
    "GG@Protein_N-term": "[NH2:1]C(=O)CNC(=O)CN",
    "SATA@Any_N-term": "[NH2:1]C(=O)CSC(C)=O",
    "SATP@Any_N-term": "[NH2:1]C(=O)CCSC(C)=O",
    "Succinamide@Any_N-term": "[NH2:1]C(=O)CCC(=O)NO",
    "SATA-Glc@Any_N-term": "[NH2:1]C(=O)CS[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O",
}


c_term_modifications = {
    "Methyl@Any_C-term": "[OH:2]C",
    "Ethyl@Any_C-term": "[OH:2]CC",
    "Propyl@Any_C-term": "[OH:2]CCC",
    "Amidated@Any_C-term": "[NH3:2]",
    "Cation:Na@Any_C-term": "[OH:2][Na]",
    "Cation:K@Any_C-term": "[OH:2][K]",
    "Cation:Cu[I]@Any_C-term": "[OH:2][Cu]",
    "Cation:Li@Any_C-term": "[OH:2][Li]",
}


ptm_dict = {
    "Carbamidomethyl@C": "[NH2:1]C(CSCC(N)=O)C(=O)[OH:2]",
    "Oxidation@M": "[NH2:1]C(CCS(C)=O)C(=O)[OH:2]",
    "Deamidated@N": "[NH2:1][C@@H](CC(=O)O)C(=O)[OH:2]",
    "Propionyl@K": "[NH2:1]C(CCCCNC(=O)CC)C(=O)[OH:2]",
    "Deamidated@Q": "[NH2:1][C@@H](CCC(=O)O)C(=O)[OH:2]",
    "Gln->pyro-Glu@Q^Any_N-term": "[NH:1]1C(=O)CC[C@H]1C(=O)[OH:2]",
    "Glu->pyro-Glu@E^Any_N-term": "[NH:1]1C(=O)CC[C@H]1C(=O)[OH:2]",
    "Phospho@S": "[NH2:1][C@@H](COP(=O)(O)O)C(=O)[OH:2]",
    "Nitro@Y": "[NH2:1][C@@H](Cc1ccc(O)c([N+](=O)[O-])c1)C(=O)[OH:2]",
    "Acetyl@K": "[NH2:1][C@@H](CCCCNC(C)=O)C(=O)[OH:2]",
    "Dimethyl@K": "[NH2:1][C@@H](CCCCN(C)C)C(=O)[OH:2]",
    "mTRAQ@K": "[NH2:1][C@@H](CCCCNC(=O)CN1CCN(C)CC1)C(=O)[OH:2]",
    "mTRAQ:13C(3)15N(1)@K": "[NH2:1][C@@H](CCCCNC(=O)[13CH2][15N]1CCN(C)[13CH2][13CH2]1)C(=O)[OH:2]",
    "mTRAQ:13C(6)15N(2)@K": "[NH2:1][C@@H](CCCCNC(=O)[13CH2][15N]1[13CH2][13CH2][15N]([13CH3])[13CH2][13CH2]1)C(=O)[OH:2]",
    "Pyridylethyl@C": "[NH2:1]C(CSCCc1ccncc1)C(=O)[OH:2]",
    "Butyryl@K": "[NH2:1]C(CCCCNC(=O)CCC)C(=O)[OH:2]",
    "Phospho@T": "[NH2:1]C(C(=O)[OH:2])C(C)OP(=O)(O)O",
    "Methylthio@C": "[NH2:1][C@@H](CSSC)C(=O)[OH:2]",
    "Carbamidomethyl@M": "[NH2:1]C(CCS(C)=CC(N)=O)C(=O)[OH:2]",
    "Succinyl@K": "[NH:1](C(=O)CCC(=O)O)C(CCCCN)C(=O)[OH:2]",
    "Crotonyl@K": "[NH2:1]C(CCCCNC(=O)C=CC)C(=O)[OH:2]",
    "Phospho@Y": "[NH2:1]C(Cc1ccc(OP(=O)(O)O)cc1)C(=O)[OH:2]",
    "Malonyl@K": "[NH2:1][C@@H](CCCCNC(=O)CC(=O)O)C(=O)[OH:2]",
    "Met->Hse@M^Any_C-term": "[NH2:1][C@@H](CCO)C(=O)[OH:2]",
    "Pro->(2S,4R)-4-fluoroproline@P": "[NH:1]1C[C@H](F)C[C@H]1C(=O)[OH:2]",
    "Pro->(2S,4S)-4fluoroproline@P": "[NH:1]1C[C@@H](F)C[C@H]1C(=O)[OH:2]",
    "Pro->(2S)-1,3-thiazolidine-2-carboxylic_acid@P": "[NH:1]1CCS[C@H]1C(=O)[OH:2]",
    "Pro->(4R)-1,3-Thiazolidine-4-carboxylic_acid@P": "[NH:1]1CSC[C@H]1C(=O)[OH:2]",
    "Pro->(2S,4R)-4-hydroxyproline@P": "[NH:1]1C[C@H](O)C[C@H]1C(=O)[OH:2]",
    "Pro->(DL)-pipecolic_acid@P": "[NH:1]1CCCCC1C(=O)[OH:2]",
    "Pro->3,4-Dehydro-L-proline@P": "[NH:1]1CC=CC1C(=O)[OH:2]",
    "Pro->(1S,3S,5S)-2-Azabicyclo[3.1.0]hexane-3-carboxylic_acid@P": "[NH:1]1[C@H](C(=O)[OH:2])C[C@@H]2C[C@@H]21",
    "Pro->(1R,3S,5R)-2-Azabicyclo[3.1.0]hexane-3-carboxylic_acid@P": "[NH:1]1[C@@H]2C[C@@H]2C[C@H]1C(=O)[OH:2]",
    "Pro->(2S,3aS,7aS)-Octahydro-1H-indole-2-carboxylic_acid@P": "[NH:1]1[C@H](C(=O)[OH:2])C[C@@H]2CCCC[C@@H]21",
    "Pro->(DL)-5-trifluoromethylproline@P": "[NH:1]1C(C(=O)[OH:2])CCC1C(F)(F)F",
    "hydroxyisobutyryl@K": "[NH2:1]C(CCCCNC(=O)C(C)(C)O)C(=O)[OH:2]",
    "GG@K": "[NH2:1][C@@H](CCCCNC(=O)CNC(=O)CN)C(=O)[OH:2]",
    "Phospho@H": "[NH2:1][C@@H](Cc1cn(P(=O)(O)O)cn1)C(=O)[OH:2]",
    "Phospho@D": "[NH2:1][C@@H](CC(=O)OP(=O)(O)O)C(=O)[OH:2]",
    "Cysteinyl@C": "[NH2:1][C@H](CSSC[C@@H](N)C(=O)O)C(=O)[OH:2]",
    "SATA@K": "[NH2:1][C@@H](CCCCNC(=O)CSC(C)=O)C(=O)[OH:2]",
    "SATP@K": "[NH2:1][C@@H](CCCCNC(=O)CCSC(C)=O)C(=O)[OH:2]",
    "Succinamide@K": "[NH2:1][C@@H](CCCCNC(=O)CCC(=O)NO)C(=O)[OH:2]",
    "SATA-Glc@K": "[NH2:1][C@@H](CCCCNC(=O)CS[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O)C(=O)[OH:2]",
    "Methyl@H": "[NH2:1][C@@H](Cc1cn(C)cn1)C(=O)[OH:2]",
    "Oxidation@P": "[NH:1]1C[C@H](O)C[C@H]1C(=O)[OH:2]",
}

for i in n_term_modifications:
    aa_modifier.n_term_modifications[i] = n_term_modifications[i]

for i in c_term_modifications:
    aa_modifier.c_term_modifications[i] = c_term_modifications[i]

for i in ptm_dict:
    aa_modifier.ptm_dict[i] = ptm_dict[i]


def main():
    ptms_to_add = process_modifications(ptm_dict, mod_type="ptm")
    add_new_modifications(ptms_to_add)

    nterms_to_add = process_modifications(
        n_term_modifications,
        mod_type="n_term",
    )
    add_new_modifications(nterms_to_add)

    cterms_to_add = process_modifications(
        c_term_modifications,
        mod_type="c_term",
    )
    add_new_modifications(cterms_to_add)

    MOD_DF["unimod_id"] = MOD_DF["unimod_id"].astype(int)
    orig_df = pd.read_csv(
        os.path.join(CONST_FILE_FOLDER, "modification.tsv"), sep="	", index_col=0
    )
    MOD_DF[["mod_name", *orig_df.columns]].to_csv(
        os.path.join(CONST_FILE_FOLDER, "modification.tsv"), sep="	", index=False
    )


if __name__ == "__main__":
    main()
