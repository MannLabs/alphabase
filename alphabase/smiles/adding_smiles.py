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
from typing import Dict, Literal

import pandas as pd

from alphabase.constants._const import CONST_FILE_FOLDER
from alphabase.constants.aa import aa_formula
from alphabase.constants.atom import ChemicalCompositonFormula
from alphabase.constants.modification import MOD_DF, add_new_modifications
from alphabase.smiles.smiles import AminoAcidModifier

aa_modifier = AminoAcidModifier()


def process_modifications(
    mods_dict: Dict[str, str], mod_type: Literal["ptm", "n_term", "c_term"] = "ptm"
) -> Dict[str, Dict[str, str]]:
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
            original_aa = ptm.split("@")[1].split("^")[0]
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
    "mTRAQ@Any_N-term": "C(=O)CN1CCN(CC1)C",
    "mTRAQ:13C(3)15N(1)@Any_N-term": "C(=O)[13C]([H])([H])[15N]1[13C]([H])([H])[13C]([H])([H])N(CC1)C",
    "mTRAQ:13C(6)15N(2)@Any_N-term": "C(=O)[13C]([H])([H])[15N]1[13C]([H])([H])[13C]([H])([H])[15N]([13C]([H])([H])[13C]1([H])([H]))[13C]([H])([H])([H])",
    "Acetyl@Any_N-term": "C(=O)C",
    "Acetyl@Protein_N-term": "C(=O)C",
    "Propionyl@Any_N-term": "C(=O)CC",
    "Biotin@Any_N-term": "C(=O)CCCCC1SCC2NC(=O)NC21",
    "Carbamidomethyl@Any_N-term": "C(=O)NC",
    "Carbamyl@Any_N-term": "C(=O)N",
    "Propionamide@Any_N-term": "CCC(N)=O",
    "Pyridylacetyl@Any_N-term": "C(=O)Cc1ccccn1",
    "Methyl@Any_N-term": "C",
    "Dimethyl@Any_N-term": "C",
    "Dimethyl:2H(6)13C(2)@Any_N-term": "[13C]([2H])([2H])([2H])",
    "Dimethyl:2H(4)@Any_N-term": "C([2H])([2H])([1H])",
    "Dimethyl:2H(4)13C(2)@Any_N-term": "[13C]([2H])([2H])([1H])",
    "GG@Protein_N-term": "C(=O)CNC(=O)CN",
    "SATA@Any_N-term": "C(=O)CSC(=O)C",
    "SATP@Any_N-term": "C(=O)CCSC(=O)C",
    "Succinamide@Any_N-term": "C(=O)CCC(=O)NO",
    "SATA-Glc@Any_N-term": "C(=O)CS[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O",
}


c_term_modifications = {
    "Methyl@Any_C-term": "OC",
    "Ethyl@Any_C-term": "OCC",
    "Propyl@Any_C-term": "OCCC",
    "Amidated@Any_C-term": "N",
    "Cation:Na@Any_C-term": "O[Na]",
    "Cation:K@Any_C-term": "O[K]",
    "Cation:Cu[I]@Any_C-term": "O[Cu]",
    "Cation:Li@Any_C-term": "O[Li]",
}


ptm_dict = {
    "Carbamidomethyl@C": "C(C(C(=O)[Ts])N([Fl])([Fl]))SCC(=O)N",
    "Oxidation@M": "O=C([Ts])C(N([Fl])([Fl]))CCS(=O)C",
    "GlyGly@K": "NCC(=O)NCC(=O)NCCCC[C@H](N([Fl])([Fl]))C([Ts])=O",
    "Deamidated@N": "C([C@@H](C(=O)[Ts])N([Fl])([Fl]))C(=O)O",
    "Propionyl@K": "CCC(=O)NCCCCC(C(=O)[Ts])N([Fl])([Fl])",
    "Deamidated@Q": "C(CC(=O)O)[C@@H](C(=O)[Ts])N([Fl])([Fl])",
    "Gln->pyro-Glu@Q^Any_N-term": "O=C([Ts])[C@H]1N([Fl])C(=O)CC1",
    "Glu->pyro-Glu@E^Any_N-term": "O=C([Ts])[C@H]1N([Fl])C(=O)CC1",
    "Phospho@S": "O=P(O)(O)OC[C@@H](C(=O)[Ts])N([Fl])([Fl])",
    "Nitro@Y": "O=[N+]([O-])c1cc(ccc1O)C[C@@H](C(=O)[Ts])N([Fl])([Fl])",
    "Acetyl@K": "CC(=O)NCCCC[C@H](N([Fl])([Fl]))C(=O)[Ts]",
    "Dimethyl@K": "CN(C)CCCC[C@H](N([Fl])([Fl]))C(=O)[Ts]",
    "mTRAQ@K": "[H]N(CCCC[C@H](N([Fl])([Fl]))C(=O)[Ts])C(=O)CN1CCN(C)CC1",
    "mTRAQ:13C(3)15N(1)@K": "[H]N(CCCC[C@H](N([Fl])([Fl]))C(=O)[Ts])C(=O)[13CH2][15N]1CCN(C)[13CH2][13CH2]1",
    "mTRAQ:13C(6)15N(2)@K": "[H]N(CCCC[C@H](N([Fl])([Fl]))C(=O)[Ts])C(=O)[13CH2][15N]1[13CH2][13CH2][15N]([13CH3])[13CH2][13CH2]1",
    "Pyridylethyl@C": "C1=CN=CC=C1CCSCC(C(=O)[Ts])N([Fl])([Fl])",
    "Butyryl@K": "CCCC(=O)NCCCCC(C(=O)[Ts])N([Fl])([Fl])",
    "Phospho@T": "CC(C(C(=O)[Ts])N([Fl])([Fl]))OP(=O)(O)O",
    "Methylthio@C": "CSSC[C@H](N([Fl])([Fl]))C([Ts])=O",
    "Carbamidomethyl@M": "CS(CCC(N([Fl])([Fl]))C([Ts])=O)=CC(N)=O",
    "Succinyl@K": "C(CCN)CC(C(=O)[Ts])N([Fl])C(=O)CCC(=O)O",
    "Crotonyl@K": "CC=CC(=O)NCCCCC(C(=O)[Ts])N([Fl])([Fl])",
    "Phospho@Y": "C1=CC(=CC=C1CC(C(=O)[Ts])N([Fl])([Fl]))OP(=O)(O)O",
    "Malonyl@K": "N([Fl])([Fl])[C@@H](CCCC(NC(=O)CC(=O)O))C(=O)[Ts]",
    "Met->Hse@M^Any_C-term": "N([Fl])([Fl])[C@H](C(=O)[Ts])CCO",
    "Pro->(2S,4R)-4-fluoroproline@P": "F[C@@H]1C[C@H](N([Fl])C1)C(=O)[Ts]",
    "Pro->(2S,4S)-4fluoroproline@P": "F[C@H]1C[C@H](N([Fl])C1)C(=O)[Ts]",
    "Pro->(2S)-1,3-thiazolidine-2-carboxylic_acid@P": "S1[C@H](N([Fl])CC1)C(=O)[Ts]",
    "Pro->(4R)-1,3-Thiazolidine-4-carboxylic_acid@P": "S1CN([Fl])[C@@H](C1)C(=O)[Ts]",
    "Pro->(2S,4R)-4-hydroxyproline@P": "O[C@@H]1C[C@H](N([Fl])C1)C(=O)[Ts]",
    "Pro->(DL)-pipecolic_acid@P": "C1CCN([Fl])C(C1)C(=O)[Ts]",
    "Pro->3,4-Dehydro-L-proline@P": "C1C=CC(N1([Fl]))C(=O)[Ts]",
    "Pro->(1S,3S,5S)-2-Azabicyclo[3.1.0]hexane-3-carboxylic_acid@P": "[C@H]12N([Fl])[C@@H](C[C@@H]2C1)C(=O)[Ts]",
    "Pro->(1R,3S,5R)-2-Azabicyclo[3.1.0]hexane-3-carboxylic_acid@P": "[C@@H]12N([Fl])[C@@H](C[C@H]2C1)C(=O)[Ts]",
    "Pro->(2S,3aS,7aS)-Octahydro-1H-indole-2-carboxylic_acid@P": "N1([Fl])[C@@H](C[C@@H]2CCCC[C@H]12)C(=O)[Ts]",
    "Pro->(DL)-5-trifluoromethylproline@P": "FC(C1CCC(N1([Fl]))C(=O)[Ts])(F)F",
    "hydroxyisobutyryl@K": "CC(C)(O)C(=O)NCCCCC(N([Fl])[Fl])C([Ts])=O",
    "GG@K": "NCC(=O)NCC(=O)NCCCC[C@H](N([Fl])([Fl]))C([Ts])=O",
    "Phospho@H": "O=P(O)(O)n1cc(nc1)C[C@@H](C(=O)[Ts])N([Fl])([Fl])",
    "Phospho@D": "O=P(O)(O)OC(=O)C[C@@H](C(=O)[Ts])N([Fl])([Fl])",
    "Cysteinyl@C": "N([Fl])([Fl])[C@@H](C(=O)[Ts])CSSC[C@@H](N)C(=O)O",
    "SATA@K": "CC(=O)SCC(=O)NCCCC[C@H](N([Fl])([Fl]))C([Ts])=O",
    "SATP@K": "CC(=O)SCCC(=O)NCCCC[C@H](N([Fl])([Fl]))C([Ts])=O",
    "Succinamide@K": "ONC(=O)CCC(=O)NCCCC[C@H](N([Fl])([Fl]))C([Ts])=O",
    "SATA-Glc@K": "[C@@H]1(O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O)SCC(=O)NCCCC[C@H](N([Fl])[Fl])C([Ts])=O",
    "Methyl@H": "Cn1cc(nc1)C[C@@H](C(=O)[Ts])N([Fl])([Fl])",
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
