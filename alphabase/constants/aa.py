import os
import typing

import numpy as np
import pandas as pd

from alphabase.constants._const import CONST_FILE_FOLDER
from alphabase.constants.atom import (
    MASS_H2O,
    calc_mass_from_formula,
    parse_formula,
    reset_elements,
)

# We use all 128 ASCII code to represent amino acids for flexible extensions in the future.
# The amino acid masses are stored in 128-lengh array :py:data:`AA_ASCII_MASS`.
# If an ASCII code is not in `aa_formula`, the mass will be set as a large value to disable MS search.
aa_formula: pd.DataFrame = pd.read_csv(
    os.path.join(CONST_FILE_FOLDER, "amino_acid.tsv"), sep="\t", index_col=0
)

#: AA mass array with ASCII code, mass of 'A' is AA_ASCII_MASS[ord('A')]
AA_ASCII_MASS: np.ndarray = np.ones(128) * 1e8

#: 128-len AA dataframe
AA_DF: pd.DataFrame = pd.DataFrame()

# AA formula to formula dict of dict. For example: {'K': {'C': n, 'O': m, ...}}
AA_Composition: dict = {}


def replace_atoms(atom_replace_dict: typing.Dict):
    for aa, row in aa_formula.iterrows():
        formula = row["formula"]
        atom_comp = dict(parse_formula(formula))
        for atom_from, atom_to in atom_replace_dict.items():
            if atom_from in atom_comp:
                atom_comp[atom_to] = atom_comp[atom_from]
                del atom_comp[atom_from]
        aa_formula.loc[aa, "formula"] = "".join(
            [f"{atom}({n})" for atom, n in atom_comp.items()]
        )


def reset_AA_mass() -> np.ndarray:
    """AA mass in np.array with shape (128,)"""
    global AA_ASCII_MASS
    for aa, row in aa_formula.iterrows():
        AA_ASCII_MASS[ord(aa)] = calc_mass_from_formula(row["formula"])
    return AA_ASCII_MASS


reset_AA_mass()


def reset_AA_df():
    global AA_DF
    AA_DF = pd.DataFrame()
    num_rows = len(AA_ASCII_MASS)
    AA_DF["aa"] = [chr(aa) for aa in range(num_rows)]
    AA_DF["formula"] = [""] * num_rows
    AA_DF["smiles"] = [""] * num_rows
    AA_DF["mass"] = AA_ASCII_MASS
    for aa, row in aa_formula.iterrows():
        AA_DF.loc[ord(aa), "formula"] = row["formula"]
        AA_DF.loc[ord(aa), "smiles"] = row["smiles"]
    return AA_DF


reset_AA_df()


def reset_AA_Composition():
    global AA_Composition
    AA_Composition = {}
    for aa, row in aa_formula.iterrows():
        AA_Composition[aa] = dict(parse_formula(row["formula"]))
    return AA_Composition


reset_AA_Composition()


def reset_AA_atoms(atom_replace_dict: typing.Dict = {}):
    reset_elements()
    replace_atoms(atom_replace_dict)
    reset_AA_mass()
    reset_AA_df()
    reset_AA_Composition()


def update_an_AA(aa: str, formula: str, smiles: str = ""):
    aa_idx = ord(aa)
    aa_formula.loc[aa, "formula"] = formula
    aa_formula.loc[aa, "smiles"] = smiles
    AA_DF.loc[aa_idx, "formula"] = formula
    AA_DF.loc[aa_idx, "smiles"] = smiles
    AA_ASCII_MASS[aa_idx] = calc_mass_from_formula(formula)
    AA_DF.loc[aa_idx, "mass"] = AA_ASCII_MASS[aa_idx]
    AA_Composition[aa] = dict(parse_formula(formula))


def calc_AA_masses(sequence: str) -> np.ndarray:
    """
    Parameters
    ----------
    sequence : str
        Unmodified peptide sequence

    Returns
    -------
    np.ndarray
        Masses of each amino acid.
    """
    return AA_ASCII_MASS[np.array(sequence, "c").view(np.int8)]


def calc_AA_masses_for_same_len_seqs(sequence_array: np.ndarray) -> np.ndarray:
    """
    Calculate AA masses for the array of same-len AA sequences.

    Parameters
    ----------
    sequence_array : np.ndarray or list
        unmodified sequences with the same length.

    Returns
    -------
    np.ndarray
        2-D (array_size, sequence_len) array of masses.

    Raises
    -------
    ValueError
        If sequences are not with the same length.
    """
    return AA_ASCII_MASS[
        # we use np.int32 here because unicode str
        # uses 4 bytes for a char.
        np.array(sequence_array).view(np.int32)
    ].reshape(len(sequence_array), -1)


def calc_sequence_masses_for_same_len_seqs(sequence_array: np.ndarray) -> np.ndarray:
    """
    Calculate sequence masses for the array of same-len AA sequences.

    Parameters
    ----------
    sequence_array : np.ndarray or list
        unmodified sequences with the same length.

    Returns
    -------
    np.ndarray
        1-D (array_size, sequence_len) array of masses.

    Raises
    -------
    ValueError
        If sequences are not with the same length.
    """
    return np.sum(calc_AA_masses_for_same_len_seqs(sequence_array), axis=1) + MASS_H2O


def calc_AA_masses_for_var_len_seqs(sequence_array: np.ndarray) -> np.ndarray:
    """
    We recommend to use `calc_AA_masses_for_same_len_seqs` as it is much faster.

    Parameters
    ----------
    sequence_array : np.ndarray
        Sequences with variable lengths.

    Returns
    -------
    np.ndarray
        1D array of masses, zero values are padded to fill the max length.
    """
    return AA_ASCII_MASS[np.array(sequence_array).view(np.int32)].reshape(
        len(sequence_array), -1
    )
