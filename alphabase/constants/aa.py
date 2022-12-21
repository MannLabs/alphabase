import os
import pandas as pd
import numpy as np

from typing import Union, Tuple

from alphabase.yaml_utils import load_yaml

from alphabase.constants.element import (
    calc_mass_from_formula, 
    MASS_H2O, parse_formula,
)

from alphabase.constants._const import CONST_FILE_FOLDER

# We use all 128 ASCII code to represent amino acids for flexible extensions in the future.
# The amino acid masses are stored in 128-lengh array :py:data:`AA_ASCII_MASS`. 
# If an ASCII code is not in `AA_CHEM`, the mass will be set as a large value to disable MS search.
AA_CHEM:dict = load_yaml(
    os.path.join(CONST_FILE_FOLDER, 'amino_acid.yaml')
)

def reset_AA_mass()->np.ndarray:
    """AA mass in np.array with shape (128,)"""
    AA_ASCII_MASS = np.ones(128)*1e8
    for aa, chem in AA_CHEM.items():
        AA_ASCII_MASS[ord(aa)] = calc_mass_from_formula(chem)
    return AA_ASCII_MASS

#: AA mass array with ASCII code, mass of 'A' is AA_ASCII_MASS[ord('A')]
AA_ASCII_MASS:np.ndarray = reset_AA_mass()

def reset_AA_df():
    AA_DF = pd.DataFrame()
    AA_DF['aa'] = [chr(aa) for aa in range(len(AA_ASCII_MASS))]
    AA_DF['formula'] = ['']*len(AA_ASCII_MASS)
    aa_idxes = []
    formulas = []
    for aa, formula in AA_CHEM.items():
        aa_idxes.append(ord(aa))
        formulas.append(formula)
    AA_DF.loc[aa_idxes, 'formula'] = formulas
    AA_DF['mass'] = AA_ASCII_MASS
    return AA_DF

#: 128-len AA dataframe
AA_DF:pd.DataFrame = reset_AA_df()

# AA to formula dict of dict. For example: {'K': {'C': n, 'O': m, ...}}
AA_formula:dict = {}
for aa, formula, mass in AA_DF.values:
    AA_formula[aa] = dict(
        parse_formula(formula)
    )

def calc_sequence_mass(
    sequence: str
)->np.ndarray:
    '''
    Parameters
    ----------
    sequence : str
        Unmodified peptide sequence

    Returns
    -------
    np.ndarray
        Masses of each amino acid.
    '''
    return AA_ASCII_MASS[np.array(sequence,'c').view(np.int8)]

def calc_AA_masses_for_same_len_seqs(
    sequence_array: np.ndarray
)->np.ndarray:
    '''
    Calculate AA masses for the array of same-len AA sequences.

    Parameters
    ----------
    sequence_array : np.ndarray
        unmodified sequences with the same length.

    Returns
    -------
    np.ndarray
        2-D (array_size, sequence_len) array of masses.

    Raises
    -------
    ValueError
        If sequences are not with the same length.
    '''
    return AA_ASCII_MASS[
        # we use np.int32 here because unicode str 
        # uses 4 bytes for a char.
        np.array(sequence_array).view(np.int32) 
    ].reshape(len(sequence_array), -1)

def calc_sequence_masses_for_same_len_seqs(
    sequence_array: np.ndarray
)->np.ndarray:
    '''
    Calculate sequence masses for the array of same-len AA sequences.

    Parameters
    ----------
    sequence_array : np.ndarray
        unmodified sequences with the same length.

    Returns
    -------
    np.ndarray
        1-D (array_size, sequence_len) array of masses.
    
    Raises
    -------
    ValueError
        If sequences are not with the same length.
    '''
    return np.sum(
        calc_AA_masses_for_same_len_seqs(sequence_array),
        axis=1
    )+MASS_H2O


def calc_AA_masses_for_var_len_seqs(
    sequence_array: np.ndarray
)->np.ndarray:
    '''
    We recommend to use `calc_AA_masses_for_same_len_seqs` as it is much faster.

    Parameters
    ----------
    sequence_array : np.ndarray
        Sequences with variable lengths.
        
    Returns
    -------
    np.ndarray
        1D array of masses, zero values are padded to fill the max length.
    '''
    return AA_ASCII_MASS[
        np.array(sequence_array).view(np.int32)
    ].reshape(len(sequence_array), -1)
