import os
import pandas as pd
import numpy as np
import typing

from alphabase.yaml_utils import load_yaml

from alphabase.constants.element import (
    calc_mass_from_formula, 
    MASS_H2O, parse_formula,
    reset_elements
)

from alphabase.constants._const import CONST_FILE_FOLDER

# We use all 128 ASCII code to represent amino acids for flexible extensions in the future.
# The amino acid masses are stored in 128-lengh array :py:data:`AA_ASCII_MASS`. 
# If an ASCII code is not in `AA_Formula`, the mass will be set as a large value to disable MS search.
AA_Formula:dict = load_yaml(
    os.path.join(CONST_FILE_FOLDER, 'amino_acid.yaml')
)
#: AA mass array with ASCII code, mass of 'A' is AA_ASCII_MASS[ord('A')]
AA_ASCII_MASS:np.ndarray = np.ones(128)*1e8

#: 128-len AA dataframe
AA_DF:pd.DataFrame = pd.DataFrame()

# AA formula to formula dict of dict. For example: {'K': {'C': n, 'O': m, ...}}
AA_Composition:dict = {}

def replace_atoms(atom_replace_dict:typing.Dict):
    for aa, formula in list(AA_Formula.items()):
        atom_comp = dict(parse_formula(formula))
        for atom_from, atom_to in atom_replace_dict.items():
            if atom_from in atom_comp:
                atom_comp[atom_to] = atom_comp[atom_from]
                del atom_comp[atom_from]
        AA_Formula[aa] = "".join([f"{atom}({n})" for atom, n in atom_comp.items()])

def reset_AA_mass()->np.ndarray:
    """AA mass in np.array with shape (128,)"""
    global AA_ASCII_MASS
    for aa, chem in AA_Formula.items():
        AA_ASCII_MASS[ord(aa)] = calc_mass_from_formula(chem)
    return AA_ASCII_MASS
reset_AA_mass()

def reset_AA_df():
    global AA_DF
    AA_DF = pd.DataFrame()
    AA_DF['aa'] = [chr(aa) for aa in range(len(AA_ASCII_MASS))]
    AA_DF['formula'] = ['']*len(AA_ASCII_MASS)
    aa_idxes = []
    formulas = []
    for aa, formula in AA_Formula.items():
        aa_idxes.append(ord(aa))
        formulas.append(formula)
    AA_DF.loc[aa_idxes, 'formula'] = formulas
    AA_DF['mass'] = AA_ASCII_MASS
    return AA_DF
reset_AA_df()

def reset_AA_Composition():
    global AA_Composition
    AA_Composition = {}
    for aa, formula, mass in AA_DF.values:
        AA_Composition[aa] = dict(
            parse_formula(formula)
        )
    return AA_Composition
reset_AA_Composition()

def reset_AA_atoms(atom_replace_dict:typing.Dict = {}):
    reset_elements()
    replace_atoms(atom_replace_dict)
    reset_AA_mass()
    reset_AA_df()
    reset_AA_Composition()

def update_an_AA(aa:str, formula:str):
    aa_idx = ord(aa)
    AA_DF.loc[aa_idx,'formula'] = formula
    AA_ASCII_MASS[aa_idx] = calc_mass_from_formula(formula)
    AA_DF.loc[aa_idx,'mass'] = AA_ASCII_MASS[aa_idx]
    AA_Formula[aa] = formula
    AA_Composition[aa] = dict(parse_formula(formula))

def calc_AA_masses(
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
