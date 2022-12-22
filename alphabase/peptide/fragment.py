import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Dict, TYPE_CHECKING
import warnings
import numba as nb
import logging

from alphabase.peptide.mass_calc import *
from alphabase.constants.modification import (
    calc_modloss_mass
)
from alphabase.constants.element import (
    MASS_H2O, MASS_PROTON, 
    MASS_NH3, CHEM_MONO_MASS
)

from alphabase.peptide.precursor import (
    refine_precursor_df,
    update_precursor_mz,
    is_precursor_sorted
)

def get_charged_frag_types(
    frag_types:List[str], 
    max_frag_charge:int = 2
)->List[str]:
    '''
    Combine fragment types and charge states.

    Parameters
    ----------
    frag_types : List[str]
        e.g. ['b','y','b_modloss','y_modloss']

    max_frag_charge : int
        max fragment charge. (default: 2)
    
    Returns
    -------
    List[str]
        charged fragment types
    
    Examples
    --------
    >>> frag_types=['b','y','b_modloss','y_modloss']
    >>> get_charged_frag_types(frag_types, 2)
    ['b_z1','b_z2','y_z1','y_z2','b_modloss_z1','b_modloss_z2','y_modloss_z1','y_modloss_z2']
    '''
    charged_frag_types = []
    for _type in frag_types:
        for _ch in range(1, max_frag_charge+1):
            charged_frag_types.append(f"{_type}_z{_ch}")
    return charged_frag_types

def parse_charged_frag_type(
    charged_frag_type: str
)->Tuple[str,int]:
    '''
    Oppsite to `get_charged_frag_types`.
    
    Parameters
    ----------
    charged_frag_type : str
        e.g. 'y_z1', 'b_modloss_z1'

    Returns
    -------
    tuple
        str. Fragment type, e.g. 'b','y'

        int. Charge state
    '''
    items = charged_frag_type.split('_')
    _ch = items[-1]
    _type = '_'.join(items[:-1])
    return _type, int(_ch[1:])

def init_zero_fragment_dataframe(
    peplen_array:np.ndarray,
    charged_frag_types:List[str], 
    dtype=np.float64
)->Tuple[pd.DataFrame, np.ndarray, np.ndarray]: 
    '''Initialize a zero dataframe based on peptide length 
    (nAA) array (peplen_array) and charge_frag_types (column number).
    The row number of returned dataframe is np.sum(peplen_array-1).

    Parameters
    ----------
    peplen_array : np.ndarray
        peptide lengths for the fragment dataframe
        
    charged_frag_types : List[str]
        `['b_z1','b_z2','y_z1','y_z2','b_modloss_z1','y_H2O_z1'...]`
    
    Returns
    -------
    tuple
        pd.DataFrame, `fragment_df` with zero values

        np.ndarray (int64), the start indices point to the `fragment_df` for each peptide

        np.ndarray (int64), the end indices point to the `fragment_df` for each peptide
    '''
    indices = np.zeros(len(peplen_array)+1, dtype=np.int64)
    indices[1:] = peplen_array-1
    indices = np.cumsum(indices)
    fragment_df = pd.DataFrame(
        np.zeros((indices[-1],len(charged_frag_types)), dtype=dtype),
        columns = charged_frag_types
    )
    return fragment_df, indices[:-1], indices[1:]

def init_fragment_dataframe_from_other(
    reference_fragment_df: pd.DataFrame,
    dtype=np.float64
):
    '''
    Init zero fragment dataframe from the `reference_fragment_df` (same rows and same columns)
    '''
    return pd.DataFrame(
        np.zeros_like(reference_fragment_df.values, dtype=dtype),
        columns = reference_fragment_df.columns
    )

def init_fragment_by_precursor_dataframe(
    precursor_df,
    charged_frag_types: List[str],
    *,
    reference_fragment_df: pd.DataFrame = None,
    dtype:np.dtype=np.float64,
    inplace_in_reference:bool=False,
):
    '''
    Init zero fragment dataframe for the `precursor_df`. If 
    the `reference_fragment_df` is provided, the result dataframe's 
    length will be the same as reference_fragment_df. Otherwise it 
    generates the dataframe from scratch.
    
    Parameters
    ----------
    precursor_df : pd.DataFrame
        precursors to generate fragment masses,
        if `precursor_df` contains the 'frag_start_idx' column, 
        it is better to provide `reference_fragment_df` as 
        `precursor_df.frag_start_idx` and `precursor.frag_end_idx` 
        point to the indices in `reference_fragment_df`

    charged_frag_types : List
        `['b_z1','b_z2','y_z1','y_z2','b_modloss_z1','y_H2O_z1'...]`

    reference_fragment_df : pd.DataFrame
        init zero fragment_mz_df based
        on this reference. If None, fragment_mz_df will be 
        initialized by :func:`alphabase.peptide.fragment.init_zero_fragment_dataframe`.
        Defaults to None.

    inplace_in_reference : bool, optional
        if calculate the fragment mz 
        inplace in the reference_fragment_df (default: False)

    Returns
    -------
    pd.DataFrame
        zero `fragment_df` with given `charged_frag_types` columns
    '''
    if 'frag_start_idx' not in precursor_df.columns:
        (
            fragment_df, start_indices, end_indices
        ) = init_zero_fragment_dataframe(
            precursor_df.nAA.values,
            charged_frag_types,
            dtype=dtype
        )
        precursor_df['frag_start_idx'] = start_indices
        precursor_df['frag_end_idx'] = end_indices
    else:
        if reference_fragment_df is None:
            # raise ValueError(
            #     "`precursor_df` contains 'frag_start_idx' column, "\
            #     "please provide `reference_fragment_df` argument"
            # )
            fragment_df = pd.DataFrame(
                np.zeros((
                    precursor_df.frag_end_idx.max(), 
                    len(charged_frag_types)
                )),
                columns = charged_frag_types
            )
        else:
            if inplace_in_reference: 
                fragment_df = reference_fragment_df[[
                    _fr for _fr in charged_frag_types 
                    if _fr in reference_fragment_df.columns
                ]]
            else:
                fragment_df = pd.DataFrame(
                    np.zeros((
                        len(reference_fragment_df), 
                        len(charged_frag_types)
                    )),
                    columns = charged_frag_types
                )
    return fragment_df

def update_sliced_fragment_dataframe(
    fragment_df: pd.DataFrame,
    values: np.ndarray,
    frag_start_end_list: List[Tuple[int,int]],
    charged_frag_types: List[str]=None,
)->pd.DataFrame:
    '''
    Set the values of the slices `frag_start_end_list=[(start,end),(start,end),...]` 
    of fragment_df.

    Parameters
    ----------
    fragment_df : pd.DataFrame
        fragment dataframe to set the values

    values : np.ndarray
        values to set

    frag_start_end_list : List[Tuple[int,int]]
        e.g. `[(start,end),(start,end),...]`

    charged_frag_types : List[str], optional
        e.g. `['b_z1','b_z2','y_z1','y_z2']`.
        If None, the columns of values should be the same as fragment_df's columns.
        It is much faster if charged_frag_types is None as we use numpy slicing, 
        otherwise we use pd.loc (much slower).
        Defaults to None.
    
    Returns
    -------
    pd.DataFrame
        fragment_df after the values are set into slices
    '''
    frag_slice_list = [slice(start,end) for start,end in frag_start_end_list]
    frag_slices = np.r_[tuple(frag_slice_list)]
    if charged_frag_types is None or len(charged_frag_types)==0:
        fragment_df.values[frag_slices, :] = values
    else:
        charged_frag_idxes = [fragment_df.columns.get_loc(c) for c in charged_frag_types]
        fragment_df.iloc[frag_slices, charged_frag_idxes] = values
    return fragment_df

def get_sliced_fragment_dataframe(
    fragment_df: pd.DataFrame,
    frag_start_end_list:Union[List,np.ndarray],
    charged_frag_types:List = None,
)->pd.DataFrame:
    '''
    Get the sliced fragment_df from `frag_start_end_list=[(start,end),(start,end),...]`.
    
    Parameters
    ----------
    fragment_df : pd.DataFrame
        fragment dataframe to get values

    frag_start_end_list : Union
        List[Tuple[int,int]], e.g. `[(start,end),(start,end),...]` or np.ndarray

    charged_frag_types : List[str]
        e.g. `['b_z1','b_z2','y_z1','y_z2']`.
        if None, all columns will be considered
    
    Returns
    -------
    pd.DataFrame
    
        sliced fragment_df. If `charged_frag_types` is None, 
        return fragment_df with all columns
    '''
    frag_slice_list = [slice(start,end) for start,end in frag_start_end_list]
    frag_slices = np.r_[tuple(frag_slice_list)]
    if charged_frag_types is None or len(charged_frag_types)==0:
        charged_frag_idxes = slice(None)
    else:
        charged_frag_idxes = [fragment_df.columns.get_loc(c) for c in charged_frag_types]
    return fragment_df.iloc[frag_slices, charged_frag_idxes]

def concat_precursor_fragment_dataframes(
    precursor_df_list: List[pd.DataFrame],
    fragment_df_list: List[pd.DataFrame],
    *other_fragment_df_lists
)->Tuple[pd.DataFrame,...]:
    '''
    Since fragment_df is indexed by precursor_df, when we concatenate multiple 
    fragment_df, the indexed positions will change for in precursor_dfs,  
    this function keeps the correct indexed positions of precursor_df when 
    concatenating multiple fragment_df dataframes.
    
    Parameters
    ----------
    precursor_df_list : List[pd.DataFrame]
        precursor dataframe list to concatenate

    fragment_df_list : List[pd.DataFrame]
        fragment dataframe list to concatenate

    other_fragment_df_lists
        arbitray other fragment dataframe list to concatenate, 
        e.g. fragment_mass_df, fragment_inten_df, ...
    
    Returns
    -------
    Tuple[pd.DataFrame,...]
        concatenated precursor_df, fragment_df, other_fragment_dfs ...
    '''
    fragment_df_lens = [len(fragment_df) for fragment_df in fragment_df_list]
    precursor_df_list = [precursor_df.copy() for precursor_df in precursor_df_list]
    cum_frag_df_lens = np.cumsum(fragment_df_lens)
    for i,precursor_df in enumerate(precursor_df_list[1:]):
        precursor_df[['frag_start_idx','frag_end_idx']] += cum_frag_df_lens[i]
    return (
        pd.concat(precursor_df_list, ignore_index=True),
        pd.concat(fragment_df_list, ignore_index=True),
        *[pd.concat(other_list, ignore_index=True)
            for other_list in other_fragment_df_lists
        ]
    )

def calc_fragment_mz_values_for_same_nAA(
    df_group:pd.DataFrame, 
    nAA:int, 
    charged_frag_types:list
):
    mod_list = df_group.mods.str.split(';').apply(
        lambda x: [m for m in x if len(m)>0]
    ).values
    site_list = df_group.mod_sites.str.split(';').apply(
        lambda x: [int(s) for s in x if len(s)>0]
    ).values

    if 'mod_deltas' in df_group.columns:
        mod_delta_list = df_group.mod_deltas.str.split(';').apply(
            lambda x: [float(m) for m in x if len(m)>0]
        ).values
        mod_delta_site_list = df_group.mod_delta_sites.str.split(';').apply(
            lambda x: [int(s) for s in x if len(s)>0]
        ).values
    else:
        mod_delta_list = None
        mod_delta_site_list = None
    (
        b_mass, y_mass, pepmass
    ) = calc_b_y_and_peptide_masses_for_same_len_seqs(
        df_group.sequence.values.astype('U'), 
        mod_list, site_list,
        mod_delta_list,
        mod_delta_site_list
    )
    b_mass = b_mass.reshape(-1)
    y_mass = y_mass.reshape(-1)

    for charged_frag_type in charged_frag_types:
        if charged_frag_type.startswith('b_modloss'):
            b_modloss = np.concatenate([
                calc_modloss_mass(nAA, mods, sites, True)
                for mods, sites in zip(mod_list, site_list)
            ])
            break
    for charged_frag_type in charged_frag_types:
        if charged_frag_type.startswith('y_modloss'):
            y_modloss = np.concatenate([
                calc_modloss_mass(nAA, mods, sites, False)
                for mods, sites in zip(mod_list, site_list)
            ])
            break

    mz_values = []
    # Neutral masses also considered for future uses
    # for example when searching with spectral with neutral masses
    for charged_frag_type in charged_frag_types:
        if charged_frag_type == 'b':
            mz_values.append(b_mass)
        elif charged_frag_type == 'y':
            mz_values.append(y_mass)
    add_proton = MASS_PROTON
    for charged_frag_type in charged_frag_types:
        frag_type, charge = parse_charged_frag_type(charged_frag_type)
        if frag_type == 'b':
            mz_values.append(b_mass/charge + add_proton)
        elif frag_type == 'y':
            mz_values.append(y_mass/charge + add_proton)
        elif frag_type == 'b_modloss':
            _mass = (b_mass-b_modloss)/charge + add_proton
            _mass[b_modloss == 0] = 0
            mz_values.append(_mass)
        elif frag_type == 'y_modloss':
            _mass = (y_mass-y_modloss)/charge + add_proton
            _mass[y_modloss == 0] = 0
            mz_values.append(_mass)
        elif frag_type == 'b_H2O':
            _mass = (b_mass-MASS_H2O)/charge + add_proton
            mz_values.append(_mass)
        elif frag_type == 'y_H2O':
            _mass = (y_mass-MASS_H2O)/charge + add_proton
            mz_values.append(_mass)
        elif frag_type == 'b_NH3':
            _mass = (b_mass-MASS_NH3)/charge + add_proton
            mz_values.append(_mass)
        elif frag_type == 'y_NH3':
            _mass = (y_mass-MASS_NH3)/charge + add_proton
            mz_values.append(_mass)
        elif frag_type == 'c':
            _mass = (b_mass+MASS_NH3)/charge + add_proton
            mz_values.append(_mass)
        elif frag_type == 'z':
            _mass = (
                y_mass-(MASS_NH3-CHEM_MONO_MASS['H'])
            )/charge + add_proton
            mz_values.append(_mass)
        else:
            raise NotImplementedError(
                f'Fragment type "{frag_type}" is not in fragment_mz_df.'
            )
    return np.array(mz_values).T

def mask_fragments_for_charge_greater_than_precursor_charge(
    fragment_df:pd.DataFrame, 
    precursor_charge_array:np.ndarray,
    nAA_array:np.ndarray,
    *,
    candidate_fragment_charges:list = [2,3,4],
):
    """Mask the fragment dataframe when 
    the fragment charge is larger than the precursor charge"""
    precursor_charge_array = np.repeat(
        precursor_charge_array, nAA_array-1
    )
    for col in fragment_df.columns:
        for charge in candidate_fragment_charges:
            if col.endswith(f'z{charge}'):
                fragment_df.loc[
                    precursor_charge_array<charge,col
                ] = 0
    return fragment_df

@nb.njit
def parse_fragment_positions(frag_directions, frag_start_idxes, frag_end_idxes):
    frag_positions = np.zeros_like(frag_directions, dtype=np.uint32)
    for frag_start, frag_end in zip(frag_start_idxes, frag_end_idxes):
        frag_positions[frag_start:frag_end] = np.arange(0,frag_end-frag_start).reshape(-1,1)
    return frag_positions

@nb.njit
def parse_fragment_numbers(frag_directions, frag_start_idxes, frag_end_idxes):
    frag_numbers = np.zeros_like(frag_directions, dtype=np.uint32)
    for frag_start, frag_end in zip(frag_start_idxes, frag_end_idxes):
        frag_numbers[frag_start:frag_end] = _parse_fragment_number_of_one_peptide(
            frag_directions[frag_start:frag_end]
        )
    return frag_numbers

@nb.njit    
def _parse_fragment_number_of_one_peptide(frag_directions):
    frag_number = np.zeros_like(frag_directions, dtype=np.uint32)
    max_index = len(frag_number)
    for (i,j), frag_direct in np.ndenumerate(frag_directions):
            if frag_direct == 1:
                frag_number[i,j] = i+1
            elif frag_direct == -1:
                frag_number[i,j] = max_index-i
            else:
                pass
    return frag_number

@nb.njit
def exclude_not_top_k(
    fragment_intensities, top_k, 
    frag_start_idxes, frag_end_idxes
)->np.ndarray:
    excluded = np.zeros_like(fragment_intensities, dtype=np.bool_)
    for frag_start, frag_end in zip(frag_start_idxes, frag_end_idxes):
        if top_k >= frag_end-frag_start: continue
        idxes = np.argsort(fragment_intensities[frag_start:frag_end])
        _excl = np.ones_like(idxes, dtype=np.bool_)
        _excl[idxes[-top_k:]] = False
        excluded[frag_start:frag_end] = _excl
    return excluded


def flatten_fragments(precursor_df: pd.DataFrame, 
    fragment_mz_df: pd.DataFrame,
    fragment_intensity_df: pd.DataFrame,
    min_fragment_intensity: float = -1.,
    keep_top_k_fragments: int = 1000,
    custom_columns:list = [
        'type','number','position','charge','loss_type'
    ],
)->Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts the tabular fragment format consisting of 
    the `fragment_mz_df` and the `fragment_intensity_df` 
    into a linear fragment format.
    The linear fragment format will only retain fragments 
    above a given intensity treshold with `mz > 0`. 
    It consists of columns: `mz`, `intensity`, 
    `type`, `number`, `charge` and `loss_type`,  
    where each column refers to:

    - mz:        float64, fragment mz value
    - intensity: float32, fragment intensity value
    - type:      int8, ASCII code of the ion type (97=a, 98=b, 99=c, 120=x, 121=y, 122=z), or more ion types in the future. See https://en.wikipedia.org/wiki/ASCII for more ASCII information
    - number:    uint32, fragment series number
    - position:  uint32, fragment position in sequence (from left to right, starts with 0)
    - charge:    int8, fragment charge
    - loss_type: int16, fragment loss type, 0=noloss, 17=NH3, 18=H2O, 98=H3PO4 (phos), ...
    
    The fragment pointers `frag_start_idx` and `frag_end_idx` 
    will be reannotated to the new fragment format.

    For ASCII code `type`, we can convert it into byte-str by using `frag_df.type.values.view('S1')`.
    
    Parameters
    ----------
    precursor_df : pd.DataFrame
        input precursor dataframe which contains the frag_start_idx and frag_end_idx columns
    
    fragment_mz_df : pd.DataFrame
        input fragment mz dataframe of shape (N, T) which contains N * T fragment mzs
    
    fragment_intensity_df : pd.DataFrame
        input fragment mz dataframe of shape (N, T) which contains N * T fragment mzs
    
    min_fragment_intensity : float, optional
        minimum intensity which should be retained. Defaults to -1.0
    
    custom_columns : list, optional
        'mz' and 'intensity' columns are required. Others could be customized. 
        Defaults to ['type','number','position','charge','loss_type']
    
    Returns
    -------
    pd.DataFrame
        precursor dataframe whith reindexed `frag_start_idx` and `frag_end_idx` columns
    pd.DataFrame
        fragment dataframe with columns: `mz`, `intensity`, `type`, `number`, 
        `charge` and `loss_type`, where each column refers to:
        
        - mz:        float, fragment mz value
        - intensity: float32, fragment intensity value
        - type:      int8, ASCII code of the ion type (97=a, 98=b, 99=c, 120=x, 121=y, 122=z), or more ion types in the future. See https://en.wikipedia.org/wiki/ASCII for more ASCII information
        - number:    uint32, fragment series number
        - position:  uint32, fragment position in sequence (from left to right, starts with 0)
        - charge:    int8, fragment charge
        - loss_type: int16, fragment loss type, 0=noloss, 17=NH3, 18=H2O, 98=H3PO4 (phos), ...
    """
    
    # new dataframes for fragments and precursors are created
    frag_df = pd.DataFrame()
    frag_df['mz'] = fragment_mz_df.values.reshape(-1)
    frag_df['intensity'] = fragment_intensity_df.values.astype(np.float32).reshape(-1)

    frag_types = []
    frag_loss_types = []
    frag_charges = []
    frag_directions = [] # 'abc': direction=1, 'xyz': direction=-1, otherwise 0
    
    for col in fragment_mz_df.columns.values:
        _types = col.split('_')
        frag_types.append(ord(_types[0])) # using ASCII code
        frag_charges.append(int(_types[-1][1:]))
        if len(_types) == 2:
            frag_loss_types.append(0)
        else:
            if _types[1] == 'NH3':
                frag_loss_types.append(17)
            elif _types[1] == 'H2O':
                frag_loss_types.append(18)
            else:
                frag_loss_types.append(98)

        if _types[0] in 'abc':
            frag_directions.append(1)
        elif _types[0] in 'xyz':
            frag_directions.append(-1)
        else:
            frag_directions.append(0)

    if 'type' in custom_columns:
        frag_df['type'] = np.array(frag_types*len(fragment_mz_df), dtype=np.int8)
    if 'loss_type' in custom_columns:    
        frag_df['loss_type'] = np.array(frag_loss_types*len(fragment_mz_df), dtype=np.int16)
    if 'charge' in custom_columns:
        frag_df['charge'] = np.array(frag_charges*len(fragment_mz_df), dtype=np.int8)
    
    frag_directions = np.array([frag_directions]*len(fragment_mz_df), dtype=np.int8)
    if 'number' in custom_columns:
        frag_df['number'] = parse_fragment_numbers(
            frag_directions, 
            precursor_df.frag_start_idx.values, 
            precursor_df.frag_end_idx.values
        ).reshape(-1)
    if 'position' in custom_columns:
        frag_df['position'] = parse_fragment_positions(
            frag_directions, 
            precursor_df.frag_start_idx.values, 
            precursor_df.frag_end_idx.values
        ).reshape(-1)

    precursor_new_df = precursor_df.copy()
    precursor_new_df[['frag_start_idx','frag_end_idx']] *= len(fragment_mz_df.columns)

    
    frag_df.intensity.mask(frag_df.mz == 0.0, 0.0, inplace=True)
    excluded = (
        frag_df.intensity.values < min_fragment_intensity
    ) | (
        frag_df.mz.values == 0
    ) | (
        exclude_not_top_k(
            frag_df.intensity.values, keep_top_k_fragments,
            precursor_new_df.frag_start_idx.values,
            precursor_new_df.frag_end_idx.values,
        )
    )
    frag_df = frag_df[~excluded]
    frag_df = frag_df.reset_index(drop=True)


    # cumulative sum counts the number of fragments before the given fragment which were removed. 
    # This sum does not include the fragment at the index position and has therefore len N +1
    cum_sum_tresh = np.zeros(shape=len(excluded)+1, dtype=np.int64)
    cum_sum_tresh[1:] = np.cumsum(excluded)

    precursor_new_df['frag_start_idx'] -= cum_sum_tresh[precursor_new_df.frag_start_idx.values]
    precursor_new_df['frag_end_idx'] -= cum_sum_tresh[precursor_new_df.frag_end_idx.values]

    return precursor_new_df, frag_df

@nb.njit()
def compress_fragment_indices(frag_idx):
    """
    recalculates fragment indices to remove unused fragments. Can be used to compress a fragment library.
    Expects fragment indices to be ordered by increasing values (!!!).
    It should be O(N) runtime with N being the number of fragment rows.
    
    >>> frag_idx = [[6,  10],
                [12, 14],
                [20, 22]]
    
    >>> frag_idx = [[0, 4],
                [4, 6],
                [6, 8]]
    >>> fragment_pointer = [6,7,8,9,12,13,20,21]
    """
    frag_idx_len = frag_idx[:,1]-frag_idx[:,0]


    # This sum does not include the fragment at the index position and has therefore len N +1
    frag_idx_cumsum = np.zeros(shape=len(frag_idx_len)+1, dtype='int64')
    frag_idx_cumsum[1:] = np.cumsum(frag_idx_len)

    fragment_pointer = np.zeros(np.sum(frag_idx_len), dtype='int64')

    for i in range(len(frag_idx)):

        start_index = frag_idx_cumsum[i]

        for j,k in enumerate(range(frag_idx[i,0],frag_idx[i,1])):
            fragment_pointer[start_index+j]=k


    new_frag_idx = np.column_stack((frag_idx_cumsum[:-1],frag_idx_cumsum[1:]))
    return new_frag_idx, fragment_pointer

def remove_unused_fragments(
        precursor_df: pd.DataFrame, 
        fragment_df_list: Tuple[pd.DataFrame, ...]
    ) -> Tuple[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
    """Removes unused fragments of removed precursors, 
    reannotates the frag_start_idx and frag_end_idx
    
    Parameters
    ----------
    precursor_df : pd.DataFrame
        Precursor dataframe which contains frag_start_idx and frag_end_idx columns
    
    fragment_df_list : List[pd.DataFrame]
        A list of fragment dataframes which should be compressed by removing unused fragments.
        Multiple fragment dataframes can be provided which will all be sliced in the same way. 
        This allows to slice both the fragment_mz_df and fragment_intensity_df. 
        At least one fragment dataframe needs to be provided. 
    
    Returns
    -------
    pd.DataFrame, List[pd.DataFrame]
        returns the reindexed precursor DataFrame and the sliced fragment DataFrames
    """

    precursor_df = precursor_df.sort_values(['frag_start_idx'], ascending=True)
    frag_idx = precursor_df[['frag_start_idx','frag_end_idx']].values

    new_frag_idx, fragment_pointer = compress_fragment_indices(frag_idx)

    precursor_df[['frag_start_idx','frag_end_idx']] = new_frag_idx
    precursor_df = precursor_df.sort_index()

    output_tuple = []

    for i in range(len(fragment_df_list)):
        output_tuple.append(fragment_df_list[i].iloc[fragment_pointer].copy().reset_index(drop=True))

    return precursor_df, tuple(output_tuple)

def create_fragment_mz_dataframe_by_sort_precursor(
    precursor_df: pd.DataFrame,
    charged_frag_types:List,
    batch_size:int=500000,
)->pd.DataFrame:
    """Sort nAA in precursor_df for faster fragment mz dataframe creation.
    
    Because the fragment mz values are continous in memory, so it is faster
    when setting values in pandas.
    
    Note that this function will change the order and index of precursor_df
    
    Parameters
    ----------
    precursor_df : pd.DataFrame
        precursor dataframe
    
    charged_frag_types : List
        fragment types list
    
    batch_size : int, optional
        Calculate fragment mz values in batch. 
        Defaults to 500000.
    """
    if 'frag_start_idx' in precursor_df.columns:
        precursor_df.drop(columns=[
            'frag_start_idx','frag_end_idx'
        ], inplace=True)

    refine_precursor_df(precursor_df)

    fragment_mz_df = init_fragment_by_precursor_dataframe(
        precursor_df, charged_frag_types
    )

    _grouped = precursor_df.groupby('nAA')
    for nAA, big_df_group in _grouped:
        for i in range(0, len(big_df_group), batch_size):
            batch_end = i+batch_size
            
            df_group = big_df_group.iloc[i:batch_end,:]

            mz_values = calc_fragment_mz_values_for_same_nAA(
                df_group, nAA, charged_frag_types
            )

            fragment_mz_df.iloc[
                df_group.frag_start_idx.values[0]:
                df_group.frag_end_idx.values[-1], :
            ] = mz_values
    return mask_fragments_for_charge_greater_than_precursor_charge(
            fragment_mz_df,
            precursor_df.charge.values,
            precursor_df.nAA.values,
        )

def create_fragment_mz_dataframe(
    precursor_df: pd.DataFrame,
    charged_frag_types:List,
    *,
    reference_fragment_df: pd.DataFrame = None,
    inplace_in_reference:bool = False,
    batch_size:int=500000,
)->pd.DataFrame:
    '''
    Generate fragment mass dataframe for the precursor_df. If 
    the `reference_fragment_df` is provided and precursor_df contains `frag_start_idx`, 
    it will generate  the mz dataframe based on the reference. Otherwise it 
    generates the mz dataframe from scratch.
    
    Parameters
    ----------
    precursor_df : pd.DataFrame
        precursors to generate fragment masses,
        if `precursor_df` contains the 'frag_start_idx' column, 
        `reference_fragment_df` must be provided
    charged_frag_types : List
        `['b_z1','b_z2','y_z1','y_z2','b_modloss_1','y_H2O_z1'...]`
    
    reference_fragment_df : pd.DataFrame
        kwargs only. Generate fragment_mz_df based on this reference, 
        as `precursor_df.frag_start_idx` and 
        `precursor.frag_end_idx` point to the indices in 
        `reference_fragment_df`.
        Defaults to None
    
    inplace_in_reference : bool
        kwargs only. Change values in place in the `reference_fragment_df`.
        Defaults to False
    
    batch_size: int
        Number of peptides for each batch, to save RAM.
    
    Returns
    -------
    pd.DataFrame
        `fragment_mz_df` with given `charged_frag_types`
    '''
    if reference_fragment_df is None:
        if 'frag_start_idx' in precursor_df.columns:
            # raise ValueError(
            #     "`precursor_df` contains 'frag_start_idx' column, "\
            #     "please provide `reference_fragment_df` argument"
            # )
            fragment_mz_df = init_fragment_by_precursor_dataframe(
                precursor_df, charged_frag_types,
            )
            return create_fragment_mz_dataframe(
                precursor_df=precursor_df, 
                charged_frag_types=charged_frag_types,
                reference_fragment_df=fragment_mz_df,
                inplace_in_reference=True,
                batch_size=batch_size,
            )
    if 'nAA' not in precursor_df.columns:
        # fast
        return create_fragment_mz_dataframe_by_sort_precursor(
            precursor_df, charged_frag_types, batch_size
        )

    if (is_precursor_sorted(precursor_df) and 
        reference_fragment_df is None
    ):
        # fast
        return create_fragment_mz_dataframe_by_sort_precursor(
            precursor_df, charged_frag_types, batch_size
        )

    else:
        # slow
        if reference_fragment_df is not None:
            if inplace_in_reference:
                fragment_mz_df = reference_fragment_df.loc[:,[
                    _fr for _fr in charged_frag_types 
                    if _fr in reference_fragment_df.columns
                ]]
            else:
                fragment_mz_df = pd.DataFrame(
                    np.zeros((
                        len(reference_fragment_df), 
                        len(charged_frag_types)
                    )),
                    columns = charged_frag_types
                )
        else:
            fragment_mz_df = init_fragment_by_precursor_dataframe(
                precursor_df, charged_frag_types,
            )

        _grouped = precursor_df.groupby('nAA')
        for nAA, big_df_group in _grouped:
            for i in range(0, len(big_df_group), batch_size):
                batch_end = i+batch_size
                
                df_group = big_df_group.iloc[i:batch_end,:]

                mz_values = calc_fragment_mz_values_for_same_nAA(
                    df_group, nAA, fragment_mz_df.columns
                )
                
                update_sliced_fragment_dataframe(
                    fragment_mz_df, mz_values, 
                    df_group[['frag_start_idx','frag_end_idx']].values, 
                )

    return mask_fragments_for_charge_greater_than_precursor_charge(
            fragment_mz_df,
            precursor_df.charge.values,
            precursor_df.nAA.values,
        )


# %% ../../nbdev_nbs/peptide/fragment.ipynb 38
@nb.njit(nogil=True)
def join_left(
    left: np.ndarray, 
    right: np.ndarray
    ):
    """joins all values in the left array to the values in the right array. 
    The index to the element in the right array is returned. 
    If the value wasn't found, -1 is returned. If the element appears more than once, the last appearance is used.

    Parameters
    ----------

    left: numpy.ndarray
        left array which should be matched

    right: numpy.ndarray
        right array which should be matched to

    Returns
    -------
    numpy.ndarray, dtype = int64
        array with length of the left array which indices pointing to the right array
        -1 is returned if values could not be found in the right array
    """
    left_indices = np.argsort(left)
    left_sorted = left[left_indices]

    right_indices = np.argsort(right)
    right_sorted = right[right_indices]

    joined_index = -np.ones(len(left), dtype='int64')
    
    # from hereon sorted arrays are expected
    lower_right = 0

    for i in range(len(joined_index)):

        for k in range(lower_right, len(right)):

            if left_sorted[i] >= right_sorted[k]:
                if left_sorted[i] == right_sorted[k]:
                    joined_index[i] = k
                    lower_right = k
            else:
                break

    # the joined_index_sorted connects indices from the sorted left array with the sorted right array
    # to get the original indices, the order of both sides needs to be restored
    # First, the indices pointing to the right side are restored by masking the array for hits and looking up the right side
    joined_index[joined_index >= 0] = right_indices[joined_index[joined_index >= 0]]

    # Next, the left side is restored by arranging the items
    joined_index[left_indices] =  joined_index

    return joined_index
