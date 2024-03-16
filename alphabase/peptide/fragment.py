import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Dict, TYPE_CHECKING
import warnings
import numba as nb
import logging

from alphabase.constants._const import (
    PEAK_MZ_DTYPE, PEAK_INTENSITY_DTYPE
)
from alphabase.peptide.mass_calc import *
from alphabase.constants.modification import (
    calc_modloss_mass
)
from alphabase.constants.element import (
    MASS_H2O, MASS_PROTON, 
    MASS_NH3, MASS_H, MASS_C, MASS_O,
)

from alphabase.peptide.precursor import (
    refine_precursor_df,
    update_precursor_mz,
    is_precursor_sorted
)

from alphabase.constants.element import (
    calc_mass_from_formula
)

frag_type_representation_dict = {
    'c': 'b+N(1)H(3)',
    'z': 'y+N(-1)H(-2)',
    'a': 'b+C(-1)O(-1)',
    'x': 'y+C(1)O(1)H(-2)',
    'b_H2O': 'b+H(-2)O(-1)',
    'y_H2O': 'y+H(-2)O(-1)',
    'b_NH3': 'b+N(-1)H(-3)',
    'y_NH3': 'y+N(-1)H(-3)',
    'c_lossH': 'b+N(1)H(2)',
    'z_addH': 'y+N(-1)H(-1)',
}
"""
Represent fragment ion types from b/y ions.
Modification neutral losses (i.e. modloss) are not here 
as they have variable atoms added to b/y ions.
"""

frag_mass_from_ref_ion_dict = {}
"""
Masses parsed from :data:`frag_type_representation_dict`.
"""

def add_new_frag_type(frag_type:str, representation:str):
    """Add new modifications into :data:`frag_type_representation_dict`
    and update :data:`frag_mass_from_ref_ion_dict`.

    Parameters
    ----------
    frag_type : str
        New fragment type
    representation : str
        The representation similar to :data:`frag_type_representation_dict`
    """
    frag_type_representation_dict[frag_type] = representation
    ref_ion, formula = representation.split('+')
    frag_mass_from_ref_ion_dict[frag_type] = dict(
        ref_ion=ref_ion, 
        add_mass=calc_mass_from_formula(formula)
    )

def parse_all_frag_type_representation():
    for frag, representation in frag_type_representation_dict.items():
        add_new_frag_type(frag, representation)

parse_all_frag_type_representation()



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
    _type, _ch = charged_frag_type.split('_z')
    return _type, int(_ch)

def init_zero_fragment_dataframe(
    peplen_array:np.ndarray,
    charged_frag_types:List[str], 
    dtype=PEAK_MZ_DTYPE
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
    dtype=PEAK_MZ_DTYPE
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
    dtype:np.dtype=PEAK_MZ_DTYPE,
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
        `precursor_df.frag_start_idx` and `precursor.frag_stop_idx` 
        point to the indices in `reference_fragment_df`

    charged_frag_types : List
        `['b_z1','b_z2','y_z1','y_z2','b_modloss_z1','y_H2O_z1'...]`

    reference_fragment_df : pd.DataFrame
        init zero fragment_mz_df based
        on this reference. If None, fragment_mz_df will be 
        initialized by :func:`alphabase.peptide.fragment.init_zero_fragment_dataframe`.
        Defaults to None.

    dtype: np.dtype
        dtype of fragment mz values, Defaults to :data:`PEAK_MZ_DTYPE`.

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
        precursor_df['frag_stop_idx'] = end_indices
    else:
        if reference_fragment_df is None:
            # raise ValueError(
            #     "`precursor_df` contains 'frag_start_idx' column, "\
            #     "please provide `reference_fragment_df` argument"
            # )
            fragment_df = pd.DataFrame(
                np.zeros((
                    precursor_df.frag_stop_idx.max(), 
                    len(charged_frag_types)
                ), dtype=dtype),
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
                    ), dtype=dtype),
                    columns = charged_frag_types
                )
    return fragment_df

def update_sliced_fragment_dataframe(
    fragment_df: pd.DataFrame,
    fragment_df_vals: np.ndarray,
    values: np.ndarray,
    frag_start_end_list: List[Tuple[int,int]],
    charged_frag_types: List[str]=None,
):
    '''
    Set the values of the slices `frag_start_end_list=[(start,end),(start,end),...]` 
    of fragment_df.

    Parameters
    ----------
    fragment_df : pd.DataFrame
        fragment dataframe to set the values

    fragment_df_vals : np.ndarray
        The `fragment_df.to_numpy(copy=True)`, to prevent readonly assignment.

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
    '''
    frag_slice_list = [slice(start,end) for start,end in frag_start_end_list]
    frag_slices = np.r_[tuple(frag_slice_list)]
    if charged_frag_types is None or len(charged_frag_types)==0:
        fragment_df_vals[frag_slices, :] = values.astype(fragment_df_vals.dtype)
    else:
        charged_frag_idxes = [fragment_df.columns.get_loc(c) for c in charged_frag_types]
        fragment_df.iloc[
            frag_slices, charged_frag_idxes
        ] = values.astype(fragment_df_vals.dtype)
        fragment_df_vals[frag_slices] = fragment_df.values[frag_slices]

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
        precursor_df[['frag_start_idx','frag_stop_idx']] += cum_frag_df_lens[i]
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

    if 'aa_mass_diffs' in df_group.columns:
        mod_diff_list = df_group.aa_mass_diffs.str.split(';').apply(
            lambda x: [float(m) for m in x if len(m)>0]
        ).values
        mod_diff_site_list = df_group.aa_mass_diff_sites.str.split(';').apply(
            lambda x: [int(s) for s in x if len(s)>0]
        ).values
    else:
        mod_diff_list = None
        mod_diff_site_list = None
    (
        b_mass, y_mass, pepmass
    ) = calc_b_y_and_peptide_masses_for_same_len_seqs(
        df_group.sequence.values.astype('U'), 
        mod_list, site_list,
        mod_diff_list,
        mod_diff_site_list
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
    add_proton = MASS_PROTON
    for charged_frag_type in charged_frag_types:
        # Neutral masses also considered for future uses
        if charged_frag_type == 'b':
            mz_values.append(b_mass)
            continue
        elif charged_frag_type == 'y':
            mz_values.append(y_mass)
            continue
        frag_type, charge = parse_charged_frag_type(charged_frag_type)
        if frag_type == 'b':
            _mass = b_mass/charge + add_proton
        elif frag_type == 'y':
            _mass = y_mass/charge + add_proton
        elif frag_type == 'b_modloss':
            _mass = (b_mass-b_modloss)/charge + add_proton
            _mass[b_modloss == 0] = 0
        elif frag_type == 'y_modloss':
            _mass = (y_mass-y_modloss)/charge + add_proton
            _mass[y_modloss == 0] = 0
        elif frag_type in frag_mass_from_ref_ion_dict:
            ref_ion = frag_mass_from_ref_ion_dict[frag_type]['ref_ion']
            add_mass = frag_mass_from_ref_ion_dict[frag_type]['add_mass']
            if ref_ion == 'b':
                _mass = (b_mass+add_mass)/charge + add_proton
            elif ref_ion == 'y':
                _mass = (y_mass+add_mass)/charge + add_proton
            else:
                raise KeyError(f"ref_ion only allows `b` and `y`, but {ref_ion} is given")
        # elif frag_type == 'b_H2O':
        #     _mass = (b_mass-MASS_H2O)/charge + add_proton
        # elif frag_type == 'y_H2O':
        #     _mass = (y_mass-MASS_H2O)/charge + add_proton
        # elif frag_type == 'b_NH3':
        #     _mass = (b_mass-MASS_NH3)/charge + add_proton
        # elif frag_type == 'y_NH3':
        #     _mass = (y_mass-MASS_NH3)/charge + add_proton
        # elif frag_type == 'c':
        #     _mass = (MASS_NH3+b_mass)/charge + add_proton
        # elif frag_type == 'c_lossH': # H rearrangement: c-1
        #     _mass = (MASS_NH3-MASS_H+b_mass)/charge + add_proton
        # elif frag_type == 'z':
        #     _mass = (MASS_H-MASS_NH3+y_mass)/charge + add_proton
        # elif frag_type == 'z_addH': # H rearrangement: z+1
        #     _mass = (MASS_H*2-MASS_NH3+y_mass)/charge + add_proton
        # elif frag_type == 'a':
        #     _mass = (-MASS_C-MASS_O+b_mass)/charge + add_proton
        # elif frag_type == 'x':
        #     _mass = (MASS_C+MASS_O-MASS_H*2+y_mass)/charge + add_proton
        else:
            raise KeyError(
                f'Fragment type "{frag_type}" is not in fragment_mz_df.'
            )
        mz_values.append(_mass)
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



@nb.njit(parallel=True)
def fill_in_indices(
    frag_start_idxes:np.ndarray,
    frag_stop_idxes: np.ndarray, 
    indices:np.ndarray, 
    max_indices:np.ndarray, 
    excluded_indices:np.ndarray, 
    top_k: int, 
    flattened_intensity:np.ndarray, 
    number_of_fragment_types:int, 
    max_frag_per_peptide:int = 300) -> None:
    """
    Fill in indices, max indices and excluded indices for each peptide.
    indices: index of fragment per peptide (from 0 to max_index-1)
    max_indices: max index of fragments per peptide (number of fragments per peptide)
    excluded_indices: not top k excluded indices per peptide

    Parameters
    ----------
    frag_start_idxes : np.ndarray
        start indices of fragments for each peptide

    frag_stop_idxes : np.ndarray
        stop indices of fragments for each peptide

    indices : np.ndarray
        index of fragment per peptide (from 0 to max_index-1) it will be filled in this function

    max_indices : np.ndarray
        max index of fragments per peptide (number of fragments per peptide) it will be filled in this function

    excluded_indices : np.ndarray
        not top k excluded indices per peptide it will be filled in this function

    top_k : int
        top k highest peaks to keep

    flattened_intensity : np.ndarray
        Flattened fragment intensities

    number_of_fragment_types : int
        number of types of fragments (e.g. b,y,b_modloss,y_modloss, ...) equals to the number of columns in fragment mz dataframe

    max_frag_per_peptide : int, optional
        maximum number of fragments per peptide, Defaults to 300

    """
    array = np.arange(0,max_frag_per_peptide).reshape(-1,1)
    ones = np.ones(max_frag_per_peptide).reshape(-1,1)
    length = len(frag_start_idxes)

    for i in nb.prange(length):
        frag_start = frag_start_idxes[i]
        frag_end = frag_stop_idxes[i]
        max_index = frag_end-frag_start
        indices[frag_start:frag_end] = array[:max_index]
        max_indices[frag_start:frag_end] = ones[:max_index]*max_index
        if flattened_intensity is None or top_k >= max_index*number_of_fragment_types: continue
        idxes = np.argsort(flattened_intensity[frag_start*number_of_fragment_types:frag_end*number_of_fragment_types])
        _excl = np.ones_like(idxes, dtype=np.bool_)
        _excl[idxes[-top_k:]] = False
        excluded_indices[frag_start*number_of_fragment_types:frag_end*number_of_fragment_types] = _excl
            


@nb.vectorize([nb.uint32(nb.int8, nb.uint32, nb.uint32, nb.uint32)],target='parallel')
def calculate_fragment_numbers(frag_direction:np.int8, frag_number:np.uint32, index:np.uint32, max_index:np.uint32):
    """
    Calculate fragment numbers for each fragment based on the fragment direction.

    Parameters
    ----------
    frag_direction : np.int8
        directions of fragments for each peptide

    frag_number : np.uint32
        fragment numbers for each peptide

    index : np.uint32
        index of fragment per peptide (from 0 to max_index-1)

    max_index : np.uint32
        max index of fragments per peptide (number of fragments per peptide) 
    """
    if frag_direction == 1:
        frag_number = index + 1
    elif frag_direction == -1:
        frag_number = max_index - index        
    return frag_number



def parse_fragment(
    frag_directions:np.ndarray, 
    frag_start_idxes:np.ndarray, 
    frag_stop_idxes: np.ndarray, 
    top_k: int, 
    intensities:np.ndarray, 
    number_of_fragment_types:int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse fragments to get fragment numbers, fragment positions and not top k excluded indices in one hit 
    faster than doing each operation individually, and makes the most of the operations that are done in parallel. 

    Parameters
    ----------
    frag_directions : np.ndarray
        directions of fragments for each peptide

    frag_start_idxes : np.ndarray
        start indices of fragments for each peptide

    frag_stop_idxes : np.ndarray
        stop indices of fragments for each peptide

    top_k : int
        top k highest peaks to keep

    intensities : np.ndarray
        Flattened fragment intensities

    number_of_fragment_types : int
        number of types of fragments (e.g. b,y,b_modloss,y_modloss, ...) equals to the number of columns in fragment mz dataframe
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of fragment numbers, fragment positions and not top k excluded indices

    """
    # Allocate memory for fragment numbers, indices, max indices and excluded indices
    frag_numbers = np.empty_like(frag_directions, dtype=np.uint32)
    indices = np.empty_like(frag_directions, dtype=np.uint32)
    max_indices = np.empty_like(frag_directions, dtype=np.uint32)
    excluded_indices = np.zeros(frag_directions.shape[0]*frag_directions.shape[1], dtype=np.bool_)

    # Fill in indices, max indices and excluded indices
    fill_in_indices(frag_start_idxes, frag_stop_idxes,indices,max_indices, excluded_indices,top_k,intensities, number_of_fragment_types)
 
    # Calculate fragment numbers
    frag_numbers = calculate_fragment_numbers(frag_directions, frag_numbers, indices, max_indices)
    return frag_numbers, indices, excluded_indices

   
def flatten_fragments(
    precursor_df: pd.DataFrame, 
    fragment_mz_df: pd.DataFrame,
    fragment_intensity_df: pd.DataFrame,
    min_fragment_intensity: float = -1,
    keep_top_k_fragments: int = 1000,
    custom_columns : list = [
        'type','number','position','charge','loss_type'
    ],
    custom_df : Dict[str, pd.DataFrame] = {}
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

    - mz:        :data:`PEAK_MZ_DTYPE`, fragment mz value
    - intensity: :data:`PEAK_INTENSITY_DTYPE`, fragment intensity value
    - type:      uint8, ASCII code of the ion type. Small caps are for regular scoring ions used during search: (97=a, 98=b, 99=c, 120=x, 121=y, 122=z).
                        Small caps subtracted by 64 are used for ions only quantified and not scored: (33=a, 34=b, 35=c, 56=x, 57=y, 58=z).
                        By default all ions are scored and quantified. It is left to the user or search engine to decide which ions to use.
    - number:    uint32, fragment series number
    - position:  uint32, fragment position in sequence (from left to right, starts with 0)
    - charge:    uint8, fragment charge
    - loss_type: int16, fragment loss type, 0=noloss, 17=NH3, 18=H2O, 98=H3PO4 (phos), ...
    
    The fragment pointers `frag_start_idx` and `frag_stop_idx` 
    will be reannotated to the new fragment format.

    For ASCII code `type`, we can convert it into byte-str by using `frag_df.type.values.view('S1')`.
    
    Parameters
    ----------
    precursor_df : pd.DataFrame
        input precursor dataframe which contains the frag_start_idx and frag_stop_idx columns
    
    fragment_mz_df : pd.DataFrame
        input fragment mz dataframe of shape (N, T) which contains N * T fragment mzs.
        Fragments with mz==0 will be excluded.
    
    fragment_intensity_df : pd.DataFrame
        input fragment intensity dataframe of shape (N, T) which contains N * T fragment mzs.
        Could be empty (len==0) to exclude intensity values.
    
    min_fragment_intensity : float, optional
        minimum intensity which should be retained. Defaults to -1
    
    custom_columns : list, optional
        'mz' and 'intensity' columns are required. Others could be customized. 
        Defaults to ['type','number','position','charge','loss_type']

    custom_df : Dict[str, pd.DataFrame], optional
        Append custom columns by providing additional dataframes of the same shape as fragment_mz_df and fragment_intensity_df. Defaults to {}.
    
    Returns
    -------
    pd.DataFrame
        precursor dataframe with added `flat_frag_start_idx` and `flat_frag_stop_idx` columns
    pd.DataFrame
        fragment dataframe with columns: `mz`, `intensity`, `type`, `number`, 
        `charge` and `loss_type`, where each column refers to:
        
        - mz:        :data:`PEAK_MZ_DTYPE`, fragment mz value
        - intensity: :data:`PEAK_INTENSITY_DTYPE`, fragment intensity value
        - type:      uint8, ASCII code of the ion type. Small caps are for regular scoring ions used during search: (97=a, 98=b, 99=c, 120=x, 121=y, 122=z).
                            Small caps subtracted by 64 are used for ions only quantified and not scored: (33=a, 34=b, 35=c, 56=x, 57=y, 58=z).
                            By default all ions are scored and quantified. It is left to the user or search engine to decide which ions to use.
        - number:    uint32, fragment series number
        - position:  uint32, fragment position in sequence (from left to right, starts with 0)
        - charge:    uint8, fragment charge
        - loss_type: int16, fragment loss type, 0=noloss, 17=NH3, 18=H2O, 98=H3PO4 (phos), ...
    """
    if len(precursor_df) == 0:
        return precursor_df, pd.DataFrame()
    # new dataframes for fragments and precursors are created
    frag_df = {}
    frag_df['mz'] = fragment_mz_df.values.reshape(-1)
    if len(fragment_intensity_df) > 0:
        frag_df['intensity'] = fragment_intensity_df.values.astype(
            PEAK_INTENSITY_DTYPE
        ).reshape(-1)
        use_intensity = True
    else:
        use_intensity = False
    # add additional columns to the fragment dataframe
    # each column in the flat fragment dataframe is a whole pandas dataframe in the dense representation
    for col_name, df in custom_df.items():
        frag_df[col_name] = df.values.reshape(-1)
    
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
        if ord(_types[0]) >= 97 and ord(_types[0]) <= 109: # a-m
            frag_directions.append(1)
        elif ord(_types[0]) >= 110 and ord(_types[0]) <= 122: #n-z
            frag_directions.append(-1)
        else:
            frag_directions.append(0)
    
    if 'type' in custom_columns:
        frag_df['type'] = np.array(np.tile(frag_types, len(fragment_mz_df)), dtype=np.int8)
    if 'loss_type' in custom_columns:    
        frag_df['loss_type'] = np.array(np.tile(frag_loss_types, len(fragment_mz_df)), dtype=np.int16)
    if 'charge' in custom_columns:
        frag_df['charge'] = np.array(np.tile(frag_charges, len(fragment_mz_df)), dtype=np.int8)
    
    frag_directions = np.array(np.tile(frag_directions,(len(fragment_mz_df),1)), dtype=np.int8)

    numbers, positions, excluded_indices = parse_fragment(
        frag_directions, 
        precursor_df.frag_start_idx.values, 
        precursor_df.frag_stop_idx.values,
        keep_top_k_fragments,
        frag_df['intensity'] if use_intensity else None,
        len(fragment_mz_df.columns)
    )
   
    if 'number' in custom_columns:

        frag_df['number'] = numbers.reshape(-1)
   
    if 'position' in custom_columns:
        frag_df['position'] = positions.reshape(-1)
    

    precursor_df['flat_frag_start_idx'] = precursor_df.frag_start_idx
    precursor_df['flat_frag_stop_idx'] = precursor_df.frag_stop_idx
    precursor_df[['flat_frag_start_idx','flat_frag_stop_idx']] *= len(fragment_mz_df.columns)

 
    if use_intensity:
        frag_df['intensity'][frag_df['mz'] == 0.0] = 0.0


    excluded = (
        frag_df['mz'] == 0 if not use_intensity else
        (
            frag_df['intensity'] < min_fragment_intensity
        ) | (
            frag_df['mz'] == 0
        ) | (
            excluded_indices
            )
        )

    frag_df = pd.DataFrame(frag_df)
    frag_df = frag_df[~excluded]
    frag_df = frag_df.reset_index(drop=True)

    # cumulative sum counts the number of fragments before the given fragment which were removed. 
    # This sum does not include the fragment at the index position and has therefore len N +1
    cum_sum_tresh = np.zeros(shape=len(excluded)+1, dtype=np.int64)
    cum_sum_tresh[1:] = np.cumsum(excluded)

    precursor_df['flat_frag_start_idx'] -= cum_sum_tresh[precursor_df.flat_frag_start_idx.values]
    precursor_df['flat_frag_stop_idx'] -= cum_sum_tresh[precursor_df.flat_frag_stop_idx.values]

    return precursor_df, frag_df

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
        fragment_df_list: Tuple[pd.DataFrame, ...],
        frag_start_col:str = 'frag_start_idx',
        frag_stop_col:str = 'frag_stop_idx',
    ) -> Tuple[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
    """Removes unused fragments of removed precursors, 
    reannotates the `frag_start_col` and `frag_stop_col`
    
    Parameters
    ----------
    precursor_df : pd.DataFrame
        Precursor dataframe which contains frag_start_idx and frag_stop_idx columns
    
    fragment_df_list : List[pd.DataFrame]
        A list of fragment dataframes which should be compressed by removing unused fragments.
        Multiple fragment dataframes can be provided which will all be sliced in the same way. 
        This allows to slice both the fragment_mz_df and fragment_intensity_df. 
        At least one fragment dataframe needs to be provided. 

    frag_start_col : str, optional
        Fragment start idx column in `precursor_df`, such as "frag_start_idx" and "peak_start_idx".
        Defaults to "frag_start_idx".

    frag_stop_col : str, optional
        Fragment stop idx column in `precursor_df`, such as "frag_stop_idx" and "peak_stop_idx".
        Defaults to "frag_stop_idx".
    
    Returns
    -------
    pd.DataFrame, List[pd.DataFrame]
        returns the reindexed precursor DataFrame and the sliced fragment DataFrames
    """

    precursor_df = precursor_df.sort_values([frag_start_col], ascending=True)
    frag_idx = precursor_df[[frag_start_col,frag_stop_col]].values

    new_frag_idx, fragment_pointer = compress_fragment_indices(frag_idx)

    precursor_df[[frag_start_col,frag_stop_col]] = new_frag_idx
    precursor_df = precursor_df.sort_index()

    output_tuple = []

    for i in range(len(fragment_df_list)):
        output_tuple.append(
            fragment_df_list[i].iloc[
                fragment_pointer
            ].copy().reset_index(drop=True)
        )

    return precursor_df, tuple(output_tuple)

def create_fragment_mz_dataframe_by_sort_precursor(
    precursor_df: pd.DataFrame,
    charged_frag_types:List,
    batch_size:int=500000,
    dtype:np.dtype=PEAK_MZ_DTYPE,
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
            'frag_start_idx','frag_stop_idx'
        ], inplace=True)

    refine_precursor_df(precursor_df)

    fragment_mz_df = init_fragment_by_precursor_dataframe(
        precursor_df, charged_frag_types, 
        dtype=dtype,
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
                df_group.frag_stop_idx.values[-1], :
            ] = mz_values.astype(PEAK_MZ_DTYPE)
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
    dtype:np.dtype=PEAK_MZ_DTYPE,
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
        `precursor.frag_stop_idx` point to the indices in 
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
                dtype=dtype,
            )
            return create_fragment_mz_dataframe(
                precursor_df=precursor_df, 
                charged_frag_types=charged_frag_types,
                reference_fragment_df=fragment_mz_df,
                inplace_in_reference=True,
                batch_size=batch_size,
                dtype=dtype,
            )
    if 'nAA' not in precursor_df.columns:
        # fast
        return create_fragment_mz_dataframe_by_sort_precursor(
            precursor_df, charged_frag_types, 
            batch_size, dtype=dtype,
        )

    if (is_precursor_sorted(precursor_df) and 
        reference_fragment_df is None
    ):
        # fast
        return create_fragment_mz_dataframe_by_sort_precursor(
            precursor_df, charged_frag_types, 
            batch_size, dtype=dtype
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
                    ), dtype=dtype),
                    columns = charged_frag_types
                )
        else:
            fragment_mz_df = init_fragment_by_precursor_dataframe(
                precursor_df, charged_frag_types,
                dtype=dtype,
            )

        frag_mz_values = fragment_mz_df.to_numpy(copy=True)

        _grouped = precursor_df.groupby('nAA')
        for nAA, big_df_group in _grouped:
            for i in range(0, len(big_df_group), batch_size):
                batch_end = i+batch_size
                
                df_group = big_df_group.iloc[i:batch_end,:]

                mz_values = calc_fragment_mz_values_for_same_nAA(
                    df_group, nAA, fragment_mz_df.columns
                )
                
                update_sliced_fragment_dataframe(
                    fragment_mz_df, frag_mz_values, mz_values, 
                    df_group[['frag_start_idx','frag_stop_idx']].values, 
                )

    fragment_mz_df.iloc[:] = frag_mz_values

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

def calc_fragment_count(
        precursor_df : pd.DataFrame, 
        fragment_intensity_df : pd.DataFrame
    ):

    """
    Calculates the number of fragments for each precursor.

    Parameters
    ----------

    precursor_df : pd.DataFrame
        precursor dataframe which contains the frag_start_idx and frag_stop_idx columns

    fragment_intensity_df : pd.DataFrame
        fragment intensity dataframe which contains the fragment intensities

    Returns
    ------- 
    numpy.ndarray
        array with the number of fragments for each precursor
    """
    if not set(['frag_start_idx', 'frag_stop_idx']).issubset(precursor_df.columns):
        raise KeyError('frag_start_idx and frag_stop_idx not in dataframe')
    
    n_fragments = []
    
    for start, stop in zip(precursor_df['frag_start_idx'].values, precursor_df['frag_stop_idx'].values):
        n_fragments += [np.sum(fragment_intensity_df.iloc[start:stop].values > 0)]

    return np.array(n_fragments)
  
def filter_fragment_number(
        precursor_df : pd.DataFrame, 
        fragment_intensity_df : pd.DataFrame,
        n_fragments_allowed_column_name : str = 'n_fragments_allowed',
        n_allowed : int = 999
    ):

    """
    Filters the number of fragments for each precursor.
        
    Parameters
    ----------

    precursor_df : pd.DataFrame

        precursor dataframe which contains the frag_start_idx and frag_stop_idx columns

    fragment_intensity_df : pd.DataFrame
        fragment intensity dataframe which contains the fragment intensities

    n_fragments_allowed_column_name : str, default = 'n_fragments_allowed'
        column name in precursor_df which contains the number of allowed fragments

    n_allowed : int, default = 999
        number of fragments which should be allowed

    Returns
    -------
    None
    """

    if not set(['frag_start_idx', 'frag_stop_idx']).issubset(precursor_df.columns):
        raise KeyError('frag_start_idx and frag_stop_idx not in dataframe')

    for i, (start_idx, stop_idx, n_allowed_lib) in enumerate(
        zip(
            precursor_df['frag_start_idx'].values, 
            precursor_df['frag_stop_idx'].values, 
            precursor_df[n_fragments_allowed_column_name].values
            )
        ):

        _allowed = min(n_allowed_lib, n_allowed)

        intensies = fragment_intensity_df.iloc[start_idx:stop_idx].values
        flat_intensities = np.sort(intensies.flatten())[::-1]
        intensies[intensies <= flat_intensities[_allowed]] = 0
        fragment_intensity_df.iloc[start_idx:stop_idx] = intensies
        
def calc_fragment_cardinality(
        precursor_df,
        fragment_mz_df,
        group_column = 'elution_group_idx',
        split_target_decoy = True
    ):

    """
    Calculate the cardinality for a given fragment across a group of precursors.
    The cardinality is the number of precursors that have a given fragment at a given position.

    All precursors within a group are expected to have the same number of fragments.    
    The precursor dataframe.

    fragment_mz_df : pd.DataFrame
        The fragment mz dataframe.

    group_column : str
        The column to group the precursors by. Integer column is expected.

    split_target_decoy : bool
        If True, the cardinality is calculated for the target and decoy precursors separately.

    """
    
    if len(precursor_df) == 0:
        raise ValueError('Precursor dataframe is empty.')
    
    if len(fragment_mz_df) == 0:
        raise ValueError('Fragment dataframe is empty.')
    
    if group_column not in precursor_df.columns:
        raise KeyError('Group column not in precursor dataframe.')
    
    if ('frag_start_idx' not in precursor_df.columns) or ('frag_stop_idx' not in precursor_df.columns):
        raise KeyError('Precursor dataframe does not contain fragment indices.')
    
    precursor_df = precursor_df.sort_values(group_column)
    fragment_mz = fragment_mz_df.values
    fragment_cardinality = np.ones(fragment_mz.shape, dtype=np.uint8)

    @nb.njit
    def _calc_fragment_cardinality(
        elution_group_idx,
        start_idx,
        stop_idx,
        fragment_mz,
        fragment_cardinality,
    ):
        elution_group = elution_group_idx[0]
        elution_group_start = 0

        for i in range(len(elution_group_idx)):
            if i == len(elution_group_idx)-1 or elution_group_idx[i] != elution_group_idx[i+1]:
                elution_group_stop = i+1

            # check if whole elution group is covered
            n_precursor = elution_group_stop - elution_group_start
            
            # Check that all precursors within a group have the same number of fragments.
            nAA = stop_idx[elution_group_start:elution_group_stop] - start_idx[elution_group_start:elution_group_stop]
            if not np.all(nAA[0] == nAA):
                raise ValueError('All precursors within a group must have the same number of fragments.')

            # within a group, check for each precursor if it has the same fragment as another precursor
            for i in range(n_precursor):

                precursor_start_idx = start_idx[elution_group_start + i]
                precursor_stop_idx = stop_idx[elution_group_start + i]

                precursor_fragment_mz = fragment_mz[precursor_start_idx:precursor_stop_idx]

                for j in range(n_precursor):
                    if i == j:
                        continue

                    other_precursor_start_idx = start_idx[elution_group_start + j]
                    other_precursor_stop_idx = stop_idx[elution_group_start + j]
                    other_precursor_fragment_mz = fragment_mz[other_precursor_start_idx:other_precursor_stop_idx]
                    
                    binary_mask = np.abs(precursor_fragment_mz - other_precursor_fragment_mz) < 0.00001
                    
                    fragment_cardinality[precursor_start_idx:precursor_stop_idx] += binary_mask.astype(np.uint8)
                    
            elution_group_start = elution_group_stop
    if ('decoy' in precursor_df.columns) and (split_target_decoy):
        decoy_classes = precursor_df['decoy'].unique()
        for decoy_class in decoy_classes:
            df = precursor_df[precursor_df['decoy'] == decoy_class]
            _calc_fragment_cardinality(
                df[group_column].values,
                df['frag_start_idx'].values,
                df['frag_stop_idx'].values,
                fragment_mz,
                fragment_cardinality,
            )
    else:
        _calc_fragment_cardinality(
            precursor_df[group_column].values,
            precursor_df['frag_start_idx'].values,
            precursor_df['frag_stop_idx'].values,
            fragment_mz,
            fragment_cardinality,
        )

    return pd.DataFrame(
        fragment_cardinality, 
        columns = fragment_mz_df.columns
    )