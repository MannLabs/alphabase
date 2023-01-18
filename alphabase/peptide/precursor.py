import pandas as pd
import numpy as np
import numba
import multiprocessing as mp

from xxhash import xxh64_intdigest
from functools import partial

from alphabase.constants.element import (
    MASS_PROTON, MASS_ISOTOPE
)
from alphabase.constants.aa import AA_Composition
from alphabase.constants.modification import MOD_Composition
from alphabase.constants.isotope import (
    IsotopeDistribution
)
from alphabase.peptide.mass_calc import (
    calc_peptide_masses_for_same_len_seqs
)

def refine_precursor_df(
    df:pd.DataFrame, 
    drop_frag_idx = True,
    ensure_data_validity = False,
)->pd.DataFrame:
    """ 
    Refine df inplace for faster precursor/fragment calculation.
    """
    if ensure_data_validity:
        df.fillna('', inplace=True)
        if 'charge' in df.columns:
            if df.charge.dtype not in [
                'int','int8','int64','int32',
                # np.int64, np.int32, np.int8,
            ]:
                df['charge'] = df['charge'].astype(np.int8)
        if 'mod_sites' in df.columns:
            if df.mod_sites.dtype not in ['O','U']:
                df['mod_sites'] = df.mod_sites.astype('U')

    if 'nAA' not in df.columns:
        df['nAA']= df.sequence.str.len().astype(np.int32)

    if drop_frag_idx and 'frag_start_idx' in df.columns:
        df.drop(columns=[
            'frag_start_idx','frag_stop_idx'
        ], inplace=True)

    if not is_precursor_refined(df):
        df.sort_values('nAA', inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df

reset_precursor_df = refine_precursor_df

def is_precursor_refined(precursor_df: pd.DataFrame):
    return (
        (len(precursor_df) == 0) or (
            (precursor_df.index.values[0] == 0) and
            precursor_df.nAA.is_monotonic_increasing and
            np.all(
                np.diff(precursor_df.index.values)==1
            )
        )
    )

is_precursor_sorted = is_precursor_refined

def update_precursor_mz(
    precursor_df: pd.DataFrame,
    batch_size = 500000,
)->pd.DataFrame:
    """
    Calculate precursor_mz inplace in the precursor_df
    
    Parameters
    ----------
    precursor_df : pd.DataFrame

        precursor_df with the 'charge' column

    Returns
    -------
    pd.DataFrame
        
        precursor_df with 'precursor_mz'
    """

    if 'nAA' not in precursor_df:
        reset_precursor_df(precursor_df)
        _calc_in_order = True
    elif is_precursor_sorted(precursor_df):
        _calc_in_order = True
    else:
        _calc_in_order = False
    precursor_df['precursor_mz'] = 0.
    _grouped = precursor_df.groupby('nAA')
    precursor_mz_idx = precursor_df.columns.get_loc(
        'precursor_mz'
    )
    for nAA, big_df_group in _grouped:
        for i in range(0, len(big_df_group), batch_size):
            batch_end = i+batch_size
            
            df_group = big_df_group.iloc[i:batch_end,:]

            pep_mzs = calc_peptide_masses_for_same_len_seqs(
                df_group.sequence.values.astype('U'),
                df_group.mods.values,
                df_group.mod_deltas.values if 
                'mod_deltas' in df_group.columns else None
            )/df_group.charge + MASS_PROTON
            if _calc_in_order:
                precursor_df.iloc[:,precursor_mz_idx].values[
                    df_group.index.values[0]:
                    df_group.index.values[-1]+1
                ] = pep_mzs
            else:
                precursor_df.loc[
                    df_group.index, 'precursor_mz'
                ] = pep_mzs
    return precursor_df

calc_precursor_mz = update_precursor_mz

def get_mod_seq_hash(
    sequence:str, mods:str, 
    mod_sites:str,
    *, seed:int=0
)->np.uint64:
    """Get hash code value for a peptide:
      (sequence, mods, mod_sites)

    Parameters
    ----------
        sequence : str
            
            Amino acid sequence

        mods : str
            
            Modification names in AlphaBase format

        mod_sites : str
        
            Modification sites in AlphaBase format

        seed : int
            
            Seed for hashing.
            Optional, by default 0

    Returns
    -------
    np.uint64

        64-bit hash code value
    """
    return np.array([
        xxh64_intdigest(sequence, seed=seed),
        xxh64_intdigest(mods, seed=seed),
        xxh64_intdigest(mod_sites, seed=seed),
    ],dtype=np.uint64).sum() # use np.sum to prevent overflow

def get_mod_seq_charge_hash(
    sequence:str, mods:str, 
    mod_sites:str, charge:int,
    *, seed=0
):
    """Get hash code value for a precursor:
      (sequence, mods, mod_sites, charge)

    Parameters
    ----------
    sequence : str

        Amino acid sequence

    mods : str
        
        Modification names in AlphaBase format

    mod_sites : str
    
        Modification sites in AlphaBase format

    charge : int
    
        Precursor charge state

    seed : int
    
        Seed for hashing.
        Optional, by default 0

    Returns
    -------
    np.uint64
    
        64-bit hash code value
    """
    return np.array([
        get_mod_seq_hash(
            sequence, mods, mod_sites, 
            seed=seed
        ),
        charge,
    ],dtype=np.uint64).sum() # use np.sum to prevent overflow 

def hash_mod_seq_df(
    precursor_df:pd.DataFrame,
    *, seed=0
):
    """ Internal function """
    hash_vals = precursor_df.sequence.apply(
        lambda x: xxh64_intdigest(x, seed=seed)
    ).astype(np.uint64).values
    hash_vals += precursor_df.mods.apply(
        lambda x: xxh64_intdigest(x, seed=seed)
    ).astype(np.uint64).values
    hash_vals += precursor_df.mod_sites.apply(
        lambda x: xxh64_intdigest(x, seed=seed)
    ).astype(np.uint64).values

    precursor_df[
        "mod_seq_hash"
    ] = hash_vals
    return precursor_df

def hash_mod_seq_charge_df(
    precursor_df:pd.DataFrame,
    *, seed=0
):
    """ Internal function """
    if "mod_seq_hash" not in precursor_df.columns:
        hash_mod_seq_df(precursor_df, seed=seed)
    if "charge" not in precursor_df.columns:
        return precursor_df
    
    precursor_df["mod_seq_charge_hash"] = (
        precursor_df["mod_seq_hash"].values
        + precursor_df["charge"].values.astype(np.uint64)
    )
    return precursor_df

def hash_precursor_df(
    precursor_df:pd.DataFrame,
    *, seed:int=0
)->pd.DataFrame:
    """Add columns 'mod_seq_hash' and 'mod_seq_charge_hash'
    into precursor_df (inplace). 
    The 64-bit hash function is from xxhash (xxhash.xxh64).

    Parameters
    ----------
    precursor_df : pd.DataFrame
        
        precursor_df
        
    Seed : int
    
        Seed for xxhash.xxh64.
        Optional, by default 0

    Returns
    -------
    pd.DataFrame

        DataFrame with columns 'mod_seq_hash' and 'mod_seq_charge_hash'
    """
    hash_mod_seq_df(precursor_df, seed=seed)

    if 'charge' in precursor_df.columns:
        hash_mod_seq_charge_df(precursor_df, seed=seed)
    return precursor_df

def get_mod_seq_formula(seq:str, mods:str)->list:
    """ 
    'PEPTIDE','Acetyl@Any N-term' --> [('C',n), ('H',m), ...] 
    """
    formula = {}
    for aa in seq:
        for chem,n in AA_Composition[aa].items():
            if chem in formula:
                formula[chem]+=n
            else:
                formula[chem]=n
    if len(mods) > 0:
        for mod in mods.split(';'):
            for chem,n in MOD_Composition[mod].items():
                if chem in formula:
                    formula[chem]+=n
                else:
                    formula[chem]=n
    return list(formula.items())

@numba.njit
def get_right_most_isotope_offset(
    intensities:np.ndarray, 
    apex_idx:int,
    min_right_most_intensity:float,
)->int:
    """Get right-most isotope index

    Parameters
    ----------
    intensities : np.ndarray

        Isotope intensities

    apex_idx : int

        The index or position of apex peak

    min_right_most_intensity : float

        Minimal intensity to consider for right-most peak relative to apex

    Returns
    -------
    int

        Index or position of the right-most peak
    """

    apex_inten = intensities[apex_idx]
    for i in range(len(intensities)-1,-1,-1):
        if intensities[i] >= apex_inten*min_right_most_intensity:
            return i
    return apex_idx

def get_mod_seq_isotope_distribution(
    seq_mods:tuple, 
    isotope_dist:IsotopeDistribution,
    min_right_most_intensity:float=0.2,
)->tuple:
    """Get isotope abundance distribution by IsotopeDistribution.
    This function is designed for multiprocessing.

    Parameters
    ----------
    seq_mods : tuple
        (sequence, mods)
    
    isotope_dist : IsotopeDistribution
        See `IsotopeDistribution` in `alphabase.constants.isotope`

    min_right_most_intensity : float
        The minimal intensity value of the right-most peak relative to apex peak. 
        Optional, by default 0.2

    Returns
    -------
    tuple
        float - Abundance of mono+1 / mono
        float - Abundance of apex / mono
        int - Apex isotope position relative to mono, i.e. apex index - mono index and 0 refers to the position of mono itself
        float - Abundance of right-most peak which has at least `min_right_most_intensity` intensity relative to the apex peak
        int - Right-most position relative to mono, i.e. right-most index - mono index
    """
    dist, mono = isotope_dist.calc_formula_distribution(
        get_mod_seq_formula(*seq_mods)
    )

    apex_idx = np.argmax(dist)

    # find right-most peak
    right_most_idx = get_right_most_isotope_offset(
        dist, apex_idx, min_right_most_intensity
    )

    return (
        dist[mono+1]/dist[mono], 
        dist[apex_idx]/dist[mono], 
        apex_idx-mono,
        dist[right_most_idx]/dist[mono],
        right_most_idx-mono,
    )

def calc_precursor_isotope(
    precursor_df:pd.DataFrame,
    min_right_most_intensity:float=0.2,
):
    """Calculate isotope mz values and relative (to M0) intensity values for precursor_df inplace.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        precursor_df to calculate

    min_right_most_intensity : float
        The minimal intensity value of the right-most peak relative to apex peak. 
        Optional, by default 0.2

    Returns
    -------
    pd.DataFrame
        precursor_df with additional columns:

        - isotope_m1_intensity: relative intensity of M1 to mono peak 
        - isotope_m1_mz: mz of M1
        - isotope_apex_intensity: relative intensity of the apex peak
        - isotope_apex_mz: mz of the apex peak
        - isotope_apex_offset: position offset of the apex peak to mono peak
        - isotope_right_most_intensity: relative intensity of the right-most peak
        - isotope_right_most_mz: mz of the right-most peak
        - isotope_right_most_offset: position offset of the right-most peak
    """

    if "precursor_mz" not in precursor_df.columns:
        update_precursor_mz(precursor_df)

    isotope_dist = IsotopeDistribution()

    (
        precursor_df['isotope_m1_intensity'], 
        precursor_df['isotope_apex_intensity'],
        precursor_df['isotope_apex_offset'],
        precursor_df['isotope_right_most_intensity'],
        precursor_df['isotope_right_most_offset'],
    ) = zip(
        *precursor_df[['sequence','mods']].apply(
            get_mod_seq_isotope_distribution, 
            axis=1, isotope_dist=isotope_dist,
            min_right_most_intensity=min_right_most_intensity,
        )
    )
    precursor_df['isotope_m1_intensity'] = precursor_df[
        'isotope_m1_intensity'
    ].astype(np.float32)
    precursor_df['isotope_apex_intensity'] = precursor_df[
        'isotope_apex_intensity'
    ].astype(np.float32)
    precursor_df['isotope_apex_offset'] = precursor_df[
        'isotope_apex_offset'
    ].astype(np.int8)
    precursor_df['isotope_right_most_intensity'] = precursor_df[
        'isotope_right_most_intensity'
    ].astype(np.float32)
    precursor_df['isotope_right_most_offset'] = precursor_df[
        'isotope_right_most_offset'
    ].astype(np.int8)

    precursor_df['isotope_m1_mz'] = (
        precursor_df.precursor_mz + 
        MASS_ISOTOPE/precursor_df.charge
    )

    precursor_df['isotope_apex_mz'] = (
        precursor_df.precursor_mz + 
        (
            MASS_ISOTOPE
            *precursor_df.isotope_apex_offset
            /precursor_df.charge
        )
    )
    precursor_df['isotope_right_most_mz'] = (
        precursor_df.precursor_mz + 
        (
            MASS_ISOTOPE
            *precursor_df.isotope_right_most_offset
            /precursor_df.charge
        )
    )

    return precursor_df

def _batchify_df(df_group, mp_batch_size):
    """Internal funciton for multiprocessing"""
    for _, df in df_group:
        for i in range(0, len(df), mp_batch_size):
            yield df.iloc[i:i+mp_batch_size,:]

def _count_batchify_df(df_group, mp_batch_size):
    """Internal funciton for multiprocessing"""
    count = 0
    for _, df in df_group:
        for _ in range(0, len(df), mp_batch_size):
            count += 1
    return count

# `process_bar` should be replaced by more advanced tqdm wrappers created by Sander
# I will leave it to alphabase.utils
def calc_precursor_isotope_mp(
    precursor_df:pd.DataFrame, 
    processes:int=8,
    mp_batch_size:int=100000,
    process_bar=None,
    min_right_most_intensity:float=0.2,
    min_precursor_num_to_run_mp:int=1000,
)->pd.DataFrame:
    """`calc_precursor_isotope` is not that fast for large dataframes, 
    so here we use multiprocessing for faster isotope pattern calculation. 
    The speed is acceptable with multiprocessing (3.8 min for 21M precursors, 8 processes).

    Parameters
    ----------
    precursor_df : pd.DataFrame
        Precursor_df to calculate
        
    processes : int
        Process number. Optional, by default 8
    
    mp_batch_size : int
        Multiprocessing batch size. Optional, by default 100000.
        
    process_bar : Callable
        The tqdm-based callback function 
        to check multiprocessing. Defaults to None.

    min_right_most_intensity : float
        The minimal intensity value of the right-most peak relative to apex peak. 
        Optional, by default 0.2

    Returns
    -------
    pd.DataFrame
        DataFrame with `isotope_*` columns, 
        see :meth:'calc_precursor_isotope()'.
    """
    if len(precursor_df) < min_precursor_num_to_run_mp:
        return calc_precursor_isotope(
            precursor_df=precursor_df,
            min_right_most_intensity=min_right_most_intensity,
        )
    df_list = []
    df_group = precursor_df.groupby('nAA')
    with mp.get_context("spawn").Pool(processes) as p:
        processing = p.imap(
            partial(
                calc_precursor_isotope,
                min_right_most_intensity=min_right_most_intensity
            ), _batchify_df(df_group, mp_batch_size)
        )
        if process_bar:
            processing = process_bar(
                processing, _count_batchify_df(
                    df_group, mp_batch_size
                )
            )
        for df in processing:
            df_list.append(df)
    return pd.concat(df_list)
