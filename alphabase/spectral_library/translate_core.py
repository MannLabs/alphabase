"""Shared machinery for translating AlphaBase spectral libraries into other formats.

The format modules own their column names and their file writers:

- :mod:`alphabase.spectral_library.translate` for the SWATH-style transition list
  (one row per precursor/fragment pair) written as tsv,
- :mod:`alphabase.spectral_library.translate_diann` for DIA-NN's 1.9.1+ parquet schema.

Everything they have in common lives here: rendering modified sequences, finding the
precursor columns, masking and flattening the dense fragment dataframes.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
import tqdm

from alphabase.constants.modification import MOD_DF, ModificationKeys
from alphabase.numba_wrapper import numba_njit
from alphabase.utils import explode_multiple_columns

# ---------------------------------------------------------------------------------------
# precursor columns
# ---------------------------------------------------------------------------------------

# Candidate precursor columns per exported value, in order of preference. Both formats
# read the same source columns, so the order is shared; the target column names are not,
# and stay in the format modules. Spelled out, because `PsmDfCols` has constants for the
# measured columns but not for the predicted ones that peptdeep writes.
RT_COLUMNS = ("irt_pred", "rt_pred", "rt_norm_pred", "rt", "irt", "rt_norm")
MOBILITY_COLUMNS = ("mobility_pred", "mobility")
CCS_COLUMNS = ("ccs_pred", "ccs")


def first_present_column(
    precursor_df: pd.DataFrame,
    candidates: Union[tuple, list],
    default: Union[str, float, None] = None,
) -> Union[pd.Series, str, float, None]:
    """Return the first present candidate column of `precursor_df`, else `default`.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        The precursor dataframe to look in.

    candidates : tuple or list
        Column names to try, in order of preference.

    default : str, float or None
        Returned when `precursor_df` holds none of the candidates.

    Returns
    -------
    pd.Series, str, float or None
        The first candidate column present, else `default`.

    """
    for col in candidates:
        if col in precursor_df.columns:
            return precursor_df[col]
    return default


# ---------------------------------------------------------------------------------------
# modified sequences
# ---------------------------------------------------------------------------------------


# @numba.njit #(cannot use numba for pd.Series)
def create_modified_sequence(
    seq_mods_sites: tuple,  # must be ('sequence','mods','mod_sites')
    translate_mod_dict: Optional[dict] = None,
    mod_sep: str = "[]",
    nterm: str = "_",
    cterm: str = "_",
) -> str:
    """Translate `(sequence, mods, mod_sites)` into a modified sequence.

    Used by `df.apply()`. For example,
    `('ABCDEFG','Mod1@A;Mod2@E','1;5')` -> `_A[Mod1@A]BCDE[Mod2@E]FG_`.

    Modifications are inserted from the C-terminus inwards, so the sites still to be
    written keep their original offsets into the sequence.

    Parameters
    ----------
    seq_mods_sites : tuple
        must be `(sequence, mods, mod_sites)`

    translate_mod_dict : dict
        A dict to map AlphaX modification names to other software,
        use unimod name if None.
        Defaults to None.

    mod_sep : str
        '[]' or '()', default '[]'

    nterm : str
        Prefix of the modified sequence, which any modification at site 0 is appended
        to. Defaults to '_'.

    cterm : str
        Suffix of the modified sequence, which any modification at site -1 is appended
        to. Defaults to '_'.

    Returns
    -------
    str
        The modified sequence, between the `nterm` and `cterm` markers.

    """
    mod_seq, mods, mod_sites = seq_mods_sites
    if mods:
        mods = mods.split(ModificationKeys.SEPARATOR)
        mod_sites = [int(i) for i in mod_sites.split(ModificationKeys.SEPARATOR)]
        rev_order = np.argsort(mod_sites)[::-1]
        mod_sites = [mod_sites[rev_order[i]] for i in range(len(mod_sites))]
        mods = [mods[rev_order[i]] for i in range(len(mods))]
        if translate_mod_dict is None:
            mods = [mod[: mod.find(ModificationKeys.SITE_SEPARATOR)] for mod in mods]
        else:
            mods = [translate_mod_dict[mod] for mod in mods]
        for _site, mod in zip(mod_sites, mods):
            if _site > 0:
                mod_seq = (
                    mod_seq[:_site] + mod_sep[0] + mod + mod_sep[1] + mod_seq[_site:]
                )
            elif _site == -1:
                cterm += mod_sep[0] + mod + mod_sep[1]
            elif _site == 0:
                nterm += mod_sep[0] + mod + mod_sep[1]
            else:
                mod_seq = (
                    mod_seq[:_site] + mod_sep[0] + mod + mod_sep[1] + mod_seq[_site:]
                )
    return nterm + mod_seq + cterm


mod_to_unimod_dict = {}
"""dict: AlphaBase modification name -> `UniMod:N`, for modifications that have a
UniMod id. Used as the `translate_mod_dict` of the formats that name modifications
by UniMod id."""
for mod_name, unimod_id in MOD_DF[["mod_name", "unimod_id"]].to_numpy():
    if unimod_id in (-1, "-1"):
        continue
    mod_to_unimod_dict[mod_name] = f"UniMod:{unimod_id}"


# ---------------------------------------------------------------------------------------
# fragment annotation
# ---------------------------------------------------------------------------------------


def is_nterm_frag(frag_type: str) -> bool:
    """Whether a fragment type is an N-terminal ion series, that is a, b or c.

    N-terminal fragments are numbered from the N-terminus of the peptide, the others
    from its C-terminus.

    Parameters
    ----------
    frag_type : str
        A fragment type, either bare ('b') or charged ('b_z1', 'y_modloss_z2').

    Returns
    -------
    bool
        True for the a/b/c series, False otherwise.

    """
    return frag_type[0] in "abc"


@numba_njit
def _get_frag_info_from_column_name(column: str) -> tuple:
    """Split a charged fragment type into its type, loss type and charge.

    Only used when converting alphabase libraries into other libraries. For example,
    'b_z1' -> ('b', 'noloss', '1') and 'y_modloss_z2' -> ('y', 'modloss', '2').

    Parameters
    ----------
    column : str
        A charged fragment type, that is a column of the dense fragment dataframes.

    Returns
    -------
    tuple
        `(frag_type, loss_type, charge)`, all as str.

    """
    idx = column.rfind("_")
    frag_type = column[:idx]
    charge = column[idx + 2 :]
    if len(frag_type) == 1:
        loss_type = "noloss"
    else:
        idx = frag_type.find("_")
        loss_type = frag_type[idx + 1 :]
        frag_type = frag_type[0]
    return frag_type, loss_type, charge


def _get_frag_num(columns: np.ndarray, rows: np.ndarray, frag_len: int) -> list:
    """Give the series number of the fragments at the given rows and columns.

    Parameters
    ----------
    columns : np.ndarray
        Charged fragment type of every fragment.

    rows : np.ndarray
        Row of every fragment within its precursor, counted from 0.

    frag_len : int
        Number of fragment rows of the precursor, that is `nAA - 1`.

    Returns
    -------
    list
        The series number of every fragment, counted from its own terminus.

    """
    frag_nums = []
    for r, c in zip(rows, columns):
        if is_nterm_frag(c):
            frag_nums.append(r + 1)
        else:
            frag_nums.append(frag_len - r)
    return frag_nums


# ---------------------------------------------------------------------------------------
# fragment masking
# ---------------------------------------------------------------------------------------


def mask_fragment_intensity_by_mz_(
    fragment_mz_df: pd.DataFrame,
    fragment_intensity_df: pd.DataFrame,
    min_frag_mz: float,
    max_frag_mz: float,
) -> None:
    """Zero the intensity of the fragments outside the m/z range, in place.

    Parameters
    ----------
    fragment_mz_df : pd.DataFrame
        Dense fragment m/z dataframe, which gives the m/z of every slot.

    fragment_intensity_df : pd.DataFrame
        Dense fragment intensity dataframe, modified in place.

    min_frag_mz, max_frag_mz : float
        The m/z range to keep.

    """
    fragment_intensity_df.mask(
        (fragment_mz_df > max_frag_mz) | (fragment_mz_df < min_frag_mz), 0, inplace=True
    )


def mask_fragment_intensity_by_frag_nAA(  # noqa: N802  public name
    fragment_intensity_df: pd.DataFrame,
    precursor_df: pd.DataFrame,
    max_mask_frag_nAA: int,  # noqa: N803  public name
) -> None:
    """Zero the intensity of the shortest fragments of every precursor, in place.

    The `max_mask_frag_nAA` fragments closest to each terminus are masked, that is the
    b/y ions with a series number at or below `max_mask_frag_nAA`.

    Parameters
    ----------
    fragment_intensity_df : pd.DataFrame
        Dense fragment intensity dataframe, modified in place.

    precursor_df : pd.DataFrame
        Precursor dataframe with the `frag_start_idx` and `frag_stop_idx` columns.

    max_mask_frag_nAA : int
        Number of fragments to mask at each terminus. Zero or less masks nothing.

    """
    if max_mask_frag_nAA <= 0:
        return
    b_mask = np.zeros(len(fragment_intensity_df), dtype=np.bool_)
    y_mask = b_mask.copy()
    for i_frag in range(max_mask_frag_nAA):
        b_mask[precursor_df.frag_start_idx.to_numpy() + i_frag] = True
        y_mask[precursor_df.frag_stop_idx.to_numpy() - i_frag - 1] = True

    masks = np.zeros(
        (len(fragment_intensity_df), len(fragment_intensity_df.columns)), dtype=np.bool_
    )
    for i, col in enumerate(fragment_intensity_df.columns.values):
        if is_nterm_frag(col):
            masks[:, i] = b_mask
        else:
            masks[:, i] = y_mask

    fragment_intensity_df.mask(masks, 0, inplace=True)


# ---------------------------------------------------------------------------------------
# fragment flattening
# ---------------------------------------------------------------------------------------


def merge_precursor_fragment_df(  # noqa: PLR0913  one argument per output column
    precursor_df: pd.DataFrame,
    fragment_mz_df: pd.DataFrame,
    fragment_inten_df: pd.DataFrame,
    top_n_inten: int,
    frag_type_head: str = "FragmentType",
    frag_mass_head: str = "FragmentMz",
    frag_inten_head: str = "RelativeIntensity",
    frag_charge_head: str = "FragmentCharge",
    frag_series_head: str = "FragmentNumber",
    frag_loss_head: str = "FragmentLossType",
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Flatten the dense fragment dataframes onto the precursors, one row per fragment.

    Every precursor keeps its `top_n_inten` most intense fragments, in descending
    intensity order, with the intensities relative to its own most intense fragment.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        Precursor dataframe with the `frag_start_idx` and `frag_stop_idx` columns. Its
        other columns are carried through, repeated once per kept fragment.

    fragment_mz_df : pd.DataFrame
        Dense fragment m/z dataframe.

    fragment_inten_df : pd.DataFrame
        Dense fragment intensity dataframe.

    top_n_inten : int
        Number of most intense fragments to keep per precursor.

    frag_type_head : str
        Name to give the fragment type column of the result.

    frag_mass_head : str
        Name to give the fragment m/z column of the result.

    frag_inten_head : str
        Name to give the relative fragment intensity column of the result.

    frag_charge_head : str
        Name to give the fragment charge column of the result.

    frag_series_head : str
        Name to give the fragment series number column of the result.

    frag_loss_head : str
        Name to give the fragment loss type column of the result.

    verbose : bool
        Show a progress bar over the precursors.

    Returns
    -------
    pd.DataFrame
        The precursor columns and the six fragment columns, one row per kept fragment.

    """
    df = precursor_df.copy()
    frag_columns = fragment_mz_df.columns.to_numpy().astype("U")
    frag_type_list = []
    frag_loss_list = []
    frag_charge_list = []
    frag_mass_list = []
    frag_inten_list = []
    frag_num_list = []
    iters = enumerate(df[["frag_start_idx", "frag_stop_idx"]].to_numpy())
    if verbose:
        iters = tqdm.tqdm(iters)
    for _i, (start, end) in iters:
        intens = fragment_inten_df.iloc[start:end, :].to_numpy(
            copy=True
        )  # is loc[start:end-1,:] faster?
        max_inten = np.amax(intens)
        if max_inten > 0:
            intens /= max_inten
        masses = fragment_mz_df.iloc[start:end, :].to_numpy()
        sorted_idx = np.argsort(intens.reshape(-1))[-top_n_inten:][::-1]
        idx_in_df = np.unravel_index(sorted_idx, masses.shape)

        frag_len = end - start
        rows = np.arange(frag_len, dtype=np.int32)[idx_in_df[0]]
        columns = frag_columns[idx_in_df[1]]

        frag_types, loss_types, charges = zip(
            *[_get_frag_info_from_column_name(_) for _ in columns]
        )

        frag_nums = _get_frag_num(columns, rows, frag_len)

        frag_type_list.append(frag_types)
        frag_loss_list.append(loss_types)
        frag_charge_list.append(charges)
        frag_mass_list.append(masses[idx_in_df])
        frag_inten_list.append(intens[idx_in_df])
        frag_num_list.append(frag_nums)

    df[frag_type_head] = frag_type_list
    df[frag_mass_head] = frag_mass_list
    df[frag_inten_head] = frag_inten_list
    df[frag_charge_head] = frag_charge_list
    df[frag_series_head] = frag_num_list
    df[frag_loss_head] = frag_loss_list

    return explode_multiple_columns(
        df,
        [
            frag_type_head,
            frag_mass_head,
            frag_inten_head,
            frag_charge_head,
            frag_series_head,
            frag_loss_head,
        ],
    )
