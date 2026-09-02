"""Shared machinery for translating AlphaBase spectral libraries into other formats.

The format modules own their column names and their file writers:

- :mod:`alphabase.spectral_library.translate` for the SWATH-style transition list
  (one row per precursor/fragment pair) written as tsv,
- :mod:`alphabase.spectral_library.translate_diann` for DIA-NN's 1.9.1+ parquet schema.

Everything they have in common lives here: rendering modified sequences, finding the
precursor columns, masking and flattening the dense fragment dataframes.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import tqdm

from alphabase.constants.modification import MOD_DF, ModificationKeys
from alphabase.numba_wrapper import numba_njit
from alphabase.peptide.precursor import update_precursor_mz
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


# columns `update_precursor_mz` reads; anything else is left out of the copy it works on
_PRECURSOR_MZ_INPUT_COLUMNS = ("sequence", "mods", "mod_sites", "charge", "nAA")


def precursor_mz_series(precursor_df: pd.DataFrame) -> pd.Series:
    """Give the precursor m/z of every precursor, without modifying `precursor_df`.

    `update_precursor_mz` writes its result into the dataframe it is given, which would
    add a column to the library being exported. Calculate it on a copy of the columns it
    reads instead, and give back only the result.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        Precursor dataframe with the `sequence`, `mods`, `mod_sites` and `charge`
        columns, or with `precursor_mz` already calculated.

    Returns
    -------
    pd.Series
        The precursor m/z, indexed like `precursor_df`.

    """
    if "precursor_mz" in precursor_df.columns:
        return precursor_df["precursor_mz"]

    present = [c for c in _PRECURSOR_MZ_INPUT_COLUMNS if c in precursor_df.columns]
    slim_df = precursor_df[present].copy()
    if "nAA" not in slim_df.columns:
        # `update_precursor_mz` would otherwise reorder the rows to add it
        slim_df["nAA"] = slim_df["sequence"].str.len()
    update_precursor_mz(slim_df)
    return slim_df["precursor_mz"]


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


@dataclass(frozen=True)
class FragmentColumns:
    """Names to give the fragment columns of a translated library."""

    frag_type: str
    mz: str
    intensity: str
    charge: str
    series_number: str
    loss_type: str

    def as_list(self) -> list:
        """The column names in the order they are added to the translated library."""
        return [
            self.frag_type,
            self.mz,
            self.intensity,
            self.charge,
            self.series_number,
            self.loss_type,
        ]


@dataclass(frozen=True)
class FragmentFilter:
    """Which fragments of a library to export.

    Attributes
    ----------
    keep_k_highest : int
        Number of most intense fragments to keep per precursor.

    min_mz, max_mz : float
        The fragment m/z range to keep. Both zero keeps every m/z.

    min_intensity : float
        Drop fragments at or below this intensity, relative to the most intense
        fragment of their precursor.

    min_nAA : int
        Drop the b/y fragments with a series number below this. Zero keeps all.

    """

    keep_k_highest: int = 12
    min_mz: float = 200.0
    max_mz: float = 2000.0
    min_intensity: float = 0.01
    min_nAA: int = 0  # noqa: N815  matches the min_frag_nAA parameter of the formats

    @property
    def limits_mz(self) -> bool:
        """Whether the m/z range excludes anything."""
        return self.min_mz > 0 or self.max_mz > 0

    @property
    def masked_frag_nAA(self) -> int:  # noqa: N802  matches min_nAA
        """Number of fragments to drop at each terminus of a precursor."""
        return max(self.min_nAA - 1, 0)


def explode_top_fragments(  # noqa: PLR0913  the frames, the column names and the filters
    precursor_df: pd.DataFrame,
    fragment_mz_df: pd.DataFrame,
    fragment_intensity_df: pd.DataFrame,
    *,
    columns: FragmentColumns,
    fragment_filter: FragmentFilter,
    modloss_label: str = "H3PO4",
    verbose: bool = True,
) -> pd.DataFrame:
    """Flatten the dense fragment dataframes onto the precursors, one row per fragment.

    Every precursor keeps the `keep_k_highest` most intense of the fragments that pass
    `fragment_filter`, in descending intensity order, with the intensities relative to
    its own most intense fragment.

    The filters are applied to a private copy of each precursor's fragments, so neither
    `fragment_mz_df` nor `fragment_intensity_df` is modified.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        Precursor dataframe with the `frag_start_idx` and `frag_stop_idx` columns, which
        index into the dense fragment dataframes. Its other columns are carried through,
        repeated once per kept fragment; the two index columns are not.

    fragment_mz_df : pd.DataFrame
        Dense fragment m/z dataframe.

    fragment_intensity_df : pd.DataFrame
        Dense fragment intensity dataframe.

    columns : FragmentColumns
        Names to give the fragment columns of the result.

    fragment_filter : FragmentFilter
        Which fragments to keep.

    modloss_label : str
        Written in place of the `modloss` loss type, which names the lost molecule
        rather than the mechanism. Default: "H3PO4"

    verbose : bool
        Show a progress bar over the precursors.

    Returns
    -------
    pd.DataFrame
        The precursor columns and the six fragment columns, one row per kept fragment.

    """
    df = precursor_df.copy()
    frag_columns = fragment_mz_df.columns.to_numpy().astype("U")
    is_nterm_column = np.array([is_nterm_frag(col) for col in frag_columns])

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
        # `copy=True`, so the masking below cannot reach the caller's library
        intens = fragment_intensity_df.iloc[start:end, :].to_numpy(copy=True)
        masses = fragment_mz_df.iloc[start:end, :].to_numpy()

        if fragment_filter.limits_mz:
            intens[
                (masses < fragment_filter.min_mz) | (masses > fragment_filter.max_mz)
            ] = 0
        n_masked = min(fragment_filter.masked_frag_nAA, len(intens))
        if n_masked > 0:
            # b ions are numbered from the start of the block, y ions from its end
            intens[:n_masked, is_nterm_column] = 0
            intens[len(intens) - n_masked :, ~is_nterm_column] = 0

        max_inten = np.amax(intens)
        if max_inten > 0:
            intens /= max_inten
        sorted_idx = np.argsort(intens.reshape(-1))[-fragment_filter.keep_k_highest :][
            ::-1
        ]
        idx_in_df = np.unravel_index(sorted_idx, masses.shape)

        frag_len = end - start
        rows = np.arange(frag_len, dtype=np.int32)[idx_in_df[0]]
        kept_columns = frag_columns[idx_in_df[1]]

        frag_types, loss_types, charges = zip(
            *[_get_frag_info_from_column_name(_) for _ in kept_columns]
        )

        frag_type_list.append(frag_types)
        frag_loss_list.append(loss_types)
        frag_charge_list.append(charges)
        frag_mass_list.append(masses[idx_in_df])
        frag_inten_list.append(intens[idx_in_df])
        frag_num_list.append(_get_frag_num(kept_columns, rows, frag_len))

    df[columns.frag_type] = frag_type_list
    df[columns.mz] = frag_mass_list
    df[columns.intensity] = frag_inten_list
    df[columns.charge] = frag_charge_list
    df[columns.series_number] = frag_num_list
    df[columns.loss_type] = frag_loss_list

    df = explode_multiple_columns(df, columns.as_list())
    df = df[df[columns.intensity] > fragment_filter.min_intensity]
    df.loc[df[columns.loss_type] == "modloss", columns.loss_type] = modloss_label
    return df.drop(["frag_start_idx", "frag_stop_idx"], axis=1)


def translate_in_batches(  # noqa: PLR0913  the frames, the two callables and the batching
    precursor_df: pd.DataFrame,
    fragment_mz_df: pd.DataFrame,
    fragment_intensity_df: pd.DataFrame,
    convert: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame],
    write: Callable[[pd.DataFrame, int], None],
    *,
    batch_size: int,
    progress: bool = True,
) -> None:
    """Convert a library in batches of precursors and write each batch out.

    One row per fragment is much larger than the dense library, so batching keeps the
    peak memory of an export bounded. Only the precursors are batched: `frag_start_idx`
    and `frag_stop_idx` are absolute offsets into the dense fragment dataframes, so
    those are passed whole for the lookup to stay in sync.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        Precursor dataframe of the library to convert.

    fragment_mz_df, fragment_intensity_df : pd.DataFrame
        Dense fragment dataframes of the library to convert.

    convert : callable
        Called as `convert(precursor_batch, fragment_mz_df, fragment_intensity_df)`,
        and gives the translated rows of that batch.

    write : callable
        Called as `write(translated_batch, batch_start)`, where `batch_start` is the
        position of the batch's first precursor. It is 0 for the first batch, which is
        how a writer knows to emit a header.

    batch_size : int
        Number of precursors to convert per batch.

    progress : bool
        Show a progress bar over the batches.

    """
    batch_starts = range(0, len(precursor_df), batch_size)
    for start in tqdm.tqdm(batch_starts, disable=not progress):
        batch_df = precursor_df.iloc[start : start + batch_size]
        write(convert(batch_df, fragment_mz_df, fragment_intensity_df), start)
