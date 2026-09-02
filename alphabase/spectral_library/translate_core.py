"""Machinery shared by the spectral library export formats.

:module:`alphabase.spectral_library.translate` writes a SWATH/Spectronaut transition list
and :module:`alphabase.spectral_library.translate_diann` a DIA-NN 1.9.1+ parquet library.
Both flatten the same alphabase library into one row per precursor/fragment pair and
differ only in dialect, so the modified-sequence rendering, the fragment selection and
the candidate precursor columns live here rather than in either format.

Before this module, they lived in ``translate.py``, which made the SWATH format the de
facto shared library: ``translate_diann`` imported five helpers from it.
"""

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import tqdm

from alphabase.constants.modification import MOD_DF, ModificationKeys
from alphabase.numba_wrapper import numba_njit
from alphabase.peptide.precursor import update_precursor_mz
from alphabase.psm_reader.keys import ConstantsClass, PsmDfCols

# Candidate precursor columns, in order of precedence.
RT_COLUMNS = [
    "irt_pred",
    "rt_pred",
    "rt_norm_pred",
    PsmDfCols.RT,
    "irt",
    PsmDfCols.RT_NORM,
]
MOBILITY_COLUMNS = ["mobility_pred", PsmDfCols.MOBILITY]
CCS_COLUMNS = ["ccs_pred", PsmDfCols.CCS]

# AlphaBase modification name -> UniMod id, for the formats that name mods by id.
# Modifications without a UniMod id are absent, so looking one up raises rather than
# writing a name the target software cannot parse.
mod_to_unimod_dict = {
    mod_name: f"UniMod:{unimod_id}"
    for mod_name, unimod_id in MOD_DF[["mod_name", "unimod_id"]].to_numpy()
    if unimod_id not in (-1, "-1")
}


class FragmentTableCols(metaclass=ConstantsClass):
    """Canonical columns of the flattened fragment table.

    Each export renames these to its own dialect. ``PRECURSOR_ROW`` is the positional
    row of the precursor a fragment belongs to; it is what joins the table back to the
    precursors and is not written out.
    """

    PRECURSOR_ROW = "precursor_row"
    FRAG_TYPE = "frag_type"
    MZ = "mz"
    INTENSITY = "intensity"
    CHARGE = "charge"
    SERIES_NUMBER = "series_number"
    LOSS_TYPE = "loss_type"


def get_precursor_mz(precursor_df: pd.DataFrame) -> pd.Series:
    """Return the precursors' m/z, leaving `precursor_df` alone.

    The read-only counterpart of
    :func:`alphabase.peptide.precursor.update_precursor_mz`, which writes its result
    into the frame it is handed -- so an export that called it left a `precursor_mz`
    column behind on the caller's library.
    """
    if PsmDfCols.PRECURSOR_MZ in precursor_df.columns:
        return precursor_df[PsmDfCols.PRECURSOR_MZ]
    return update_precursor_mz(precursor_df.copy())[PsmDfCols.PRECURSOR_MZ]


def first_present_column(
    precursor_df: pd.DataFrame,
    candidates: list[str],
    default: Union[str, float, None] = None,
) -> Union[pd.Series, str, float, None]:
    """Return the first present candidate column of `precursor_df`, else `default`.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        The precursor frame to look in.

    candidates : list of str
        Column names in order of precedence.

    default : str or float or None
        Returned when the frame carries none of the candidates. Defaults to None, which
        lets a caller tell "absent" from a legitimate value and omit the output column.

    Returns
    -------
    pd.Series or str or float or None
        The first candidate column present, else `default`.

    """
    for column in candidates:
        if column in precursor_df.columns:
            return precursor_df[column]
    return default


# @numba.njit #(cannot use numba for pd.Series)
def create_modified_sequence(
    seq_mods_sites: tuple,  # must be ('sequence','mods','mod_sites')
    translate_mod_dict: Optional[dict] = None,
    mod_sep: str = "[]",
    nterm: str = "_",
    cterm: str = "_",
) -> str:
    """Translate `(sequence, mods, mod_sites)` into a modified sequence.

    Used by `df.apply()`. For example, `('ABCDEFG','Mod1@A;Mod2@E','1;5')` ->
    `_A[Mod1@A]BCDE[Mod2@E]FG_`.

    Sites are 1-based and applied from the C-terminal end inwards, so an earlier
    insertion cannot shift a later site. Site 0 is the N-terminus and -1 the
    C-terminus; both are rendered onto `nterm`/`cterm`, which puts an N-terminal mod
    inside the leading separator and a C-terminal one after the trailing separator.

    Parameters
    ----------
    seq_mods_sites : tuple
        Must be `(sequence, mods, mod_sites)`.

    translate_mod_dict : dict
        A dict to map AlphaX modification names to other software; the bare AlphaBase
        name (everything before the `@`) is used if None. Defaults to None.

    mod_sep : str
        '[]' or '()', default '[]'.

    nterm : str
        Rendered before the sequence, and carries a site-0 modification.

    cterm : str
        Rendered after the sequence, and carries a site--1 modification.

    Returns
    -------
    str
        The modified sequence.

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
            if _site == -1:
                cterm += mod_sep[0] + mod + mod_sep[1]
            elif _site == 0:
                nterm += mod_sep[0] + mod + mod_sep[1]
            else:
                mod_seq = (
                    mod_seq[:_site] + mod_sep[0] + mod + mod_sep[1] + mod_seq[_site:]
                )
    return nterm + mod_seq + cterm


def is_nterm_frag(frag_type: str) -> bool:
    """Whether a fragment column name is an N-terminal (a/b/c) series."""
    return frag_type[0] in "abc"


@numba_njit
def _get_frag_info_from_column_name(column: str) -> tuple:
    """Split a fragment column name into `(frag_type, loss_type, charge)`.

    For example `y_modloss_z2` -> `('y', 'modloss', '2')` and `b_z1` -> `('b',
    'noloss', '1')`. The charge stays a string because numba cannot parse one;
    :func:`fragment_table` casts the collected column.
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
    """Number each fragment within its series.

    N-terminal series are numbered from the start of the peptide and C-terminal ones
    from the end, so row `r` of a precursor with `frag_len` fragment rows is `r + 1`
    for a b-ion and `frag_len - r` for a y-ion.
    """
    return [
        row + 1 if is_nterm_frag(column) else frag_len - row
        for row, column in zip(rows, columns)
    ]


def fragment_table(  # noqa: PLR0913
    frag_start_idx: np.ndarray,
    frag_stop_idx: np.ndarray,
    fragment_mz_df: pd.DataFrame,
    fragment_intensity_df: pd.DataFrame,
    *,
    keep_k_highest: int,
    min_frag_mz: float = 0,
    max_frag_mz: float = np.inf,
    min_frag_nAA: int = 0,  # noqa: N803
    verbose: bool = True,
) -> pd.DataFrame:
    """Flatten each precursor's most intense fragments into one row per fragment.

    Filtering, normalization and selection all happen on a per-precursor copy, so the
    library's fragment frames are left exactly as they were. Fragments outside the m/z
    window are dropped rather than zeroed, as are empty fragment slots -- a `*_modloss`
    column of a precursor whose modification has no loss carries m/z 0, and selecting it
    would export a fragment that does not exist. An unbounded window is expressed by the
    bounds themselves, `0` and `np.inf`, so no combination of them is a special case.

    Intensities are normalized to the precursor's most intense kept fragment, and the
    `keep_k_highest` highest are kept in descending order. The result carries the
    canonical columns of :class:`FragmentTableCols`, including `precursor_row` -- the
    positional row of the precursor -- so it needs no precursor frame to be built and
    no output frame to be built into. :func:`join_fragments` attaches the precursors.

    Parameters
    ----------
    frag_start_idx, frag_stop_idx : np.ndarray
        Per precursor, the half-open row range into the fragment frames. These are
        absolute offsets, so batching the precursors leaves the fragment frames whole.

    fragment_mz_df : pd.DataFrame
        The library's fragment m/z frame.

    fragment_intensity_df : pd.DataFrame
        The library's fragment intensity frame.

    keep_k_highest : int
        Keep this many fragments per precursor.

    min_frag_mz : float
        Drop fragments below this m/z. 0 for no lower bound, as m/z is positive.

    max_frag_mz : float
        Drop fragments above this m/z. `np.inf` for no upper bound.

    min_frag_nAA : int
        Drop the smallest `min_frag_nAA - 1` fragments of each series; 0 disables. The
        off-by-one is the existing meaning of the export parameter of the same name.

    verbose : bool
        Show a progress bar over the precursors.

    Returns
    -------
    pd.DataFrame
        One row per kept fragment, in :class:`FragmentTableCols` columns. `mz` and
        `intensity` keep the dtype of the frame they came from; the charge and series
        number are integers.

    """
    frag_columns = fragment_mz_df.columns.to_numpy().astype("U")
    is_nterm = np.array([is_nterm_frag(column) for column in frag_columns])
    n_masked_per_terminus = max(min_frag_nAA - 1, 0)

    if min_frag_mz == 0 and max_frag_mz == 0:
        warnings.warn(
            "Disabling the fragment m/z window with min_frag_mz=0, max_frag_mz=0 is "
            "deprecated; pass max_frag_mz=np.inf instead. min_frag_mz=0 already means "
            "no lower bound, as m/z is positive.",
            FutureWarning,
        )
        max_frag_mz = np.inf

    frag_types: list = []
    frag_losses: list = []
    frag_charges: list = []
    frag_numbers: list = []
    frag_masses: list = []
    frag_intensities: list = []
    iters = zip(frag_start_idx, frag_stop_idx)
    if verbose:
        iters = tqdm.tqdm(iters)
    for start, end in iters:
        masses = fragment_mz_df.iloc[start:end, :].to_numpy()
        keep = (masses > 0) & (masses >= min_frag_mz) & (masses <= max_frag_mz)
        if n_masked_per_terminus:
            # b numbers count from the first row, y numbers from the last, so the
            # smallest of each series sit at opposite ends of the block. `max(..., 0)`
            # because a negative slice start wraps rather than clamping.
            keep[:n_masked_per_terminus, is_nterm] = False
            keep[max(len(keep) - n_masked_per_terminus, 0) :, ~is_nterm] = False

        # `copy=True`, so normalizing and zeroing below cannot reach the library
        intens = fragment_intensity_df.iloc[start:end, :].to_numpy(copy=True)
        intens[~keep] = 0
        max_inten = np.amax(intens)
        if max_inten > 0:
            intens /= max_inten

        sorted_idx = np.argsort(intens.reshape(-1))[-keep_k_highest:][::-1]
        # a filtered-out slot can still be selected when a precursor has fewer than
        # `keep_k_highest` fragments left, so drop those rather than export them
        sorted_idx = sorted_idx[keep.reshape(-1)[sorted_idx]]
        idx_in_df = np.unravel_index(sorted_idx, masses.shape)

        frag_len = end - start
        rows = np.arange(frag_len, dtype=np.int32)[idx_in_df[0]]
        columns = frag_columns[idx_in_df[1]]

        infos = [_get_frag_info_from_column_name(column) for column in columns]
        types, losses, charges = zip(*infos) if infos else ((), (), ())
        frag_types.extend(types)
        frag_losses.extend(losses)
        frag_charges.extend(charges)
        frag_numbers.extend(_get_frag_num(columns, rows, frag_len))
        frag_masses.append(masses[idx_in_df])
        frag_intensities.append(intens[idx_in_df])

    return pd.DataFrame(
        {
            FragmentTableCols.PRECURSOR_ROW: np.repeat(
                np.arange(len(frag_start_idx)),
                [len(kept) for kept in frag_masses],
            ),
            FragmentTableCols.FRAG_TYPE: frag_types,
            FragmentTableCols.MZ: np.concatenate(frag_masses),
            FragmentTableCols.INTENSITY: np.concatenate(frag_intensities),
            FragmentTableCols.CHARGE: np.array(frag_charges, dtype=np.int64),
            FragmentTableCols.SERIES_NUMBER: np.array(frag_numbers, dtype=np.int64),
            FragmentTableCols.LOSS_TYPE: frag_losses,
        }
    )


def join_fragments(
    precursor_df: pd.DataFrame,
    fragment_df: pd.DataFrame,
    columns: dict,
) -> pd.DataFrame:
    """Repeat each precursor row across its fragments, renamed to `columns`.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        The export's precursor rows, in the order `fragment_df`'s `precursor_row`
        indexes them.

    fragment_df : pd.DataFrame
        A :func:`fragment_table` result.

    columns : dict
        Maps :class:`FragmentTableCols` names to this format's output names. Its order
        is the order the fragment columns are appended in.

    Returns
    -------
    pd.DataFrame
        One row per precursor/fragment pair, keeping `precursor_df`'s index.

    """
    rows = fragment_df[FragmentTableCols.PRECURSOR_ROW].to_numpy()
    joined = precursor_df.iloc[rows].copy()
    for canonical, name in columns.items():
        joined[name] = fragment_df[canonical].to_numpy()
    return joined
