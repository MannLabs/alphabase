"""Machinery shared by the spectral library export formats.

:module:`alphabase.spectral_library.translate` writes a SWATH/Spectronaut transition list
and :module:`alphabase.spectral_library.translate_diann` a DIA-NN 1.9.1+ parquet library.
Both flatten the same alphabase library into one row per precursor/fragment pair and
differ only in dialect, so the modified-sequence rendering, the fragment selection and
the candidate precursor columns live here rather than in either format.

Before this module, they lived in ``translate.py``, which made the SWATH format the de
facto shared library: ``translate_diann`` imported five helpers from it.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
import tqdm

from alphabase.constants.modification import MOD_DF, ModificationKeys
from alphabase.numba_wrapper import numba_njit
from alphabase.psm_reader.keys import ConstantsClass, PsmDfCols
from alphabase.utils import explode_multiple_columns

# Candidate precursor columns in order of precedence, for libraries that carry more than
# one. The `*_pred` names are peptdeep's prediction outputs, which take priority over a
# measured value; `irt_pred` outranks `rt_pred` because an indexed RT is what a
# third-party library wants.
RT_COLUMNS = ["irt_pred", "rt_pred", PsmDfCols.RT, "irt", PsmDfCols.RT_NORM]
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


# the per-fragment columns, in the order the exports emit them
FRAGMENT_VALUE_COLUMNS = [
    FragmentTableCols.FRAG_TYPE,
    FragmentTableCols.MZ,
    FragmentTableCols.INTENSITY,
    FragmentTableCols.CHARGE,
    FragmentTableCols.SERIES_NUMBER,
    FragmentTableCols.LOSS_TYPE,
]


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
    'noloss', '1')`. The charge is left as a string, as it is only written out.
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
    verbose: bool = True,
) -> pd.DataFrame:
    """Flatten each precursor's most intense fragments into one row per fragment.

    Intensities are normalized to the precursor's most intense fragment, and the
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

    verbose : bool
        Show a progress bar over the precursors.

    Returns
    -------
    pd.DataFrame
        One row per kept fragment, in :class:`FragmentTableCols` columns.

    """
    frag_columns = fragment_mz_df.columns.to_numpy().astype("U")
    frag_types = []
    frag_losses = []
    frag_charges = []
    frag_masses = []
    frag_intensities = []
    frag_numbers = []
    iters = zip(frag_start_idx, frag_stop_idx)
    if verbose:
        iters = tqdm.tqdm(iters)
    for start, end in iters:
        intens = fragment_intensity_df.iloc[start:end, :].to_numpy(copy=True)
        max_inten = np.amax(intens)
        if max_inten > 0:
            intens /= max_inten
        masses = fragment_mz_df.iloc[start:end, :].to_numpy()
        sorted_idx = np.argsort(intens.reshape(-1))[-keep_k_highest:][::-1]
        idx_in_df = np.unravel_index(sorted_idx, masses.shape)

        frag_len = end - start
        rows = np.arange(frag_len, dtype=np.int32)[idx_in_df[0]]
        columns = frag_columns[idx_in_df[1]]

        types, losses, charges = zip(
            *[_get_frag_info_from_column_name(_) for _ in columns]
        )
        frag_types.append(types)
        frag_losses.append(losses)
        frag_charges.append(charges)
        frag_masses.append(masses[idx_in_df])
        frag_intensities.append(intens[idx_in_df])
        frag_numbers.append(_get_frag_num(columns, rows, frag_len))

    table = pd.DataFrame(
        {
            FragmentTableCols.PRECURSOR_ROW: np.arange(len(frag_start_idx)),
            FragmentTableCols.FRAG_TYPE: frag_types,
            FragmentTableCols.MZ: frag_masses,
            FragmentTableCols.INTENSITY: frag_intensities,
            FragmentTableCols.CHARGE: frag_charges,
            FragmentTableCols.SERIES_NUMBER: frag_numbers,
            FragmentTableCols.LOSS_TYPE: frag_losses,
        }
    )
    return explode_multiple_columns(table, FRAGMENT_VALUE_COLUMNS)


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


def mask_fragment_intensity_by_mz_(
    fragment_mz_df: pd.DataFrame,
    fragment_intensity_df: pd.DataFrame,
    min_frag_mz: float,
    max_frag_mz: float,
) -> None:
    """Zero the intensity of fragments outside [`min_frag_mz`, `max_frag_mz`], in place.

    Note that this edits the intensity frame it is given, and does not remove any
    fragment: what drops a fragment from an export is the caller's `min_frag_intensity`
    filter afterwards.
    """
    fragment_intensity_df.mask(
        (fragment_mz_df > max_frag_mz) | (fragment_mz_df < min_frag_mz), 0, inplace=True
    )


def mask_fragment_intensity_by_frag_nAA(  # noqa: N802
    fragment_intensity_df: pd.DataFrame,
    precursor_df: pd.DataFrame,
    max_mask_frag_nAA: int,  # noqa: N803
) -> None:
    """Zero the intensity of the smallest fragments of each precursor, in place.

    The `max_mask_frag_nAA` fragments nearest each terminus are masked: the lowest b
    numbers from `frag_start_idx` forwards, and the lowest y numbers from
    `frag_stop_idx` backwards.
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
    for i, col in enumerate(fragment_intensity_df.columns.to_numpy()):
        masks[:, i] = b_mask if is_nterm_frag(col) else y_mask

    fragment_intensity_df.mask(masks, 0, inplace=True)
