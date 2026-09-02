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
from alphabase.psm_reader.keys import LibPsmDfCols, PsmDfCols
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


def merge_precursor_fragment_df(  # noqa: PLR0913
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
    verbose: bool = True,  # noqa: FBT001, FBT002
) -> pd.DataFrame:
    """Attach each precursor's most intense fragments and explode to one row each.

    `precursor_df` is the half-built *output* frame and must carry `frag_start_idx` and
    `frag_stop_idx` to look the fragments up with; the caller drops them afterwards.
    Intensities are normalized to the precursor's most intense fragment, and the
    `top_n_inten` highest are kept in descending order.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        The output frame so far, one row per precursor.

    fragment_mz_df, fragment_inten_df : pd.DataFrame
        The library's fragment frames, indexed by the precursor's index range.

    top_n_inten : int
        Keep this many fragments per precursor.

    frag_type_head : str
        Output column name for the fragment series letter.

    frag_mass_head : str
        Output column name for the fragment m/z.

    frag_inten_head : str
        Output column name for the normalized fragment intensity.

    frag_charge_head : str
        Output column name for the fragment charge.

    frag_series_head : str
        Output column name for the fragment number within its series.

    frag_loss_head : str
        Output column name for the fragment loss type.

    verbose : bool
        Show a progress bar over the precursors.

    Returns
    -------
    pd.DataFrame
        One row per kept precursor/fragment pair.

    """
    df = precursor_df.copy()
    frag_columns = fragment_mz_df.columns.to_numpy().astype("U")
    frag_type_list = []
    frag_loss_list = []
    frag_charge_list = []
    frag_mass_list = []
    frag_inten_list = []
    frag_num_list = []
    iters = enumerate(
        df[[LibPsmDfCols.FRAG_START_IDX, LibPsmDfCols.FRAG_STOP_IDX]].to_numpy()
    )
    if verbose:
        iters = tqdm.tqdm(iters)
    for _i, (start, end) in iters:
        intens = fragment_inten_df.iloc[start:end, :].to_numpy(copy=True)
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

        frag_type_list.append(frag_types)
        frag_loss_list.append(loss_types)
        frag_charge_list.append(charges)
        frag_mass_list.append(masses[idx_in_df])
        frag_inten_list.append(intens[idx_in_df])
        frag_num_list.append(_get_frag_num(columns, rows, frag_len))

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
