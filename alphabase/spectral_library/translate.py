import contextlib
import os
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import tqdm
from pandas.io.common import _is_binary_mode, get_handle

from alphabase.constants.modification import MOD_DF, ModificationKeys
from alphabase.spectral_library.base import SpecLibBase


# @numba.njit #(cannot use numba for pd.Series)
def create_modified_sequence(
    seq_mods_sites: tuple,  # must be ('sequence','mods','mod_sites')
    translate_mod_dict: dict = None,
    mod_sep="[]",
    nterm="_",
    cterm="_",
):
    """
    Translate `(sequence, mods, mod_sites)` into a modified sequence. Used by `df.apply()`.
    For example, `('ABCDEFG','Mod1@A;Mod2@E','1;5')`->`_A[Mod1@A]BCDE[Mod2@E]FG_`.

    Parameters
    ----------
    seq_mods_sites : List
        must be `(sequence, mods, mod_sites)`

    translate_mod_dict : dict
        A dict to map AlphaX modification names to other software,
        use unimod name if None.
        Defaults to None.

    mod_sep : str
        '[]' or '()', default '[]'

    """
    mod_seq, mods, mod_sites = seq_mods_sites
    if mods:
        mods = mods.split(ModificationKeys.SEPARATOR)
        mod_sites = [int(i) for i in mod_sites.split(ModificationKeys.SEPARATOR)]
        rev_order = sorted(range(len(mod_sites)), key=mod_sites.__getitem__)[::-1]
        mod_sites = [mod_sites[i] for i in rev_order]
        mods = [mods[i] for i in rev_order]
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


def _get_frag_info_from_column_name(column: str):
    """
    Only used when converting alphabase libraries into other libraries
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


class _Fragments(NamedTuple):
    precursor: np.ndarray  # position of the precursor in the set
    column: np.ndarray  # position of the fragment type column
    number: np.ndarray  # fragment series number
    mz: np.ndarray
    intensity: np.ndarray  # relative to the base peak of the precursor


def _top_fragments(
    frag_start: np.ndarray,
    frag_stop: np.ndarray,
    fragment_mz: np.ndarray,
    fragment_intensity: np.ndarray,
    frag_columns: list,
    *,
    top_k: int,
    min_frag_mz: float = -np.inf,
    max_frag_mz: float = np.inf,
    min_frag_intensity: float = -np.inf,
    min_frag_nAA: int = 0,
) -> _Fragments:
    if len(frag_start) == 0:
        return _Fragments(
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=fragment_mz.dtype),
            np.empty(0, dtype=fragment_intensity.dtype),
        )

    # Lay the fragments out as a block of precursor x position x column, padded
    # to the longest peptide, so that every step below is an operation on an axis.
    lengths = frag_stop - frag_start
    position = np.arange(lengths.max(initial=0))
    in_peptide = position < lengths[:, None]
    rows = (frag_start[:, None] + position)[in_peptide]
    shape = (*in_peptide.shape, len(frag_columns))
    mz = np.zeros(shape, fragment_mz.dtype)
    mz[in_peptide] = fragment_mz[rows]
    intensity = np.zeros(shape, fragment_intensity.dtype)
    intensity[in_peptide] = fragment_intensity[rows]
    number = np.where(
        [is_nterm_frag(column) for column in frag_columns],
        position[:, None] + 1,
        lengths[:, None, None] - position[:, None],
    )

    # Excluded fragments must not set the base peak either.
    intensity[(mz > max_frag_mz) | (mz < min_frag_mz) | (number < min_frag_nAA)] = 0
    base_peak = intensity.max(axis=(1, 2), initial=0)
    base_peak[~(base_peak > 0)] = 1
    intensity /= base_peak[:, None, None]
    intensity[~in_peptide] = -np.inf  # padding is never picked

    # Sorting ascending and reversing, as before, keeps the order of fragments
    # with equal intensity.
    intensity = intensity.reshape(len(lengths), -1)
    top = np.argsort(intensity, axis=1, kind="stable")[:, ::-1][:, :top_k]
    top_intensity = np.take_along_axis(intensity, top, axis=1)
    picked = top_intensity > min_frag_intensity
    precursor = np.nonzero(picked)[0]
    cell = top[picked]
    return _Fragments(
        precursor,
        cell % len(frag_columns),
        number.reshape(len(lengths), -1)[precursor, cell],
        mz.reshape(len(lengths), -1)[precursor, cell],
        top_intensity[picked],
    )


def _fragment_df(
    precursor_df: pd.DataFrame,
    fragment_mz_df: pd.DataFrame,
    fragment_inten_df: pd.DataFrame,
    heads: tuple,
    **fragment_filters,
) -> pd.DataFrame:
    frag_columns = list(fragment_mz_df.columns)
    frag_types, loss_types, charges = (
        np.array(values, dtype=object)
        for values in zip(*map(_get_frag_info_from_column_name, frag_columns))
    )
    frag_start = precursor_df["frag_start_idx"].to_numpy()
    frag_stop = precursor_df["frag_stop_idx"].to_numpy()
    fragment_mz = fragment_mz_df.to_numpy()
    fragment_intensity = fragment_inten_df.to_numpy()
    # Bound the padded selection arrays; retain only selected fragments per batch.
    batch_size = 10000
    batches = []
    for start in range(0, max(len(precursor_df), 1), batch_size):
        fragments = _top_fragments(
            frag_start[start : start + batch_size],
            frag_stop[start : start + batch_size],
            fragment_mz,
            fragment_intensity,
            frag_columns,
            **fragment_filters,
        )
        fragments.precursor[:] += start
        batches.append(fragments)
    fragments = _Fragments(*(np.concatenate(arrays) for arrays in zip(*batches)))
    type_head, mass_head, inten_head, charge_head, series_head, loss_head = heads
    df = precursor_df.iloc[fragments.precursor].copy()
    df[type_head] = frag_types[fragments.column]
    df[mass_head] = fragments.mz
    df[inten_head] = fragments.intensity
    df[charge_head] = charges[fragments.column]
    df[series_head] = fragments.number
    df[loss_head] = loss_types[fragments.column]
    return df


def merge_precursor_fragment_df(
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
    verbose=True,
):
    """
    Convert alphabase library into a single dataframe.
    This method is not important, as it will be only
    used by DiaNN, or spectronaut, or others
    """
    return _fragment_df(
        precursor_df,
        fragment_mz_df,
        fragment_inten_df,
        (
            frag_type_head,
            frag_mass_head,
            frag_inten_head,
            frag_charge_head,
            frag_series_head,
            frag_loss_head,
        ),
        top_k=top_n_inten,
    )


mod_to_unimod_dict = {}
for mod_name, unimod_id in MOD_DF[["mod_name", "unimod_id"]].values:
    if unimod_id == -1 or unimod_id == "-1":
        continue
    mod_to_unimod_dict[mod_name] = f"UniMod:{unimod_id}"


def is_nterm_frag(frag_type: str):
    return frag_type[0] in "abc"


def mask_fragment_intensity_by_mz_(
    fragment_mz_df: pd.DataFrame,
    fragment_intensity_df: pd.DataFrame,
    min_frag_mz,
    max_frag_mz,
):
    fragment_intensity_df.mask(
        (fragment_mz_df > max_frag_mz) | (fragment_mz_df < min_frag_mz), 0, inplace=True
    )


def mask_fragment_intensity_by_frag_nAA(
    fragment_intensity_df: pd.DataFrame, precursor_df: pd.DataFrame, max_mask_frag_nAA
):
    if max_mask_frag_nAA <= 0:
        return
    b_mask = np.zeros(len(fragment_intensity_df), dtype=np.bool_)
    y_mask = b_mask.copy()
    for i_frag in range(max_mask_frag_nAA):
        b_mask[precursor_df.frag_start_idx.values + i_frag] = True
        y_mask[precursor_df.frag_stop_idx.values - i_frag - 1] = True

    masks = np.zeros(
        (len(fragment_intensity_df), len(fragment_intensity_df.columns)), dtype=np.bool_
    )
    for i, col in enumerate(fragment_intensity_df.columns.values):
        if is_nterm_frag(col):
            masks[:, i] = b_mask
        else:
            masks[:, i] = y_mask

    fragment_intensity_df.mask(masks, 0, inplace=True)


def _precursor_table(precursor_df: pd.DataFrame, translate_mod_dict: dict = None):
    df = pd.DataFrame(index=precursor_df.index)
    df["ModifiedPeptide"] = [
        create_modified_sequence(
            values, translate_mod_dict=translate_mod_dict, mod_sep="[]"
        )
        for values in zip(
            precursor_df["sequence"], precursor_df["mods"], precursor_df["mod_sites"]
        )
    ]

    df["frag_start_idx"] = precursor_df["frag_start_idx"]
    df["frag_stop_idx"] = precursor_df["frag_stop_idx"]

    df["PrecursorCharge"] = precursor_df["charge"]

    for rt_col in ["irt_pred", "rt_pred", "rt", "irt", "rt_norm"]:
        if rt_col in precursor_df.columns:
            df["RT"] = precursor_df[rt_col]
            break
    if "RT" not in df.columns:
        raise ValueError("precursor_df must contain the RT columns")

    for im_col in ["mobility_pred", "mobility"]:
        if im_col in precursor_df.columns:
            df["IonMobility"] = precursor_df[im_col]
            break

    for ccs_col in ["ccs_pred", "ccs"]:
        if ccs_col in precursor_df.columns:
            df["CCS"] = precursor_df[ccs_col]
            break

    # df['LabelModifiedSequence'] = df['ModifiedPeptide']
    df["StrippedPeptide"] = precursor_df["sequence"]
    df["PrecursorMz"] = precursor_df["precursor_mz"]

    for prot_col in ["uniprot_ids", "proteins"]:
        if prot_col in precursor_df.columns:
            df["ProteinID"] = precursor_df[prot_col]
            break

    if "genes" in precursor_df.columns:
        df["Genes"] = precursor_df["genes"]

    if "decoy" in precursor_df.columns:
        df["Decoy"] = precursor_df["decoy"]

    # if 'protein_group' in speclib._precursor_df.columns:
    #     df['ProteinGroups'] = speclib._precursor_df['protein_group']

    return df


def speclib_to_single_df(
    speclib: SpecLibBase,
    *,
    translate_mod_dict: dict = None,
    keep_k_highest_fragments: int = 12,
    min_frag_mz=200,
    max_frag_mz=2000,
    min_frag_intensity=0.01,
    min_frag_nAA=0,
    modloss: str = "H3PO4",
    frag_type_head: str = "FragmentType",
    frag_mass_head: str = "FragmentMz",
    frag_inten_head: str = "RelativeIntensity",
    frag_charge_head: str = "FragmentCharge",
    frag_loss_head: str = "FragmentLossType",
    frag_series_head: str = "FragmentNumber",
    verbose=True,
) -> pd.DataFrame:
    """
    Convert alphabase library to diann (or Spectronaut) library dataframe
    This method is not important, as it will be only
    used by DiaNN, or spectronaut, or others

    Parameters
    ----------
    translate_mod_dict : dict
        A dict to map AlphaX modification names to other software,
        use unimod name if None.
        Defaults to None.

    keep_k_highest_peaks : int
        only keep highest fragments for each precursor. Default: 12

    Returns
    -------
    pd.DataFrame
        a single dataframe in the SWATH-like format

    """
    if "precursor_mz" not in speclib._precursor_df.columns:
        speclib.calc_precursor_mz()
    if not (min_frag_mz > 0 or max_frag_mz > 0):
        min_frag_mz, max_frag_mz = -np.inf, np.inf

    df = _fragment_df(
        _precursor_table(speclib._precursor_df, translate_mod_dict),
        speclib._fragment_mz_df,
        speclib._fragment_intensity_df,
        (
            frag_type_head,
            frag_mass_head,
            frag_inten_head,
            frag_charge_head,
            frag_series_head,
            frag_loss_head,
        ),
        top_k=keep_k_highest_fragments,
        min_frag_mz=min_frag_mz,
        max_frag_mz=max_frag_mz,
        min_frag_intensity=min_frag_intensity,
        min_frag_nAA=min_frag_nAA,
    )
    df.loc[df[frag_loss_head] == "modloss", frag_loss_head] = modloss

    return df.drop(["frag_start_idx", "frag_stop_idx"], axis=1)


def speclib_to_swath_df(
    speclib: SpecLibBase,
    *,
    keep_k_highest_fragments: int = 12,
    min_frag_mz=200,
    max_frag_mz=2000,
    min_frag_intensity=0.01,
) -> pd.DataFrame:
    speclib_to_single_df(
        speclib,
        translate_mod_dict=None,
        keep_k_highest_fragments=keep_k_highest_fragments,
        min_frag_mz=min_frag_mz,
        max_frag_mz=max_frag_mz,
        min_frag_intensity=min_frag_intensity,
    )


def _float_text(values: np.ndarray) -> pa.Array:
    array = pa.array(values)
    return pc.if_else(pc.is_nan(array), "", pc.cast(array, pa.string()))


def _column_text(values: pd.Series) -> pa.Array:
    if values.dtype.kind == "f":
        return _float_text(values.to_numpy())
    if values.dtype.kind in "iu":
        return pc.cast(pa.array(values.to_numpy()), pa.string())
    text = pa.array(np.where(values.isna(), "", values.astype(str)), pa.string())
    needs_quotes = pc.match_substring_regex(text, '[\t\n"]')
    if pc.any(needs_quotes).as_py():
        quoted = pc.replace_substring(text, '"', '""')
        quoted = pc.binary_join_element_wise('"', quoted, '"', "")
        text = pc.if_else(needs_quotes, quoted, text)
    return text


def _tsv_text(
    precursor_df: pd.DataFrame,
    fragment_mz: np.ndarray,
    fragment_intensity: np.ndarray,
    frag_columns: list,
    *,
    translate_mod_dict: dict,
    with_header: bool,
    **fragment_filters,
) -> memoryview:
    table = _precursor_table(precursor_df, translate_mod_dict)
    fragments = _top_fragments(
        table.pop("frag_start_idx").to_numpy(),
        table.pop("frag_stop_idx").to_numpy(),
        fragment_mz,
        fragment_intensity,
        frag_columns,
        **fragment_filters,
    )
    frag_types, loss_types, charges = zip(
        *map(_get_frag_info_from_column_name, frag_columns)
    )
    modloss = "H3PO4"
    row_ends = [(modloss if loss == "modloss" else loss) + "\n" for loss in loss_types]

    precursor_text = pc.binary_join_element_wise(
        *[_column_text(table[name]) for name in table.columns], "\t"
    )
    # Arrow holds at most 2 GiB per string array. The join below checks that,
    # but `take` silently overflows.
    repeated_bytes = pc.binary_length(precursor_text).to_numpy()[fragments.precursor]
    if repeated_bytes.sum(dtype=np.int64) >= 2**31:
        raise ValueError("batch_size is too large for one tsv batch")
    rows = pc.binary_join_element_wise(
        precursor_text.take(fragments.precursor),
        pa.array(frag_types).take(fragments.column),
        _float_text(fragments.mz),
        _float_text(fragments.intensity),
        pa.array(charges).take(fragments.column),
        pc.cast(pa.array(fragments.number), pa.string()),
        pa.array(row_ends).take(fragments.column),
        "\t",
    )
    if with_header:
        fragment_heads = (
            "FragmentType",
            "FragmentMz",
            "RelativeIntensity",
            "FragmentCharge",
            "FragmentNumber",
            "FragmentLossType",
        )
        header = "\t".join([*table.columns, *fragment_heads]) + "\n"
        rows = pa.concat_arrays([pa.array([header]), rows])
    # A string array stores its strings contiguously, so its data buffer is the tsv text.
    _, offsets, text = rows.buffers()
    text_end = np.frombuffer(offsets, dtype=np.int32)[rows.offset + len(rows)]
    return memoryview(text)[:text_end]


def translate_to_tsv(
    speclib: SpecLibBase,
    tsv: str,
    *,
    keep_k_highest_fragments: int = 12,
    min_frag_mz: float = 200,
    max_frag_mz: float = 2000,
    min_frag_intensity: float = 0.01,
    min_frag_nAA: int = 0,
    batch_size: int = 10000,
    translate_mod_dict: dict = None,
    multiprocessing: bool = True,
):
    if "precursor_mz" not in speclib._precursor_df.columns:
        speclib.calc_precursor_mz()
    precursor_df = speclib._precursor_df
    frag_columns = list(speclib._fragment_mz_df.columns)
    assert frag_columns == list(speclib._fragment_intensity_df.columns)
    fragment_mz = speclib._fragment_mz_df.to_numpy()
    fragment_intensity = speclib._fragment_intensity_df.to_numpy()
    assert fragment_mz.shape == fragment_intensity.shape

    def render(batch_start: int) -> memoryview:
        return _tsv_text(
            precursor_df.iloc[batch_start : batch_start + batch_size],
            fragment_mz,
            fragment_intensity,
            frag_columns,
            translate_mod_dict=translate_mod_dict,
            with_header=batch_start == 0,
            top_k=keep_k_highest_fragments,
            min_frag_mz=min_frag_mz,
            max_frag_mz=max_frag_mz,
            min_frag_intensity=min_frag_intensity,
            min_frag_nAA=min_frag_nAA,
        )

    # More threads than this were not faster in measurements.
    max_threads = 4
    n_threads = min(os.cpu_count(), max_threads) if multiprocessing else 1
    batch_starts = range(0, len(precursor_df), batch_size)
    with contextlib.ExitStack() as stack:
        out = tsv
        if isinstance(tsv, (str, os.PathLike)):
            out = stack.enter_context(
                get_handle(tsv, "wb", compression="infer", is_text=False)
            ).handle
        is_text = not _is_binary_mode(out, "")
        pool = stack.enter_context(ThreadPoolExecutor(n_threads))
        # Rendering one wave of batches at a time bounds the text held in memory
        # while the batches are still written in order.
        for wave in tqdm.tqdm(range(0, len(batch_starts), n_threads)):
            for text in pool.map(render, batch_starts[wave : wave + n_threads]):
                out.write(str(text, "utf-8") if is_text else text)
