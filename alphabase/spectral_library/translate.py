import contextlib
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import tqdm
from pandas.io.common import _is_binary_mode, get_handle

from alphabase.constants.modification import MOD_DF, ModificationKeys
from alphabase.spectral_library.base import SpecLibBase
from alphabase.utils import explode_multiple_columns


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


def _get_frag_num(columns, rows, frag_len):
    frag_nums = []
    for r, c in zip(rows, columns):
        if is_nterm_frag(c):
            frag_nums.append(r + 1)
        else:
            frag_nums.append(frag_len - r)
    return frag_nums


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
    df = precursor_df.copy()
    frag_columns = fragment_mz_df.columns.values.astype("U")
    frag_type_list = []
    frag_loss_list = []
    frag_charge_list = []
    frag_mass_list = []
    frag_inten_list = []
    frag_num_list = []
    iters = enumerate(df[["frag_start_idx", "frag_stop_idx"]].values)
    if verbose:
        iters = tqdm.tqdm(iters)
    for _i, (start, end) in iters:
        intens = fragment_inten_df.iloc[start:end, :].to_numpy(
            copy=True
        )  # is loc[start:end-1,:] faster?
        max_inten = np.amax(intens)
        if max_inten > 0:
            intens /= max_inten
        masses = fragment_mz_df.iloc[start:end, :].values
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

    # try:
    #     return df.explode([
    #         frag_type_head,
    #         frag_mass_head,
    #         frag_inten_head,
    #         frag_charge_head,
    #         frag_loss_head,
    #         frag_num_head
    #     ])
    # except ValueError:
    #     # df.explode does not allow mulitple columns before pandas version 1.x.x.
    #     df = df.explode(frag_type_head)

    #     df[frag_mass_head] = _flatten(frag_mass_list)
    #     df[frag_inten_head] = _flatten(frag_inten_list)
    #     df[frag_charge_head] = _flatten(frag_charge_list)
    #     df[frag_loss_head] = _flatten(frag_loss_list)
    #     df[frag_num_head] = _flatten(frag_num_list)
    #     return df


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


def _precursor_table(speclib: SpecLibBase, modified_peptides):
    df = pd.DataFrame(index=speclib._precursor_df.index)
    df["ModifiedPeptide"] = modified_peptides
    df["frag_start_idx"] = speclib._precursor_df["frag_start_idx"]
    df["frag_stop_idx"] = speclib._precursor_df["frag_stop_idx"]

    df["PrecursorCharge"] = speclib._precursor_df["charge"]

    for rt_col in ["irt_pred", "rt_pred", "rt", "irt", "rt_norm"]:
        if rt_col in speclib.precursor_df.columns:
            df["RT"] = speclib.precursor_df[rt_col]
            break
    if "RT" not in df.columns:
        raise ValueError("precursor_df must contain the RT columns")

    for im_col in ["mobility_pred", "mobility"]:
        if im_col in speclib.precursor_df.columns:
            df["IonMobility"] = speclib.precursor_df[im_col]
            break

    for ccs_col in ["ccs_pred", "ccs"]:
        if ccs_col in speclib.precursor_df.columns:
            df["CCS"] = speclib.precursor_df[ccs_col]
            break

    # df['LabelModifiedSequence'] = df['ModifiedPeptide']
    df["StrippedPeptide"] = speclib.precursor_df["sequence"]

    if "precursor_mz" not in speclib._precursor_df.columns:
        speclib.calc_precursor_mz()
    df["PrecursorMz"] = speclib._precursor_df["precursor_mz"]

    for prot_col in ["uniprot_ids", "proteins"]:
        if prot_col in speclib.precursor_df.columns:
            df["ProteinID"] = speclib.precursor_df[prot_col]
            break

    if "genes" in speclib._precursor_df.columns:
        df["Genes"] = speclib._precursor_df["genes"]

    if "decoy" in speclib._precursor_df.columns:
        df["Decoy"] = speclib._precursor_df["decoy"]

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
    modified_peptides = speclib._precursor_df[["sequence", "mods", "mod_sites"]].apply(
        create_modified_sequence,
        axis=1,
        translate_mod_dict=translate_mod_dict,
        mod_sep="[]",
    )
    df = _precursor_table(speclib, modified_peptides)
    if min_frag_mz > 0 or max_frag_mz > 0:
        mask_fragment_intensity_by_mz_(
            speclib._fragment_mz_df,
            speclib._fragment_intensity_df,
            min_frag_mz,
            max_frag_mz,
        )

    if min_frag_nAA > 0:
        mask_fragment_intensity_by_frag_nAA(
            speclib._fragment_intensity_df,
            speclib._precursor_df,
            max_mask_frag_nAA=min_frag_nAA - 1,
        )

    df = merge_precursor_fragment_df(
        df,
        speclib._fragment_mz_df,
        speclib._fragment_intensity_df,
        top_n_inten=keep_k_highest_fragments,
        frag_type_head=frag_type_head,
        frag_mass_head=frag_mass_head,
        frag_inten_head=frag_inten_head,
        frag_charge_head=frag_charge_head,
        frag_loss_head=frag_loss_head,
        frag_series_head=frag_series_head,
        verbose=verbose,
    )
    df = df[df["RelativeIntensity"] > min_frag_intensity]
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


def _top_fragments(
    frag_start: np.ndarray,
    frag_stop: np.ndarray,
    fragment_mz: np.ndarray,
    fragment_intensity: np.ndarray,
    frag_columns: list,
    *,
    top_k: int,
    min_frag_intensity: float,
) -> tuple[np.ndarray, ...]:
    lengths = frag_stop - frag_start
    n_columns = len(frag_columns)
    max_cells = lengths.max() * n_columns
    # The original [-top_k:] slice takes all fragments when top_k is zero.
    max_selected = len(range(max_cells)[-top_k:])
    cells = np.full((len(lengths), max_selected), -1, dtype=np.intp)
    intensities = np.zeros(cells.shape, dtype=fragment_intensity.dtype)

    # Sort unpadded rows so ties select the same ions as a per-precursor argsort.
    for length in np.unique(lengths):
        precursors = np.flatnonzero(lengths == length)
        starts = frag_start[precursors, None]
        rows = starts + np.arange(length)
        values = fragment_intensity[rows].reshape(len(precursors), -1)
        base_peak = values.max(axis=1, keepdims=True)
        np.divide(values, base_peak, out=values, where=base_peak > 0)
        order = np.argsort(values, axis=1)[:, -top_k:][:, ::-1]
        cells[precursors, : order.shape[1]] = starts * n_columns + order
        intensities[precursors, : order.shape[1]] = np.take_along_axis(
            values, order, axis=1
        )

    picked = (cells >= 0) & (intensities > min_frag_intensity)
    precursor = np.nonzero(picked)[0]
    row, column = np.divmod(cells[picked], n_columns)
    n_terminal = np.array([is_nterm_frag(name) for name in frag_columns])
    number = np.where(
        n_terminal[column], row - frag_start[precursor] + 1, frag_stop[precursor] - row
    )
    return precursor, column, number, fragment_mz[row, column], intensities[picked]


def _numeric_text(values) -> pa.Array:
    array = pa.array(values, from_pandas=True)
    return pc.fill_null(pc.cast(array, pa.string()), "")


def _column_text(values: pd.Series) -> pa.Array:
    if values.dtype.kind in "fiu":
        return _numeric_text(values)
    text = pa.array(np.where(values.isna(), "", values.astype(str)), pa.string())
    needs_quotes = pc.match_substring_regex(text, '[\t\n"]')
    if pc.any(needs_quotes).as_py():
        quoted = pc.replace_substring(text, '"', '""')
        quoted = pc.binary_join_element_wise('"', quoted, '"', "")
        text = pc.if_else(needs_quotes, quoted, text)
    return text


def _tsv_text(
    table: pd.DataFrame,
    fragment_mz: np.ndarray,
    fragment_intensity: np.ndarray,
    frag_columns: list,
    *,
    with_header: bool,
    top_k: int,
    min_frag_intensity: float,
) -> memoryview:
    precursor, column, number, mz, intensity = _top_fragments(
        table.pop("frag_start_idx").to_numpy(),
        table.pop("frag_stop_idx").to_numpy(),
        fragment_mz,
        fragment_intensity,
        frag_columns,
        top_k=top_k,
        min_frag_intensity=min_frag_intensity,
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
    repeated_bytes = pc.binary_length(precursor_text).to_numpy()[precursor]
    if repeated_bytes.sum(dtype=np.int64) >= 2**31:
        raise ValueError("batch_size is too large for one tsv batch")
    rows = pc.binary_join_element_wise(
        precursor_text.take(precursor),
        pa.array(frag_types).take(column),
        _numeric_text(mz),
        _numeric_text(intensity),
        pa.array(charges).take(column),
        pc.cast(pa.array(number), pa.string()),
        pa.array(row_ends).take(column),
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
    mask_fragment_intensity_by_mz_(
        speclib._fragment_mz_df,
        speclib._fragment_intensity_df,
        min_frag_mz,
        max_frag_mz,
    )
    if min_frag_nAA > 0:
        mask_fragment_intensity_by_frag_nAA(
            speclib._fragment_intensity_df,
            speclib._precursor_df,
            max_mask_frag_nAA=min_frag_nAA - 1,
        )
    precursor_df = speclib._precursor_df
    frag_columns = list(speclib._fragment_mz_df.columns)
    fragment_mz = speclib._fragment_mz_df.to_numpy()
    fragment_intensity = speclib._fragment_intensity_df.to_numpy()

    def render(batch_start: int) -> memoryview:
        batch = precursor_df.iloc[batch_start : batch_start + batch_size]
        batch_speclib = SpecLibBase()
        batch_speclib._precursor_df = batch
        modified_peptides = [
            create_modified_sequence(values, translate_mod_dict)
            for values in zip(batch["sequence"], batch["mods"], batch["mod_sites"])
        ]
        return _tsv_text(
            _precursor_table(batch_speclib, modified_peptides),
            fragment_mz,
            fragment_intensity,
            frag_columns,
            with_header=batch_start == 0,
            top_k=keep_k_highest_fragments,
            min_frag_intensity=min_frag_intensity,
        )

    # More threads than this were not faster in measurements.
    max_threads = 4
    n_threads = min(os.cpu_count(), max_threads) if multiprocessing else 1
    batch_starts = range(0, len(precursor_df), batch_size)
    if isinstance(tsv, str):
        with open(tsv, "w"):
            pass
    if not batch_starts:
        return
    with contextlib.ExitStack() as stack:
        out = tsv
        if isinstance(tsv, (str, os.PathLike)):
            out = stack.enter_context(
                get_handle(tsv, "ab", compression="infer", is_text=False)
            ).handle
        is_text = not _is_binary_mode(out, "")
        pool = stack.enter_context(ThreadPoolExecutor(n_threads))
        # Rendering one wave of batches at a time bounds the text held in memory
        # while the batches are still written in order.
        for wave in tqdm.tqdm(range(0, len(batch_starts), n_threads)):
            for text in pool.map(render, batch_starts[wave : wave + n_threads]):
                out.write(str(text, "utf-8") if is_text else text)
