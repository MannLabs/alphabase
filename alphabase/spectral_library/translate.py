import multiprocessing as mp
import typing
from typing import List

import numba
import numpy as np
import pandas as pd
import tqdm

from alphabase.constants.modification import MOD_DF, ModificationKeys
from alphabase.psm_reader.keys import LibPsmDfCols, PsmDfCols
from alphabase.spectral_library.base import SpecLibBase
from alphabase.utils import explode_multiple_columns


# @numba.njit #(cannot use numba for pd.Series)
def create_modified_sequence(
    seq_mods_sites: typing.Tuple,  # must be ('sequence','mods','mod_sites')
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


@numba.njit
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
    df = pd.DataFrame()
    df["ModifiedPeptide"] = speclib._precursor_df[
        ["sequence", "mods", "mod_sites"]
    ].apply(
        create_modified_sequence,
        axis=1,
        translate_mod_dict=translate_mod_dict,
        mod_sep="[]",
    )

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


# fragment column names passed to `merge_precursor_fragment_df`
DIANN_PARQUET_FRAG_HEADS = {
    "frag_type_head": "Fragment.Type",
    "frag_mass_head": "Product.Mz",
    "frag_inten_head": "Relative.Intensity",
    "frag_charge_head": "Fragment.Charge",
    "frag_series_head": "Fragment.Series.Number",
    "frag_loss_head": "Fragment.Loss.Type",
}

# dtype tokens for DIANN_PARQUET_SCHEMA (INT64 / FLOAT=float32 / str)
DIANN_DTYPE_INT = "int"
DIANN_DTYPE_FLOAT = "float"
DIANN_DTYPE_STR = "str"

# DIA-NN 1.9.1+ `.parquet` library schema as ordered `(column, dtype)` pairs. Drives column
# order, dtype casting and the pyarrow schema. `Signature` is omitted, as DIA-NN requires
# for third-party libraries.
DIANN_PARQUET_SCHEMA = [
    ("Precursor.Id", DIANN_DTYPE_STR),
    ("Modified.Sequence", DIANN_DTYPE_STR),
    ("Stripped.Sequence", DIANN_DTYPE_STR),
    ("Precursor.Charge", DIANN_DTYPE_INT),
    ("Proteotypic", DIANN_DTYPE_INT),
    ("Decoy", DIANN_DTYPE_INT),
    ("N.Term", DIANN_DTYPE_INT),
    ("C.Term", DIANN_DTYPE_INT),
    ("RT", DIANN_DTYPE_FLOAT),
    ("IM", DIANN_DTYPE_FLOAT),
    ("Q.Value", DIANN_DTYPE_FLOAT),
    ("Peptidoform.Q.Value", DIANN_DTYPE_FLOAT),
    ("PTM.Site.Confidence", DIANN_DTYPE_FLOAT),
    ("PG.Q.Value", DIANN_DTYPE_FLOAT),
    ("Precursor.Mz", DIANN_DTYPE_FLOAT),
    ("Product.Mz", DIANN_DTYPE_FLOAT),
    ("Relative.Intensity", DIANN_DTYPE_FLOAT),
    ("Fragment.Type", DIANN_DTYPE_STR),
    ("Fragment.Charge", DIANN_DTYPE_INT),
    ("Fragment.Series.Number", DIANN_DTYPE_INT),
    ("Fragment.Loss.Type", DIANN_DTYPE_STR),
    ("Fragment.Score", DIANN_DTYPE_FLOAT),
    ("Exclude.From.Quant", DIANN_DTYPE_INT),
    ("Protein.Ids", DIANN_DTYPE_STR),
    ("Protein.Group", DIANN_DTYPE_STR),
    ("Protein.Names", DIANN_DTYPE_STR),
    ("Genes", DIANN_DTYPE_STR),
    ("Flags", DIANN_DTYPE_INT),
    ("Source.Id", DIANN_DTYPE_STR),
]
DIANN_PARQUET_COLUMN_ORDER = [name for name, _ in DIANN_PARQUET_SCHEMA]
_DIANN_TO_PANDAS_DTYPE = {
    DIANN_DTYPE_INT: "int64",
    DIANN_DTYPE_FLOAT: "float32",
    DIANN_DTYPE_STR: "str",
}

# `Flags` bitfield: bit 0 on every fragment, bit 4 on each precursor's base peak.
_DIANN_FLAG_BASE = 1 << 0
_DIANN_FLAG_FIRST_FRAGMENT = 1 << 4


def _first_present(precursor_df: pd.DataFrame, candidates: List[str], default=None):
    """Return the first present candidate column of `precursor_df`, else `default`."""
    for col in candidates:
        if col in precursor_df.columns:
            return precursor_df[col]
    return default


def speclib_to_diann_df(
    speclib: SpecLibBase,
    *,
    translate_mod_dict: dict = None,
    keep_k_highest_fragments: int = 12,
    min_frag_mz: float = 200,
    max_frag_mz: float = 2000,
    min_frag_intensity: float = 0.01,
    min_frag_nAA: int = 0,
    modloss: str = "H3PO4",
    verbose: bool = True,
) -> pd.DataFrame:
    """Convert an alphabase library to a DIA-NN 1.9.1+ parquet-format dataframe.

    Emits DIA-NN's report-style dot-notation columns (see ``DIANN_PARQUET_SCHEMA``) and
    ``(UniMod:N)`` modified sequences, importable by DIA-NN 1.9.1+ and readable back with
    :class:`alphabase.spectral_library.reader.LibraryReaderBase`. Columns a ``SpecLibBase``
    has no value for are filled with defaults matching a DIA-NN predicted library (q-values
    and scores 0, ``PTM.Site.Confidence`` 1, ``Source.Id`` empty).

    ``N.Term``/``C.Term`` are protein-terminus flags taken from ``is_prot_nterm``/
    ``is_prot_cterm`` (alphabase FASTA digestion) if present, else 0. ``Signature`` is not
    written, as DIA-NN requires for third-party libraries.

    Parameters
    ----------
    translate_mod_dict : dict
        Maps AlphaBase modification names to other software; defaults to UniMod ids.

    keep_k_highest_fragments : int
        Keep only the k most intense fragments per precursor. Default: 12

    Returns
    -------
    pd.DataFrame
        A long-format dataframe in the DIA-NN parquet library schema.
    """
    if translate_mod_dict is None:
        translate_mod_dict = mod_to_unimod_dict

    if PsmDfCols.PRECURSOR_MZ not in speclib._precursor_df.columns:
        speclib.calc_precursor_mz()
    precursor_df = speclib._precursor_df

    df = pd.DataFrame(index=precursor_df.index)

    df["Modified.Sequence"] = precursor_df[
        [PsmDfCols.SEQUENCE, PsmDfCols.MODS, PsmDfCols.MOD_SITES]
    ].apply(
        create_modified_sequence,
        axis=1,
        translate_mod_dict=translate_mod_dict,
        mod_sep="()",
        nterm="",
        cterm="",
    )
    df["Stripped.Sequence"] = precursor_df[PsmDfCols.SEQUENCE]
    df["Precursor.Charge"] = precursor_df[PsmDfCols.CHARGE]
    df["Precursor.Id"] = df["Modified.Sequence"] + df["Precursor.Charge"].astype(str)
    df["Precursor.Mz"] = precursor_df[PsmDfCols.PRECURSOR_MZ]

    rt = _first_present(
        precursor_df, ["irt_pred", "rt_pred", PsmDfCols.RT, "irt", PsmDfCols.RT_NORM]
    )
    if rt is None:
        raise ValueError("precursor_df must contain a retention time column")
    df["RT"] = rt
    df["IM"] = _first_present(precursor_df, ["mobility_pred", PsmDfCols.MOBILITY], 0.0)

    df["Protein.Group"] = _first_present(
        precursor_df, [PsmDfCols.PROTEINS, PsmDfCols.UNIPROT_IDS], ""
    )
    df["Protein.Ids"] = _first_present(
        precursor_df, [PsmDfCols.UNIPROT_IDS, PsmDfCols.PROTEINS], ""
    )
    df["Protein.Names"] = _first_present(precursor_df, ["protein_names"], "")
    df["Genes"] = _first_present(precursor_df, [PsmDfCols.GENES], "")
    df["Decoy"] = _first_present(precursor_df, [PsmDfCols.DECOY], 0)

    # N.Term/C.Term mark peptides at the protein N-/C-terminus (from FASTA digestion)
    df["N.Term"] = _first_present(precursor_df, ["is_prot_nterm"], 0)
    df["C.Term"] = _first_present(precursor_df, ["is_prot_cterm"], 0)

    # proteotypic unless the peptide maps to multiple (';'-joined) proteins
    df["Proteotypic"] = (
        ~df["Protein.Ids"].astype(str).str.contains(";", regex=False)
    ).astype("int64")

    # constant defaults for columns a SpecLibBase has no value for
    df["Q.Value"] = 0.0
    df["Peptidoform.Q.Value"] = 0.0
    df["PTM.Site.Confidence"] = 1.0
    df["PG.Q.Value"] = 0.0
    df["Fragment.Score"] = 0.0
    df["Exclude.From.Quant"] = 0
    df["Source.Id"] = ""

    df[LibPsmDfCols.FRAG_START_IDX] = precursor_df[LibPsmDfCols.FRAG_START_IDX]
    df[LibPsmDfCols.FRAG_STOP_IDX] = precursor_df[LibPsmDfCols.FRAG_STOP_IDX]

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
        verbose=verbose,
        **DIANN_PARQUET_FRAG_HEADS,
    )
    df = df[df["Relative.Intensity"] > min_frag_intensity]
    df.loc[df["Fragment.Loss.Type"] == "modloss", "Fragment.Loss.Type"] = modloss
    df = df.reset_index(drop=True)

    # Flags: base bit on all fragments, base-peak bit on each precursor's most intense one
    df["Flags"] = _DIANN_FLAG_BASE
    if len(df):
        base_peak_idx = df.groupby("Precursor.Id", sort=False)[
            "Relative.Intensity"
        ].idxmax()
        df.loc[base_peak_idx, "Flags"] |= _DIANN_FLAG_FIRST_FRAGMENT

    df = df.drop([LibPsmDfCols.FRAG_START_IDX, LibPsmDfCols.FRAG_STOP_IDX], axis=1)

    for name, dtype in DIANN_PARQUET_SCHEMA:
        if dtype == "str":
            df[name] = df[name].fillna("").astype(str)
        else:
            df[name] = df[name].astype(_DIANN_TO_PANDAS_DTYPE[dtype])
    return df[DIANN_PARQUET_COLUMN_ORDER]


def translate_to_parquet(
    speclib: SpecLibBase,
    parquet_path: str,
    *,
    keep_k_highest_fragments: int = 12,
    min_frag_mz: float = 200,
    max_frag_mz: float = 2000,
    min_frag_intensity: float = 0.01,
    min_frag_nAA: int = 0,
    batch_size: int = 100000,
    translate_mod_dict: dict = None,
) -> None:
    """Translate an alphabase library into a DIA-NN 1.9.1+ parquet spectral library.

    The written parquet uses DIA-NN's report-style column schema (see
    :func:`speclib_to_diann_df`) and can be imported by DIA-NN or read back with
    :class:`alphabase.spectral_library.reader.LibraryReaderBase`.

    Precursors are processed in batches and streamed to a single parquet file, so large
    libraries do not need to be held in memory at once.

    Parameters
    ----------
    speclib : SpecLibBase
        The alphabase spectral library to translate.

    parquet_path : str
        Path of the parquet file to write.

    batch_size : int
        Number of precursors to convert per batch. Default: 100000

    translate_mod_dict : dict
        A dict to map AlphaBase modification names to other software.
        Defaults to None, which uses UniMod ids, matching DIA-NN.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrow_type = {
        DIANN_DTYPE_INT: pa.int64,
        DIANN_DTYPE_FLOAT: pa.float32,
        DIANN_DTYPE_STR: pa.string,
    }
    schema = pa.schema(
        [(name, arrow_type[dtype]()) for name, dtype in DIANN_PARQUET_SCHEMA]
    )

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

    _speclib = SpecLibBase()
    _speclib._fragment_intensity_df = speclib._fragment_intensity_df
    _speclib._fragment_mz_df = speclib._fragment_mz_df
    precursor_df = speclib._precursor_df

    writer = pq.ParquetWriter(parquet_path, schema)
    try:
        for i in tqdm.tqdm(range(0, len(precursor_df), batch_size)):
            _speclib._precursor_df = precursor_df.iloc[i : i + batch_size]
            df = speclib_to_diann_df(
                _speclib,
                translate_mod_dict=translate_mod_dict,
                keep_k_highest_fragments=keep_k_highest_fragments,
                min_frag_mz=0,
                max_frag_mz=0,
                min_frag_intensity=min_frag_intensity,
                min_frag_nAA=0,
                verbose=False,
            )
            table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            writer.write_table(table)
    finally:
        writer.close()


class WritingProcess(mp.Process):
    def __init__(self, task_queue, tsv, *args, **kwargs):
        self.task_queue: mp.Queue = task_queue
        self.tsv = tsv
        super().__init__(*args, **kwargs)

    def run(self):
        while True:
            df, batch = self.task_queue.get()
            if df is None:
                break
            if tuple([int(i) for i in pd.__version__.split(".")[:2]]) >= (1, 5):
                newline = dict(lineterminator="\n")
            else:
                newline = dict(line_terminator="\n")
            df.to_csv(
                self.tsv,
                header=(batch == 0),
                sep="\t",
                mode="a",
                index=False,
                **newline,
            )


def translate_to_tsv(
    speclib: SpecLibBase,
    tsv: str,
    *,
    keep_k_highest_fragments: int = 12,
    min_frag_mz: float = 200,
    max_frag_mz: float = 2000,
    min_frag_intensity: float = 0.01,
    min_frag_nAA: int = 0,
    batch_size: int = 100000,
    translate_mod_dict: dict = None,
    multiprocessing: bool = True,
):
    if multiprocessing:
        queue_size = 1000000 // batch_size
        if queue_size < 2:
            queue_size = 2
        elif queue_size > 10:
            queue_size = 10
        df_head_queue = mp.Queue(maxsize=queue_size)
        writing_process = WritingProcess(df_head_queue, tsv)
        writing_process.start()
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
    if isinstance(tsv, str):
        with open(tsv, "w"):
            pass
    _speclib = SpecLibBase()
    _speclib._fragment_intensity_df = speclib._fragment_intensity_df
    _speclib._fragment_mz_df = speclib._fragment_mz_df
    precursor_df = speclib._precursor_df
    for i in tqdm.tqdm(range(0, len(precursor_df), batch_size)):
        _speclib._precursor_df = precursor_df.iloc[i : i + batch_size]
        df = speclib_to_single_df(
            _speclib,
            translate_mod_dict=translate_mod_dict,
            keep_k_highest_fragments=keep_k_highest_fragments,
            min_frag_mz=0,
            max_frag_mz=0,
            min_frag_intensity=min_frag_intensity,
            min_frag_nAA=0,
            verbose=False,
        )
        if multiprocessing:
            df_head_queue.put((df, i))
        else:
            if tuple([int(i) for i in pd.__version__.split(".")[:2]]) >= (1, 5):
                newline = dict(lineterminator="\n")
            else:
                newline = dict(line_terminator="\n")
            df.to_csv(tsv, header=(i == 0), sep="\t", mode="a", index=False, **newline)
    if multiprocessing:
        df_head_queue.put((None, None))
        print(
            "Translation finished, it will take several minutes to export the rest precursors to the tsv file..."
        )
        writing_process.join()
