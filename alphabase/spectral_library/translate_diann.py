"""Translate AlphaBase spectral libraries to DIA-NN 1.9.1+ parquet format.

This reuses shared export helpers from
:mod:`alphabase.spectral_library.translate`; it is kept as a separate module in
preparation for a larger refactor of the library-export code.
"""

from typing import List, Optional, Union

import pandas as pd
import tqdm

from alphabase.psm_reader.keys import LibPsmDfCols, PsmDfCols
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.translate import (
    create_modified_sequence,
    mask_fragment_intensity_by_frag_nAA,
    mask_fragment_intensity_by_mz_,
    merge_precursor_fragment_df,
    mod_to_unimod_dict,
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


def _first_present(
    precursor_df: pd.DataFrame,
    candidates: List[str],
    default: Union[str, float, None] = None,
) -> Union[pd.Series, str, float, None]:
    """Return the first present candidate column of `precursor_df`, else `default`."""
    for col in candidates:
        if col in precursor_df.columns:
            return precursor_df[col]
    return default


def speclib_to_diann_df(  # noqa: PLR0913, PLR0915
    speclib: SpecLibBase,
    *,
    translate_mod_dict: Optional[dict] = None,
    keep_k_highest_fragments: int = 12,
    min_frag_mz: float = 200,
    max_frag_mz: float = 2000,
    min_frag_intensity: float = 0.01,
    min_frag_nAA: int = 0,  # noqa: N803
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
    speclib : SpecLibBase
        The alphabase spectral library to convert.

    translate_mod_dict : dict
        Maps AlphaBase modification names to other software; defaults to UniMod ids.

    keep_k_highest_fragments : int
        Keep only the k most intense fragments per precursor. Default: 12

    min_frag_mz, max_frag_mz : float
        Fragment m/z range; fragments outside it are dropped. Set both to 0 to disable.

    min_frag_intensity : float
        Drop fragments whose relative intensity is at or below this value.

    min_frag_nAA : int
        Mask the smallest ``min_frag_nAA - 1`` b/y fragments per precursor; 0 disables.

    modloss : str
        Loss label written for modification-loss fragments. Default: "H3PO4"

    verbose : bool
        Show a progress bar while exploding fragments.

    Returns
    -------
    pd.DataFrame
        A long-format dataframe in the DIA-NN parquet library schema.

    """
    if translate_mod_dict is None:
        translate_mod_dict = mod_to_unimod_dict

    if PsmDfCols.PRECURSOR_MZ not in speclib.precursor_df.columns:
        speclib.calc_precursor_mz()
    precursor_df = speclib.precursor_df

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
            speclib.fragment_mz_df,
            speclib.fragment_intensity_df,
            min_frag_mz,
            max_frag_mz,
        )
    if min_frag_nAA > 0:
        mask_fragment_intensity_by_frag_nAA(
            speclib.fragment_intensity_df,
            speclib.precursor_df,
            max_mask_frag_nAA=min_frag_nAA - 1,
        )

    df = merge_precursor_fragment_df(
        df,
        speclib.fragment_mz_df,
        speclib.fragment_intensity_df,
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


def translate_to_parquet(  # noqa: PLR0913
    speclib: SpecLibBase,
    parquet_path: str,
    *,
    keep_k_highest_fragments: int = 12,
    min_frag_mz: float = 200,
    max_frag_mz: float = 2000,
    min_frag_intensity: float = 0.01,
    min_frag_nAA: int = 0,  # noqa: N803
    batch_size: int = 100000,
    translate_mod_dict: Optional[dict] = None,
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

    keep_k_highest_fragments : int
        Keep only the k most intense fragments per precursor. Default: 12

    min_frag_mz, max_frag_mz : float
        Fragment m/z range; fragments outside it are dropped. Set both to 0 to disable.

    min_frag_intensity : float
        Drop fragments whose relative intensity is at or below this value.

    min_frag_nAA : int
        Mask the smallest ``min_frag_nAA - 1`` b/y fragments per precursor; 0 disables.

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
            speclib.fragment_mz_df,
            speclib.fragment_intensity_df,
            min_frag_mz,
            max_frag_mz,
        )
    if min_frag_nAA > 0:
        mask_fragment_intensity_by_frag_nAA(
            speclib.fragment_intensity_df,
            speclib.precursor_df,
            max_mask_frag_nAA=min_frag_nAA - 1,
        )

    # process precursors in batches: the flat (one row per fragment) format is much larger
    # than the compact library, so batching keeps peak memory bounded for large libraries.
    # SpecLibBase has no public setters for the fragment frames, and its precursor_df setter
    # would refine/reorder the batch, so the private frames are assigned directly here.
    batch_speclib = SpecLibBase()
    batch_speclib._fragment_intensity_df = speclib.fragment_intensity_df  # noqa: SLF001
    batch_speclib._fragment_mz_df = speclib.fragment_mz_df  # noqa: SLF001
    precursor_df = speclib.precursor_df

    writer = pq.ParquetWriter(parquet_path, schema)
    try:
        for i in tqdm.tqdm(range(0, len(precursor_df), batch_size)):
            batch_speclib._precursor_df = precursor_df.iloc[i : i + batch_size]  # noqa: SLF001
            df = speclib_to_diann_df(
                batch_speclib,
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
