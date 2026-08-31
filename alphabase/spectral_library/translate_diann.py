"""Translate AlphaBase spectral libraries to DIA-NN 1.9.1+ parquet format.

This reuses shared export helpers from
:mod:`alphabase.spectral_library.translate`; it is kept as a separate module in
preparation for a larger refactor of the library-export code.
"""

from typing import Optional, Union

import pandas as pd
import tqdm

from alphabase.psm_reader.keys import ConstantsClass, LibPsmDfCols, PsmDfCols
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.translate import (
    create_modified_sequence,
    mask_fragment_intensity_by_frag_nAA,
    mask_fragment_intensity_by_mz_,
    merge_precursor_fragment_df,
    mod_to_unimod_dict,
)


class DiannParquetCols(metaclass=ConstantsClass):
    """Column names of a DIA-NN 1.9.1+ `.parquet` spectral library.

    DIA-NN uses its report-style dot notation here. ``SIGNATURE`` is listed for
    completeness but is not written: DIA-NN requires third-party libraries to omit it.
    """

    PRECURSOR_ID = "Precursor.Id"
    MODIFIED_SEQUENCE = "Modified.Sequence"
    STRIPPED_SEQUENCE = "Stripped.Sequence"
    PRECURSOR_CHARGE = "Precursor.Charge"
    PROTEOTYPIC = "Proteotypic"
    DECOY = "Decoy"
    N_TERM = "N.Term"
    C_TERM = "C.Term"
    RT = "RT"
    IM = "IM"
    Q_VALUE = "Q.Value"
    PEPTIDOFORM_Q_VALUE = "Peptidoform.Q.Value"
    PTM_SITE_CONFIDENCE = "PTM.Site.Confidence"
    PG_Q_VALUE = "PG.Q.Value"
    PRECURSOR_MZ = "Precursor.Mz"
    PRODUCT_MZ = "Product.Mz"
    RELATIVE_INTENSITY = "Relative.Intensity"
    FRAGMENT_TYPE = "Fragment.Type"
    FRAGMENT_CHARGE = "Fragment.Charge"
    FRAGMENT_SERIES_NUMBER = "Fragment.Series.Number"
    FRAGMENT_LOSS_TYPE = "Fragment.Loss.Type"
    FRAGMENT_SCORE = "Fragment.Score"
    EXCLUDE_FROM_QUANT = "Exclude.From.Quant"
    PROTEIN_IDS = "Protein.Ids"
    PROTEIN_GROUP = "Protein.Group"
    PROTEIN_NAMES = "Protein.Names"
    GENES = "Genes"
    FLAGS = "Flags"
    SOURCE_ID = "Source.Id"
    SIGNATURE = "Signature"


# fragment column names passed to `merge_precursor_fragment_df`
DIANN_PARQUET_FRAG_HEADS = {
    "frag_type_head": DiannParquetCols.FRAGMENT_TYPE,
    "frag_mass_head": DiannParquetCols.PRODUCT_MZ,
    "frag_inten_head": DiannParquetCols.RELATIVE_INTENSITY,
    "frag_charge_head": DiannParquetCols.FRAGMENT_CHARGE,
    "frag_series_head": DiannParquetCols.FRAGMENT_SERIES_NUMBER,
    "frag_loss_head": DiannParquetCols.FRAGMENT_LOSS_TYPE,
}

# dtype tokens for DIANN_PARQUET_SCHEMA (INT64 / FLOAT=float32 / str)
DIANN_DTYPE_INT = "int"
DIANN_DTYPE_FLOAT = "float"
DIANN_DTYPE_STR = "str"

# DIA-NN 1.9.1+ `.parquet` library schema as ordered `(column, dtype)` pairs. Drives column
# order, dtype casting and the pyarrow schema. `Signature` is omitted, as DIA-NN requires
# for third-party libraries.
DIANN_PARQUET_SCHEMA = [
    (DiannParquetCols.PRECURSOR_ID, DIANN_DTYPE_STR),
    (DiannParquetCols.MODIFIED_SEQUENCE, DIANN_DTYPE_STR),
    (DiannParquetCols.STRIPPED_SEQUENCE, DIANN_DTYPE_STR),
    (DiannParquetCols.PRECURSOR_CHARGE, DIANN_DTYPE_INT),
    (DiannParquetCols.PROTEOTYPIC, DIANN_DTYPE_INT),
    (DiannParquetCols.DECOY, DIANN_DTYPE_INT),
    (DiannParquetCols.N_TERM, DIANN_DTYPE_INT),
    (DiannParquetCols.C_TERM, DIANN_DTYPE_INT),
    (DiannParquetCols.RT, DIANN_DTYPE_FLOAT),
    (DiannParquetCols.IM, DIANN_DTYPE_FLOAT),
    (DiannParquetCols.Q_VALUE, DIANN_DTYPE_FLOAT),
    (DiannParquetCols.PEPTIDOFORM_Q_VALUE, DIANN_DTYPE_FLOAT),
    (DiannParquetCols.PTM_SITE_CONFIDENCE, DIANN_DTYPE_FLOAT),
    (DiannParquetCols.PG_Q_VALUE, DIANN_DTYPE_FLOAT),
    (DiannParquetCols.PRECURSOR_MZ, DIANN_DTYPE_FLOAT),
    (DiannParquetCols.PRODUCT_MZ, DIANN_DTYPE_FLOAT),
    (DiannParquetCols.RELATIVE_INTENSITY, DIANN_DTYPE_FLOAT),
    (DiannParquetCols.FRAGMENT_TYPE, DIANN_DTYPE_STR),
    (DiannParquetCols.FRAGMENT_CHARGE, DIANN_DTYPE_INT),
    (DiannParquetCols.FRAGMENT_SERIES_NUMBER, DIANN_DTYPE_INT),
    (DiannParquetCols.FRAGMENT_LOSS_TYPE, DIANN_DTYPE_STR),
    (DiannParquetCols.FRAGMENT_SCORE, DIANN_DTYPE_FLOAT),
    (DiannParquetCols.EXCLUDE_FROM_QUANT, DIANN_DTYPE_INT),
    (DiannParquetCols.PROTEIN_IDS, DIANN_DTYPE_STR),
    (DiannParquetCols.PROTEIN_GROUP, DIANN_DTYPE_STR),
    (DiannParquetCols.PROTEIN_NAMES, DIANN_DTYPE_STR),
    (DiannParquetCols.GENES, DIANN_DTYPE_STR),
    (DiannParquetCols.FLAGS, DIANN_DTYPE_INT),
    (DiannParquetCols.SOURCE_ID, DIANN_DTYPE_STR),
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


def _get_first_present_column(
    precursor_df: pd.DataFrame,
    candidates: list[str],
    default: Union[str, float, None] = None,
) -> Union[pd.Series, str, float, None]:
    """Return the first present candidate column of `precursor_df`, else `default`."""
    for col in candidates:
        if col in precursor_df.columns:
            return precursor_df[col]
    return default


# TODO: go for an OOP approach: a writer class holding the export settings as state, with
# the precursor mapping / fragment explosion / dtype casting as methods.
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

    df[DiannParquetCols.MODIFIED_SEQUENCE] = precursor_df[
        [PsmDfCols.SEQUENCE, PsmDfCols.MODS, PsmDfCols.MOD_SITES]
    ].apply(
        create_modified_sequence,
        axis=1,
        translate_mod_dict=translate_mod_dict,
        mod_sep="()",
        nterm="",
        cterm="",
    )
    df[DiannParquetCols.STRIPPED_SEQUENCE] = precursor_df[PsmDfCols.SEQUENCE]
    df[DiannParquetCols.PRECURSOR_CHARGE] = precursor_df[PsmDfCols.CHARGE]
    df[DiannParquetCols.PRECURSOR_ID] = df[DiannParquetCols.MODIFIED_SEQUENCE] + df[
        DiannParquetCols.PRECURSOR_CHARGE
    ].astype(str)
    df[DiannParquetCols.PRECURSOR_MZ] = precursor_df[PsmDfCols.PRECURSOR_MZ]

    rt = _get_first_present_column(
        precursor_df, ["irt_pred", "rt_pred", PsmDfCols.RT, "irt", PsmDfCols.RT_NORM]
    )
    if rt is None:
        raise ValueError("precursor_df must contain a retention time column")
    df[DiannParquetCols.RT] = rt
    df[DiannParquetCols.IM] = _get_first_present_column(
        precursor_df, ["mobility_pred", PsmDfCols.MOBILITY], 0.0
    )

    df[DiannParquetCols.PROTEIN_GROUP] = _get_first_present_column(
        precursor_df, [PsmDfCols.PROTEINS, PsmDfCols.UNIPROT_IDS], ""
    )
    df[DiannParquetCols.PROTEIN_IDS] = _get_first_present_column(
        precursor_df, [PsmDfCols.UNIPROT_IDS, PsmDfCols.PROTEINS], ""
    )
    df[DiannParquetCols.PROTEIN_NAMES] = _get_first_present_column(
        precursor_df, ["protein_names"], ""
    )
    df[DiannParquetCols.GENES] = _get_first_present_column(
        precursor_df, [PsmDfCols.GENES], ""
    )
    df[DiannParquetCols.DECOY] = _get_first_present_column(
        precursor_df, [PsmDfCols.DECOY], 0
    )

    # N.Term/C.Term mark peptides at the protein N-/C-terminus (from FASTA digestion)
    df[DiannParquetCols.N_TERM] = _get_first_present_column(
        precursor_df, ["is_prot_nterm"], 0
    )
    df[DiannParquetCols.C_TERM] = _get_first_present_column(
        precursor_df, ["is_prot_cterm"], 0
    )

    # proteotypic unless the peptide maps to multiple (';'-joined) proteins
    df[DiannParquetCols.PROTEOTYPIC] = (
        ~df[DiannParquetCols.PROTEIN_IDS].astype(str).str.contains(";", regex=False)
    ).astype("int64")

    # constant defaults for columns a SpecLibBase has no value for
    df[DiannParquetCols.Q_VALUE] = 0.0
    df[DiannParquetCols.PEPTIDOFORM_Q_VALUE] = 0.0
    df[DiannParquetCols.PTM_SITE_CONFIDENCE] = 1.0
    df[DiannParquetCols.PG_Q_VALUE] = 0.0
    df[DiannParquetCols.FRAGMENT_SCORE] = 0.0
    df[DiannParquetCols.EXCLUDE_FROM_QUANT] = 0
    df[DiannParquetCols.SOURCE_ID] = ""

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
    df = df[df[DiannParquetCols.RELATIVE_INTENSITY] > min_frag_intensity]
    df.loc[
        df[DiannParquetCols.FRAGMENT_LOSS_TYPE] == "modloss",
        DiannParquetCols.FRAGMENT_LOSS_TYPE,
    ] = modloss
    df = df.reset_index(drop=True)

    # Flags: base bit on all fragments, base-peak bit on each precursor's most intense one
    df[DiannParquetCols.FLAGS] = _DIANN_FLAG_BASE
    if len(df):
        base_peak_idx = df.groupby(DiannParquetCols.PRECURSOR_ID, sort=False)[
            DiannParquetCols.RELATIVE_INTENSITY
        ].idxmax()
        df.loc[base_peak_idx, DiannParquetCols.FLAGS] |= _DIANN_FLAG_FIRST_FRAGMENT

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
            # Only the precursors are batched: frag_start_idx/frag_stop_idx are absolute offsets into
            # the full fragment frames, so those stay whole for the lookup to stay in sync.
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
