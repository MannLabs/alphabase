"""Translate AlphaBase spectral libraries to DIA-NN 1.9.1+ parquet format.

Shared export helpers live in :mod:`alphabase.spectral_library.translate_core`.
"""

import functools
from typing import Optional

import numpy as np
import pandas as pd

from alphabase.psm_reader.keys import ConstantsClass, LibPsmDfCols, PsmDfCols
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.translate_core import (
    MOBILITY_COLUMNS,
    RT_COLUMNS,
    FragmentColumns,
    FragmentFilter,
    create_modified_sequence,
    explode_top_fragments,
    first_present_column,
    mod_to_unimod_dict,
    precursor_mz_series,
    translate_in_batches,
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


# fragment column names given to `explode_top_fragments`
DIANN_FRAGMENT_COLUMNS = FragmentColumns(
    frag_type=DiannParquetCols.FRAGMENT_TYPE,
    mz=DiannParquetCols.PRODUCT_MZ,
    intensity=DiannParquetCols.RELATIVE_INTENSITY,
    charge=DiannParquetCols.FRAGMENT_CHARGE,
    series_number=DiannParquetCols.FRAGMENT_SERIES_NUMBER,
    loss_type=DiannParquetCols.FRAGMENT_LOSS_TYPE,
)

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

# internal column naming the source precursor row of a fragment, dropped before returning
_PRECURSOR_ROW = "_precursor_row"


# precursor columns of the DIA-NN schema that map to more than one alphabase column
PROTEIN_GROUP_COLUMNS = (PsmDfCols.PROTEINS, PsmDfCols.UNIPROT_IDS)
PROTEIN_IDS_COLUMNS = (PsmDfCols.UNIPROT_IDS, PsmDfCols.PROTEINS)


def speclib_to_diann_df(  # noqa: PLR0913
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
    return _precursors_to_diann_df(
        speclib.precursor_df,
        speclib.fragment_mz_df,
        speclib.fragment_intensity_df,
        translate_mod_dict=translate_mod_dict,
        fragment_filter=FragmentFilter(
            keep_k_highest=keep_k_highest_fragments,
            min_mz=min_frag_mz,
            max_mz=max_frag_mz,
            min_intensity=min_frag_intensity,
            min_nAA=min_frag_nAA,
        ),
        modloss=modloss,
        verbose=verbose,
    )


def _precursors_to_diann_df(  # noqa: PLR0913  the frames and the export settings
    precursor_df: pd.DataFrame,
    fragment_mz_df: pd.DataFrame,
    fragment_intensity_df: pd.DataFrame,
    *,
    translate_mod_dict: Optional[dict],
    fragment_filter: FragmentFilter,
    modloss: str,
    verbose: bool,
) -> pd.DataFrame:
    """Translate precursors and their fragments into DIA-NN parquet rows.

    The dataframe-level implementation of :func:`speclib_to_diann_df`, so that a batch
    of precursors can be translated without building a `SpecLibBase` around it.
    """
    if translate_mod_dict is None:
        translate_mod_dict = mod_to_unimod_dict

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
    df[DiannParquetCols.PRECURSOR_MZ] = precursor_mz_series(precursor_df)

    rt = first_present_column(precursor_df, RT_COLUMNS)
    if rt is None:
        raise ValueError("precursor_df must contain a retention time column")
    df[DiannParquetCols.RT] = rt
    df[DiannParquetCols.IM] = first_present_column(precursor_df, MOBILITY_COLUMNS, 0.0)

    df[DiannParquetCols.PROTEIN_GROUP] = first_present_column(
        precursor_df, PROTEIN_GROUP_COLUMNS, ""
    )
    df[DiannParquetCols.PROTEIN_IDS] = first_present_column(
        precursor_df, PROTEIN_IDS_COLUMNS, ""
    )
    df[DiannParquetCols.PROTEIN_NAMES] = first_present_column(
        precursor_df, ("protein_names",), ""
    )
    df[DiannParquetCols.GENES] = first_present_column(
        precursor_df, (PsmDfCols.GENES,), ""
    )
    df[DiannParquetCols.DECOY] = first_present_column(
        precursor_df, (PsmDfCols.DECOY,), 0
    )

    # N.Term/C.Term mark peptides at the protein N-/C-terminus (from FASTA digestion)
    df[DiannParquetCols.N_TERM] = first_present_column(
        precursor_df, ("is_prot_nterm",), 0
    )
    df[DiannParquetCols.C_TERM] = first_present_column(
        precursor_df, ("is_prot_cterm",), 0
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

    # carried through the explode, so every kept fragment names its source precursor
    # row. A position rather than an index label, which need not be unique.
    df[_PRECURSOR_ROW] = np.arange(len(precursor_df))

    df = explode_top_fragments(
        df,
        fragment_mz_df,
        fragment_intensity_df,
        columns=DIANN_FRAGMENT_COLUMNS,
        fragment_filter=fragment_filter,
        modloss_label=modloss,
        verbose=verbose,
    )

    # exploding repeats the precursor's index, so make it unique before addressing rows
    df = df.reset_index(drop=True)

    # Flags: base bit on all fragments, base-peak bit on each precursor's most intense
    # one. Grouped by the source precursor row, not by `Precursor.Id`: that is only the
    # modified sequence and the charge, so two precursor rows sharing a peptidoform
    # would share one base-peak bit between them and one of them would get none.
    df[DiannParquetCols.FLAGS] = _DIANN_FLAG_BASE
    if len(df):
        base_peak_idx = df.groupby(_PRECURSOR_ROW, sort=False)[
            DiannParquetCols.RELATIVE_INTENSITY
        ].idxmax()
        df.loc[base_peak_idx, DiannParquetCols.FLAGS] |= _DIANN_FLAG_FIRST_FRAGMENT
    df = df.drop(columns=_PRECURSOR_ROW)

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

    convert = functools.partial(
        _precursors_to_diann_df,
        translate_mod_dict=translate_mod_dict,
        fragment_filter=FragmentFilter(
            keep_k_highest=keep_k_highest_fragments,
            min_mz=min_frag_mz,
            max_mz=max_frag_mz,
            min_intensity=min_frag_intensity,
            min_nAA=min_frag_nAA,
        ),
        modloss="H3PO4",
        verbose=False,
    )

    writer = pq.ParquetWriter(parquet_path, schema)

    def write(df: pd.DataFrame, _batch_start: int) -> None:
        writer.write_table(
            pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        )

    try:
        translate_in_batches(
            speclib.precursor_df,
            speclib.fragment_mz_df,
            speclib.fragment_intensity_df,
            convert,
            write,
            batch_size=batch_size,
        )
    finally:
        writer.close()
