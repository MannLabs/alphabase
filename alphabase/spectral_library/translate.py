"""Translate AlphaBase spectral libraries to a SWATH-style transition list.

The exported table is a *transition list*: one row per precursor/fragment pair, the
shape SRM/DIA tools use for an assay library. It carries Spectronaut's column names
(`ModifiedPeptide`, `StrippedPeptide`, `RelativeIntensity`, ...), which both Spectronaut
and DIA-NN's legacy tsv reader accept, and which
:class:`alphabase.spectral_library.reader.SWATHLibraryReader` reads back.

Shared export helpers live in :mod:`alphabase.spectral_library.translate_core`.
"""

import functools
import multiprocessing as mp
import warnings
from typing import IO, Optional, Union

import pandas as pd

from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.translate_core import (
    CCS_COLUMNS,
    MOBILITY_COLUMNS,
    RT_COLUMNS,
    FragmentColumns,
    FragmentFilter,
    create_modified_sequence,
    explode_top_fragments,
    first_present_column,
    is_nterm_frag,  # noqa: F401  re-exported for backwards compatibility
    mask_fragment_intensity_by_frag_nAA,  # noqa: F401  re-exported, no longer used here
    mask_fragment_intensity_by_mz_,  # noqa: F401  re-exported, no longer used here
    mod_to_unimod_dict,  # noqa: F401  re-exported for backwards compatibility
    precursor_mz_series,
    translate_in_batches,
)

# precursor columns of the transition list that map to more than one alphabase column
PROTEIN_ID_COLUMNS = ("uniprot_ids", "proteins")

SWATH_FRAGMENT_COLUMNS = FragmentColumns(
    frag_type="FragmentType",
    mz="FragmentMz",
    intensity="RelativeIntensity",
    charge="FragmentCharge",
    series_number="FragmentNumber",
    loss_type="FragmentLossType",
)

# `line_terminator` was renamed to `lineterminator` in pandas 1.5
_TO_CSV_NEWLINE = (
    {"lineterminator": "\n"}
    if tuple(int(i) for i in pd.__version__.split(".")[:2]) >= (1, 5)
    else {"line_terminator": "\n"}
)


def _precursors_to_swath_df(
    precursor_df: pd.DataFrame,
    fragment_mz_df: pd.DataFrame,
    fragment_intensity_df: pd.DataFrame,
    *,
    translate_mod_dict: Optional[dict],
    fragment_filter: FragmentFilter,
    columns: FragmentColumns,
    modloss: str,
    verbose: bool,
) -> pd.DataFrame:
    """Translate precursors and their fragments into transition list rows.

    The dataframe-level implementation of :func:`translate_to_transition_df`, so that a batch
    of precursors can be translated without building a `SpecLibBase` around it.
    """
    df = pd.DataFrame(index=precursor_df.index)
    df["ModifiedPeptide"] = precursor_df[["sequence", "mods", "mod_sites"]].apply(
        create_modified_sequence,
        axis=1,
        translate_mod_dict=translate_mod_dict,
        mod_sep="[]",
    )

    df["frag_start_idx"] = precursor_df["frag_start_idx"]
    df["frag_stop_idx"] = precursor_df["frag_stop_idx"]

    df["PrecursorCharge"] = precursor_df["charge"]

    rt = first_present_column(precursor_df, RT_COLUMNS)
    if rt is None:
        raise ValueError("precursor_df must contain the RT columns")
    df["RT"] = rt

    mobility = first_present_column(precursor_df, MOBILITY_COLUMNS)
    if mobility is not None:
        df["IonMobility"] = mobility

    ccs = first_present_column(precursor_df, CCS_COLUMNS)
    if ccs is not None:
        df["CCS"] = ccs

    # df['LabelModifiedSequence'] = df['ModifiedPeptide']
    df["StrippedPeptide"] = precursor_df["sequence"]
    df["PrecursorMz"] = precursor_mz_series(precursor_df)

    protein_id = first_present_column(precursor_df, PROTEIN_ID_COLUMNS)
    if protein_id is not None:
        df["ProteinID"] = protein_id

    genes = first_present_column(precursor_df, ("genes",))
    if genes is not None:
        df["Genes"] = genes

    decoy = first_present_column(precursor_df, ("decoy",))
    if decoy is not None:
        df["Decoy"] = decoy

    return explode_top_fragments(
        df,
        fragment_mz_df,
        fragment_intensity_df,
        columns=columns,
        fragment_filter=fragment_filter,
        modloss_label=modloss,
        verbose=verbose,
    )


def translate_to_transition_df(  # noqa: PLR0913  one argument per output column
    speclib: SpecLibBase,
    *,
    translate_mod_dict: Optional[dict] = None,
    keep_k_highest_fragments: int = 12,
    min_frag_mz: float = 200,
    max_frag_mz: float = 2000,
    min_frag_intensity: float = 0.01,
    min_frag_nAA: int = 0,  # noqa: N803  public name
    modloss: str = "H3PO4",
    frag_type_head: str = "FragmentType",
    frag_mass_head: str = "FragmentMz",
    frag_inten_head: str = "RelativeIntensity",
    frag_charge_head: str = "FragmentCharge",
    frag_loss_head: str = "FragmentLossType",
    frag_series_head: str = "FragmentNumber",
    verbose: bool = True,
) -> pd.DataFrame:
    """Convert an alphabase library into a transition list dataframe.

    One row per precursor/fragment pair, in Spectronaut's column names. The library is
    not modified.

    Parameters
    ----------
    speclib : SpecLibBase
        The alphabase spectral library to convert.

    translate_mod_dict : dict
        A dict to map AlphaX modification names to other software,
        use the alphabase name without its site if None.
        Defaults to None.

    keep_k_highest_fragments : int
        only keep highest fragments for each precursor. Default: 12

    min_frag_mz, max_frag_mz : float
        Fragment m/z range; fragments outside it are dropped. Set both to 0 to disable.

    min_frag_intensity : float
        Drop fragments whose relative intensity is at or below this value.

    min_frag_nAA : int
        Drop the b/y fragments with a series number below this; 0 keeps all.

    modloss : str
        Loss label written for modification-loss fragments. Default: "H3PO4"

    frag_type_head, frag_mass_head, frag_inten_head, frag_charge_head, frag_loss_head,
    frag_series_head : str
        Names to give the fragment columns of the result.

    verbose : bool
        Show a progress bar while exploding fragments.

    Returns
    -------
    pd.DataFrame
        a single dataframe in the SWATH-like format

    """
    return _precursors_to_swath_df(
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
        columns=FragmentColumns(
            frag_type=frag_type_head,
            mz=frag_mass_head,
            intensity=frag_inten_head,
            charge=frag_charge_head,
            series_number=frag_series_head,
            loss_type=frag_loss_head,
        ),
        modloss=modloss,
        verbose=verbose,
    )


def speclib_to_swath_df(
    speclib: SpecLibBase,
    *,
    keep_k_highest_fragments: int = 12,
    min_frag_mz: float = 200,
    max_frag_mz: float = 2000,
    min_frag_intensity: float = 0.01,
) -> pd.DataFrame:
    translate_to_transition_df(
        speclib,
        translate_mod_dict=None,
        keep_k_highest_fragments=keep_k_highest_fragments,
        min_frag_mz=min_frag_mz,
        max_frag_mz=max_frag_mz,
        min_frag_intensity=min_frag_intensity,
    )


def speclib_to_single_df(speclib: SpecLibBase, **kwargs) -> pd.DataFrame:
    """Deprecated alias of :func:`translate_to_transition_df`."""
    warnings.warn(
        "`alphabase.spectral_library.translate.speclib_to_single_df()` is deprecated. "
        "Please use "
        "`alphabase.spectral_library.translate.translate_to_transition_df()` instead.",
        FutureWarning,
    )
    return translate_to_transition_df(speclib, **kwargs)


def _write_tsv(tsv: Union[str, IO], df: pd.DataFrame, batch_start: int) -> None:
    """Append a batch to the tsv, with a header only for the first batch."""
    df.to_csv(
        tsv,
        header=(batch_start == 0),
        sep="\t",
        mode="a",
        index=False,
        **_TO_CSV_NEWLINE,
    )


class WritingProcess(mp.Process):
    """Writes the translated batches it is given to a tsv, off the translating process."""

    def __init__(self, task_queue: mp.Queue, tsv: Union[str, IO], *args, **kwargs):
        self.task_queue: mp.Queue = task_queue
        self.tsv = tsv
        super().__init__(*args, **kwargs)

    def run(self) -> None:
        """Write batches until the queue gives a None, which ends the process."""
        while True:
            df, batch = self.task_queue.get()
            if df is None:
                break
            _write_tsv(self.tsv, df, batch)


def translate_to_tsv(  # noqa: PLR0913  one argument per export setting
    speclib: SpecLibBase,
    tsv: Union[str, IO],
    *,
    keep_k_highest_fragments: int = 12,
    min_frag_mz: float = 200,
    max_frag_mz: float = 2000,
    min_frag_intensity: float = 0.01,
    min_frag_nAA: int = 0,  # noqa: N803  public name
    batch_size: int = 100000,
    translate_mod_dict: Optional[dict] = None,
    multiprocessing: bool = True,  # noqa: FBT001, FBT002  public name
) -> None:
    """Translate an alphabase library into a transition list tsv.

    Precursors are converted in batches and streamed to one file, so a large library
    does not need to be held in the flat format all at once. The library is not
    modified.

    Parameters
    ----------
    speclib : SpecLibBase
        The alphabase spectral library to translate.

    tsv : str or file object
        Path of the tsv to write, or an open file to write to.

    keep_k_highest_fragments : int
        Keep only the k most intense fragments per precursor. Default: 12

    min_frag_mz, max_frag_mz : float
        Fragment m/z range; fragments outside it are dropped. Set both to 0 to disable.

    min_frag_intensity : float
        Drop fragments whose relative intensity is at or below this value.

    min_frag_nAA : int
        Drop the b/y fragments with a series number below this; 0 keeps all.

    batch_size : int
        Number of precursors to convert per batch. Default: 100000

    translate_mod_dict : dict
        A dict to map AlphaX modification names to other software.
        Defaults to None, which uses the alphabase names without their sites.

    multiprocessing : bool
        Write from a separate process, so writing overlaps with translating. Needs
        `tsv` to be a path.

    """
    convert = functools.partial(
        _precursors_to_swath_df,
        translate_mod_dict=translate_mod_dict,
        fragment_filter=FragmentFilter(
            keep_k_highest=keep_k_highest_fragments,
            min_mz=min_frag_mz,
            max_mz=max_frag_mz,
            min_intensity=min_frag_intensity,
            min_nAA=min_frag_nAA,
        ),
        columns=SWATH_FRAGMENT_COLUMNS,
        modloss="H3PO4",
        verbose=False,
    )

    if isinstance(tsv, str):
        with open(tsv, "w"):
            pass

    writing_process = None
    if multiprocessing:
        queue_size = min(max(1000000 // batch_size, 2), 10)
        df_head_queue = mp.Queue(maxsize=queue_size)
        writing_process = WritingProcess(df_head_queue, tsv)
        writing_process.start()

        def write(df: pd.DataFrame, batch_start: int) -> None:
            df_head_queue.put((df, batch_start))
    else:
        write = functools.partial(_write_tsv, tsv)

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
        if writing_process is not None:
            df_head_queue.put((None, None))
            print(  # noqa: T201  kept from the original
                "Translation finished, it will take several minutes to export the rest precursors to the tsv file..."
            )
            writing_process.join()
