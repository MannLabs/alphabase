"""Translate AlphaBase spectral libraries to a SWATH/Spectronaut transition list.

The shared export machinery lives in :mod:`alphabase.spectral_library.translate_core`;
this module holds the SWATH column names, the precursor mapping and the tsv writer. The
names it used to define are re-exported below, so importing them from here keeps
working.
"""

import multiprocessing as mp

import pandas as pd
import tqdm

from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.translate_core import (
    CCS_COLUMNS,
    MOBILITY_COLUMNS,
    RT_COLUMNS,
    FragmentTableCols,
    create_modified_sequence,
    first_present_column,
    fragment_table,
    get_precursor_mz,
    is_nterm_frag,
    join_fragments,
    mod_to_unimod_dict,
)

# the SWATH names for the canonical fragment columns, in output order
SWATH_FRAGMENT_COLUMNS = {
    FragmentTableCols.FRAG_TYPE: "FragmentType",
    FragmentTableCols.MZ: "FragmentMz",
    FragmentTableCols.INTENSITY: "RelativeIntensity",
    FragmentTableCols.CHARGE: "FragmentCharge",
    FragmentTableCols.SERIES_NUMBER: "FragmentNumber",
    FragmentTableCols.LOSS_TYPE: "FragmentLossType",
}

__all__ = [
    # re-exported from translate_core, where these now live
    "create_modified_sequence",
    "is_nterm_frag",
    "mod_to_unimod_dict",
    "get_precursor_mz",
    # this module's own
    "WritingProcess",
    "speclib_to_single_df",
    "speclib_to_swath_df",
    "translate_to_tsv",
]

# pandas renamed `to_csv`'s newline argument in 1.5; alphabase does not pin a minimum
_CSV_NEWLINE = (
    {"lineterminator": "\n"}
    if tuple(int(part) for part in pd.__version__.split(".")[:2]) >= (1, 5)
    else {"line_terminator": "\n"}
)


def _precursors_to_swath_df(  # noqa: PLR0913
    precursor_df: pd.DataFrame,
    fragment_mz_df: pd.DataFrame,
    fragment_intensity_df: pd.DataFrame,
    *,
    translate_mod_dict: dict = None,
    keep_k_highest_fragments: int = 12,
    min_frag_mz=200,
    max_frag_mz=2000,
    min_frag_intensity=0.01,
    min_frag_nAA=0,
    modloss: str = "H3PO4",
    verbose=True,
) -> pd.DataFrame:
    """Convert precursor and fragment frames to a SWATH transition list.

    The dataframe-in form of :func:`speclib_to_single_df`, so that one batch of
    precursors can be converted without standing up a `SpecLibBase` around it.
    """
    df = pd.DataFrame()
    df["ModifiedPeptide"] = precursor_df[["sequence", "mods", "mod_sites"]].apply(
        create_modified_sequence,
        axis=1,
        translate_mod_dict=translate_mod_dict,
        mod_sep="[]",
    )

    df["PrecursorCharge"] = precursor_df["charge"]

    rt = first_present_column(precursor_df, RT_COLUMNS)
    if rt is None:
        raise ValueError("precursor_df must contain the RT columns")
    df["RT"] = rt

    # ion mobility and CCS are optional: the column is omitted, not defaulted
    mobility = first_present_column(precursor_df, MOBILITY_COLUMNS)
    if mobility is not None:
        df["IonMobility"] = mobility

    ccs = first_present_column(precursor_df, CCS_COLUMNS)
    if ccs is not None:
        df["CCS"] = ccs

    df["StrippedPeptide"] = precursor_df["sequence"]

    df["PrecursorMz"] = get_precursor_mz(precursor_df)

    # this format prefers uniprot_ids; the DIA-NN one splits them over two columns
    proteins = first_present_column(precursor_df, ["uniprot_ids", "proteins"])
    if proteins is not None:
        df["ProteinID"] = proteins

    if "genes" in precursor_df.columns:
        df["Genes"] = precursor_df["genes"]

    if "decoy" in precursor_df.columns:
        df["Decoy"] = precursor_df["decoy"]

    fragments = fragment_table(
        precursor_df["frag_start_idx"].to_numpy(),
        precursor_df["frag_stop_idx"].to_numpy(),
        fragment_mz_df,
        fragment_intensity_df,
        keep_k_highest=keep_k_highest_fragments,
        min_frag_mz=min_frag_mz,
        max_frag_mz=max_frag_mz,
        min_frag_nAA=min_frag_nAA,
        verbose=verbose,
    )
    df = join_fragments(df, fragments, SWATH_FRAGMENT_COLUMNS)
    df = df[df["RelativeIntensity"] > min_frag_intensity]
    df.loc[df["FragmentLossType"] == "modloss", "FragmentLossType"] = modloss

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

    min_frag_mz, max_frag_mz : float
        Fragment m/z range; fragments outside it are dropped. Pass 0 for no lower bound
        and `np.inf` for no upper bound.

    Returns
    -------
    pd.DataFrame
        a single dataframe in the SWATH-like format

    """
    return _precursors_to_swath_df(
        speclib._precursor_df,
        speclib._fragment_mz_df,
        speclib._fragment_intensity_df,
        translate_mod_dict=translate_mod_dict,
        keep_k_highest_fragments=keep_k_highest_fragments,
        min_frag_mz=min_frag_mz,
        max_frag_mz=max_frag_mz,
        min_frag_intensity=min_frag_intensity,
        min_frag_nAA=min_frag_nAA,
        modloss=modloss,
        verbose=verbose,
    )


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
            df.to_csv(
                self.tsv,
                header=(batch == 0),
                sep="\t",
                mode="a",
                index=False,
                **_CSV_NEWLINE,
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
    """Translate an alphabase library into a SWATH/Spectronaut transition-list tsv.

    Precursors are converted in batches and appended to the tsv, so large libraries do
    not need to be held in memory at once in the flat, one-row-per-fragment format.

    Parameters
    ----------
    min_frag_mz, max_frag_mz : float
        Fragment m/z range; fragments outside it are dropped. Pass 0 for no lower bound
        and `np.inf` for no upper bound.

    See :func:`speclib_to_single_df`, whose parameters this shares, for the rest.
    """
    if multiprocessing:
        queue_size = 1000000 // batch_size
        if queue_size < 2:
            queue_size = 2
        elif queue_size > 10:
            queue_size = 10
        df_head_queue = mp.Queue(maxsize=queue_size)
        writing_process = WritingProcess(df_head_queue, tsv)
        writing_process.start()
    if isinstance(tsv, str):
        with open(tsv, "w"):
            pass

    # only the precursors are batched -- the fragment indices are absolute
    precursor_df = speclib._precursor_df
    for first_row in tqdm.tqdm(range(0, len(precursor_df), batch_size)):
        df = _precursors_to_swath_df(
            precursor_df.iloc[first_row : first_row + batch_size],
            speclib._fragment_mz_df,
            speclib._fragment_intensity_df,
            translate_mod_dict=translate_mod_dict,
            keep_k_highest_fragments=keep_k_highest_fragments,
            min_frag_mz=min_frag_mz,
            max_frag_mz=max_frag_mz,
            min_frag_intensity=min_frag_intensity,
            min_frag_nAA=min_frag_nAA,
            verbose=False,
        )
        if multiprocessing:
            df_head_queue.put((df, first_row))
        else:
            df.to_csv(
                tsv,
                header=(first_row == 0),
                sep="\t",
                mode="a",
                index=False,
                **_CSV_NEWLINE,
            )
    if multiprocessing:
        df_head_queue.put((None, None))
        print(
            "Translation finished, it will take several minutes to export the rest precursors to the tsv file..."
        )
        writing_process.join()
        if writing_process.exitcode:
            raise RuntimeError(
                f"the process writing {tsv} exited with code "
                f"{writing_process.exitcode}, so the file is incomplete; its traceback "
                'is above. A script with no `if __name__ == "__main__":` guard is the '
                "usual cause on macOS and Windows. Pass multiprocessing=False to write "
                "from this process instead."
            )
