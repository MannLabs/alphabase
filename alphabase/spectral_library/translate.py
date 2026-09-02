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
    create_modified_sequence,
    first_present_column,
    is_nterm_frag,
    mask_fragment_intensity_by_frag_nAA,
    mask_fragment_intensity_by_mz_,
    merge_precursor_fragment_df,
    mod_to_unimod_dict,
)

__all__ = [
    # re-exported from translate_core, where these now live
    "create_modified_sequence",
    "is_nterm_frag",
    "mask_fragment_intensity_by_frag_nAA",
    "mask_fragment_intensity_by_mz_",
    "merge_precursor_fragment_df",
    "mod_to_unimod_dict",
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

    rt = first_present_column(speclib.precursor_df, RT_COLUMNS)
    if rt is None:
        raise ValueError("precursor_df must contain the RT columns")
    df["RT"] = rt

    # ion mobility and CCS are optional: the column is omitted, not defaulted
    mobility = first_present_column(speclib.precursor_df, MOBILITY_COLUMNS)
    if mobility is not None:
        df["IonMobility"] = mobility

    ccs = first_present_column(speclib.precursor_df, CCS_COLUMNS)
    if ccs is not None:
        df["CCS"] = ccs

    df["StrippedPeptide"] = speclib.precursor_df["sequence"]

    if "precursor_mz" not in speclib._precursor_df.columns:
        speclib.calc_precursor_mz()
    df["PrecursorMz"] = speclib._precursor_df["precursor_mz"]

    # this format prefers uniprot_ids; the DIA-NN one splits them over two columns
    proteins = first_present_column(speclib.precursor_df, ["uniprot_ids", "proteins"])
    if proteins is not None:
        df["ProteinID"] = proteins

    if "genes" in speclib._precursor_df.columns:
        df["Genes"] = speclib._precursor_df["genes"]

    if "decoy" in speclib._precursor_df.columns:
        df["Decoy"] = speclib._precursor_df["decoy"]

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
    )
    df = df[df["RelativeIntensity"] > min_frag_intensity]
    df.loc[df["FragmentLossType"] == "modloss", "FragmentLossType"] = modloss

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
    # process precursors in batches: the flat (one row per fragment) format is much larger
    # than the compact library, so batching keeps peak memory bounded for large libraries
    batch_speclib = SpecLibBase()
    batch_speclib._fragment_intensity_df = speclib._fragment_intensity_df
    batch_speclib._fragment_mz_df = speclib._fragment_mz_df
    precursor_df = speclib._precursor_df
    for i in tqdm.tqdm(range(0, len(precursor_df), batch_size)):
        batch_speclib._precursor_df = precursor_df.iloc[i : i + batch_size]
        df = speclib_to_single_df(
            batch_speclib,
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
            df.to_csv(
                tsv, header=(i == 0), sep="\t", mode="a", index=False, **_CSV_NEWLINE
            )
    if multiprocessing:
        df_head_queue.put((None, None))
        print(
            "Translation finished, it will take several minutes to export the rest precursors to the tsv file..."
        )
        writing_process.join()
