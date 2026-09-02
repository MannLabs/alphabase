"""Translate AlphaBase spectral libraries to a SWATH-style transition list.

The exported table is a *transition list*: one row per precursor/fragment pair, the
shape SRM/DIA tools use for an assay library. It carries Spectronaut's column names
(`ModifiedPeptide`, `StrippedPeptide`, `RelativeIntensity`, ...), which both Spectronaut
and DIA-NN's legacy tsv reader accept, and which
:class:`alphabase.spectral_library.reader.SWATHLibraryReader` reads back.

Shared export helpers live in :mod:`alphabase.spectral_library.translate_core`.
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
    is_nterm_frag,  # noqa: F401  re-exported for backwards compatibility
    mask_fragment_intensity_by_frag_nAA,
    mask_fragment_intensity_by_mz_,
    merge_precursor_fragment_df,
    mod_to_unimod_dict,  # noqa: F401  re-exported for backwards compatibility
)

# precursor columns of the transition list that map to more than one alphabase column
PROTEIN_ID_COLUMNS = ("uniprot_ids", "proteins")


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
    precursor_df = speclib.precursor_df

    df = pd.DataFrame()
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

    if "precursor_mz" not in precursor_df.columns:
        speclib.calc_precursor_mz()
    df["PrecursorMz"] = speclib.precursor_df["precursor_mz"]

    protein_id = first_present_column(precursor_df, PROTEIN_ID_COLUMNS)
    if protein_id is not None:
        df["ProteinID"] = protein_id

    genes = first_present_column(precursor_df, ("genes",))
    if genes is not None:
        df["Genes"] = genes

    decoy = first_present_column(precursor_df, ("decoy",))
    if decoy is not None:
        df["Decoy"] = decoy

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
