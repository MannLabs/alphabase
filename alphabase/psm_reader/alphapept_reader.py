import os

import h5py
import numba
import numpy as np
import pandas as pd

from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.psm_reader import (
    PSMReaderBase,
    psm_reader_provider,
    psm_reader_yaml,
)


@numba.njit
def parse_ap(precursor):
    """
    Parser to parse peptide strings
    """
    items = precursor.split("_")
    decoy = 1 if len(items) == 3 else 0
    modseq = items[0]
    charge = items[-1]

    parsed = []
    mods = []
    sites = []
    string = ""

    for i in range(len(modseq)):
        if modseq[i].isupper():
            break
    if i > 0:
        sites.append("0")
        mods.append(modseq[:i])
        modseq = modseq[i:]

    for i in modseq:
        string += i
        if i.isupper():
            parsed.append(i)
            if len(string) > 1:
                sites.append(str(len(parsed)))
                mods.append(string)
            string = ""

    return "".join(parsed), ";".join(mods), ";".join(sites), charge, decoy


class AlphaPeptReader(PSMReaderBase):
    def __init__(
        self,
        *,
        column_mapping: dict = None,
        modification_mapping: dict = None,
        fdr=0.01,
        keep_decoy=False,
        **kwargs,
    ):
        """
        Reading PSMs from alphapept's *.ms_data.hdf
        """
        super().__init__(
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            fdr=fdr,
            keep_decoy=keep_decoy,
            **kwargs,
        )
        self.hdf_dataset = "identifications"

    def _init_column_mapping(self):
        self.column_mapping = psm_reader_yaml["alphapept"]["column_mapping"]

    def _init_modification_mapping(self):
        self.modification_mapping = psm_reader_yaml["alphapept"]["modification_mapping"]

    def _load_file(self, filename):
        with h5py.File(filename, "r") as _hdf:
            dataset = _hdf[self.hdf_dataset]
            df = pd.DataFrame({col: dataset[col] for col in dataset})
            df[PsmDfCols.RAW_NAME] = os.path.basename(filename)[: -len(".ms_data.hdf")]
            df["precursor"] = df["precursor"].str.decode("utf-8")
            # df['naked_sequence'] = df['naked_sequence'].str.decode('utf-8')
            if "scan_no" in df.columns:
                df["scan_no"] = df["scan_no"].astype("int")
                df["raw_idx"] = df["scan_no"] - 1  # if thermo, use scan-1 as spec_idx
            df[PsmDfCols.CHARGE] = df[PsmDfCols.CHARGE].astype(int)
        return df

    def _load_modifications(self, df: pd.DataFrame):
        if len(df) == 0:
            self._psm_df[PsmDfCols.SEQUENCE] = ""
            self._psm_df[PsmDfCols.MODS] = ""
            self._psm_df[PsmDfCols.MOD_SITES] = ""
            self._psm_df[PsmDfCols.DECOY] = 0
            return

        (
            self._psm_df[PsmDfCols.SEQUENCE],
            self._psm_df[PsmDfCols.MODS],
            self._psm_df[PsmDfCols.MOD_SITES],
            _charges,
            self._psm_df[PsmDfCols.DECOY],
        ) = zip(*df["precursor"].apply(parse_ap))
        self._psm_df[PsmDfCols.DECOY] = self._psm_df[PsmDfCols.DECOY].astype(np.int8)


def register_readers():
    psm_reader_provider.register_reader("alphapept", AlphaPeptReader)
