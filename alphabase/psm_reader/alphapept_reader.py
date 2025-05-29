"""Reader for AlphaPept's .ms_data.hdf files."""

from pathlib import Path
from typing import Tuple

import h5py
import numba
import numpy as np
import pandas as pd

from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.psm_reader import (
    PSMReaderBase,
    psm_reader_provider,
)


@numba.njit
def parse_ap(precursor: str) -> Tuple[str, str, str, str, int]:
    """Parser to parse peptide strings."""
    items = precursor.split("_")
    decoy = 1 if len(items) == 3 else 0  # noqa: PLR2004 magic value
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
    """Reader for AlphaPept's .ms_data.hdf files."""

    _reader_type = "alphapept"

    def _load_file(self, filename: str) -> pd.DataFrame:
        """Load an AlphaPept output file to a DataFrame."""
        with h5py.File(filename, "r") as _hdf:
            dataset = _hdf[
                "identifications"
            ]  # TODO: "identifications" could be moved to yaml
            df = pd.DataFrame({col: dataset[col] for col in dataset})

        # TODO: make this more stable
        df[PsmDfCols.RAW_NAME] = Path(filename).name[: -len(".ms_data.hdf")]

        return df

    def _pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """AlphaPept-specific preprocessing of output data."""
        df["precursor"] = df["precursor"].str.decode("utf-8")
        # df['naked_sequence'] = df['naked_sequence'].str.decode('utf-8')
        if "scan_no" in df.columns:
            df["scan_no"] = df["scan_no"].astype("int")
            df["raw_idx"] = df["scan_no"] - 1  # if thermo, use scan-1 as spec_idx
        df[PsmDfCols.CHARGE] = df[PsmDfCols.CHARGE].astype(int)
        return df

    def _load_modifications(self, origin_df: pd.DataFrame) -> None:
        (
            self._psm_df[PsmDfCols.SEQUENCE],
            self._psm_df[PsmDfCols.MODS],
            self._psm_df[PsmDfCols.MOD_SITES],
            _charges,
            self._psm_df[PsmDfCols.DECOY],
        ) = zip(*origin_df["precursor"].apply(parse_ap))

        self._psm_df[PsmDfCols.DECOY] = self._psm_df[PsmDfCols.DECOY].astype(np.int8)


def register_readers() -> None:
    """Register readers for AlphaPept's .ms_data.hdf files."""
    psm_reader_provider.register_reader("alphapept", AlphaPeptReader)
