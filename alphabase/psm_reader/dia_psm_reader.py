"""Readers for Spectronaut's output library and reports, Swath data and DIANN data."""

import numpy as np
import pandas as pd

from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.maxquant_reader import ModifiedSequenceReader
from alphabase.psm_reader.psm_reader import psm_reader_provider


class SpectronautReader(ModifiedSequenceReader):
    """Reader for Spectronaut's output library TSV/CSV."""

    _reader_type = "spectronaut"
    _add_unimod_to_mod_mapping = True
    _min_max_rt_norm = True

    def _pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Spectronaut-specific preprocessing of output data."""
        if "ReferenceRun" in df.columns:
            df.drop_duplicates(
                ["ReferenceRun", self.mod_seq_column, "PrecursorCharge"], inplace=True
            )
        else:
            df.drop_duplicates([self.mod_seq_column, "PrecursorCharge"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df


class SwathReader(SpectronautReader):
    """Reader for SWATH or OpenSWATH library TSV/CSV."""

    _reader_type = "spectronaut"  # no typo
    _add_unimod_to_mod_mapping = True


class DiannReader(ModifiedSequenceReader):
    """Reader for DIANN data."""

    _reader_type = "diann"
    _add_unimod_to_mod_mapping = True
    _min_max_rt_norm = False

    def _post_process(self, origin_df: pd.DataFrame) -> None:
        self._psm_df.rename(
            columns={PsmDfCols.SPEC_IDX: PsmDfCols.DIANN_SPEC_INDEX}, inplace=True
        )

        super()._post_process(origin_df)


class SpectronautReportReader(ModifiedSequenceReader):
    """Reader for Spectronaut's report TSV/CSV."""

    _reader_type = "spectronaut_report"
    _add_unimod_to_mod_mapping = True
    _min_max_rt_norm = False

    def _pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Spectronaut report-specific preprocessing of output data."""
        df[[self.mod_seq_column, PsmDfCols.CHARGE]] = df[
            self._precursor_id_column
        ].str.split(".", expand=True, n=2)
        df[PsmDfCols.CHARGE] = df[PsmDfCols.CHARGE].astype(np.int8)
        return df


def register_readers() -> None:
    """Register readers for Spectronaut's output library and reports, Swath data and DIANN data."""
    psm_reader_provider.register_reader("spectronaut", SpectronautReader)
    psm_reader_provider.register_reader("speclib_tsv", SpectronautReader)
    psm_reader_provider.register_reader("openswath", SwathReader)
    psm_reader_provider.register_reader("swath", SwathReader)
    psm_reader_provider.register_reader("diann", DiannReader)
    psm_reader_provider.register_reader("spectronaut_report", SpectronautReportReader)
