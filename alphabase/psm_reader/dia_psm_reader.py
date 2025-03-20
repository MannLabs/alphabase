"""Readers for Spectronaut's output library and reports, Swath data and DIANN data."""

from typing import List, Optional

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

    def __init__(  # noqa: PLR0913, D417 # too many arguments in function definition, missing argument descriptions
        self,
        *,
        column_mapping: Optional[dict] = None,
        modification_mapping: Optional[dict] = None,
        mod_seq_columns: Optional[List[str]] = None,
        fdr: float = 0.01,
        keep_decoy: bool = False,
        rt_unit: Optional[str] = None,
        # DIANN reader-specific
        filter_first_search_fdr: bool = False,
        filter_second_search_fdr: bool = False,
        **kwargs,
    ):
        """Reader for DIANN data.

        See documentation of `PSMReaderBase` for more information.

        Parameters
        ----------
        filter_first_search_fdr : bool, optional
            If True, the FDR filtering will be done also to the first search columns (fdr1_search1 and fdr2_search1)

        filter_second_search_fdr : bool, optional
            If True, the FDR filtering will be done also to the second columns (fdr1_search2 and fdr2_search2)

        See documentation of `PSMReaderBase` for the rest of parameters.

        """
        super().__init__(
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            mod_seq_columns=mod_seq_columns,
            fdr=fdr,
            keep_decoy=keep_decoy,
            rt_unit=rt_unit,
            **kwargs,
        )

        self._filter_first_search_fdr = filter_first_search_fdr
        self._filter_second_search_fdr = filter_second_search_fdr

    def _post_process(self, origin_df: pd.DataFrame) -> None:
        self._psm_df.rename(
            columns={PsmDfCols.SPEC_IDX: PsmDfCols.DIANN_SPEC_INDEX}, inplace=True
        )

        super()._post_process(origin_df)

    def _filter_fdr(self) -> None:
        """Filter PSMs based on additional FDR columns if requested.

        If a column is not present in the dataframe, it is ignored.
        """
        super()._filter_fdr()

        extra_fdr_columns = []

        if self._filter_first_search_fdr:
            extra_fdr_columns += [PsmDfCols.FDR1_SEARCH1, PsmDfCols.FDR2_SEARCH1]

        if self._filter_second_search_fdr:
            extra_fdr_columns += [PsmDfCols.FDR1_SEARCH2, PsmDfCols.FDR2_SEARCH2]

        mask = np.ones(len(self._psm_df), dtype=bool)
        for col in extra_fdr_columns:
            if col in self._psm_df.columns:
                mask &= self._psm_df[col] <= self._fdr_threshold

        if not all(mask):
            self._psm_df = self._psm_df[mask]


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
