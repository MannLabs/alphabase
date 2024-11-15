"""Readers for Spectronaut's output library and reports, Swath data and DIANN data."""

from typing import List, Optional

import numpy as np
import pandas as pd

from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.maxquant_reader import MaxQuantReader
from alphabase.psm_reader.psm_reader import psm_reader_provider, psm_reader_yaml


class SpectronautReader(MaxQuantReader):
    """Reader for Spectronaut's output library TSV/CSV.

    Other parameters, please see `MaxQuantReader`
    in `alphabase.psm_reader.maxquant_reader`

    Parameters
    ----------
    csv_sep : str, optional
        Delimiter for TSV/CSV, by default 'tab'

    """

    def __init__(  # noqa: PLR0913 many arguments in function definition
        self,
        *,
        column_mapping: Optional[dict] = None,
        modification_mapping: Optional[dict] = None,
        fdr: float = 0.01,
        keep_decoy: bool = False,
        fixed_C57: bool = False,  # noqa: N803 TODO: make this  *,fixed_c57  (breaking)
        mod_seq_columns: Optional[List[str]] = None,
        rt_unit: str = "minute",
        **kwargs,
    ):
        if mod_seq_columns is None:
            mod_seq_columns = psm_reader_yaml["spectronaut"]["mod_seq_columns"]

        super().__init__(
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            fdr=fdr,
            keep_decoy=keep_decoy,
            mod_seq_columns=mod_seq_columns,
            fixed_C57=fixed_C57,
            rt_unit=rt_unit,
            **kwargs,
        )

        self.mod_seq_column = "ModifiedPeptide"
        self._min_max_rt_norm = True

    def _init_column_mapping(self) -> None:
        self.column_mapping = psm_reader_yaml["spectronaut"]["column_mapping"]

    def _load_file(self, filename: str) -> pd.DataFrame:
        self.csv_sep = self._get_table_delimiter(filename)
        df = pd.read_csv(filename, sep=self.csv_sep, keep_default_na=False)
        self._find_mod_seq_column(df)
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

    def __init__(  # noqa: PLR0913 many arguments in function definition
        self,
        *,
        column_mapping: Optional[dict] = None,
        modification_mapping: Optional[dict] = None,
        fdr: float = 0.01,
        keep_decoy: bool = False,
        fixed_C57: bool = False,  # noqa: N803 TODO: make this  *,fixed_c57  (breaking)
        mod_seq_columns: Optional[List[str]] = None,
        **kwargs,
    ):
        """SWATH or OpenSWATH library, similar to `SpectronautReader`."""
        if mod_seq_columns is None:
            mod_seq_columns = psm_reader_yaml["spectronaut"]["mod_seq_columns"]

        super().__init__(
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            fdr=fdr,
            keep_decoy=keep_decoy,
            fixed_C57=fixed_C57,
            mod_seq_columns=mod_seq_columns,
            **kwargs,
        )


class DiannReader(SpectronautReader):
    """Reader for DIANN data."""

    def __init__(  # noqa: PLR0913 many arguments in function definition
        self,
        *,
        column_mapping: Optional[dict] = None,
        modification_mapping: Optional[dict] = None,
        fdr: float = 0.01,
        keep_decoy: bool = False,
        fixed_C57: bool = False,  # noqa: N803 TODO: make this  *,fixed_c57  (breaking)
        rt_unit: str = "minute",
        **kwargs,
    ):
        """Similar to `SpectronautReader` but different in column_mapping and modification_mapping."""
        super().__init__(
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            fdr=fdr,
            keep_decoy=keep_decoy,
            fixed_C57=fixed_C57,
            rt_unit=rt_unit,
            **kwargs,
        )

        self.mod_seq_column = "Modified.Sequence"
        self._min_max_rt_norm = False

    def _init_column_mapping(self) -> None:
        self.column_mapping = psm_reader_yaml["diann"]["column_mapping"]

    def _load_file(self, filename: str) -> pd.DataFrame:
        self.csv_sep = self._get_table_delimiter(filename)
        return pd.read_csv(filename, sep=self.csv_sep, keep_default_na=False)

    def _post_process(self) -> None:
        super()._post_process()
        self._psm_df.rename(
            columns={PsmDfCols.SPEC_IDX: PsmDfCols.DIANN_SPEC_INDEX}, inplace=True
        )


class SpectronautReportReader(MaxQuantReader):
    """Reader for Spectronaut's report TSV/CSV.

    Other parameters, please see `MaxQuantReader`
    in `alphabase.psm_reader.maxquant_reader`

    Parameters
    ----------
    csv_sep : str, optional
        Delimiter for TSV/CSV, by default ','

    """

    def __init__(  # noqa: PLR0913 many arguments in function definition
        self,
        *,
        column_mapping: Optional[dict] = None,
        modification_mapping: Optional[dict] = None,
        fdr: float = 0.01,
        keep_decoy: bool = False,
        fixed_C57: bool = False,  # noqa: N803 TODO: make this  *,fixed_c57  (breaking)
        rt_unit: str = "minute",
        **kwargs,
    ):
        super().__init__(
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            fdr=fdr,
            keep_decoy=keep_decoy,
            fixed_C57=fixed_C57,
            rt_unit=rt_unit,
            **kwargs,
        )

        self.precursor_column = "EG.PrecursorId"

        self._min_max_rt_norm = False

    def _init_column_mapping(self) -> None:
        self.column_mapping = psm_reader_yaml["spectronaut_report"]["column_mapping"]

    def _load_file(self, filename: str) -> pd.DataFrame:
        self.mod_seq_column = "ModifiedSequence"
        self.csv_sep = self._get_table_delimiter(filename)
        df = pd.read_csv(filename, sep=self.csv_sep, keep_default_na=False)
        df[[self.mod_seq_column, PsmDfCols.CHARGE]] = df[
            self.precursor_column
        ].str.split(".", expand=True, n=2)
        df[PsmDfCols.CHARGE] = df[PsmDfCols.CHARGE].astype(np.int8)
        return df


def register_readers() -> None:
    psm_reader_provider.register_reader("spectronaut", SpectronautReader)
    psm_reader_provider.register_reader("speclib_tsv", SpectronautReader)
    psm_reader_provider.register_reader("openswath", SwathReader)
    psm_reader_provider.register_reader("swath", SwathReader)
    psm_reader_provider.register_reader("diann", DiannReader)
    psm_reader_provider.register_reader("spectronaut_report", SpectronautReportReader)
