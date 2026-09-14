"""Readers for Spectronaut's output library and reports, Swath data and DIANN data."""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.maxquant_reader import ModifiedSequenceReader
from alphabase.psm_reader.psm_reader import psm_reader_provider
from alphabase.psm_reader.utils import get_column_mapping_for_df


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
        mod_seq_columns: Optional[list[str]] = None,
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

    # "EG.PrecursorId" is "<modified sequence>.<charge>", e.g. "_DATM[Oxidation (M)]EVQR_.2"
    _PRECURSOR_ID_SEPARATOR = "."

    def _extract_charge_from_precursor_column(
        self, precursor_ids: pd.Series
    ) -> pd.Series:
        """Extract the charge state from the last field of Spectronaut's precursor id."""
        charge = precursor_ids.str.rsplit(self._PRECURSOR_ID_SEPARATOR, n=1).str[-1]

        # a value without the separator yields the full precursor id here
        if not charge.str.fullmatch(r"\d+").all():
            raise ValueError(
                f"Cannot extract charge: not all values of '{precursor_ids.name}' end "
                f"in '{self._PRECURSOR_ID_SEPARATOR}<charge>'."
            )

        return charge.astype(np.int8)

    def _extract_mod_seq_from_precursor_column(
        self, precursor_ids: pd.Series
    ) -> pd.Series:
        """Extract the modified sequence from all but the last field of the precursor id."""
        if not precursor_ids.str.contains(
            self._PRECURSOR_ID_SEPARATOR, regex=False
        ).all():
            raise ValueError(
                f"Cannot extract modified sequence: not all values of "
                f"'{precursor_ids.name}' contain '{self._PRECURSOR_ID_SEPARATOR}'."
            )

        return precursor_ids.str.rsplit(self._PRECURSOR_ID_SEPARATOR, n=1).str[0]

    def _pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Spectronaut report-specific preprocessing of output data.

        Derives
            - modified sequence
            - charge

        from the precursor id column, unless the report already provides them as dedicated columns.
        """
        precursor_id_column = self._get_actual_column(
            self.column_mapping.get(PsmDfCols.PRECURSOR_ID, []),
            df,
        )

        # No need to overwrite charge or modified sequence if the report contains them
        charge_available = PsmDfCols.CHARGE in get_column_mapping_for_df(
            self.column_mapping, df
        )
        mod_seq_available = self.mod_seq_column is not None

        # Return early if charge and modified sequence are available
        if charge_available and mod_seq_available:
            return df

        # If precursor_id is missing, there is no way to extract the information
        if precursor_id_column is None:
            warnings.warn(
                "Cannot extract charge and modified sequence column from available columns."
                "Please provide charge column and modified sequence column explicitly via the column_mapping.",
                category=UserWarning,
                stacklevel=2,
            )
            return df

        if not charge_available:
            df[PsmDfCols.CHARGE] = self._extract_charge_from_precursor_column(
                df[precursor_id_column]
            )

        if not mod_seq_available:
            # `import_file()` sets mod_seq_column to None. Update to extracted column (standardized name)
            # otherwise `_load_modifications()` looks up a `None` column
            self.mod_seq_column = PsmDfCols.MODIFIED_SEQUENCE
            df[self.mod_seq_column] = self._extract_mod_seq_from_precursor_column(
                df[precursor_id_column]
            )

        return df


def register_readers() -> None:
    """Register readers for Spectronaut's output library and reports, Swath data and DIANN data."""
    psm_reader_provider.register_reader("spectronaut", SpectronautReader)
    psm_reader_provider.register_reader("speclib_tsv", SpectronautReader)
    psm_reader_provider.register_reader("openswath", SwathReader)
    psm_reader_provider.register_reader("swath", SwathReader)
    psm_reader_provider.register_reader("diann", DiannReader)
    psm_reader_provider.register_reader("spectronaut_report", SpectronautReportReader)
