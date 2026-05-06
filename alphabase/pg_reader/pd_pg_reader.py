"""Proteome Discoverer protein group reader."""

from typing import Any, Optional

from .pg_reader import PGReaderBase, pg_reader_provider


class ProteomeDiscovererReader(PGReaderBase):
    """Reader for protein group matrices from the proteome discoverer search engine.

    Reads output from the proteome discoverer search engine by Thermo Fisher Scientific.
    Reads files with the `__Proteins.txt` suffix.

    References
    ----------
    - Orsburn BC. Proteome Discoverer-A Community Enhanced Data Processing Suite for Protein Informatics. Proteomes. 2021 Mar 23;9(1):15. doi: 10.3390/proteomes9010015. PMID: 33806881; PMCID: PMC8006021.

    """

    _reader_type = "proteome_discoverer"

    def __init__(
        self,
        *,
        column_mapping: Optional[dict[str, Any]] = None,
        measurement_regex: Optional[str] = "abundances_grouped",
    ) -> None:
        """Read protein group (PG) matrices into the standardized alphabase format.

        Parameters
        ----------
        column_mapping
            A dictionary of mapping alphabase columns (keys) to the corresponding columns in the other
            search engine (values). If `None` will be loaded from the `column_mapping` key of the respective
            search engine in `pg_reader.yaml`
        measurement_regex
            Regular expression that identifies correct measurement type. Only relevant if PG matrix contains multiple
            measurement types. For example, alphapept returns the raw protein intensity per sample in column `A` and the
            LFQ corrected value in `A_LFQ`. If `None` uses all columns.


        Attributes
        ----------
        column_mapping
            Dictionary structure mapping alphabase columns (keys) to the corresponding columns in the other
            search engine (values), see parameters.
        measurement_regex
            Regular expression that matches quantity of interest for all samples

        Notes
        -----
        Standardizes protein group reports to a protein group dataframe (features x samples) in wide format. Contains at least
            - sample (run) identifier: :att:`pg_reader.keys.PGCols.SAMPLE_NAME` as column index
            - protein group identifier: :att:`pg_reader.keys.PGCols.protein` as index
            - protein group intensity: :att:`pg_reader.keys.PGCols.INTENSITY` as values

        Additional feature-level metadata might be available in the index.

        """
        super().__init__(
            column_mapping=column_mapping, measurement_regex=measurement_regex
        )


pg_reader_provider.register_reader(
    "proteome_discoverer", reader_class=ProteomeDiscovererReader
)
