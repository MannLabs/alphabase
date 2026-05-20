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
        measurement_regex: Optional[str] = "abundances",
    ) -> None:
        """Read protein group (PG) matrices from the proteome discoverer search engine into the standardized alphabase format.

        Parameters
        ----------
        column_mapping
            Dictionary mapping alphabase column names (keys) to Proteome Discoverer column names (values).
            If `None`, uses default mapping from configuration file.
        measurement_regex
            Pattern to select quantity columns from Proteome Discoverer output

                - `abundance` (default): Raw intensities per sample
                - `abundances_normalized`: Normalized intensities per sample
                - `abundances_scaled`: Scaled intensities per sample
                - `abundances_grouped`: Aggregated abundance values of a user-defined sample group.

            See :meth:`alphabase.pg_reader.ProteomeDiscovererReader.get_preconfigured_regex()` for all available patterns.

        See Also
        --------
        :class:`alphabase.pg_reader.pg_reader.PGReaderBase`

        """
        super().__init__(
            column_mapping=column_mapping, measurement_regex=measurement_regex
        )


pg_reader_provider.register_reader(
    "proteome_discoverer", reader_class=ProteomeDiscovererReader
)
