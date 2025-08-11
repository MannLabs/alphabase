"""AlphaPept protein group reader."""

import re
from typing import Any, Literal, Optional, Union

import pandas as pd

from .keys import PGCols
from .pg_reader import PGReaderBase, pg_reader_provider


class AlphaPeptPGReader(PGReaderBase):
    """Reader for protein group matrices from the alphapept search engine.

    Per default, the reader will read raw intensities from the protein group matrix. By passing a
    suitable regular expression, it is also possible to extract LFQ corrected intensities from the
    reader.

    Notes:
    -----
    AlphaPept protein group matrices contain both raw intensities and LFQ-corrected intensities.
    The LFQ-corrected intensities are marked by an `_LFQ` suffix.

    Example:
    -------
    Get example data

    .. code-block:: python

        import os
        import tempfile
        from alphabase.tools.data_downloader import DataShareDownloader
        from alphabase.pg_reader import AlphaPeptPGReader


        # Download to temporary directory
        URL = "https://datashare.biochem.mpg.de/s/6G6KHJqwcRPQiOO"
        download_dir = tempfile.mkdtemp()

        download_path = DataShareDownloader(url=URL, output_dir=download_dir).download()


    Per default, the reader will return the raw intensities. Additional protein features are stored
    in the dataframe index, samples are stored as columns.

    .. code-block:: python

        # Get raw intensities
        reader = AlphaPeptPGReader()
        results = reader.import_file(download_path)
        results.index.names
        > FrozenList(['proteins', 'uniprot_ids', 'ensembl_ids', 'source_db', 'is_decoy'])
        results.columns
        > Index(['A', 'B'], dtype='object')

    To read the LFQ values, pass the pre-configured key `lfq` to the reader, which represents a regular expression
    that automatically extracts the `LFQ` columns from the protein group table.

    .. code-block:: python

        # Get raw intensities
        reader = AlphaPeptPGReader(measurement_regex="lfq")
        results = reader.import_file(download_path)
        results.index.names
        > FrozenList(['proteins', 'uniprot_ids', 'ensembl_ids', 'source_db', 'is_decoy'])
        results.columns
        > Index(['A_LFQ', 'B_LFQ'], dtype='object')


    To checkout all preconfigured regular expressions, use the `get_preconfigured_regex` method:

    .. code-block:: python

        AlphaPeptPGReader.get_preconfigured_regex()
        > {'raw': '^.*(?<!_LFQ)$', 'lfq': '_LFQ$'}


    To get all columns (both raw or LFQ), pass a custom regular expression that matches any column name (e.g. `.*` )

    .. code-block:: python

        reader = AlphaPeptPGReader(measurement_regex=".*")
        results.columns
        > Index(['A_LFQ', 'B_LFQ', 'A', 'B'], dtype='object')


    If desired, remove the test data

    .. code-block:: python

        # Clean up
        os.rmdir(download_dir)

    """

    _reader_type: str = "alphapept"

    # Report file settings (delimiter + index column)
    _FILE_DELIMITER: str = ","
    # alphapept does not set a name for the feature column, i.e. it is set to the pandas default
    _INDEX_COL: str = "Unnamed: 0"

    # Feature settings
    # Decoys are prefixed with REV__ in alphapept
    _DECOY_REGEX: str = "^REV__"
    # Ensembl IDs are identified with a ENSEMBL prefix
    _ENSEMBL_REGEX: str = "^ENSEMBL:"
    # The expected length of fasta headers is 3 (sp|Uniprot ID|Uniprot Name)
    _FASTA_HEADER_DEFAULT_LENGTH: int = 3

    def __init__(  # noqa: D107 from base class
        self,
        *,
        column_mapping: Optional[dict[str, Any]] = None,
        measurement_regex: Union[str, Literal["raw", "lfq"], None] = "raw",  # noqa: PYI051 raw and lfq are special casees and not equivalent to string
    ):
        super().__init__(
            column_mapping=column_mapping, measurement_regex=measurement_regex
        )

    def _pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess of alphapept protein group report and return modified copy of the dataframe.

        Processes feature index to a parsed, streamlined version.

        Parameters
        ----------
        df
            alphapept protein group report.

        Returns
        -------
        :class:`pd.DataFrame`
            Modified copy of protein group report with parsed index. The index contains the levels
                - proteins: str
                - uniprot_ids: str
                - ensembl_ids: str
                - source_db: str
                - is_decoy: bool

        """
        df = df.copy()

        # alphapept does not set a name for the feature column
        # load it as regular column and set it to index afterwards
        df = df.set_index(self._INDEX_COL)

        # Parse index
        parsed_index: list[dict[str, str]] = list(
            df.index.map(lambda idx: self._parse_alphapept_index(idx))
        )

        # Overwrite index with streamlined version
        df.index = pd.MultiIndex.from_frame(pd.DataFrame(parsed_index))

        return df

    def _parse_alphapept_index(self, identifier: str) -> dict[str, str]:
        """Parse protein identifier from AlphaPept protein group table.

        Parameters
        ----------
        identifier : str
            Protein identifier string from AlphaPept

        Returns
        -------
        dict
            Dictionary with parsed components:
            - proteins: str, semicolon-separated protein names or "na"
            - uniprot_ids: str, semicolon-separated UniProt IDs or "na"
            - ensembl_ids: str, semicolon-separated ENSEMBL IDs or "na"
            - source_db: str, semicolon-separated data sources or "na"
            - is_decoy: bool, True if any identifier in a protein group starts with "REV__"

        Examples
        --------

        .. code-block:: python

            # sp|Q9NQT4|EXOS5_HUMAN
            {"source_db": "sp", "uniprot_ids": "Q9NQT4", "ensembl_ids": "na", "proteins": "EXOS5_HUMAN", "is_decoy": False}

            # Q0IIK2
            {"source_db": "na", "uniprot_ids": "Q0IIK2", "ensembl_ids": "na", "proteins": "na", "is_decoy": False}

            # "sp|Q9H2K8|TAOK3_HUMAN,sp|Q7L7X3|TAOK1_HUMAN"
            {"source_db": "sp;sp", "uniprot_ids": "Q9H2K8;Q7L7X3", "ensembl_ids": "na;na", "proteins": "TAOK3_HUMAN;TAOK1_HUMAN", "is_decoy": False}

            # ENSEMBL:ENSBTAP00000024146
            {"source_db": "ENSEMBL", "uniprot_ids": "na", "ensembl_ids": "ENSBTAP00000024146", "proteins": "na", "is_decoy": False}

            # ENSEMBL:ENSBTAP00000024146,sp|P35520|CBS_HUMAN
            {"source_db": "ENSEMBL;sp", "uniprot_ids": "P35520", "ensembl_ids": "ENSBTAP00000024146", "proteins": "CBS_HUMAN", "is_decoy": False}

            # REV__sp|Q13085|ACACA_HUMAN
            {"source_db": "REV__sp", "uniprot_ids": "Q13085", "ensembl_ids": "na", "proteins": "ACACA_HUMAN", "is_decoy": True}

        """
        decoy_pattern = re.compile(self._DECOY_REGEX)
        ensembl_pattern = re.compile(self._ENSEMBL_REGEX)

        # Multiple proteins are separted by comma
        protein_entries = identifier.split(",")

        source_db: list[str] = []
        uniprot_ids: list[str] = []
        ensembl_ids: list[str] = []
        proteins: list[str] = []
        is_decoy: list[bool] = []

        for entry in protein_entries:
            # Decoys
            # Identify decoys and remove decoy prefix if present
            entry_is_decoy = bool(decoy_pattern.search(entry))
            is_decoy.append(entry_is_decoy)

            # Check for ENSEMBL format (ENSEMBL:IDENTIFIER)
            if re.search(ensembl_pattern, entry):
                source_db.append("ENSEMBL")

                # Remove "ENSEMBL:" prefix
                uniprot_ids.append("na")
                proteins.append("na")
                ensembl_ids.append(re.sub(ensembl_pattern, "", entry))

            # Check if entry contains pipe separators (UniProt format)
            # Options:
            # sp|Q9H2K8|TAOK3_HUMAN
            # Q9H2K8

            # TODO: How to handle REV sequences here?
            # Currently they are only marked by the DECOY_INDICATOR flag, but should the individual identifiers be flagged as well?
            elif "|" in entry:
                parts = entry.split("|")
                if len(parts) == self._FASTA_HEADER_DEFAULT_LENGTH:
                    source_db.append(parts[0])
                    uniprot_ids.append(parts[1])
                    proteins.append(parts[2])
                    ensembl_ids.append("na")
                else:
                    # Handle unexpected format
                    source_db.append("na")
                    uniprot_ids.append("na")
                    proteins.append(entry)
                    ensembl_ids.append("na")
            else:
                # No pipes or ENSEMBL prefix, assume it's just a UniProt ID
                uniprot_ids.append(entry)
                source_db.append("na")
                proteins.append("na")
                ensembl_ids.append("na")

        # Join with semicolons or use "na" if empty
        source_db_str = ";".join(source_db) if source_db else "na"
        uniprot_ids_str = ";".join(uniprot_ids) if uniprot_ids else "na"
        ensembl_ids_str = ";".join(ensembl_ids) if ensembl_ids else "na"
        proteins_str = ";".join(proteins) if proteins else "na"
        is_decoy = any(is_decoy)

        return {
            PGCols.PROTEINS: proteins_str,
            PGCols.UNIPROT_IDS: uniprot_ids_str,
            PGCols.ENSEMBL_IDS: ensembl_ids_str,
            PGCols.SOURCE_DB: source_db_str,
            PGCols.DECOY_INDICATOR: is_decoy,
        }


pg_reader_provider.register_reader("alphapept", reader_class=AlphaPeptPGReader)
