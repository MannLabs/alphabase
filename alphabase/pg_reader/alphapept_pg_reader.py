"""AlphaPept protein group reader."""

import re

import pandas as pd

from .keys import PGCols
from .pg_reader import PGReaderBase, pg_reader_provider


class AlphaPeptPGReader(PGReaderBase):
    """Reader for protein group matrices from the alphapept search engine."""

    _reader_type: str = "alphapept"

    # Report file settings (delimiter + index column)
    _FILE_DELIMITER: str = ","
    _INDEX_COL: str = 0

    # Feature settings
    # Decoys are prefixed with REV__ in alphapept
    _DECOY_REGEX: str = "^REV__"
    # Ensembl IDs are identified with a ENSEMBL prefix
    _ENSEMBL_REGEX: str = "^ENSEMBL:"
    # The expected length of fasta headers is 3 (sp|Uniprot ID|Uniprot Name)
    _FASTA_HEADER_DEFAULT_LENGTH: int = 3

    def _load_file(self, file_path: str) -> pd.DataFrame:
        """Load protein group (PG) file from alphapept into a dataframe.

        Parameters
        ----------
        file_path
            File path, must be an alphapept `.csv` proteing roup report.

        Returns
        -------
        :class:`pd.DataFrame`
            Protein group matrix

        """
        # alphapept does not name its feature index column, i.e. its not possible to use the standard
        # column mapping procedure
        # The easiest way is to overwrite the `_load_file` method and set the feature column directly as index
        return pd.read_csv(
            file_path,
            sep=self._FILE_DELIMITER,
            keep_default_na=False,
            index_col=self._INDEX_COL,
        )

    def _pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess of alphapept protein group report and return modified copy of the dataframe.

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
        # Parse index
        parsed_index: list[dict[str, str]] = list(
            df.index.map(lambda idx: self._parse_alphapept_index(idx))
        )

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
            - is_decoy: bool, True if identifier starts with "REV__"

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
            {"source_db": "sp", "uniprot_ids": "Q13085", "ensembl_ids": "na", "proteins": "ACACA_HUMAN", "is_decoy": True}

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
            entry_is_decoy = bool(decoy_pattern.match(identifier))

            if entry_is_decoy:
                identifier = re.sub(decoy_pattern, "", identifier)

            is_decoy.append(entry_is_decoy)

            # Check for ENSEMBL format (ENSEMBL:IDENTIFIER)
            if re.match(ensembl_pattern, entry):
                source_db.append("ENSEMBL")

                # Remove "ENSEMBL:" prefix
                ensembl_ids.append(re.sub(ensembl_pattern, entry, ""))

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
                else:
                    # Handle unexpected format
                    source_db.append("na")
                    uniprot_ids.append("na")
                    proteins.append(entry)
            else:
                # No pipes or ENSEMBL prefix, assume it's just a UniProt ID
                uniprot_ids.append(entry)
                source_db.append("na")
                proteins.append("na")

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
