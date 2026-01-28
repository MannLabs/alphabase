"""PEAKS protein group and peptided wide table reader."""

from typing import Any, Literal, Optional, Union

import pandas as pd

from .keys import PGCols
from .pg_reader import PGReaderBase, pg_reader_provider

PROTEIN_GENE_SEPARATOR = "|"
ACCESSION_COLUMN = "Accession"
EXPECTED_PAIR_LENGTH = 2


def _assemble_pg_gene_groups(
    accession: str,
    id_separator: str = PROTEIN_GENE_SEPARATOR,
    group_separator: str = ":",
) -> tuple[str, str]:
    """Assemble protein and gene groups from PEAKS accession string.

    Parameters
    ----------
    accession
        Accession string containing protein|gene pairs separated by colons.
        Format: "protein1|gene1:protein2|gene2:protein3|gene3"
    id_separator
        Character separating protein from gene within each pair (default "|").
    group_separator
        Character separating multiple protein|gene pairs (default ":").

    Returns
    -------
    tuple[str, str]
        Tuple containing (protein_group, gene_group) as semicolon-separated strings.

    Example
    -------
        >>> accession = "P1|G1:P2|G2:P3|G3"
        >>> _assemble_pg_gene_groups(accession)
        ("P1;P2;P3", "G1;G2;G3")

    """
    pg_gene_list = [
        pair.split(id_separator) for pair in accession.split(group_separator)
    ]

    proteins = []
    genes = []

    for pair in pg_gene_list:
        if len(pair) >= EXPECTED_PAIR_LENGTH:
            proteins.append(pair[0])
            genes.append(pair[1])
        elif len(pair) == 1:
            # If no gene provided, just add the protein
            proteins.append(pair[0])

    return ";".join(proteins), ";".join(genes)


def _split_peaks_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Split PEAKS protein group identifiers into separate protein and gene columns.

    The 'Accession' column contains both the protein and gene identifiers separated by
    a pipe character. There may be multiple protein|gene pairs separated by a colon (??).
    This function assembles all proteins into one protein group and all genes into a gene
    group, separated by semicolons.

    Parameters
    ----------
    df
        DataFrame containing PEAKS protein group report.

    Returns
    -------
    pd.DataFrame
        Modified copy of the input DataFrame with separate 'protein' and 'gene' columns.

    """
    df_copy = df.copy()

    protein_groups = []
    gene_groups = []

    for accession in df_copy[ACCESSION_COLUMN]:
        protein_group, gene_group = _assemble_pg_gene_groups(accession)
        protein_groups.append(protein_group)
        gene_groups.append(gene_group)

    df_copy[PGCols.PROTEINS] = protein_groups
    df_copy[PGCols.GENES] = gene_groups

    return df_copy


def _reformat_observation_indices(
    df: pd.DataFrame,
    modality: Literal["peptide", "protein"],
) -> pd.DataFrame:
    """Parse PEAKS sample column names to standardized format.

    The peptide table contains sample names like Area Sample 1, Area Sample 2, etc.
    The protein table contains sample names like Sample 1 Area, Sample 2 Area, etc.

    We parse both to appropriate sample names like sample_1, sample_2, etc.

    Parameters
    ----------
    df
        DataFrame with PEAKS sample columns.
    modality
        Type of PEAKS table ("peptide" or "protein") to determine parsing strategy.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names.

    """
    if modality == "peptide":
        # Only rename columns that match "Area Sample X" pattern
        def parser(col_name: str) -> str:
            if col_name.startswith("Area Sample"):
                return col_name.lower().replace("area ", "").replace(" ", "_")
            return col_name
    elif modality == "protein":
        # Only rename columns that match "Sample X Area" pattern
        def parser(col_name: str) -> str:
            if col_name.startswith("Sample") and col_name.endswith("Area"):
                return col_name.lower().replace(" area", "").replace(" ", "_")
            return col_name
    else:
        raise ValueError(f"Unknown modality: {modality}")

    return df.rename(columns=parser)


class PeaksPGReader(PGReaderBase):
    """Reader for protein group matrices created by PEAKS."""

    _reader_type = "peaks_proteins"

    def __init__(
        self,
        *,
        column_mapping: Optional[dict[str, Any]] = None,
        measurement_regex: Union[str, Literal["lfq"], None] = "lfq",  # noqa: PYI051 lfq is a special case and not equivalent to string
    ):
        """Initialize PEAKS protein group matrix reader.

        Parameters
        ----------
        column_mapping
            Dictionary mapping alphabase column names (keys) to PEAKS column names (values).
            If `None`, uses default mapping from configuration file.
        measurement_regex
            Pattern to select quantity columns

                - "lfq": LFQ-corrected intensities (_LFQ suffix)

            See class documentation for usage examples and `get_preconfigured_regex()` for available patterns.

        """
        super().__init__(
            column_mapping=column_mapping, measurement_regex=measurement_regex
        )

    def _pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess PEAKS protein group report and return modified copy of the dataframe.

        The 'Accession' column contains both the protein and gene identifiers separated by
        a pipe character. This method splits these identifiers into a 'protein' and a 'gene' column.

        Additionally, parse the sample names to a better format.

        """
        df_processed = _split_peaks_identifiers(df)
        return _reformat_observation_indices(
            df_processed,
            modality="protein",
        )


pg_reader_provider.register_reader("peaks_proteins", reader_class=PeaksPGReader)


class PeaksPeptidesReader(PGReaderBase):
    """Reader for peptide matrices created by PEAKS."""

    _reader_type = "peaks_peptides"

    def __init__(
        self,
        *,
        column_mapping: Optional[dict[str, Any]] = None,
        measurement_regex: Union[str, Literal["lfq"], None] = "lfq",  # noqa: PYI051 lfq is a special case and not equivalent to string
    ):
        """Initialize PEAKS peptide matrix reader.

        Parameters
        ----------
        column_mapping
            Dictionary mapping alphabase column names (keys) to PEAKS column names (values).
            If `None`, uses default mapping from configuration file.
        measurement_regex
            Pattern to select quantity columns

                - "lfq": LFQ-corrected intensities (_LFQ suffix)

            See class documentation for usage examples and `get_preconfigured_regex()` for available patterns.

        """
        super().__init__(
            column_mapping=column_mapping, measurement_regex=measurement_regex
        )

    def _pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess PEAKS peptide report and return modified copy of the dataframe.

        The 'Accession' column contains both the protein and gene identifiers separated by
        a pipe character. This method splits these identifiers into a 'protein' and a 'gene' column.

        Additionally, parse the sample names to a better format.

        """
        df_processed = _split_peaks_identifiers(df)
        return _reformat_observation_indices(
            df_processed,
            modality="peptide",
        )


pg_reader_provider.register_reader("peaks_peptides", reader_class=PeaksPeptidesReader)
