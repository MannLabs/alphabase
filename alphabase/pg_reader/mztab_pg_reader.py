"""FragPipe protein group reader."""

from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd

from .pg_reader import PGReaderBase, pg_reader_provider


class MZTabPGReader(PGReaderBase):
    """Reader for MZTab search engine output.

    MZTab is a standardized tab-delimited format for reporting proteomics and metabolomics results.
    The format organizes data into distinct sections: metadata (MTD), protein groups (PRH/PRT),
    peptides (PEH/PEP), PSMs (PSH/PSM), and small molecules (SMH/SML), with each section identified
    by specific three-letter prefixes. This reader extracts protein-level quantification data from
    the PRT lines, which contain protein abundances across samples or study variables.

    Example:
    -------
    Per default, the reader will return the raw intensities from the `razor` method. Additional protein features are stored
    in the dataframe index, samples are stored as columns.

    .. code-block:: python

        from alphabase.pg_reader import MZTabPGReader

        # Get raw intensities
        reader = MZTabPGReader()
        results = reader.import_file(path)


    References:
    ----------
    - Griss, J. et al. The mzTab Data Exchange Format: Communicating Mass-spectrometry-based Proteomics and Metabolomics Experimental Results to a Wider Audience*. Molecular & Cellular Proteomics 13, 2765-2775 (2014).
    - Official MZTab Repository: https://github.com/HUPO-PSI/mzTab.git
    - Official documentation: https://hupo-psi.github.io/mzTab/

    """

    _reader_type: str = "mztab"

    _PROTEIN_ROW_INDICATOR: str = "PRT"
    _PROTEIN_HEADER_INDICATOR: str = "PRH"
    _SEPARATOR: str = "\t"

    def __init__(  # noqa: D107 inherited from base class
        self,
        *,
        column_mapping: Optional[dict[str, str]] = None,
        measurement_regex: Union[
            str, Literal["assay", "study_variable"], None  # noqa: PYI051 raw and lfq are special cases and not equivalent to string
        ] = "assay",
    ):
        super().__init__(
            column_mapping=column_mapping, measurement_regex=measurement_regex
        )

    def _load_file(self, file_path: str) -> pd.DataFrame:
        """Load MZTab file and extract protein data section.

        Parameters
        ----------
        file_path : str
            Path to MZTab file

        Returns
        -------
        pd.DataFrame
            DataFrame containing protein data from MZTab file

        Notes
        -----
        Protein lines are indicated with a leading `PRT`. The protein metadata header is
        indicated with a leading `PRH`. The file is tab separated.

        Raises
        ------
        ValueError
            If no protein data or metadata is found in the file

        """
        file_path = Path(file_path)
        protein_header = None
        protein_rows = []

        with file_path.open() as f:
            for line in f:
                line_stripped = line.strip()

                if line_stripped.startswith(self._PROTEIN_HEADER_INDICATOR):
                    # Protein header line - remove 'PRH' prefix and parse columns
                    header_content = line_stripped[3:].strip()
                    protein_header = header_content.split(self._SEPARATOR)

                elif line_stripped.startswith(self._PROTEIN_ROW_INDICATOR):
                    # Protein data line - remove 'PRT' prefix and parse data
                    row_content = line_stripped[3:].strip()
                    protein_rows.append(row_content.split(self._SEPARATOR))

        # Validate that we found protein data
        if protein_header is None:
            raise ValueError(
                f"No protein header ({self._PROTEIN_HEADER_INDICATOR}) found in MZTab file"
            )

        if not protein_rows:
            raise ValueError(
                f"No protein data rows ({self._PROTEIN_ROW_INDICATOR}) found in MZTab file"
            )

        return pd.DataFrame(protein_rows, columns=protein_header)


pg_reader_provider.register_reader("mztab", reader_class=MZTabPGReader)
