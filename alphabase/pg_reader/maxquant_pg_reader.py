"""MaxQuant Protein Group Reader."""

from typing import Literal, Optional, Union

import pandas as pd

from .keys import PGCols
from .pg_reader import PGReaderBase, pg_reader_provider


class MaxQuantPGReader(PGReaderBase):
    r"""Reader for protein group matrices from the MaxQuant search engine.

    By default, the reader will read raw protein intensities from the protein group matrix. By passing
    a suitable regular expression, it is also possible to extract LFQ

    Examples
    --------
    Get example data

    .. code-block:: python

        import os
        import tempfile
        from alphabase.tools.data_downloader import DataShareDownloader
        from alphabase.pg_reader import MaxQuantPGReader


        # Download to temporary directory
        URL = "https://datashare.biochem.mpg.de/s/KvToteOu0zzH17C"
        download_dir = tempfile.mkdtemp()

        download_path = DataShareDownloader(url=URL, output_dir=download_dir).download()


    Per default, the reader will return the raw intensities. Additional protein features are stored
    in the dataframe index, samples are stored as columns.

    .. code-block:: python

        # Get raw intensities
        reader = MaxQuantPGReader()
        results = reader.import_file(download_path)
        results.index.names
        > FrozenList(['proteins', 'uniprot_ids', 'genes', 'is_decoy'])
        results.columns
        > Index([...], dtype='object', length=312)


    To read the LFQ values, pass the pre-configured key `lfq` to the reader, which represents a regular expression
    that automatically extracts the `LFQ` columns from the protein group table.

    .. code-block:: python

        # Get raw intensities
        reader = MaxQuantPGReader(measurement_regex="lfq")
        results = reader.import_file(download_path)

    To checkout all preconfigured regular expressions that enable you to retrieve different intensity modalities,
    use the `get_preconfigured_regex` method:

    .. code-block:: python

        MaxQuantPGReader.get_preconfigured_regex()
        > {
            'raw': '^Intensity(?!\\s[LHM]\\s).+$',
            'lfq': '^LFQ intensity(?!\\s[LHM]\\s).+$',
            'ibaq': '^iBAQ(?!\\s[LHM]\\s).+$'
        }

    You can also pass a custom regular expression, e.g. to retrieve specific channels in TMT experiments

    .. code-block:: python

        # Match "Intensity H+ <sample>"
        reader = MaxQuantPGReader(measurement_regex="^Intensity H .+")


    If desired, remove the test data

    .. code-block:: python

        # Clean up
        os.rmdir(download_dir)


    References
    ----------
    - MaxQuant Documentation (Cox Lab, 2024-06-27): https://cox-labs.github.io/coxdocs/output_tables.html#protein-groups,
    (last viewed 2025-08)

    """

    _reader_type = "maxquant"

    def __init__(  # noqa: D107 inherited from base class
        self,
        *,
        column_mapping: Optional[dict[str, str]] = None,
        measurement_regex: Union[str, Literal["raw", "lfq", "ibaq"], None] = "raw",  # noqa: PYI051 raw and lfq are special casees and not equivalent to string
    ):
        super().__init__(
            column_mapping=column_mapping, measurement_regex=measurement_regex
        )

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process MaxQuant protein group table after standardization.

        Convert MaxQuant-specific decoy indicator (+) to standardized boolean series.

        Notes
        -----
        MaxQuant marks peptides/proteins that were found to be part of a protein derived from the reversed part of the decoy database
        with +. These should be removed for further data analysis.

        References
        ----------
        https://cox-labs.github.io/coxdocs/output_tables.html#protein-groups (Status: 2025-08)

        """
        # Convert `+` indicator to boolean
        if PGCols.DECOY_INDICATOR in df.columns:
            df[PGCols.DECOY_INDICATOR] = df[PGCols.DECOY_INDICATOR].apply(
                lambda x: x == "+"
            )
        return df


pg_reader_provider.register_reader("maxquant", reader_class=MaxQuantPGReader)
