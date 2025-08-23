"""FragPipe protein group reader."""

from typing import Literal, Optional, Union

from .pg_reader import PGReaderBase, pg_reader_provider


class FragPipePGReader(PGReaderBase):
    """Reader for `protein.tsv` reports from FragPipe.

    Example:
    -------
    Per default, the reader will return the raw intensities from the `razor` method. Additional protein features are stored
    in the dataframe index, samples are stored as columns.

    .. code-block:: python

        # Get raw intensities
        reader = FragPipePGReader()
        results = reader.import_file(download_path)


    References:
    ----------
    - FragPipe Documentation https://fragpipe.nesvilab.org/docs/tutorial_fragpipe_outputs.html#proteintsv

    """

    _reader_type: str = "fragpipe"

    def __init__(  # noqa: D107 inherited from base class
        self,
        *,
        column_mapping: Optional[dict[str, str]] = None,
        measurement_regex: Union[
            Literal[
                "raw", "razor", "unique", "total", "lfq", "lfq_unique", "lfq_total"
            ],
            None,
        ] = "razor",
    ):
        super().__init__(
            column_mapping=column_mapping, measurement_regex=measurement_regex
        )


pg_reader_provider.register_reader("fragpipe", reader_class=FragPipePGReader)
