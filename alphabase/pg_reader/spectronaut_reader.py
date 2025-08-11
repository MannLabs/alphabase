"""Spectronaut Protein Group Reader."""

import re
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd

from .pg_reader import PGReaderBase, pg_reader_provider


class SpectronautPGReader(PGReaderBase):
    """Reader for protein group matrices from the alphaDIA search engine."""

    _reader_type: str = "spectronaut"

    _to_nan_values: tuple[Any] = ("Filtered",)

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
        """Process Spectronaut protein group table after standardization.

        Notes
        -----
        Spectronaut reports might contain "Filtered" as values. Replace these values with NAN
        and assure that floating point values are returned

        """
        # Only modify the intensity columns, as defined by the `measurement_regex`
        pattern = re.compile(self.measurement_regex)
        regex_columns = [col for col in df.columns if re.search(pattern, col)]
        df[regex_columns] = df[regex_columns].replace(self._to_nan_values, np.nan)
        df[regex_columns] = df[regex_columns].astype(float)

        return df


pg_reader_provider.register_reader("spectronaut", reader_class=SpectronautPGReader)
