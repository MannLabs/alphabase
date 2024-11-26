"""Reader for AlphaDia data."""

from abc import ABC

import pandas as pd

from alphabase.psm_reader.psm_reader import PSMReaderBase, psm_reader_provider


class AlphaDiaReader(PSMReaderBase, ABC):
    """Reader for AlphaDia data."""

    _reader_type = "alphadia"

    def _translate_modifications(self) -> None:
        """Nothing to translate for AlphaDIA."""


class AlphaDiaReaderTsv(AlphaDiaReader):
    """Reader for AlphaDia TSV files."""


class AlphaDiaReaderParquet(AlphaDiaReader):
    """Reader for AlphaDia parquet files."""

    def _load_file(self, filename: str) -> pd.DataFrame:
        """Read a parquet file."""
        return pd.read_parquet(filename)


def register_readers() -> None:
    """Register AlphaDIA reader."""
    psm_reader_provider.register_reader("alphadia", AlphaDiaReaderTsv)
    psm_reader_provider.register_reader("alphadia_parquet", AlphaDiaReaderParquet)
