"""DIANN protein group reader."""

from .pg_reader import PGReaderBase, pg_reader_provider


class DiannPGReader(PGReaderBase):
    """Reader for protein group matrices from the DIANN search engine."""

    _reader_type = "diann"


pg_reader_provider.register_reader("diann", reader_class=DiannPGReader)
