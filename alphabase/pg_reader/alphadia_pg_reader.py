"""AlphaDIA protein group reader."""

from .pg_reader import PGReaderBase, pg_reader_provider


class AlphaDiaPGReader(PGReaderBase):
    """Reader for protein group matrices from the alphaDIA search engine."""

    _reader_type = "alphadia"


pg_reader_provider.register_reader("alphadia", reader_class=AlphaDiaPGReader)
