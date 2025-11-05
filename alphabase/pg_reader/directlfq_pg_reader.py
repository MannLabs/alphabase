"""AlphaDIA protein group reader."""

from .pg_reader import PGReaderBase, pg_reader_provider


class DirectLFQReader(PGReaderBase):
    """Reader for protein group matrices created by directLFQ."""

    _reader_type = "directlfq"


pg_reader_provider.register_reader("directlfq", reader_class=DirectLFQReader)
