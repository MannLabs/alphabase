"""Proteome Discoverer protein group reader."""

from .pg_reader import PGReaderBase, pg_reader_provider


class ProteomeDiscovererReader(PGReaderBase):
    """Reader for protein group matrices from the proteome discoverer search engine."""

    _reader_type = "proteome_discoverer"


pg_reader_provider.register_reader(
    "proteome_discoverer", reader_class=ProteomeDiscovererReader
)
