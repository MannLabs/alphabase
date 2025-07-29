"""Integration tests for protein group reader provider."""

from alphabase.pg_reader import AlphaDiaPGReader, pg_reader_provider
from alphabase.pg_reader.keys import PGCols


class TestAlphaDiaPGReaderProvider:
    def test_reader_provider(self) -> None:
        """Test whether reader provider initializes alphadia PG reader correctly."""
        reader = pg_reader_provider.get_reader("alphadia")

        assert isinstance(reader, AlphaDiaPGReader)

    def test_reader_provider_import(self, example_alphadia_tsv: str) -> None:
        """Test if import works via `pg_reader_provider`"""
        reader = pg_reader_provider.get_reader("alphadia")

        result_df = reader.import_file(example_alphadia_tsv)

        assert result_df.shape == (9364, 6)
        assert result_df.index.name == PGCols.PROTEINS
