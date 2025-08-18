"""Integration tests for protein group reader provider."""

from alphabase.pg_reader import AlphaDiaPGReader, DiannPGReader, pg_reader_provider


class TestAlphaDiaPGReaderProvider:
    def test_reader_provider(self) -> None:
        """Test whether reader provider initializes alphadia PG reader correctly."""
        reader = pg_reader_provider.get_reader("alphadia")

        assert isinstance(reader, AlphaDiaPGReader)


class TestDiannPGReaderProvider:
    def test_reader_provider(self) -> None:
        """Test whether reader provider initializes DIANN PG reader correctly."""
        reader = pg_reader_provider.get_reader("diann")

        assert isinstance(reader, DiannPGReader)
