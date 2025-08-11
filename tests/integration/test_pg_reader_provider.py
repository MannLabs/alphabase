"""Integration tests for protein group reader provider."""

from alphabase.pg_reader import (
    AlphaDiaPGReader,
    AlphaPeptPGReader,
    DiannPGReader,
    MaxQuantPGReader,
    SpectronautPGReader,
    pg_reader_provider,
)


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


class TestAlphapeptPGReaderProvider:
    def test_reader_provider(self) -> None:
        """Test whether reader provider initializes alphapept protein group reader correctly."""
        reader = pg_reader_provider.get_reader("alphapept")

        assert isinstance(reader, AlphaPeptPGReader)


class TestMaxQuantPGReaderProvider:
    def test_reader_provider(self) -> None:
        """Test whether reader provider initializes MaxQuant protein group reader correctly."""
        reader = pg_reader_provider.get_reader("maxquant")

        assert isinstance(reader, MaxQuantPGReader)


class TestSpectronautPGReaderProvider:
    def test_reader_provider(self) -> None:
        """Test whether reader provider initializes spectronaut protein group reader correctly."""
        reader = pg_reader_provider.get_reader("spectronaut")

        assert isinstance(reader, SpectronautPGReader)
