"""Integration tests for protein group reader provider."""

from alphabase.pg_reader import AlphaDiaPGReader, DiannPGReader, pg_reader_provider
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
        assert result_df.index.name == PGCols.UNIPROT_IDS


class TestDiannPGReaderProvider:
    def test_reader_provider(self) -> None:
        """Test whether reader provider initializes DIANN PG reader correctly."""
        reader = pg_reader_provider.get_reader("diann")

        assert isinstance(reader, DiannPGReader)

    def test_reader_provider_import(self, example_diann_tsv: str) -> None:
        """Test if DIANN protein group report import works via `pg_reader_provider`"""
        reader = pg_reader_provider.get_reader("diann")

        result_df = reader.import_file(example_diann_tsv)

        assert result_df.shape == (10, 20)
        assert result_df.index.names == [
            PGCols.PROTEINS,
            PGCols.UNIPROT_IDS,
            PGCols.GENES,
            PGCols.PROTEIN_CANDIDATES,
            PGCols.DESCRIPTION,
        ]
