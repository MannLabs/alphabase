"""Integration tests for protein group reader provider."""

import pytest

from alphabase.pg_reader import (
    AlphaDiaPGReader,
    AlphaPeptPGReader,
    DiannPGReader,
    pg_reader_provider,
)
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
        """Test if diann protein group report import works via `pg_reader_provider`"""
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


class TestAlphapeptPGReaderProvider:
    def test_reader_provider(self) -> None:
        """Test whether reader provider initializes alphapept PG reader correctly."""
        reader = pg_reader_provider.get_reader("alphapept")

        assert isinstance(reader, AlphaPeptPGReader)

    @pytest.mark.parametrize(
        ("measurement_regex", "expected_shape", "expected_colums"),
        [
            # Default
            (None, (3781, 2), ["A", "B"]),
            # LFQ
            ("LFQ", (3781, 2), ["A_LFQ", "B_LFQ"]),
            # Get all
            (".*", (3781, 4), ["A_LFQ", "B_LFQ", "A", "B"]),
        ],
    )
    def test_reader_provider_import(
        self,
        example_alphapept_csv: str,
        measurement_regex: str,
        expected_shape: tuple[int, int],
        expected_colums: list[str],
    ) -> None:
        """Test if diann protein group report import works via `pg_reader_provider`"""
        reader = pg_reader_provider.get_reader(
            "alphapept", measurement_regex=measurement_regex
        )

        result_df = reader.import_file(example_alphapept_csv)

        assert result_df.shape == expected_shape
        assert list(result_df.columns) == expected_colums
        assert result_df.index.names == [
            PGCols.PROTEINS,
            PGCols.UNIPROT_IDS,
            PGCols.ENSEMBL_IDS,
            PGCols.SOURCE_DB,
            PGCols.DECOY_INDICATOR,
        ]
