"""Integration tests for protein group reader."""

import pytest

from alphabase.pg_reader import AlphaDiaPGReader, AlphaPeptPGReader, DiannPGReader
from alphabase.pg_reader.keys import PGCols


class TestAlphaDiaPGReaderImportIntegration:
    def test_import_real_file(self, example_alphadia_tsv: str) -> None:
        reader = AlphaDiaPGReader()

        result_df = reader.import_file(example_alphadia_tsv)

        assert result_df.shape == (9364, 6)
        assert result_df.index.name == PGCols.UNIPROT_IDS


class TestDiannPGReaderImportIntegration:
    def test_import_real_file(self, example_diann_tsv: str) -> None:
        """Test import of real DIANN file"""
        reader = DiannPGReader()

        result_df = reader.import_file(example_diann_tsv)

        assert result_df.shape == (10, 20)
        assert result_df.index.names == [
            PGCols.PROTEINS,
            PGCols.UNIPROT_IDS,
            PGCols.GENES,
            PGCols.PROTEIN_CANDIDATES,
            PGCols.DESCRIPTION,
        ]


class TestAlphapeptPGReaderImportIntegration:
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
    def test_import_real_file(
        self,
        example_alphapept_csv: str,
        measurement_regex: str,
        expected_shape: tuple[int, int],
        expected_colums: list[str],
    ) -> None:
        """Test alphapept protein group reader import with real data.

        Tests whether the reader can import raw data (default), LFQ data, and all columns
        """
        reader = AlphaPeptPGReader(measurement_regex=measurement_regex)

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
