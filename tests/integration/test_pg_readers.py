"""Integration tests for protein group reader."""

from alphabase.pg_reader import AlphaDiaPGReader, DiannPGReader
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
