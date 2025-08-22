"""Integration tests for protein group reader."""

from alphabase.pg_reader import AlphaDiaPGReader
from alphabase.pg_reader.keys import PGCols


class TestAlphaDiaPGReaderImportIntegration:
    def test_import_real_file(self, example_alphadia_tsv: str) -> None:
        reader = AlphaDiaPGReader()

        result_df = reader.import_file(example_alphadia_tsv)

        assert result_df.shape == (9364, 6)
        assert result_df.index.name == PGCols.UNIPROT_IDS
