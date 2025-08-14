"""Integration tests for protein group reader."""

import pandas as pd

from alphabase.pg_reader import AlphaDiaPGReader, DiannPGReader


class TestAlphaDiaPGReaderImportIntegration:
    def test_import_real_file(self, example_alphadia_tsv: str) -> None:
        """Test import of real AlphaDIA file"""
        file_path, reference = example_alphadia_tsv
        reader = AlphaDiaPGReader()

        result_df = reader.import_file(file_path)

        pd.testing.assert_frame_equal(result_df, reference)


class TestDiannPGReaderImportIntegration:
    def test_import_real_file(self, example_diann_tsv: str) -> None:
        """Test import of real DIANN file"""
        file_path, reference = example_diann_tsv
        reader = DiannPGReader()

        result_df = reader.import_file(file_path)

        pd.testing.assert_frame_equal(result_df, reference)
