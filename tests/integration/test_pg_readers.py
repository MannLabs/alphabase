"""Integration tests for protein group reader."""

import pandas as pd
import pytest

from alphabase.pg_reader import (
    AlphaDiaPGReader,
    AlphaPeptPGReader,
    DiannPGReader,
    FragPipePGReader,
    MaxQuantPGReader,
    SpectronautPGReader,
)
from alphabase.pg_reader.keys import PGCols


class TestAlphaDiaPGReaderImportIntegration:
    def test_import_real_file(
        self, example_alphadia_tsv: tuple[str, pd.DataFrame]
    ) -> None:
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


class TestAlphapeptPGReaderImportIntegration:
    def test_import_csv_file_equivalent(
        self, example_alphapept_csv: tuple[str, pd.DataFrame]
    ):
        """Test that AlphaPeptPGReader default import is exactly equivalent to reference"""
        file_path, reference = example_alphapept_csv
        reader = AlphaPeptPGReader()

        result_df = reader.import_file(file_path)

        pd.testing.assert_frame_equal(result_df, reference)

    def test_import_hdf_file_equivalent(
        self, example_alphapept_hdf: tuple[str, pd.DataFrame]
    ):
        """Test that AlphaPeptPGReader default import is exactly equivalent to reference"""

        file_path, reference = example_alphapept_hdf
        reader = AlphaPeptPGReader()

        result_df = reader.import_file(file_path)

        pd.testing.assert_frame_equal(result_df, reference)

    @pytest.mark.parametrize(
        ("measurement_regex", "expected_shape", "expected_colums"),
        [
            # Default
            ("raw", (9, 2), ["A", "B"]),
            # Match lfq key in config
            ("lfq", (9, 2), ["A_LFQ", "B_LFQ"]),
            # custom - match LFQ
            ("LFQ", (9, 2), ["A_LFQ", "B_LFQ"]),
            # Get all
            (".*", (9, 4), ["A_LFQ", "B_LFQ", "A", "B"]),
            # Pass None
            (None, (9, 4), ["A_LFQ", "B_LFQ", "A", "B"]),
        ],
    )
    def test_import_csv_file(
        self,
        example_alphapept_csv: tuple[str, pd.DataFrame],
        measurement_regex: str,
        expected_shape: tuple[int, int],
        expected_colums: list[str],
    ) -> None:
        """Test alphapept protein group reader import with real data from alphapept csv report and different parameter combinations.

        Tests whether the reader can import raw data (default), LFQ data, and all columns
        """
        file_path, _ = example_alphapept_csv

        reader = AlphaPeptPGReader(measurement_regex=measurement_regex)

        result_df = reader.import_file(file_path)

        assert result_df.shape == expected_shape
        assert list(result_df.columns) == expected_colums
        assert result_df.index.names == [
            PGCols.PROTEINS,
            PGCols.UNIPROT_IDS,
            PGCols.ENSEMBL_IDS,
            PGCols.SOURCE_DB,
            PGCols.DECOY_INDICATOR,
        ]

    @pytest.mark.parametrize(
        ("measurement_regex", "expected_shape", "expected_colums"),
        [
            # Default
            ("raw", (3781, 2), ["A", "B"]),
            # Match lfq key in config
            ("lfq", (3781, 2), ["A_LFQ", "B_LFQ"]),
            # custom - match LFQ
            ("LFQ", (3781, 2), ["A_LFQ", "B_LFQ"]),
            # Get all
            (".*", (3781, 4), ["A_LFQ", "B_LFQ", "A", "B"]),
            # Pass None
            (None, (3781, 4), ["A_LFQ", "B_LFQ", "A", "B"]),
        ],
    )
    def test_import_hdf_file(
        self,
        example_alphapept_hdf: tuple[str, pd.DataFrame],
        measurement_regex: str,
        expected_shape: tuple[int, int],
        expected_colums: list[str],
    ) -> None:
        """Test alphapept protein group reader import with real data from alphapept hdf report and different parameter combinations.

        Tests whether the reader can import raw data (default), LFQ data, and all columns
        """
        file_path, _ = example_alphapept_hdf
        reader = AlphaPeptPGReader(measurement_regex=measurement_regex)

        result_df = reader.import_file(file_path)

        assert result_df.shape == expected_shape
        assert list(result_df.columns) == expected_colums
        assert result_df.index.names == [
            PGCols.PROTEINS,
            PGCols.UNIPROT_IDS,
            PGCols.ENSEMBL_IDS,
            PGCols.SOURCE_DB,
            PGCols.DECOY_INDICATOR,
        ]


class TestMaxQuantPGReader:
    def test_import(self, example_maxquant_tsv: str) -> None:
        """Test import of real MaxQuant file"""
        file_path, reference = example_maxquant_tsv

        reader = MaxQuantPGReader()
        result_df = reader.import_file(file_path=file_path)

        pd.testing.assert_frame_equal(result_df, reference)

    @pytest.mark.parametrize(("measurement_regex",), [("raw",), ("lfq",)])
    def test_measurement_regex(
        self, example_maxquant_tsv: tuple[str, pd.DataFrame], measurement_regex: str
    ) -> None:
        """Test import with different regular expressions"""
        file_path, _ = example_maxquant_tsv

        reader = MaxQuantPGReader(measurement_regex=measurement_regex)

        result_df = reader.import_file(file_path=file_path)

        assert result_df.shape == (9, 312)
        assert result_df.index.names == [
            PGCols.PROTEINS,
            PGCols.UNIPROT_IDS,
            PGCols.GENES,
            PGCols.DECOY_INDICATOR,
        ]


class TestSpectronautPGReader:
    def test_import_real_file_tsv(self, example_spectronaut_tsv: str) -> None:
        """Test import of real spectronaut file"""
        file_path, reference = example_spectronaut_tsv

        reader = SpectronautPGReader()

        result_df = reader.import_file(file_path=file_path)

        pd.testing.assert_frame_equal(result_df, reference)

    def test_import_real_file_parqet(self, example_spectronaut_parquet: str) -> None:
        """Test import of real spectronaut file"""
        file_path, reference = example_spectronaut_parquet

        reader = SpectronautPGReader()

        result_df = reader.import_file(file_path=file_path)

        pd.testing.assert_frame_equal(result_df, reference)


class TestFragPipePGReader:
    def test_import_real_file(self, example_fragpipe_tsv: str) -> None:
        """Test import of real FragPipe file"""
        file_path, reference = example_fragpipe_tsv

        reader = FragPipePGReader()

        result_df = reader.import_file(file_path=file_path)

        pd.testing.assert_frame_equal(result_df, reference)
