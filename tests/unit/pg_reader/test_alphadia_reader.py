"""Unit tests for AlphaDia PG reader."""

import os
from pathlib import Path
from typing import Any, Generator
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from alphabase.pg_reader import AlphaDiaPGReader
from alphabase.pg_reader.keys import PGCols


@pytest.fixture
def mock_yaml_config() -> dict[str, Any]:
    """Mock YAML configuration for AlphaDia."""
    return {
        "alphadia": {
            "reader_type": "alphadia",
            "column_mapping": {
                "proteins": "pg",
            },
            "measurement_regex": None,  # AlphaDia typically has minimal format
        }
    }


@pytest.fixture
def minimal_alphadia_df() -> pd.DataFrame:
    """Create a sample AlphaDia protein group dataframe.

    alphaDIA produces minimal output: features x samples
    with sample names as columns and protein IDs as index.
    """
    return pd.DataFrame(
        {
            "pg": ["P12345", "Q67890", "R11111"],
            "sample_01": [1000.5, 2000.3, 3000.7],
            "sample_02": [1100.2, 2100.8, 3100.4],
            "sample_03": [1200.9, 2200.1, 3200.5],
        }
    )


@pytest.fixture
def minimal_alphadia_df_standardized(minimal_alphadia_df: pd.DataFrame) -> pd.DataFrame:
    """Expected alphabase output from minimal alphadia input."""
    return minimal_alphadia_df.rename(columns={"pg": PGCols.PROTEINS}).set_index(
        PGCols.PROTEINS
    )


@pytest.fixture
def minimal_alphadia_tsv(
    tmp_path: Path, minimal_alphadia_df: pd.DataFrame
) -> Generator[str, None, None]:
    """Create temporary tsv file of PG matrix"""
    # Setup
    path = tmp_path / "alphadia.pg.tsv"
    minimal_alphadia_df.to_csv(path, sep="\t", index=False)

    # Yield
    yield str(path)

    # Tear down
    os.remove(path)


class TestAlphaDiaPGReaderInit:
    """Test initialization of AlphaDiaPGReader."""

    @patch("alphabase.pg_reader.pg_reader.pg_reader_yaml")
    def test_init_with_custom_params(
        self, mock_yaml: Mock, mock_yaml_config: dict[str, Any]
    ) -> None:
        """Test initialization with custom parameters."""
        mock_yaml.__getitem__.return_value = mock_yaml_config["alphadia"]

        custom_mapping = {"protein": "CustomProteinCol"}
        custom_regex = r"sample_\d+"

        reader = AlphaDiaPGReader(
            column_mapping=custom_mapping, measurement_regex=custom_regex
        )

        assert reader.column_mapping == custom_mapping
        assert reader.measurement_regex == custom_regex


class TestAlphaDiaPGReaderImport:
    """Test file import functionality."""

    @patch("alphabase.pg_reader.pg_reader.pg_reader_yaml")
    def test_import_standard_file(
        self,
        mock_yaml: Mock,
        mock_yaml_config: dict[str, Any],
        minimal_alphadia_df_standardized: pd.DataFrame,
        minimal_alphadia_tsv: str,
    ):
        """Test importing a standard AlphaDia PG file."""
        mock_yaml.__getitem__.return_value = mock_yaml_config["alphadia"]

        reader = AlphaDiaPGReader()

        result_df = reader.import_file(minimal_alphadia_tsv)

        # Check structure
        pd.testing.assert_frame_equal(result_df, minimal_alphadia_df_standardized)
