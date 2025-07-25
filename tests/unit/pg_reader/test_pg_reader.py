"""Unit tests for PGReaderBase class."""

import os
from typing import Any, Generator
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from alphabase.pg_reader.pg_reader import PGReaderBase


class ExamplePGReader(PGReaderBase):
    """Test class"""

    _reader_type = "test_reader"


@pytest.fixture
def mock_yaml_data() -> dict[str, Any]:
    """Mock YAML configuration data."""
    return {
        "test_reader": {
            "column_mapping": {
                "protein": "Protein ID",
                "gene": "Gene Name",
                "description": "Description",
            },
            "measurement_regex": r"Sample_\d+_LFQ",
        }
    }


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample dataframe for testing."""
    return pd.DataFrame(
        {
            "Protein ID": ["P001", "P002", "P003"],
            "Gene Name": ["GENE1", "GENE2", "GENE3"],
            "Description": ["Desc1", "Desc2", "Desc3"],
            "Sample_1_LFQ": [100.0, 200.0, 300.0],
            "Sample_2_LFQ": [150.0, 250.0, 350.0],
            "Sample_1_raw": [90.0, 190.0, 290.0],
            "Sample_2_raw": [140.0, 240.0, 340.0],
        }
    )


@pytest.fixture
def sample_csv(tmp_path, sample_df: pd.DataFrame) -> Generator[str, None, None]:
    """Create temporary csv file of PG matrix"""
    # Setup
    path = tmp_path / "sample.pg.csv"
    sample_df.to_csv(path, index=True)

    # Yield
    yield str(path)

    # Tear down
    os.remove(path)


@pytest.fixture
def sample_tsv(tmp_path, sample_df: pd.DataFrame) -> Generator[str, None, None]:
    """Create temporary tsv file of PG matrix"""
    # Setup
    path = tmp_path / "sample.pg.tsv"
    sample_df.to_csv(path, sep="\t", index=True)

    # Yield
    yield str(path)

    # Tear down
    os.remove(path)


@pytest.fixture
def sample_hdf(tmp_path, sample_df: pd.DataFrame) -> Generator[str, None, None]:
    """Create temporary hdf file of PG matrix"""
    # Setup
    path = tmp_path / "sample.pg.hdf"
    sample_df.reset_index().to_hdf(path, key=None)

    # Yield
    yield str(path)

    # Tear down
    os.remove(path)


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Create an empty dataframe."""
    return pd.DataFrame()


class TestPGReaderBaseInit:
    """Test initialization of PGReaderBase."""

    @patch("alphabase.pg_reader.pg_reader.pg_reader_yaml")
    def test_init_with_defaults(
        self, mock_yaml: Mock, mock_yaml_data: dict[str, Any]
    ) -> None:
        """Test initialization with default values from YAML."""
        mock_yaml.__getitem__.return_value = mock_yaml_data["test_reader"]

        reader = ExamplePGReader()

        assert reader.column_mapping == mock_yaml_data["test_reader"]["column_mapping"]
        assert (
            reader.measurement_regex
            == mock_yaml_data["test_reader"]["measurement_regex"]
        )

    @patch("alphabase.pg_reader.pg_reader.pg_reader_yaml")
    def test_init_with_custom_column_mapping(
        self, mock_yaml: Mock, mock_yaml_data: dict[str, Any]
    ) -> None:
        """Test initialization with custom column mapping."""
        mock_yaml.__getitem__.return_value = mock_yaml_data["test_reader"]
        custom_mapping = {"protein": "ProteinCol", "gene": "GeneCol"}

        reader = ExamplePGReader(column_mapping=custom_mapping)

        assert reader.column_mapping == custom_mapping
        assert (
            reader.measurement_regex
            == mock_yaml_data["test_reader"]["measurement_regex"]
        )

    @patch("alphabase.pg_reader.pg_reader.pg_reader_yaml")
    def test_init_with_custom_measurement_regex(
        self, mock_yaml: Mock, mock_yaml_data: dict[str, Any]
    ) -> None:
        """Test initialization with custom measurement regex."""
        mock_yaml.__getitem__.return_value = mock_yaml_data["test_reader"]
        custom_regex = r".*_intensity$"

        reader = ExamplePGReader(measurement_regex=custom_regex)

        assert reader.column_mapping == mock_yaml_data["test_reader"]["column_mapping"]
        assert reader.measurement_regex == custom_regex


class TestAddColumnMapping:
    """Test add_column_mapping method."""

    @patch("alphabase.pg_reader.pg_reader.pg_reader_yaml")
    def test_add_new_mapping(self, mock_yaml, mock_yaml_data):
        """Test adding new column mappings."""
        mock_yaml.__getitem__.return_value = mock_yaml_data["test_reader"]
        reader = ExamplePGReader()

        new_mapping = {"peptides": "Peptide Count"}
        reader.add_column_mapping(new_mapping)

        expected = {**mock_yaml_data["test_reader"]["column_mapping"], **new_mapping}
        assert reader.column_mapping == expected

    @patch("alphabase.pg_reader.pg_reader.pg_reader_yaml")
    def test_override_existing_mapping(self, mock_yaml, mock_yaml_data):
        """Test overriding existing column mappings."""
        mock_yaml.__getitem__.return_value = mock_yaml_data["test_reader"]
        reader = ExamplePGReader()

        override_mapping = {"protein": "New Protein Col"}
        reader.add_column_mapping(override_mapping)

        assert reader.column_mapping["protein"] == "New Protein Col"
        assert reader.column_mapping["gene"] == "Gene Name"  # Unchanged


class TestPreProcess:
    """Test _pre_process method."""

    @patch("alphabase.pg_reader.pg_reader.pg_reader_yaml")
    def test_pre_process_default(self, mock_yaml, mock_yaml_data, sample_df):
        """Test default pre-processing (should return unchanged df)."""
        mock_yaml.__getitem__.return_value = mock_yaml_data["test_reader"]
        reader = ExamplePGReader()

        result_df = reader._pre_process(sample_df)
        pd.testing.assert_frame_equal(result_df, sample_df)


class TestTranslateColumns:
    """Test _translate_columns method."""

    @patch("alphabase.pg_reader.pg_reader.pg_reader_yaml")
    def test_translate_columns(self, mock_yaml, mock_yaml_data, sample_df):
        """Test column translation."""
        mock_yaml.__getitem__.return_value = mock_yaml_data["test_reader"]
        reader = ExamplePGReader()

        # Should only find the standardized names (values) and not the original names (keys)
        mapping = {"Protein ID": "protein", "Gene Name": "gene"}
        result_df = reader._translate_columns(sample_df, mapping)

        assert "protein" in result_df.columns
        assert "gene" in result_df.columns
        assert "Protein ID" not in result_df.columns
        assert "Gene Name" not in result_df.columns
        # Description was not changed by Translation as it is not contained in the mapping
        assert "Description" in result_df.columns


class TestFilterMeasurement:
    """Test _filter_measurement method."""

    @patch("alphabase.pg_reader.pg_reader.pg_reader_yaml")
    def test_filter_with_regex(self, mock_yaml, mock_yaml_data, sample_df):
        """Test filtering columns with regex."""
        mock_yaml.__getitem__.return_value = mock_yaml_data["test_reader"]
        reader = ExamplePGReader()

        regex = r"Sample_\d+_LFQ"
        result_df = reader._filter_measurement(sample_df, regex)

        assert list(result_df.columns) == ["Sample_1_LFQ", "Sample_2_LFQ"]

    @patch("alphabase.pg_reader.pg_reader.pg_reader_yaml")
    def test_filter_with_extra_columns(self, mock_yaml, mock_yaml_data, sample_df):
        """Test filtering with extra columns."""
        mock_yaml.__getitem__.return_value = mock_yaml_data["test_reader"]
        reader = ExamplePGReader()

        regex = r"Sample_\d+_LFQ"
        extra_cols = ["Protein ID", "Gene Name"]
        result_df = reader._filter_measurement(
            sample_df, regex, extra_columns=extra_cols
        )

        expected_cols = ["Sample_1_LFQ", "Sample_2_LFQ", "Protein ID", "Gene Name"]
        assert list(result_df.columns) == expected_cols

    @patch("alphabase.pg_reader.pg_reader.pg_reader_yaml")
    def test_filter_no_matches_warning(self, mock_yaml, mock_yaml_data, sample_df):
        """Test warning when regex matches no columns."""
        mock_yaml.__getitem__.return_value = mock_yaml_data["test_reader"]
        reader = ExamplePGReader()

        regex = r"NonExistent.*"
        with pytest.warns(UserWarning, match="regex .* did not match any columns"):
            result_df = reader._filter_measurement(sample_df, regex)

        assert result_df.empty
