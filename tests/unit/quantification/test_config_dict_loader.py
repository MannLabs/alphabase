"""Unit tests for config_dict_loader module."""

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from alphabase.quantification.quant_reader.config_dict_loader import (
    get_input_type_and_config_dict,
)


class TestGetInputTypeAndConfigDict:
    """Test cases for get_input_type_and_config_dict function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary config file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")

        # Sample config data
        self.sample_config = {
            "test_type_1": {
                "format": "longtable",
                "sample_ID": "sample_col",
                "quant_ID": "intensity_col",
                "protein_cols": ["protein_col"],
                "ion_cols": ["sequence_col"],
            },
            "test_type_2": {
                "format": "widetable",
                "protein_cols": ["protein_col"],
                "ion_cols": ["sequence_col", "charge_col"],
                "filters": {
                    "test_filter": {
                        "param": "filter_col",
                        "comparator": "!=",
                        "value": "test_value",
                    }
                },
            },
        }

        # Write config to file
        with open(self.config_file, "w") as f:
            yaml.dump(self.sample_config, f)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("alphabase.quantification.quant_reader.config_dict_loader._load_config")
    @patch(
        "alphabase.quantification.quant_reader.config_dict_loader.quantreader_utils.read_columns_from_file"
    )
    def test_successful_match_without_input_type(
        self, mock_read_columns, mock_load_config
    ):
        """Test successful matching when no specific input_type_to_use is provided."""
        # Setup
        mock_load_config.return_value = self.sample_config
        mock_read_columns.return_value = [
            "sample_col",
            "intensity_col",
            "protein_col",
            "sequence_col",
        ]

        input_file = "test_file.tsv"

        # Execute
        result = get_input_type_and_config_dict(input_file)

        # Verify
        input_type, config_dict, sep = result
        assert input_type == "test_type_1"
        assert config_dict == self.sample_config["test_type_1"]
        assert sep == "\t"
        mock_read_columns.assert_called_once_with(input_file, sep="\t")

    @patch("alphabase.quantification.quant_reader.config_dict_loader._load_config")
    @patch(
        "alphabase.quantification.quant_reader.config_dict_loader.quantreader_utils.read_columns_from_file"
    )
    def test_successful_match_with_input_type(
        self, mock_read_columns, mock_load_config
    ):
        """Test successful matching when specific input_type_to_use is provided."""
        # Setup
        mock_load_config.return_value = self.sample_config
        mock_read_columns.return_value = [
            "protein_col",
            "sequence_col",
            "charge_col",
            "filter_col",
        ]

        input_file = "test_file.tsv"
        input_type_to_use = "test_type_2"

        # Execute
        result = get_input_type_and_config_dict(input_file, input_type_to_use)

        # Verify
        input_type, config_dict, sep = result
        assert input_type == "test_type_2"
        assert config_dict == self.sample_config["test_type_2"]
        assert sep == "\t"

    @patch("alphabase.quantification.quant_reader.config_dict_loader._load_config")
    @patch(
        "alphabase.quantification.quant_reader.config_dict_loader.quantreader_utils.read_columns_from_file"
    )
    def test_aq_reformat_file_processing(self, mock_read_columns, mock_load_config):
        """Test processing of aq_reformat.tsv files."""
        # Setup
        mock_load_config.return_value = self.sample_config
        mock_read_columns.return_value = [
            "sample_col",
            "intensity_col",
            "protein_col",
            "sequence_col",
        ]

        input_file = "original_file.raw.aq_reformat.tsv"

        # Execute
        result = get_input_type_and_config_dict(input_file)

        # Verify
        input_type, config_dict, sep = result
        assert input_type == "test_type_1"
        assert config_dict == self.sample_config["test_type_1"]
        # Verify that the original file name was used for column reading
        mock_read_columns.assert_called_once_with("original_file", sep="\t")

    @patch("alphabase.quantification.quant_reader.config_dict_loader._load_config")
    @patch(
        "alphabase.quantification.quant_reader.config_dict_loader.quantreader_utils.read_columns_from_file"
    )
    def test_csv_file_separator(self, mock_read_columns, mock_load_config):
        """Test CSV file separator detection."""
        # Setup
        mock_load_config.return_value = self.sample_config
        mock_read_columns.return_value = [
            "sample_col",
            "intensity_col",
            "protein_col",
            "sequence_col",
        ]

        input_file = "test_file.csv"

        # Execute
        result = get_input_type_and_config_dict(input_file)

        # Verify
        input_type, config_dict, sep = result
        assert sep == ","
        mock_read_columns.assert_called_once_with(input_file, sep=",")

    @patch("alphabase.quantification.quant_reader.config_dict_loader._load_config")
    @patch(
        "alphabase.quantification.quant_reader.config_dict_loader.quantreader_utils.read_columns_from_file"
    )
    def test_unknown_file_extension_separator(
        self, mock_read_columns, mock_load_config
    ):
        """Test separator detection for unknown file extensions."""
        # Setup
        mock_load_config.return_value = self.sample_config
        mock_read_columns.return_value = [
            "sample_col",
            "intensity_col",
            "protein_col",
            "sequence_col",
        ]

        input_file = "test_file.unknown"

        # Execute
        result = get_input_type_and_config_dict(input_file)

        # Verify
        input_type, config_dict, sep = result
        assert sep == "\t"  # Should default to tab
        mock_read_columns.assert_called_once_with(input_file, sep="\t")

    @patch("alphabase.quantification.quant_reader.config_dict_loader._load_config")
    @patch(
        "alphabase.quantification.quant_reader.config_dict_loader.quantreader_utils.read_columns_from_file"
    )
    def test_no_matching_input_type(self, mock_read_columns, mock_load_config):
        """Test when no input type matches the available columns."""
        # Setup
        mock_load_config.return_value = self.sample_config
        mock_read_columns.return_value = ["unrelated_col1", "unrelated_col2"]

        input_file = "test_file.tsv"

        # Execute & Verify
        with pytest.raises(ValueError) as exc_info:
            get_input_type_and_config_dict(input_file)

        error_message = str(exc_info.value)
        assert "No suitable input type found" in error_message
        assert "unrelated_col1" in error_message
        assert "unrelated_col2" in error_message
        assert "test_type_1" in error_message
        assert "test_type_2" in error_message

    @patch("alphabase.quantification.quant_reader.config_dict_loader._load_config")
    @patch(
        "alphabase.quantification.quant_reader.config_dict_loader.quantreader_utils.read_columns_from_file"
    )
    def test_specific_input_type_missing_columns(
        self, mock_read_columns, mock_load_config
    ):
        """Test when specific input_type_to_use is provided but required columns are missing."""
        # Setup
        mock_load_config.return_value = self.sample_config
        mock_read_columns.return_value = [
            "protein_col",
            "sequence_col",
        ]  # Missing charge_col and filter_col

        input_file = "test_file.tsv"
        input_type_to_use = "test_type_2"

        # Execute & Verify
        with pytest.raises(ValueError) as exc_info:
            get_input_type_and_config_dict(input_file, input_type_to_use)

        error_message = str(exc_info.value)
        assert (
            f"Input type '{input_type_to_use}' requires columns that are missing"
            in error_message
        )
        assert "charge_col" in error_message  # Missing column
        assert "filter_col" in error_message  # Missing column
        assert "protein_col" in error_message  # Available column
        assert "sequence_col" in error_message  # Available column

    @patch("alphabase.quantification.quant_reader.config_dict_loader._load_config")
    @patch(
        "alphabase.quantification.quant_reader.config_dict_loader.quantreader_utils.read_columns_from_file"
    )
    def test_none_values_in_relevant_columns_filtered(
        self, mock_read_columns, mock_load_config
    ):
        """Test that None values in relevant columns are properly filtered out."""
        # Setup - modify config to include None values
        config_with_none = {
            "test_type_with_none": {
                "format": "longtable",
                "sample_ID": "sample_col",
                "quant_ID": "intensity_col",
                "protein_cols": ["protein_col", None],  # Include None
                "ion_cols": ["sequence_col"],
            }
        }

        mock_load_config.return_value = config_with_none
        mock_read_columns.return_value = [
            "sample_col",
            "intensity_col",
            "protein_col",
            "sequence_col",
        ]

        input_file = "test_file.tsv"

        # Execute
        result = get_input_type_and_config_dict(input_file)

        # Verify
        input_type, config_dict, sep = result
        assert input_type == "test_type_with_none"
        assert config_dict == config_with_none["test_type_with_none"]

    @patch("alphabase.quantification.quant_reader.config_dict_loader._load_config")
    @patch(
        "alphabase.quantification.quant_reader.config_dict_loader.quantreader_utils.read_columns_from_file"
    )
    def test_multiple_matching_types_returns_first(
        self, mock_read_columns, mock_load_config
    ):
        """Test that when multiple types match, the first one is returned."""
        # Setup - create config where both types could match
        config_multiple_match = {
            "first_match": {
                "format": "longtable",
                "sample_ID": "sample_col",
                "quant_ID": "intensity_col",
                "protein_cols": ["protein_col"],
                "ion_cols": ["sequence_col"],
            },
            "second_match": {
                "format": "longtable",
                "sample_ID": "sample_col",
                "quant_ID": "intensity_col",
                "protein_cols": ["protein_col"],
                "ion_cols": ["sequence_col"],
            },
        }

        mock_load_config.return_value = config_multiple_match
        mock_read_columns.return_value = [
            "sample_col",
            "intensity_col",
            "protein_col",
            "sequence_col",
        ]

        input_file = "test_file.tsv"

        # Execute
        result = get_input_type_and_config_dict(input_file)

        # Verify
        input_type, config_dict, sep = result
        assert input_type == "first_match"  # Should return the first matching type
        assert config_dict == config_multiple_match["first_match"]

    @patch("alphabase.quantification.quant_reader.config_dict_loader._load_config")
    @patch(
        "alphabase.quantification.quant_reader.config_dict_loader.quantreader_utils.read_columns_from_file"
    )
    def test_input_type_to_use_skips_other_types(
        self, mock_read_columns, mock_load_config
    ):
        """Test that when input_type_to_use is specified, other types are skipped."""
        # Setup
        mock_load_config.return_value = self.sample_config
        mock_read_columns.return_value = [
            "sample_col",
            "intensity_col",
            "protein_col",
            "sequence_col",
        ]

        input_file = "test_file.tsv"
        input_type_to_use = "test_type_2"  # This type requires more columns

        # Execute & Verify
        with pytest.raises(ValueError) as exc_info:
            get_input_type_and_config_dict(input_file, input_type_to_use)

        # Should fail because test_type_2 requires charge_col and filter_col
        error_message = str(exc_info.value)
        assert (
            f"Input type '{input_type_to_use}' requires columns that are missing"
            in error_message
        )
