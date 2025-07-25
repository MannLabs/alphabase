from typing import Any
from unittest.mock import Mock, patch

import pytest

from alphabase.pg_reader.pg_reader import PGReaderBase, PGReaderProvider


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


class TestPGReaderProvider:
    """Test PGReaderProvider class."""

    @patch("alphabase.pg_reader.pg_reader.pg_reader_yaml")
    def test_register_and_get_reader(
        self, pg_reader_yaml: Mock, mock_yaml_data: dict[str, Any]
    ) -> None:
        """Test registering and getting a reader."""

        pg_reader_yaml.__getitem__.test_reader = mock_yaml_data
        provider = PGReaderProvider()
        provider.register_reader("test_reader", ExamplePGReader)

        reader = provider.get_reader("test_reader")

        assert isinstance(reader, ExamplePGReader)

    def test_get_unknown_reader_raises_error(self) -> None:
        """Test that getting an unknown reader raises KeyError."""
        provider = PGReaderProvider()

        with pytest.raises(KeyError, match="Unknown reader type 'unknown'"):
            provider.get_reader("unknown")

    @patch("alphabase.pg_reader.pg_reader.pg_reader_yaml")
    def test_get_reader_with_kwargs(
        self, mock_yaml: Mock, mock_yaml_data: dict[str, Any]
    ) -> None:
        """Test getting a reader with custom parameters."""
        mock_yaml.__getitem__.return_value = mock_yaml_data["test_reader"]
        provider = PGReaderProvider()
        provider.register_reader("test_reader", ExamplePGReader)

        custom_mapping = {"protein": "CustomProtein"}
        reader = provider.get_reader("test_reader", column_mapping=custom_mapping)

        assert reader.column_mapping == custom_mapping

    @patch("alphabase.pg_reader.pg_reader.pg_reader_yaml")
    def test_get_reader_by_yaml(
        self, mock_yaml: Mock, mock_yaml_data: dict[str, Any]
    ) -> None:
        """Test getting a reader from a YAML configuration."""
        mock_yaml.__getitem__.return_value = mock_yaml_data["test_reader"]
        provider = PGReaderProvider()
        provider.register_reader("test_reader", ExamplePGReader)

        yaml_dict = {
            "reader_type": "test_reader",
            "column_mapping": {"protein": "YAMLProtein"},
        }
        reader = provider.get_reader_by_yaml(yaml_dict)

        assert isinstance(reader, ExamplePGReader)
        assert reader.column_mapping == {"protein": "YAMLProtein"}
