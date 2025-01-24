"""Unit tests for the AnnDataFactory class."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alphabase.anndata.anndata_factory import AnnDataFactory
from alphabase.psm_reader.keys import PsmDfCols


def _get_test_psm_df():
    """Return a test PSM DataFrame."""
    return pd.DataFrame(
        {
            PsmDfCols.RAW_NAME: ["raw1", "raw1", "raw2"],
            PsmDfCols.PROTEINS: ["protein1", "protein2", "protein1"],
            PsmDfCols.INTENSITY: [100, 200, 300],
        }
    )


def test_initialization_with_missing_columns():
    """Test that an error is raised when the input DataFrame is missing required columns."""
    psm_df = _get_test_psm_df().drop(columns=[PsmDfCols.INTENSITY])

    with pytest.raises(ValueError, match="Missing required columns: \['intensity'\]"):
        # when
        AnnDataFactory(psm_df)


def test_create_anndata_with_valid_dataframe():
    """Test that an AnnData object is created correctly from a valid input DataFrame."""
    psm_df = _get_test_psm_df()
    factory = AnnDataFactory(psm_df)

    # when
    adata = factory.create_anndata()

    assert adata.shape == (2, 2)
    assert adata.obs_names.tolist() == ["raw1", "raw2"]
    assert adata.var_names.tolist() == ["protein1", "protein2"]
    assert np.array_equal(
        adata.X, np.array([[100, 200], [300, np.nan]]), equal_nan=True
    )


def test_create_anndata_with_missing_intensity_values():
    """Test that missing intensity values are replaced with NaNs in the AnnData object."""
    psm_df = pd.DataFrame(
        {
            PsmDfCols.RAW_NAME: ["raw1", "raw2"],
            PsmDfCols.PROTEINS: ["protein1", "protein2"],
            PsmDfCols.INTENSITY: [100, np.nan],
        }
    )
    factory = AnnDataFactory(psm_df)

    # when
    adata = factory.create_anndata()

    assert adata.shape == (2, 2)
    assert adata.obs_names.tolist() == ["raw1", "raw2"]
    assert adata.var_names.tolist() == ["protein1", "protein2"]
    assert np.array_equal(
        adata.X, np.array([[100.0, np.nan], [np.nan, np.nan]]), equal_nan=True
    )


def test_create_anndata_with_duplicate_proteins():
    """Test that intensity values for duplicate proteins in the same raw file are aggregated correctly."""
    psm_df = pd.DataFrame(
        {
            PsmDfCols.RAW_NAME: ["raw1", "raw1", "raw2"],
            PsmDfCols.PROTEINS: ["protein1", "protein1", "protein1"],
            PsmDfCols.INTENSITY: [100, 200, 300],
        }
    )
    factory = AnnDataFactory(psm_df)

    # when
    adata = factory.create_anndata()

    assert adata.shape == (2, 1)
    assert adata.obs_names.tolist() == ["raw1", "raw2"]
    assert adata.var_names.tolist() == ["protein1"]
    assert np.array_equal(
        adata.X,
        np.array([[100], [300]]),  # first is taken
    )


def test_create_anndata_with_empty_dataframe():
    """Test that an empty AnnData object is created when the input DataFrame is empty."""
    psm_df = pd.DataFrame(
        columns=[PsmDfCols.RAW_NAME, PsmDfCols.PROTEINS, PsmDfCols.INTENSITY]
    )
    factory = AnnDataFactory(psm_df)

    # when
    adata = factory.create_anndata()

    assert adata.shape == (0, 0)


@patch("alphabase.psm_reader.psm_reader.psm_reader_provider.get_reader")
@patch("alphabase.anndata.anndata_factory.AnnDataFactory._get_reader_configuration")
def test_from_files(mock_get_reader_configuration, mock_reader):
    mock_reader.return_value.load.return_value = _get_test_psm_df()

    mock_get_reader_configuration.return_value = {"extra_key": "extra_value"}
    factory = AnnDataFactory.from_files(
        ["file1", "file2"], reader_type="some_reader_type"
    )

    # when
    adata = factory.create_anndata()

    assert adata.shape == (2, 2)
    assert adata.obs_names.tolist() == ["raw1", "raw2"]
    assert adata.var_names.tolist() == ["protein1", "protein2"]
    assert np.array_equal(
        adata.X, np.array([[100, 200], [300, np.nan]]), equal_nan=True
    )

    mock_reader.assert_called_once_with("some_reader_type", extra_key="extra_value")


@patch("alphabase.psm_reader.psm_reader.psm_reader_provider.get_reader")
def test_from_files_nan(mock_reader):
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    PsmDfCols.RAW_NAME: ["raw2"],
                    PsmDfCols.PROTEINS: ["protein2"],
                    PsmDfCols.INTENSITY: [np.nan],
                }
            ),
            _get_test_psm_df(),
        ]
    )
    mock_reader.return_value.load.return_value = df

    factory = AnnDataFactory.from_files(
        ["file1", "file2"], reader_type="some_reader_type"
    )

    # when
    adata = factory.create_anndata()

    assert adata.shape == (2, 2)
    assert adata.obs_names.tolist() == ["raw1", "raw2"]
    assert adata.var_names.tolist() == ["protein1", "protein2"]
    assert np.array_equal(
        adata.X, np.array([[100, 200], [300, np.nan]]), equal_nan=True
    )

    mock_reader.assert_called_once_with("some_reader_type")


def test_get_reader_configuration_with_valid_reader_type():
    """Test that the correct configuration is returned for a valid reader type."""
    config = AnnDataFactory._get_reader_configuration(
        "diann"
    )  # diann is taken as an example here

    assert config == {
        "filter_first_search_fdr": True,
        "filter_second_search_fdr": True,
    }


def test_get_reader_configuration_with_unknown_reader_type():
    """Test that a reader type without special config is handled correctly."""

    config = AnnDataFactory._get_reader_configuration("invalid_reader_type")
    assert config == {}
