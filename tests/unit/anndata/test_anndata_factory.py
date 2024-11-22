"""Unit tests for the AnnDataFactory class."""

import numpy as np
import pandas as pd
import pytest

from alphabase.anndata.anndata_factory import AnnDataFactory
from alphabase.psm_reader.keys import PsmDfCols


def test_initialization_with_missing_columns():
    """Test that an error is raised when the input DataFrame is missing required columns."""
    psm_df = pd.DataFrame(
        {
            PsmDfCols.RAW_NAME: ["raw1", "raw2"],
            PsmDfCols.PROTEINS: ["protein1", "protein2"],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns: \['intensity'\]"):
        # when
        AnnDataFactory(psm_df)


def test_create_anndata_with_valid_dataframe():
    """Test that an AnnData object is created correctly from a valid input DataFrame."""
    psm_df = pd.DataFrame(
        {
            PsmDfCols.RAW_NAME: ["raw1", "raw1", "raw2"],
            PsmDfCols.PROTEINS: ["protein1", "protein2", "protein1"],
            PsmDfCols.INTENSITY: [100, 200, 300],
        }
    )
    factory = AnnDataFactory(psm_df)

    # when
    adata = factory.create_anndata()

    assert adata.shape == (2, 2)
    assert adata.obs_names.tolist() == ["raw1", "raw2"]
    assert adata.var_names.tolist() == ["protein1", "protein2"]
    assert np.array_equal(adata.X, np.array([[100, 200], [300, np.nan]]))


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
    """Test that intensity values for duplicate proteins in the same raw file are aggregated."""
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
        adata.X, np.array([[150], [300]])
    )  # mean of 100 and 200 is 150


def test_create_anndata_with_empty_dataframe():
    """Test that an empty AnnData object is created when the input DataFrame is empty."""
    psm_df = pd.DataFrame(
        columns=[PsmDfCols.RAW_NAME, PsmDfCols.PROTEINS, PsmDfCols.INTENSITY]
    )
    factory = AnnDataFactory(psm_df)

    # when
    adata = factory.create_anndata()

    assert adata.shape == (0, 0)
