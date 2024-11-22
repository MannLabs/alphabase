"""Integration tests for anndata creation."""

from io import StringIO

from alphabase.anndata.anndata_factory import AnnDataFactory
from alphabase.psm_reader import DiannReader
from tests.integration.test_psm_readers import (
    TEST_DATA_DIANN,
    _assert_reference_df_equal,
)


def test_anndata():
    """Test creating anndata from diann files."""
    # Create directly from files
    factory = AnnDataFactory.from_files(
        StringIO(TEST_DATA_DIANN),
        reader_type="diann",
    )

    adata = factory.create_anndata()

    _assert_reference_df_equal(adata.to_df(), "anndata", check_psf_df_columns=False)


def test_anndata_2():
    """Test creating anndata from diann psm df."""
    # Create directly from files
    reader = DiannReader()
    reader.load(StringIO(TEST_DATA_DIANN))

    factory = AnnDataFactory(reader.psm_df)
    adata = factory.create_anndata()

    _assert_reference_df_equal(adata.to_df(), "anndata", check_psf_df_columns=False)
