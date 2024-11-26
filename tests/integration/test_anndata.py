"""Integration tests for anndata creation."""

import os
import tempfile
from pathlib import Path

from alphabase.anndata.anndata_factory import AnnDataFactory
from alphabase.tools.data_downloader import DataShareDownloader
from tests.integration.test_psm_readers import (
    _assert_reference_df_equal,
)

current_file_directory = os.path.dirname(os.path.abspath(__file__))
test_data_path = Path(f"{current_file_directory}/reference_data")


def test_anndata_alphadia_181():
    """Test creating anndata from alphadia files."""
    # Create directly from files
    url = "https://datashare.biochem.mpg.de/s/Hk41INtwBvBl0kP/download?files=alphadia_1.8.1_report_head.tsv"
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = DataShareDownloader(url=url, output_dir=temp_dir).download()

        factory = AnnDataFactory.from_files(
            file_paths=file_path, reader_type="alphadia"
        )

    adata = factory.create_anndata()

    # TODO compare the whole anndata object here not only the df
    _assert_reference_df_equal(
        adata.to_df(), "ad_alphadia_181", check_psf_df_columns=False
    )


def test_anndata_diann_181():
    """Test creating anndata from diann files."""
    # Create directly from files
    url = "https://datashare.biochem.mpg.de/s/Hk41INtwBvBl0kP/download?files=diann_1.8.1_report_head.tsv"
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = DataShareDownloader(url=url, output_dir=temp_dir).download()

        factory = AnnDataFactory.from_files(
            file_paths=file_path,
            reader_type="diann",
            intensity_column="PG.Quantity",
        )

    adata = factory.create_anndata()

    # TODO compare the whole anndata object here not only the df
    _assert_reference_df_equal(
        adata.to_df(), "ad_diann_181", check_psf_df_columns=False
    )


def test_anndata_diann_1901():
    """Test creating anndata from diann files."""
    # Create directly from files
    url = "https://datashare.biochem.mpg.de/s/Hk41INtwBvBl0kP/download?files=diann_1.9.0_report_head.tsv"
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = DataShareDownloader(url=url, output_dir=temp_dir).download()

        factory = AnnDataFactory.from_files(
            file_paths=file_path,
            reader_type="diann",
            intensity_column="PG.Quantity",
        )

    adata = factory.create_anndata()

    # TODO compare the whole anndata object here not only the df
    _assert_reference_df_equal(
        adata.to_df(), "ad_diann_190", check_psf_df_columns=False
    )
