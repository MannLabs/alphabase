"""Shared logic for integration tests."""

from pathlib import Path

import pandas as pd
import pytest

from alphabase.tools.data_downloader import DataShareDownloader


@pytest.fixture(scope="function")
def example_alphadia_tsv(tmp_path) -> tuple[Path, pd.DataFrame]:
    """Get and parse real alphadia PG report matrix."""
    URL = "https://datashare.biochem.mpg.de/s/cN1tmElfgKOe1cW"
    REF_URL = "https://datashare.biochem.mpg.de/s/vtgitlrJNeaWl9U"

    download_path = DataShareDownloader(url=URL, output_dir=tmp_path).download()
    reference_download_path = DataShareDownloader(
        url=REF_URL, output_dir=tmp_path
    ).download()

    reference = pd.read_parquet(reference_download_path)

    return download_path, reference


@pytest.fixture(scope="function")
def example_diann_tsv(tmp_path) -> tuple[Path, pd.DataFrame]:
    """Get and parse real DIANN PG report matrix."""
    URL = "https://datashare.biochem.mpg.de/s/R7GYhwArBO2NS9J"
    REF_URL = "https://datashare.biochem.mpg.de/s/g5rsaGeGkbyKNam"  # Ground truth

    download_path = DataShareDownloader(url=URL, output_dir=tmp_path).download()
    reference_download_path = DataShareDownloader(
        url=REF_URL, output_dir=tmp_path
    ).download()

    reference = pd.read_parquet(reference_download_path)

    return (download_path, reference)
