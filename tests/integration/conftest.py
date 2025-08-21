"""Shared logic for integration tests."""

from pathlib import Path

import pandas as pd
import pytest
import requests

from alphabase.tools.data_downloader import DataShareDownloader


def check_url(url: str, timeout: int = 5) -> None:
    """Skip test if URL is not accessible."""
    try:
        response = requests.head(url, timeout=timeout)
        if response.status_code != 200:
            pytest.skip(
                f"Skipping test: URL not reachable (status {response.status_code}) -> {url}"
            )
    except requests.RequestException as e:
        pytest.skip(f"Skipping test: Cannot reach URL -> {url}. Error: {e}")


def get_example_data_with_ref(
    url: str, ref_url: str, directory: Path
) -> tuple[str, pd.DataFrame]:
    """Utility function to get test and reference data for utility tests from MPIB datashare

    Parameters
    ----------
    url
        URL to test data on MPIB datashare
    ref_url
        Reference URL pointing to a tabular parquet file.
    directory
        Directory to which the data is written

    Returns
    -------
    tuple[str, pd.DataFrame]
        - `str`: Path to test data
        - :class:`pd.DataFrame` Reference data
    """
    download_path = DataShareDownloader(url=url, output_dir=directory).download()
    reference_download_path = DataShareDownloader(
        url=ref_url, output_dir=directory
    ).download()

    reference = pd.read_parquet(reference_download_path)

    return download_path, reference


@pytest.fixture(scope="function")
def example_alphadia_tsv(tmp_path) -> tuple[Path, pd.DataFrame]:
    """Get and parse real alphadia PG report matrix."""
    URL = "https://datashare.biochem.mpg.de/s/cN1tmElfgKOe1cW"
    REF_URL = "https://datashare.biochem.mpg.de/s/vtgitlrJNeaWl9U"

    check_url(URL, timeout=5)
    check_url(REF_URL, timeout=5)

    return get_example_data_with_ref(url=URL, ref_url=REF_URL, directory=tmp_path)


@pytest.fixture(scope="function")
def example_diann_tsv(tmp_path) -> tuple[Path, pd.DataFrame]:
    """Get and parse real DIANN PG report matrix."""
    URL = "https://datashare.biochem.mpg.de/s/R7GYhwArBO2NS9J"
    REF_URL = "https://datashare.biochem.mpg.de/s/g5rsaGeGkbyKNam"  # Ground truth

    check_url(URL, timeout=5)
    check_url(REF_URL, timeout=5)

    return get_example_data_with_ref(url=URL, ref_url=REF_URL, directory=tmp_path)


@pytest.fixture(scope="function")
def example_alphapept_csv(tmp_path) -> tuple[Path, pd.DataFrame]:
    """Get and parse real alphapept protein group report matrix."""
    URL = "https://datashare.biochem.mpg.de/s/6G6KHJqwcRPQiOO"
    REF_URL = "https://datashare.biochem.mpg.de/s/Re35ygdblh2T7si"

    check_url(URL, timeout=5)
    check_url(REF_URL, timeout=5)

    return get_example_data_with_ref(url=URL, ref_url=REF_URL, directory=tmp_path)


@pytest.fixture(scope="function")
def example_alphapept_hdf(tmp_path) -> tuple[Path, pd.DataFrame]:
    """Get and parse real alphapept protein group report matrix."""
    URL = "https://datashare.biochem.mpg.de/s/ZKwmZGssk9dHtic"
    REF_URL = "https://datashare.biochem.mpg.de/s/gVhEy0mjrEE9F5f"

    check_url(URL, timeout=5)
    check_url(REF_URL, timeout=5)

    return get_example_data_with_ref(url=URL, ref_url=REF_URL, directory=tmp_path)
