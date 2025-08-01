"""Shared logic for integration tests."""

from pathlib import Path

import pytest

from alphabase.tools.data_downloader import DataShareDownloader


@pytest.fixture(scope="function")
def example_alphadia_tsv(tmp_path) -> Path:
    """Get and parse real alphadia PG report matrix."""
    URL = "https://datashare.biochem.mpg.de/s/cN1tmElfgKOe1cW"

    download_path = DataShareDownloader(url=URL, output_dir=tmp_path).download()
    return download_path


@pytest.fixture(scope="function")
def example_diann_tsv(tmp_path) -> Path:
    """Get and parse real DIANN PG report matrix."""
    URL = "https://datashare.biochem.mpg.de/s/R7GYhwArBO2NS9J"

    download_path = DataShareDownloader(url=URL, output_dir=tmp_path).download()
    return download_path


@pytest.fixture(scope="function")
def example_alphapept_csv(tmp_path) -> Path:
    """Get and parse real alphapept protein group report matrix."""
    URL = "https://datashare.biochem.mpg.de/s/6G6KHJqwcRPQiOO"

    download_path = DataShareDownloader(url=URL, output_dir=tmp_path).download()
    return download_path
