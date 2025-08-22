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
