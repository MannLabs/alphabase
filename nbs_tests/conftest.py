"""Shared test configuration for notebook tests."""

import importlib.util

import pytest

NUMBA_UNAVAILABLE = importlib.util.find_spec("numba") is None


def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001
    items: list[pytest.Item],
) -> None:
    """Skip all notebook tests when numba is not installed."""
    if not NUMBA_UNAVAILABLE:
        return

    skip_numba = pytest.mark.skip(
        reason="numba package not installed. Install with `pip install alphabase[full]`"
    )
    for item in items:
        if item.fspath.ext == ".ipynb":
            item.add_marker(skip_numba)
