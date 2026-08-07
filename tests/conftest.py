"""Shared test configuration and fixtures."""

import importlib.util

import pytest

NUMBA_UNAVAILABLE = importlib.util.find_spec("numba") is None

pytest.mark.requires_numba = pytest.mark.skipif(
    NUMBA_UNAVAILABLE,
    reason="numba package not installed. Install with `pip install alphabase[full]`",
)
