"""Shared test configuration and fixtures."""

import importlib.util

import pytest

NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None

pytest.mark.requires_numba = pytest.mark.skipif(
    NUMBA_AVAILABLE,
    reason="numba package not installed",
)
