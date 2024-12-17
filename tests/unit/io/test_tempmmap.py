"""Unit tests for the tempmmap module.

This module has a nontrivial lifecycle (top-level code), therefore the test setup is a bit unusual
(explicit handling of module (un)loading in setup_function()/teardown_function())
"""

import sys

import numpy as np


def setup_function(function):
    """Import the module before evert"""


def teardown_function(function):
    # simulating @atexit.register
    sys.modules["alphabase.io.tempmmap"].clear()
    del sys.modules["alphabase.io.tempmmap"]


def test_create_array_with_valid_shape_and_dtype():
    """Test creating and accessing an array."""
    tempmmap = sys.modules["alphabase.io.tempmmap"]
    arr = tempmmap.array((10, 10), np.float32)
    arr[1, 1] = 1

    assert arr.shape == (10, 10)
    assert arr.dtype == np.float32
    assert arr[1, 1] == 1
