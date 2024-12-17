"""Unit tests for the tempmmap module.

This module has a nontrivial lifecycle (top-level code), therefore the test setup is a bit unusual
(explicit handling of module (un)loading in setup_function()/teardown_function())
"""

import os
import sys
import tempfile

import numpy as np


def setup_function(function):
    """Import the module before every test."""
    import alphabase.io.tempmmap  # noqa


def teardown_function(function):
    """Simulate `atexit.register` and delete the module before every test."""
    tempmmap = sys.modules["alphabase.io.tempmmap"]
    tempmmap.clear()  # simulating @atexit.register

    # # later:
    # assert tempmmap._TEMP_DIR is None
    # assert tempmmap.TEMP_DIR_NAME is None

    del sys.modules["alphabase.io.tempmmap"]


def test_create_array():
    """Test creating and accessing an array."""
    tempmmap = sys.modules["alphabase.io.tempmmap"]

    # when
    arr = tempmmap.array((5, 5), np.float32)
    assert arr[1, 1] == 0
    assert arr.shape == (5, 5)
    assert arr.dtype == np.float32

    # test rw access to array
    arr[1, 1] = 1
    assert arr[1, 1] == 1

    assert tempmmap._TEMP_DIR is not None


def test_create_array_with_custom_temp_dir():
    tempmmap = sys.modules["alphabase.io.tempmmap"]

    with tempfile.TemporaryDirectory() as temp_dir:
        # when
        arr = tempmmap.array((5, 5), np.int32, tmp_dir_abs_path=temp_dir)

        assert arr.shape == (5, 5)
        assert arr.dtype == np.int32
        assert arr[1, 1] == 0

        # test rw access to array
        arr[1, 1] = 1
        assert arr[1, 1] == 1

    assert tempmmap._TEMP_DIR is not None
    assert temp_dir == tempmmap.TEMP_DIR_NAME


def test_mmap_array_from_path():
    tempmmap = sys.modules["alphabase.io.tempmmap"]

    with tempfile.TemporaryDirectory() as temp_dir:
        # first create an array ..
        arr = tempmmap.array((5, 5), np.int32, tmp_dir_abs_path=temp_dir)
        arr[1, 1] = 1

        # then reconnect to it
        temp_dir = tempmmap.TEMP_DIR_NAME
        # this is a bit hacky but given all the randomness in the temp file name, should be ok:
        temp_file_name = os.listdir(temp_dir)[0]

        reconnected_arr = tempmmap.mmap_array_from_path(f"{temp_dir}/{temp_file_name}")
        assert reconnected_arr[1, 1] == 1

        reconnected_arr[1, 1] = 2
        assert reconnected_arr[1, 1] == 2

        arr[1, 1] = 3
        assert reconnected_arr[1, 1] == 3


def test_create_zeros():
    """Test creating and accessing an array of zeros."""
    tempmmap = sys.modules["alphabase.io.tempmmap"]

    # when
    arr = tempmmap.zeros((5, 5), np.int32)
    assert arr[1, 1] == 0
    assert arr.shape == (5, 5)
    assert arr.dtype == np.int32

    assert tempmmap._TEMP_DIR is not None


def test_create_ones():
    """Test creating and accessing an array of ones."""
    tempmmap = sys.modules["alphabase.io.tempmmap"]

    # when
    arr = tempmmap.ones((5, 5), np.int32)
    assert arr[1, 1] == 1
    assert arr.shape == (5, 5)
    assert arr.dtype == np.int32

    assert tempmmap._TEMP_DIR is not None
