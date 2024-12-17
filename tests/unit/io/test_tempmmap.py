"""Unit tests for the tempmmap module.

This module has a nontrivial lifecycle (top-level code), therefore the test setup is a bit unusual
(explicit handling of module (un)loading in setup_function()/teardown_function())
"""

import os
import sys
import tempfile

import numpy as np
import pytest


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


def test_create_empty():
    """Test creating and accessing an empty array."""
    tempmmap = sys.modules["alphabase.io.tempmmap"]

    # when
    file_path = tempmmap.create_empty_mmap((5, 5), np.float32)
    arr = tempmmap.mmap_array_from_path(file_path)

    assert arr[1, 1] == 0
    assert arr.shape == (5, 5)
    assert arr.dtype == np.float32

    # test rw access to array
    arr[1, 1] = 1
    assert arr[1, 1] == 1

    assert tempmmap._TEMP_DIR is not None


def test_create_empty_with_custom_temp_dir():
    """Test creating and accessing an empty array."""
    tempmmap = sys.modules["alphabase.io.tempmmap"]

    # when
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = tempmmap.create_empty_mmap(
            (5, 5), np.float32, tmp_dir_abs_path=temp_dir
        )
        arr = tempmmap.mmap_array_from_path(file_path)

        assert arr[1, 1] == 0
        assert arr.shape == (5, 5)
        assert arr.dtype == np.float32

        # test rw access to array
        arr[1, 1] = 1
        assert arr[1, 1] == 1

        assert tempmmap._TEMP_DIR is not None
        assert temp_dir == tempmmap.TEMP_DIR_NAME


def test_create_empty_with_custom_file_path():
    """Test creating and accessing an empty array."""
    tempmmap = sys.modules["alphabase.io.tempmmap"]

    # when
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = f"{temp_dir}/temp_mmap.hdf"
        file_path = tempmmap.create_empty_mmap(
            (5, 5), np.float32, file_path=temp_file_path
        )
        arr = tempmmap.mmap_array_from_path(file_path)

        assert arr[1, 1] == 0
        assert arr.shape == (5, 5)
        assert arr.dtype == np.float32

        # test rw access to array
        arr[1, 1] = 1
        assert arr[1, 1] == 1

        assert tempmmap._TEMP_DIR is not None
        assert temp_dir != tempmmap.TEMP_DIR_NAME


def test_create_empty_with_custom_file_path_error_cases():
    """Test creating and accessing an empty array."""
    tempmmap = sys.modules["alphabase.io.tempmmap"]

    # when
    temp_file_path = "temp_dir/temp_mmap.hdfx"
    with pytest.raises(
        ValueError, match="The chosen file name 'temp_mmap.hdfx' needs to end with .hdf"
    ):
        _ = tempmmap.create_empty_mmap((5, 5), np.float32, file_path=temp_file_path)

    temp_file_path = "/temp_dir/temp_mmap.hdf"
    with pytest.raises(
        ValueError,
        match="The directory '/temp_dir' in which the file should be created does not exist.",
    ):
        _ = tempmmap.create_empty_mmap((5, 5), np.float32, file_path=temp_file_path)


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
