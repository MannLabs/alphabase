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
    tempmmap._clear()  # simulating @atexit.register

    assert tempmmap._TEMP_DIR is None
    assert tempmmap.TEMP_DIR_NAME is None

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


def test_check_temp_dir_deletion():
    """Test that tempdir is deleted at exit."""
    tempmmap = sys.modules["alphabase.io.tempmmap"]

    _ = tempmmap.array((5, 5), np.float32)
    temp_dir_name = tempmmap._TEMP_DIR.name

    # check presence of temp dir first
    assert os.path.exists(temp_dir_name)

    # when
    tempmmap._clear()

    assert not os.path.exists(temp_dir_name)


def test_create_array_with_custom_temp_dir():
    """Test creating and accessing an array with custom temp dir."""
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


def test_create_array_with_custom_temp_dir_nonexisting():
    """Test creating an array with custom temp dir: not existing."""
    tempmmap = sys.modules["alphabase.io.tempmmap"]

    temp_dir = "nonexisting_dir"
    # when
    with pytest.raises(
        ValueError,
        match="The directory 'nonexisting_dir' in which the file should be created does not exist.",
    ):
        _ = tempmmap.array((5, 5), np.int32, tmp_dir_abs_path=temp_dir)


def test_create_array_with_custom_temp_dir_not_a_dir():
    """Test creating an array with custom temp dir: not a directory."""
    tempmmap = sys.modules["alphabase.io.tempmmap"]

    with tempfile.TemporaryFile() as temp_file, pytest.raises(
        ValueError,
        match=f"The path '{temp_file.name}' does not point to a directory.",
    ):
        # when
        _ = tempmmap.create_empty_mmap(
            (5, 5), np.int32, tmp_dir_abs_path=temp_file.name
        )


def test_mmap_array_from_path():
    """Test reconnecting to an existing array."""
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
    """Test creating and accessing an empty array with custom temp dir."""
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
    """Test creating and accessing an empty array with custom file path."""
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


def test_create_empty_with_custom_file_path_exists():
    """Test creating and accessing an empty array with custom file path that exists."""
    tempmmap = sys.modules["alphabase.io.tempmmap"]

    # when
    with tempfile.TemporaryFile() as temp_file, pytest.raises(
        ValueError,
        match=f"The file '{temp_file.name}' already exists. Set overwrite to True to overwrite the file or choose a different name.",
    ):
        _ = tempmmap.create_empty_mmap((5, 5), np.float32, file_path=temp_file.name)

    # when 2
    with tempfile.TemporaryDirectory() as temp_dir, open(
        f"{temp_dir}/temp_mmap.hdf", "w"
    ) as temp_file:
        _ = tempmmap.create_empty_mmap(
            (5, 5), np.float32, file_path=temp_file.name, overwrite=True
        )
        # did not raise -> OK


def test_create_empty_with_custom_file_path_error_cases():
    """Test creating and accessing an empty array: error cases."""
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


def test_redefine_location():
    """Test redefining temp location."""
    tempmmap = sys.modules["alphabase.io.tempmmap"]

    old_temp_dir = tempmmap._TEMP_DIR

    with tempfile.TemporaryDirectory() as temp_dir:
        tempmmap.redefine_temp_location(temp_dir)

    assert old_temp_dir != tempmmap._TEMP_DIR
    assert os.path.dirname(tempmmap.TEMP_DIR_NAME) == temp_dir
