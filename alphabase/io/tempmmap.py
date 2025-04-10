"""This module allows to create temporary memory-mapped arrays."""

import atexit
import logging
import mmap
import os
import shutil
import tempfile
from pathlib import PosixPath
from typing import Optional, Union

import h5py
import numpy as np

_TEMP_DIR: Optional[tempfile.TemporaryDirectory] = None
TEMP_DIR_NAME: Optional[Union[str, PosixPath]] = None


def _init_temp_dir(prefix: str = "temp_mmap_") -> str:
    """Initialize the temporary directory for the temp mmap arrays if not already done."""

    global _TEMP_DIR, TEMP_DIR_NAME

    if _TEMP_DIR is None:
        _TEMP_DIR = tempfile.TemporaryDirectory(prefix=prefix)
        TEMP_DIR_NAME = _TEMP_DIR.name

        logging.info(
            f"Memory-mapped arrays are written to temporary directory {TEMP_DIR_NAME}. "
            "Cleanup of this folder is OS dependent and might need to be triggered manually!"
        )

    return TEMP_DIR_NAME


def _change_temp_dir_location(abs_path: str) -> None:
    """
    Check if the directory to which the temp arrays should be written exists, if so defines this as the new temp dir location. If not raise a value error.

    Parameters
    ----------
    abs_path : str
        The absolute path to the new temporary directory.

    """

    global TEMP_DIR_NAME

    # ensure that the path exists
    if os.path.exists(abs_path):
        # ensure that the path points to a directory
        if os.path.isdir(abs_path):
            TEMP_DIR_NAME = abs_path
        else:
            raise ValueError(f"The path '{abs_path}' does not point to a directory.")
    else:
        raise ValueError(
            f"The directory '{abs_path}' in which the file should be created does not exist."
        )


def _get_file_location(abs_file_path: str, overwrite: bool = False) -> str:
    """
    Check if the path specified for the new temporary file is valid. If not raise a value error.

    Valid file paths need to:
    1. be contained in directories that exist
    2. end in .hdf
    3. not exist if overwrite is set to False

    Parameters
    ----------
    abs_file_path : str
        The absolute path to the new temporary file.

    Returns
    ------
    str
        The file path if it is valid.
    """
    if not overwrite and os.path.exists(abs_file_path):
        raise ValueError(
            f"The file '{abs_file_path}' already exists. Set overwrite to True to overwrite the file or choose a different name."
        )

    if not os.path.basename(abs_file_path).endswith(".hdf"):
        raise ValueError(
            f"The chosen file name '{os.path.basename(abs_file_path)}' needs to end with .hdf"
        )

    if not os.path.isdir(os.path.dirname(abs_file_path)):
        raise ValueError(
            f"The directory '{os.path.dirname(abs_file_path)}' in which the file should be created does not exist."
        )

    return abs_file_path


def redefine_temp_location(path: str) -> str:
    """Redefine the location where the temp arrays are written to.

    Parameters
    ----------
    path : string

    Returns
    ------
    str
        the location of the new temporary directory.

    """

    global TEMP_DIR_NAME

    _clear()

    # cleanup old temporary directory
    if TEMP_DIR_NAME is not None:
        # in python 3.12, ignore_errors does not work if None is passed
        shutil.rmtree(TEMP_DIR_NAME, ignore_errors=True)

    # create new tempfile at desired location
    temp_dir_name = _init_temp_dir(prefix=os.path.join(path, "temp_mmap_"))

    return temp_dir_name


def array(shape: tuple, dtype: np.dtype, tmp_dir_abs_path: str = None) -> np.ndarray:
    """Create a writable temporary memory-mapped array.

    Parameters
    ----------
    shape : tuple
        A tuple with the shape of the array.
    dtype : type
        The np.dtype of the array.
    tmp_dir_abs_path : str, optional
        If specified the memory mapped array will be created in this directory.
        An absolute path is expected.
        Defaults to None. If not specified the global TEMP_DIR_NAME location will be used.

    Returns
    -------
    type
        A writable temporary memory-mapped array.
    """
    temp_dir_name = _init_temp_dir()

    # redefine the temporary directory if a new location is given otherwise read from global variable
    # this allows you to ensure that the correct temp directory location is used when working with multiple threads
    if tmp_dir_abs_path is not None:
        _change_temp_dir_location(tmp_dir_abs_path)
        temp_dir_name = tmp_dir_abs_path

    temp_file_path = os.path.join(
        temp_dir_name, f"temp_mmap_{np.random.randint(2**63, dtype=np.int64)}.hdf"
    )

    with h5py.File(temp_file_path, "w") as hdf_file:
        created_array = hdf_file.create_dataset("array", shape=shape, dtype=dtype)
        created_array[0] = (
            np.string_("") if isinstance(dtype, np.dtypes.StrDType) else 0
        )
        offset = created_array.id.get_offset()

    with open(temp_file_path, "rb+") as raw_hdf_file:
        mmap_obj = mmap.mmap(raw_hdf_file.fileno(), 0, access=mmap.ACCESS_WRITE)

        return np.frombuffer(
            mmap_obj, dtype=dtype, count=np.prod(shape), offset=offset
        ).reshape(shape)


def create_empty_mmap(
    shape: tuple,
    dtype: np.dtype,
    file_path: str = None,
    overwrite: bool = False,
    tmp_dir_abs_path: str = None,
):
    """Initialize a new HDF5 file compatible with mmap. Returns the path to the initialized file.
    File can be mapped using the mmap_array_from_path function.

    Parameters
    ----------
    shape : tuple
        A tuple with the shape of the array.
    dtype : type
        The np.dtype of the array.
    file_path : str, optional
        The absolute path to the file that should be created. This includes the file name.
        Defaults to None.
        If None a random file name will be generated in the default tempdir location.
    overwrite : bool , optional
        If True the file will be overwritten if it already exists.
        Defaults to False.
    tmp_dir_abs_path : str, optional
        If specified the default tempdir location will be updated to this path. Defaults to None. An absolute path to a directory is expected.

    Returns
    -------
    str
        path to the newly created file.
    """

    temp_dir_name = _init_temp_dir()

    # redefine the temporary directory if a new location is given otherwise read from global variable
    # this allows you to ensure that the correct temp directory location is used when working with multiple threads
    if tmp_dir_abs_path is not None:
        _change_temp_dir_location(tmp_dir_abs_path)
        temp_dir_name = tmp_dir_abs_path

    # if path does not exist generate a random file name in the TEMP directory
    if file_path is None:
        temp_file_path = os.path.join(
            temp_dir_name, f"temp_mmap_{np.random.randint(2**63, dtype=np.int64)}.hdf"
        )
    else:
        temp_file_path = _get_file_location(file_path, overwrite=overwrite)

    with h5py.File(temp_file_path, "w") as hdf_file:
        created_array = hdf_file.create_dataset("array", shape=shape, dtype=dtype)
        created_array[0] = (
            np.string_("") if isinstance(dtype, np.dtypes.StrDType) else 0
        )

    return temp_file_path


def mmap_array_from_path(hdf_file: str) -> np.ndarray:
    """reconnect to an exisiting HDF5 file to generate a writable temporary memory-mapped array.

    Parameters
    ----------
    hdf_file : str
        path to the array that should be reconnected to.

    Returns
    -------
    type
        A writable temporary memory-mapped array.
    """

    path = os.path.join(hdf_file)

    # read parameters required to reinitialize the mmap object
    with h5py.File(path, "r") as hdf_file:
        array_ = hdf_file["array"]
        offset = array_.id.get_offset()
        shape = array_.shape
        dtype = array_.dtype

    # reinitialize the mmap object
    with open(path, "rb+") as raw_hdf_file:
        mmap_obj = mmap.mmap(raw_hdf_file.fileno(), 0, access=mmap.ACCESS_WRITE)
        return np.frombuffer(
            mmap_obj, dtype=dtype, count=np.prod(shape), offset=offset
        ).reshape(shape)


def zeros(shape: tuple, dtype: np.dtype) -> np.ndarray:
    """Create a writable temporary memory-mapped array filled with zeros.

    Parameters
    ----------
    shape : tuple
        A tuple with the shape of the array.
    dtype : type
        The np.dtype of the array.

    Returns
    -------
    type
        A writable temporary memory-mapped array filled with zeros.
    """
    array_ = array(shape, dtype)
    array_[:] = 0
    return array_


def ones(shape: tuple, dtype: np.dtype) -> np.ndarray:
    """Create a writable temporary memory-mapped array filled with ones.

    Parameters
    ----------
    shape : tuple
        A tuple with the shape of the array.
    dtype : type
        The np.dtype of the array.

    Returns
    -------
    type
        A writable temporary memory-mapped array filled with ones.
    """
    array_ = array(shape, dtype)
    array_[:] = 1
    return array_


@atexit.register
def _clear() -> None:
    """Reset the temporary folder containing temp memory-mapped arrays.

    WARNING: All existing temp mmapp arrays will be unusable!
    """
    global _TEMP_DIR, TEMP_DIR_NAME

    if _TEMP_DIR is not None:
        logging.info(
            f"Temporary folder {TEMP_DIR_NAME} with memory-mapped arrays is being deleted. "
            "All existing memory-mapped arrays will be unusable!"
        )

        _TEMP_DIR = None  # TempDirectory will take care of the cleanup
        if os.path.exists(TEMP_DIR_NAME):
            logging.warning(
                f"Temporary folder {TEMP_DIR_NAME} still exists, manual removal necessary."
            )
        TEMP_DIR_NAME = None


def clear() -> str:
    """Reset the temporary folder containing temp memory-mapped arrays and create a new one.

    WARNING: All existing temp mmapp arrays will be unusable!

    Returns
    -------
    str
        The name of the new temporary folder.
    """
    _clear()

    temp_dir_name = _init_temp_dir()

    return temp_dir_name
