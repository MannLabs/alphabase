#!python
"""This module allows to create temporary mmapped arrays."""

# builtin
import atexit
import logging
import mmap
import os
import shutil
import tempfile

import h5py

# external
import numpy as np

# TODO initialize temp_dir not on import but when it is first needed
_TEMP_DIR = tempfile.TemporaryDirectory(prefix="temp_mmap_")
TEMP_DIR_NAME = _TEMP_DIR.name

is_cleanup_info_logged = False


def _log_cleanup_info_once() -> None:
    """Logs a info on temp array cleanup once."""
    global is_cleanup_info_logged
    if not is_cleanup_info_logged:
        logging.info(
            f"Temp mmap arrays are written to {TEMP_DIR_NAME}. "
            "Cleanup of this folder is OS dependent and might need to be triggered manually!"
        )
        is_cleanup_info_logged = True


def _change_temp_dir_location(abs_path: str) -> str:
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
            raise ValueError(f"The path {abs_path} does not point to a directory.")
    else:
        raise ValueError(
            f"The directory {abs_path} in which the file should be created does not exist."
        )


def _get_file_location(abs_file_path: str, overwrite=False) -> str:
    """
    Check if the path specified for the new temporary file is valid. If not raise a value error.

    Valid file paths need to:
    1. be contained in directories that exist
    2. end in .hdf
    3. not exist if overwrite is set to False

    Parameters
    ----------
    abs_path : str
        The absolute path to the new temporary file.

    Returns
    ------
    str
        The file path if it is valid.
    """
    # check overwrite status and existence of file
    if not overwrite and os.path.exists(abs_file_path):
        raise ValueError(
            "The file already exists. Set overwrite to True to overwrite the file or choose a different name."
        )

    # ensure that the filename conforms to the naming convention
    if not os.path.basename(abs_file_path).endswith(".hdf"):
        raise ValueError("The chosen file name needs to end with .hdf")

    # ensure that the directory in which the file should be created exists
    if os.path.isdir(os.path.commonpath(abs_file_path)):
        return abs_file_path
    else:
        raise ValueError(
            f"The directory {os.path.commonpath(abs_file_path)} in which the file should be created does not exist."
        )


def redefine_temp_location(path):
    """
    Redfine the location where the temp arrays are written to.

    Parameters
    ----------
    path : string

    Returns
    ------
    str
        the location of the new temporary directory.

    """

    global _TEMP_DIR, TEMP_DIR_NAME

    logging.warning(
        f"""Folder {TEMP_DIR_NAME} with temp mmap arrays is being deleted. All existing temp mmapp arrays will be unusable!"""
    )

    # cleaup old temporary directory
    shutil.rmtree(TEMP_DIR_NAME, ignore_errors=True)

    # create new tempfile at desired location
    _TEMP_DIR = tempfile.TemporaryDirectory(prefix=os.path.join(path, "temp_mmap_"))
    TEMP_DIR_NAME = _TEMP_DIR.name

    logging.warning(
        f"""New temp folder location. Temp mmap arrays are written to {TEMP_DIR_NAME}. Cleanup of this folder is OS dependant, and might need to be triggered manually!"""
    )

    return TEMP_DIR_NAME


def array(shape: tuple, dtype: np.dtype, tmp_dir_abs_path: str = None) -> np.ndarray:
    """Create a writable temporary mmapped array.

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
        A writable temporary mmapped array.
    """
    global TEMP_DIR_NAME

    _log_cleanup_info_once()

    # redefine the temporary directory if a new location is given otherwise read from global variable
    # this allows you to ensure that the correct temp directory location is used when working with multiple threads
    if tmp_dir_abs_path is not None:
        _change_temp_dir_location(tmp_dir_abs_path)

    temp_file_name = os.path.join(
        TEMP_DIR_NAME, f"temp_mmap_{np.random.randint(2**63, dtype=np.int64)}.hdf"
    )

    with h5py.File(temp_file_name, "w") as hdf_file:
        array = hdf_file.create_dataset("array", shape=shape, dtype=dtype)
        array[0] = np.string_("") if isinstance(dtype, np.dtypes.StrDType) else 0
        offset = array.id.get_offset()

    with open(temp_file_name, "rb+") as raw_hdf_file:
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
    global TEMP_DIR_NAME

    _log_cleanup_info_once()

    # redefine the temporary directory if a new location is given otherwise read from global variable
    # this allows you to ensure that the correct temp directory location is used when working with multiple threads
    if tmp_dir_abs_path is not None:
        _change_temp_dir_location(tmp_dir_abs_path)

    # if path does not exist generate a random file name in the TEMP directory
    if file_path is None:
        temp_file_name = os.path.join(
            TEMP_DIR_NAME, f"temp_mmap_{np.random.randint(2**63, dtype=np.int64)}.hdf"
        )
    else:
        temp_file_name = _get_file_location(file_path, overwrite=False)

    with h5py.File(temp_file_name, "w") as hdf_file:
        array = hdf_file.create_dataset("array", shape=shape, dtype=dtype)
        array[0] = np.string_("") if isinstance(dtype, np.dtypes.StrDType) else 0

    return temp_file_name


def mmap_array_from_path(hdf_file: str) -> np.ndarray:
    """reconnect to an exisiting HDF5 file to generate a writable temporary mmapped array.

    Parameters
    ----------
    hdf_file : str
        path to the array that should be reconnected to.

    Returns
    -------
    type
        A writable temporary mmapped array.
    """
    _log_cleanup_info_once()

    path = os.path.join(hdf_file)

    # read parameters required to reinitialize the mmap object
    with h5py.File(path, "r") as hdf_file:
        array = hdf_file["array"]
        offset = array.id.get_offset()
        shape = array.shape
        dtype = array.dtype

    # reinitialize the mmap object
    with open(path, "rb+") as raw_hdf_file:
        mmap_obj = mmap.mmap(raw_hdf_file.fileno(), 0, access=mmap.ACCESS_WRITE)
        return np.frombuffer(
            mmap_obj, dtype=dtype, count=np.prod(shape), offset=offset
        ).reshape(shape)


def zeros(shape: tuple, dtype: np.dtype) -> np.ndarray:
    """Create a writable temporary mmapped array filled with zeros.

    Parameters
    ----------
    shape : tuple
        A tuple with the shape of the array.
    dtype : type
        The np.dtype of the array.

    Returns
    -------
    type
        A writable temporary mmapped array filled with zeros.
    """
    _array = array(shape, dtype)
    _array[:] = 0
    return _array


def ones(shape: tuple, dtype: np.dtype) -> np.ndarray:
    """Create a writable temporary mmapped array filled with ones.

    Parameters
    ----------
    shape : tuple
        A tuple with the shape of the array.
    dtype : type
        The np.dtype of the array.

    Returns
    -------
    type
        A writable temporary mmapped array filled with ones.
    """
    _array = array(shape, dtype)
    _array[:] = 1
    return _array


@atexit.register
def clear() -> str:
    """Reset the temporary folder containing temp mmapped arrays.

    WARNING: All existing temp mmapp arrays will be unusable!

    Returns
    -------
    str
        The name of the new temporary folder.
    """
    global _TEMP_DIR, TEMP_DIR_NAME

    logging.warning(
        f"Folder {TEMP_DIR_NAME} with temp mmap arrays is being deleted. "
        "All existing temp mmapp arrays will be unusable!"
    )

    del _TEMP_DIR

    _TEMP_DIR = tempfile.TemporaryDirectory(prefix="temp_mmap_")
    TEMP_DIR_NAME = _TEMP_DIR.name
    return TEMP_DIR_NAME
