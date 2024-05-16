#!python
"""This module allows to create temporary mmapped arrays."""

# builtin
import os
import logging
import atexit

# external
import numpy as np
import mmap
import h5py
import tempfile
import shutil

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

    global _TEMP_DIR, _TEMP_DIR, TEMP_DIR_NAME
    logging.warning(
        f"""Folder {TEMP_DIR_NAME} with temp mmap arrays is being deleted.All existing temp mmapp arrays will be unusable!"""
    )

    # cleaup old temporary directory
    shutil.rmtree(TEMP_DIR_NAME, ignore_errors=True)
    del _TEMP_DIR

    # create new tempfile at desired location
    _TEMP_DIR = tempfile.TemporaryDirectory(prefix=os.path.join(path, "temp_mmap"))
    TEMP_DIR_NAME = _TEMP_DIR.name
    logging.warning(
        f"""New temp folder location. Temp mmap arrays are written to {TEMP_DIR_NAME}. Cleanup of this folder is OS dependant, and might need to be triggered manually!"""
    )
    return TEMP_DIR_NAME


def array(shape: tuple, dtype: np.dtype) -> np.ndarray:
    """Create a writable temporary mmapped array.

    Parameters
    ----------
    shape : tuple
        A tuple with the shape of the array.
    dtype : type
        The np.dtype of the array.

    Returns
    -------
    type
        A writable temporary mmapped array.
    """
    _log_cleanup_info_once()

    temp_file_name = os.path.join(
        TEMP_DIR_NAME, f"temp_mmap_{np.random.randint(2**63)}.hdf"
    )

    with h5py.File(temp_file_name, "w") as hdf_file:
        array = hdf_file.create_dataset("array", shape=shape, dtype=dtype)
        array[0] = 0
        offset = array.id.get_offset()

    with open(temp_file_name, "rb+") as raw_hdf_file:
        mmap_obj = mmap.mmap(raw_hdf_file.fileno(), 0, access=mmap.ACCESS_WRITE)
        return np.frombuffer(
            mmap_obj, dtype=dtype, count=np.prod(shape), offset=offset
        ).reshape(shape)


def create_empty_mmap(shape: tuple, dtype: np.dtype, path: str = None, overwrite=False):
    """Initialize a new HDF5 file compatible with mmap. Returns the path to the initialized file.
    File can be mapped using the mmap_array_from_path function.

    Parameters
    ----------
    shape : tuple
        A tuple with the shape of the array.
    dtype : type
        The np.dtype of the array.
    path : str, optional
        The path to the file that should be created.
        Defaults to None.
        If None a random file name will be generated.
    overwrite : bool , optional
        If True the file will be overwritten if it already exists.
        Defaults to False.

    Returns
    -------
    str
        path to the newly created file.
    """
    _log_cleanup_info_once()

    # if path does not exist generate a random file name in the TEMP directory
    if path is None:
        temp_file_name = os.path.join(
            TEMP_DIR_NAME, f"temp_mmap_{np.random.randint(2**63)}.hdf"
        )
    else:
        # check that if overwrite is false the file does not already exist
        if not overwrite:
            if os.path.exists(path):
                raise ValueError(
                    "The file already exists. Set overwrite to True to overwrite the file or choose a different name."
                )
        if not os.path.basename.endswith(".hdf"):
            raise ValueError("The chosen file name needs to end with .hdf")
        if os.path.isdir(os.path.commonpath(path)):
            temp_file_name = path
        else:
            raise ValueError(
                "The directory in which the file should be created does not exist."
            )

    with h5py.File(temp_file_name, "w") as hdf_file:
        array = hdf_file.create_dataset("array", shape=shape, dtype=dtype)
        array[0] = 0

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
    global _TEMP_DIR
    global TEMP_DIR_NAME
    logging.warning(
        f"Folder {TEMP_DIR_NAME} with temp mmap arrays is being deleted. "
        "All existing temp mmapp arrays will be unusable!"
    )
    del _TEMP_DIR
    _TEMP_DIR = tempfile.TemporaryDirectory(prefix="temp_mmap_")
    TEMP_DIR_NAME = _TEMP_DIR.name
    return TEMP_DIR_NAME
