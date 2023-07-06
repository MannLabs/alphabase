import os
import numpy as np

from alphabase.yaml_utils import load_yaml

CONST_FILE_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "const_files"
)

common_const_dict:dict = load_yaml(
    os.path.join(CONST_FILE_FOLDER, "common_constants.yaml")
)

# Only applied in peak and fragment dataframes to save RAM. 
# Using float32 still keeps 0.1 ppm precision in any value range.
# Default float dtype is "float64" for value calculation and other senarios.
PEAK_MZ_DTYPE:np.dtype = np.dtype(
    common_const_dict["PEAK_MZ_DTYPE"]
).type
PEAK_INTENSITY_DTYPE:np.dtype = np.dtype(
    common_const_dict["PEAK_INTENSITY_DTYPE"]
).type