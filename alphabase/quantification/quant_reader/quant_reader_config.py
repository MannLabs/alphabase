import logging
import pandas as pd
from . import config_dict_loader
import pathlib
import os


def setup_logging():
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
setup_logging()

##########################
PROTEIN_ID = 'protein'
QUANT_ID = 'ion'

def set_global_protein_and_ion_id(protein_id = 'protein', quant_id = 'ion'):
    global PROTEIN_ID
    global QUANT_ID
    PROTEIN_ID = protein_id
    QUANT_ID = quant_id

##########################

DEFAULT_CONFIG_PATH = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "../../../alphabase/constants/const_files/quant_reader_config.yaml",
)  # the yaml config is located one directory below the python library files
INTABLE_CONFIG = DEFAULT_CONFIG_PATH

def set_intable_config_file(intable_config_file = DEFAULT_CONFIG_PATH):
	global INTABLE_CONFIG
	INTABLE_CONFIG = intable_config_file
	

##########################
COPY_NUMPY_ARRAYS_DERIVED_FROM_PANDAS = False

def check_wether_to_copy_numpy_arrays_derived_from_pandas():
    global COPY_NUMPY_ARRAYS_DERIVED_FROM_PANDAS
    try:
        _manipulate_numpy_array_without_copy()
        COPY_NUMPY_ARRAYS_DERIVED_FROM_PANDAS = False
    except:
        logging.info('Some numpy arrays derived from pandas will be copied.')
        COPY_NUMPY_ARRAYS_DERIVED_FROM_PANDAS = True

def _manipulate_numpy_array_without_copy():
    
    protein_profile_df = pd.DataFrame({
    'ProteinA': [10, 20, 30, 40],
    'ProteinB': [15, 25, 35, 45],
    'ProteinC': [20, 30, 40, 50]
    }, index=['Sample1', 'Sample2', 'Sample3', 'Sample4'])
    
    protein_profile_df = protein_profile_df.iloc[1:3]
    protein_profile_numpy = protein_profile_df.to_numpy(copy=False)

    protein_profile_numpy[0] = protein_profile_numpy[0] +2
