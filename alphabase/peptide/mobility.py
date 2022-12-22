import numpy as np
import pandas as pd

from alphabase.peptide.fragment import update_precursor_mz
from alphabase.constants.element import common_const_dict

CCS_IM_COEF = common_const_dict['MOBILITY']['CCS_IM_COEF']
IM_GAS_MASS = common_const_dict['MOBILITY']['IM_GAS_MASS']

def get_reduced_mass(
    precursor_mzs: np.ndarray, 
    charges: np.ndarray
)->np.ndarray:
    """ Reduced mass for CCS and mobility calculation """
    reduced_masses = precursor_mzs*charges
    return reduced_masses*IM_GAS_MASS/(reduced_masses+IM_GAS_MASS)

def ccs_to_mobility_bruker(
    ccs_values: np.ndarray, 
    charges: np.ndarray, 
    precursor_mzs: np.ndarray
)->np.ndarray:
    """ Convert CCS to mobility for Bruker (timsTOF) """
    reduced_masses = get_reduced_mass(precursor_mzs, charges)
    return ccs_values*np.sqrt(reduced_masses)/charges/CCS_IM_COEF

def mobility_to_ccs_bruker(
    im_values: np.ndarray, 
    charges: np.ndarray, 
    precursor_mzs: np.ndarray
)->np.ndarray:
    """ Convert mobility to CCS for Bruker (timsTOF) """
    reduced_masses = get_reduced_mass(precursor_mzs, charges)
    return im_values*charges*CCS_IM_COEF/np.sqrt(reduced_masses)

def ccs_to_mobility_for_df(
    precursor_df:pd.DataFrame,
    ccs_column:str,
    *,
    vendor="bruker"
)->np.ndarray:
    """

    Parameters
    ----------
    precursor_df : pd.DataFrame
        precursor_df

    ccs_column : str
        CCS column name in precursor_df

    vendor : str, optional
        Different vender may have different IM calculation. 
        Defaults to "bruker".
        Note that other vendors are not implemented currently.

    Returns
    -------
    np.ndarray
        mobility values
    """
    if 'precursor_mz' not in precursor_df.columns:
        precursor_df = update_precursor_mz(precursor_df)
    return ccs_to_mobility_bruker(
        precursor_df[ccs_column].values, 
        precursor_df.charge.values,
        precursor_df.precursor_mz.values
    )

def mobility_to_ccs_for_df(
    precursor_df:pd.DataFrame,
    mobility_column:str,
    *,
    vendor="bruker"
)->np.ndarray:
    """

    Parameters
    ----------
    precursor_df : pd.DataFrame
        precursor_df

    mobility_column : str
        mobility column name in precursor_df

    vendor : str, optional
        Different vender may have different IM calculation. 
        Defaults to "bruker".
        Note that other vendors are not implemented currently.

    Returns
    -------
    np.ndarray
        CCS values
    """
    
    if 'precursor_mz' not in precursor_df.columns:
        precursor_df = update_precursor_mz(precursor_df)
    return mobility_to_ccs_bruker(
        precursor_df[mobility_column].values,
        precursor_df.charge.values,
        precursor_df.precursor_mz.values
    )
