import os
import numba
import numpy as np
import pandas as pd
from typing import Union, List

from alphabase.constants.element import (
    calc_mass_from_formula, parse_formula,
)

from alphabase.constants._const import CONST_FILE_FOLDER


#: Main entry of modification infomation (DataFrame fotmat).
MOD_DF:pd.DataFrame = pd.DataFrame()


MOD_INFO_DICT:dict = {}
#: Modification to formula str dict. {mod_name: formula str ('H(1)C(2)O(3)')}
MOD_CHEM:dict = {}
#: Modification to mass dict.
MOD_MASS:dict = {}
#: Modification to modification neutral loss dict.
MOD_LOSS_MASS:dict = {}
#Modification to formula dict of dict, i.e. {modname: {'C': n, 'H': m, ...}}
MOD_Composition:dict = {}
#: Modification loss importance
MOD_LOSS_IMPORTANCE:dict = {}

def update_all_by_MOD_DF():
    """
    As DataFrame is more conveneint in data operation, 
    we can also process MOD_DF and then update all global 
    modification variables from MOD_DF
    """
    
    MOD_INFO_DICT.clear()
    MOD_INFO_DICT.update(MOD_DF.to_dict(orient='index'))
    MOD_CHEM.clear()
    MOD_CHEM.update(MOD_DF['composition'].to_dict())
    MOD_MASS.clear()
    MOD_MASS.update(MOD_DF['mass'].to_dict())
    MOD_LOSS_MASS.clear()
    MOD_LOSS_MASS.update(MOD_DF['modloss'].to_dict())
    MOD_LOSS_IMPORTANCE.clear()
    MOD_LOSS_IMPORTANCE.update(MOD_DF['modloss_importance'].to_dict())

    MOD_Composition.clear()
    for mod, chem in MOD_CHEM.items():
        MOD_Composition[mod] = dict(parse_formula(chem))

def add_modifications_for_lower_case_AA():
    """ Add modifications for lower-case AAs for advanced usages """
    global MOD_DF
    lower_case_df = MOD_DF.copy()
    
    def _mod_lower_case(modname):
        modname, site = modname.split('@')
        if len(site) == 1:
            return modname+'@'+site.lower()
        elif '^' in site:
            site = site[0].lower()+site[1:]
            return modname+'@'+site
        else:
            return ''
    lower_case_df['mod_name'] = lower_case_df['mod_name'].apply(_mod_lower_case)
    lower_case_df = lower_case_df[lower_case_df['mod_name']!='']
    lower_case_df.set_index('mod_name', drop=False, inplace=True)
    lower_case_df['lower_case_AA'] = True
    MOD_DF['lower_case_AA'] = False
    MOD_DF = pd.concat([MOD_DF, lower_case_df])
    update_all_by_MOD_DF()

def keep_modloss_by_importance(modloss_importance_level:float=1.0):
    MOD_DF['modloss'] = MOD_DF['modloss_original']
    MOD_DF.loc[MOD_DF.modloss_importance<modloss_importance_level,"modloss"] = 0
    MOD_LOSS_MASS.clear()
    MOD_LOSS_MASS.update(MOD_DF['modloss'].to_dict())

def load_mod_df(
    tsv:str=os.path.join(CONST_FILE_FOLDER, 'modification.tsv'),
    *,
    modloss_importance_level=1,
):
    global MOD_DF
    MOD_DF = pd.read_table(tsv)
    MOD_DF.fillna('',inplace=True)
    MOD_DF['unimod_id'] = MOD_DF.unimod_id.astype(np.int32)
    MOD_DF.set_index('mod_name', drop=False, inplace=True)
    MOD_DF['mass'] = MOD_DF.composition.apply(calc_mass_from_formula)
    MOD_DF['modloss_original'] = MOD_DF.modloss_composition.apply(calc_mass_from_formula)
    MOD_DF['modloss'] = MOD_DF['modloss_original']
    keep_modloss_by_importance(modloss_importance_level)
    update_all_by_MOD_DF()
    
load_mod_df()

def calc_modification_mass(
    nAA:int, 
    mod_names:List[str], 
    mod_sites:List[int]
)->np.ndarray:
    '''
    Calculate modification masses for the given peptide length (`nAA`), 
    and modified site list.
    
    Parameters
    ----------
    nAA : int
        Peptide length

    mod_names : list
        List[str]. Modification name list

    mod_sites : list
        List[int]. Modification site list corresponding to `mod_names`.
        * `site=0` refers to an N-term modification
        * `site=-1` refers to a C-term modification
        * `1<=site<=peplen` refers to a normal modification
    
    Returns
    -------
    np.ndarray
        1-D array with length=`nAA`. 
        Masses of modifications through the peptide, 
        `0` if sites has no modifications
    '''
    masses = np.zeros(nAA)
    for site, mod in zip(mod_sites, mod_names):
        if site == 0:
            masses[site] += MOD_MASS[mod]
        elif site == -1:
            masses[site] += MOD_MASS[mod]
        else:
            masses[site-1] += MOD_MASS[mod]
    return masses

def calc_mod_masses_for_same_len_seqs(
    nAA:int, 
    mod_names_list:List[List[str]], 
    mod_sites_list:List[List[int]]
)->np.ndarray:
    '''
    Calculate modification masses for the given peptides with same peptide length (`nAA`).
    
    Parameters
    ----------
    nAA : int
    
        Peptide length

    mod_names_list : List[List[str]]
        List (pep_count) of modification list (n_mod on each peptide)

    mod_sites_list : List[List[int]]
        List of modification site list corresponding to `mod_names_list`.
        * `site=0` refers to an N-term modification
        * `site=-1` refers to a C-term modification
        * `1<=site<=peplen` refers to a normal modification
    
    Returns
    -------
    np.ndarray
        2-D array with shape=`(nAA, pep_count or len(mod_names_list)))`. 
        Masses of modifications through all the peptides, 
        `0` if sites without modifications.
    '''
    masses = np.zeros((len(mod_names_list),nAA))
    for i, (mod_names, mod_sites) in enumerate(
        zip(mod_names_list, mod_sites_list)
    ):
        for mod, site in zip(mod_names, mod_sites): 
            if site == 0:
                masses[i,site] += MOD_MASS[mod]
            elif site == -1:
                masses[i,site] += MOD_MASS[mod]
            else:
                masses[i,site-1] += MOD_MASS[mod]
    return masses

def calc_modification_mass_sum(
    mod_names:List[str]
)->float:
    """
    Calculate summed mass of the given modification 
    without knowing the sites and peptide length.
    It is useful to calculate peptide mass.
    
    Parameters
    ----------
    mod_names : List[str]
        Modification name list

    Returns
    -------
    float
        Total mass
    """
    return np.sum([
        MOD_MASS[mod] for mod in mod_names
    ])


@numba.jit(nopython=True, nogil=True)
def _calc_modloss_with_importance(
    mod_losses: np.ndarray, 
    _loss_importance: np.ndarray
)->np.ndarray:
    '''
    Calculate modification loss masses (e.g. -98 Da for Phospho@S/T). 
    Modification with higher `_loss_importance` has higher priorities. 
    For example, `AM(Oxidation@M)S(Phospho@S)...`,
    importance of Phospho@S > importance of Oxidation@M, so the modloss of 
    b3 ion will be -98 Da, not -64 Da.
    
    Parameters
    ----------
    mod_losses : np.ndarray
        Mod loss masses of each AA position

    _loss_importance : np.ndarray
        Mod loss importance of each AA position
    
    Returns
    -------
    np.ndarray
        New mod_loss masses selected by `_loss_importance`
    '''
    prev_importance = _loss_importance[0]
    prev_most = 0
    for i, _curr_imp in enumerate(_loss_importance[1:],1):
        if _curr_imp > prev_importance:
            prev_most = i
            prev_importance = _curr_imp
        else:
            mod_losses[i] = mod_losses[prev_most]
    return mod_losses

def calc_modloss_mass_with_importance(
    nAA: int, 
    mod_names: List, 
    mod_sites: List,
    for_nterm_frag: bool,
)->np.ndarray:
    '''
    Calculate modification loss masses (e.g. -98 Da for Phospho@S/T, 
    -64 Da for Oxidation@M). Modifications with higher `MOD_LOSS_IMPORTANCE` 
    have higher priorities. For example, `AS(Phospho@S)M(Oxidation@M)...`,
    importance of Phospho@S > importance of Oxidation@M, so the modloss of 
    b3 ion will be -98 Da, not -64 Da.
    
    Parameters
    ----------
    nAA : int
        Peptide length

    mod_names : List[str]
        Modification name list

    mod_sites : List[int]
        Modification site list  

    for_nterm_frag : bool
        If `True`, the loss will be on the 
        N-term fragments (mainly `b` ions); 
        If `False`, the loss will be on the 
        C-term fragments (mainly `y` ions)
    
    Returns
    -------
    np.ndarray
        mod_loss masses
    '''
    if not mod_names: return np.zeros(nAA-1)
    mod_losses = np.zeros(nAA+2)
    mod_losses[mod_sites] = [MOD_LOSS_MASS[mod] for mod in mod_names]
    _loss_importance = np.zeros(nAA+2)
    _loss_importance[mod_sites] = [
        MOD_LOSS_IMPORTANCE[mod] if mod in MOD_LOSS_IMPORTANCE else 0 
        for mod in mod_names
    ]
    
    # Will not consider the modloss if the corresponding modloss_importance is 0
    mod_losses[_loss_importance==0] = 0

    if for_nterm_frag:
        return _calc_modloss_with_importance(mod_losses, _loss_importance)[1:-2]
    else:
        return _calc_modloss_with_importance(mod_losses[::-1], _loss_importance[::-1])[-3:0:-1]

@numba.njit
def _calc_modloss(
    mod_losses: np.ndarray
)->np.ndarray:
    '''
    Calculate modification loss masses (e.g. -98 Da for Phospho@S/T). 
    
    Parameters
    ----------
    mod_losses : np.ndarray
        Mod loss masses of each AA position
    
    Returns
    -------
    np.ndarray
        New mod_loss masses 
    '''
    for i, _curr_loss in enumerate(mod_losses[1:],1):
        if _curr_loss == 0:
            mod_losses[i] = mod_losses[i-1]
        else:
            mod_losses[i] = _curr_loss
    return mod_losses
    
def calc_modloss_mass(
    nAA: int, 
    mod_names: List, 
    mod_sites: List,
    for_nterm_frag: bool,
)->np.ndarray:
    '''
    Calculate modification loss masses (e.g. -98 Da for Phospho@S/T, 
    -64 Da for Oxidation@M). The mod loss mass is calculated by the
    modification closer to the fragment sites. For example, 
    the modloss of the b3 ion for `AS(Phospho@S)M(Oxidation@M)...`
    will be -64 Da.
    
    Parameters
    ----------
    nAA : int
        Peptide length

    mod_names : List[str]
        Modification name list

    mod_sites : List[int]
        Modification site list corresponding 

    for_nterm_frag : bool
        If `True`, the loss will be on the 
        N-term fragments (mainly `b` ions); 
        If `False`, the loss will be on the 
        C-term fragments (mainly `y` ions)
    
    Returns
    -------
    np.ndarray
        mod_loss masses
    '''
    if len(mod_names) == 0: return np.zeros(nAA-1)
    mod_losses = np.zeros(nAA+2)
    mod_losses[mod_sites] = [MOD_LOSS_MASS[mod] for mod in mod_names]

    if for_nterm_frag:
        return _calc_modloss(mod_losses)[1:-2]
    else:
        return _calc_modloss(mod_losses[::-1])[-3:0:-1]

def _add_a_new_modification(
    mod_name:str, composition:str,
    modloss_composition:str=''
):
    """
    Add a new modification into :data:`MOD_DF`.
    """
    MOD_DF.loc[mod_name,[
        'mod_name','composition','modloss_composition',
        'classification','unimod_id'
    ]] = [
        mod_name, composition, modloss_composition,
        'User-added', 0
    ]
    MOD_DF.loc[mod_name,['mass','modloss']] = (
        calc_mass_from_formula(composition),
        calc_mass_from_formula(modloss_composition)
    )
    if MOD_DF.loc[mod_name, 'modloss'] > 0:
        MOD_DF.loc[mod_name, 'modloss_importance'] = 1e10
    MOD_DF.fillna(0, inplace=True)
    # update_all_by_MOD_DF()

def add_new_modifications(new_mods:Union[list,dict]):
    """Add new modifications into :data:`MOD_DF`.

    Parameters
    ----------
    new_mods : list or dict
        list of tuples example:
        ```
        [(
        mod@site:str (e.g. Mod@S), 
        composition:str (e.g. "H(4)O(2)"),
        [optional] modloss composition:str (e.g. "H(2)O(1)"),
        ), ...]
        ```,
        dict example:
        ```
        {
        "mod@site": {
        "composition":"H(4)O(2)",
        "modloss_composition":"H(2)O(1)"
        }, ...
        }
        ```
    """
    if isinstance(new_mods, list):
        for items in new_mods:
            _add_a_new_modification(*items)
    else:
        for mod_name, mod_info in new_mods.items():
            _add_a_new_modification(mod_name, **mod_info)
    update_all_by_MOD_DF()
