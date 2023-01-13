import numpy as np
from typing import List, Tuple

from alphabase.constants.aa import (
    calc_AA_masses, 
    calc_AA_masses_for_same_len_seqs,
    calc_sequence_masses_for_same_len_seqs
)
from alphabase.constants.modification import (
    calc_modification_mass,
    calc_modification_mass_sum,
    calc_mod_masses_for_same_len_seqs
)
from alphabase.constants.element import MASS_H2O

def calc_delta_modification_mass(
    pep_len:int,
    mass_deltas:List[float],
    mass_delta_sites:List[int]
)->np.ndarray:
    '''
    For open-search, we may also get modification 
    mass deltas other than mod names. This function calculate
    modification masses from these delta masses.
    
    Parameters
    ----------
    pep_len : int
        nAA
    
    mass_deltas : List[float]
        mass deltas on the peptide

    mass_delta_sites : List[int]
        localized sites of corresponding mass deltas

    Returns
    -------
    np.ndarray
        1-D array with length=`peplen`.
        Masses of modifications (mass deltas) through the peptide,
        `0` if sites has no modifications
    '''
    masses = np.zeros(pep_len)
    for site, mass in zip(mass_delta_sites, mass_deltas):
        if site == 0:
            masses[site] += mass
        elif site == -1:
            masses[site] += mass
        else:
            masses[site-1] += mass
    return masses

def calc_mod_delta_masses_for_same_len_seqs(
    nAA:int, 
    mod_deltas_list:List[List[float]], 
    mod_sites_list:List[List[int]]
)->np.ndarray:
    '''
    Calculate delta modification masses for the given peptide length (`nAA`), 
    For open-search, we may also get modification 
    mass deltas other than mod names. This function calculate
    modification masses from these delta masses.
    
    Parameters
    ----------
    nAA : int
        peptide length

    mod_names_list : List[List[str]]
        list of modification list

    mod_sites_list : List[List[int]]
        list of modification site list corresponding 
        to `mod_names_list`.
        * `site=0` refers to an N-term modification
        * `site=-1` refers to a C-term modification
        * `1<=site<=peplen` refers to a normal modification
    
    Returns
    -------
    np.ndarray
        2-D array with shape=`(nAA, pep_count or len(mod_names_list)))`. 
        Masses of modifications through all the peptides, 
        `0` if sites has no modifications
    '''
    masses = np.zeros((len(mod_deltas_list),nAA))
    for i, (mod_deltas, mod_sites) in enumerate(
        zip(mod_deltas_list, mod_sites_list)
    ):
        for mod_delta, site in zip(mod_deltas, mod_sites): 
            if site == 0:
                masses[i,site] += mod_delta
            elif site == -1:
                masses[i,site] += mod_delta
            else:
                masses[i,site-1] += mod_delta
    return masses

def calc_b_y_and_peptide_mass(
    sequence: str,
    mod_names: List[str],
    mod_sites: List[int],
    mod_deltas: List[float] = None,
    mod_delta_sites: List[int] = None,
)->Tuple[np.ndarray,np.ndarray,float]:
    '''
    It is highly recommend to use 
    `calc_b_y_and_peptide_masses_for_same_len_seqs`
    as it is much faster
    '''
    residue_masses = calc_AA_masses(sequence)
    mod_masses = calc_modification_mass(
        len(sequence), mod_names, mod_sites
    )
    residue_masses += mod_masses
    if mod_deltas is not None:
        mod_masses = calc_delta_modification_mass(
            len(sequence), mod_deltas, mod_delta_sites
        )
        residue_masses += mod_masses
    #residue_masses = residue_masses[np.newaxis, ...]
    b_masses = np.cumsum(residue_masses)
    b_masses, pepmass = b_masses[:-1], b_masses[-1]
        
    pepmass += MASS_H2O
    y_masses = pepmass - b_masses
    return b_masses, y_masses, pepmass

def calc_peptide_masses_for_same_len_seqs(
    sequences: np.ndarray,
    mod_list: List[str],
    mod_delta_list: List[str]=None
)->np.ndarray:
    '''
    Calculate peptide masses for peptide sequences with same lengths.
    We need 'same_len' here because numpy can process AA sequences 
    with same length very fast. 
    See `alphabase.aa.calc_sequence_masses_for_same_len_seqs`

    Parameters
    ----------
    mod_list : List[str]

        list of modifications, 
        e.g. `['Oxidation@M;Phospho@S','Phospho@S;Deamidated@N']`

    mass_delta_list : List[str]
    
        List of modifications as mass deltas,
        e.g. `['15.9xx;79.9xxx','79.9xx;0.98xx']`
    
    Returns
    -------
        np.ndarray
            
            peptide masses (1-D array, H2O already added)
    '''
    seq_masses = calc_sequence_masses_for_same_len_seqs(
        sequences
    )
    mod_masses = np.zeros_like(seq_masses)
    for i, mods in enumerate(mod_list):
        if len(mods) > 0:
            mod_masses[i] = calc_modification_mass_sum(
                mods.split(';')
            )
    if mod_delta_list is not None:
        for i, mass_deltas in enumerate(mod_delta_list):
            if len(mass_deltas) > 0:
                mod_masses[i] += np.sum([
                    float(mass) for mass in mass_deltas.split(';')
                ])
    return seq_masses+mod_masses
    

def calc_b_y_and_peptide_masses_for_same_len_seqs(
    sequences: np.ndarray,
    mod_list: List[List[str]],
    site_list: List[List[int]],
    mod_delta_list: List[List[float]]=None,
    mod_delta_site_list: List[List[int]]=None,
)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''
    Calculate b/y fragment masses and peptide masses 
    for peptide sequences with same lengths.
    We need 'same_len' here because numpy can process AA sequences 
    with same length very fast.

    Parameters
    ----------
    sequence : np.ndarray of str
        np.ndarray of peptie sequences with same length.

    mod_list : List[List[str]]
        list of modifications , 
        e.g. `[['Oxidation@M','Phospho@S'],['Phospho@S','Deamidated@N']]` 

    site_list : List[List[int]]
        list of modification sites
        corresponding to `mod_list`, e.g. `[[3,6],[4,17]]`

    mod_delta_list : List[List[float]]
        list of modifications, 
        e.g. `[[15.994915,79.966331],[79.966331,0.984016]]` 

    mod_delta_site_list : List[List[int]]
        list of modification mass delta sites
        corresponding to `mod_list`, e.g. `[[3,6],[4,17]]`
    
    Returns
    -------
    np.ndarray
        neutral b fragment masses (2-D array)

    np.ndarray
        neutral y fragmnet masses (2-D array)

    np.ndarray
        neutral peptide masses (1-D array)
    '''
    aa_masses = calc_AA_masses_for_same_len_seqs(sequences)
    nAA = len(sequences[0])

    # mod_masses = np.zeros_like(aa_masses)
    # for i, (mods, sites) in enumerate(zip(mod_list, site_list)):
    #     if len(mods) != 0:
    #         mod_masses[i,:] = calc_modification_mass(
    #             seq_len, 
    #             mods, 
    #             sites,
    #         )
    mod_masses = calc_mod_masses_for_same_len_seqs(nAA, mod_list, site_list)
    if mod_delta_list is not None:
        mod_masses += calc_mod_delta_masses_for_same_len_seqs(
            nAA, mod_delta_list, mod_delta_site_list
        )
        # for i, (mass_deltas, sites) in enumerate(zip(
        #     mass_delta_list, mass_delta_site_list
        # )):
        #     if len(mass_deltas) != 0:
        #         mod_masses[i,:] += calc_delta_modification_mass(
        #             seq_len, 
        #             mass_deltas, 
        #             sites,
        #         )
    aa_masses += mod_masses

    b_masses = np.cumsum(aa_masses, axis=1)
    b_masses, pepmass = b_masses[:,:-1], b_masses[:,-1:]
        
    pepmass += MASS_H2O
    y_masses = pepmass - b_masses
    return b_masses, y_masses, pepmass.flatten()
