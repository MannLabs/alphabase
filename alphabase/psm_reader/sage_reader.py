import numpy as np
import pandas as pd
import typing
import re
from functools import partial

from alphabase.psm_reader.psm_reader import (
    PSMReaderBase, psm_reader_provider,
    psm_reader_yaml
)

from alphabase.constants.modification import MOD_DF

def sage_spec_idx_from_scannr(scannr: str) -> int:
    """Extract the spectrum index from the scannr field in Sage output.

    Parameters
    ----------

    scannr : str
        The scannr field in Sage output.
    
    """
    return int(scannr.split('=')[-1])

def lookup_modification(
        mass_observed: float, 
        previous_aa: str, 
        mod_annotated_df: pd.DataFrame, 
        ppm_tolerance:int=10,
    ) -> str:
    """
    Look up a single modification based on the observed mass and the previous amino acid.

    Parameters
    ----------

    mass_observed : float
        The observed mass of the modification.

    previous_aa : str
        The previous amino acid.

    mod_annotated_df : pd.DataFrame
        The annotated modification dataframe.

    ppm_tolerance : int
        The ppm tolerance for matching the observed mass to the annotated modification mass.

    Returns
    -------

    str
        The name of the matched modification in alphabase format.
    
    """

    mass_distance = mod_annotated_df['mass'].values - mass_observed
    ppm_distance = mass_distance / mass_observed * 1e6
    ppm_distance = np.abs(ppm_distance)

    ppm_tolerance = min(np.min(ppm_distance), ppm_tolerance)

    # get index of matches
    mass_match = ppm_distance <= ppm_tolerance
    sequence_match = mod_annotated_df['location'] == previous_aa


    filtered_mod_df = mod_annotated_df[mass_match & sequence_match]
    if len(filtered_mod_df) == 0:
        print(np.min(ppm_distance))
        return None

    matched_mod = filtered_mod_df.sort_values(by='unimod_id').iloc[0]
    
    return matched_mod['mod_name']

def capture_modifications(
        sequence: str,
        mod_annotated_df: pd.DataFrame,
        ppm_tolerance: int=10
    ) -> typing.Tuple[str, str]:
    """ Capture modifications from a sequence string.

    Parameters
    ----------

    sequence : str
        The modified sequence string.

    mod_annotated_df : pd.DataFrame
        The annotated modification dataframe.

    ppm_tolerance : int
        The ppm tolerance for matching the observed mass to the annotated modification mass.

    Returns
    -------

    typing.Tuple[str, str]
        A tuple of two strings, the first string is the list of modification sites, and the second string is the list of modifications.

    """

    # get index of matches
    matches = re.finditer(r'\[(\+|-)(\d+\.\d+)\]', sequence)

    site_list = []
    mod_list = []

    error = False

    match_delta = 0

    for match in matches:
        match_start, match_end = match.start(), match.end()
        previous_aa = sequence[match_start-1] if match_start > 0 else 'Any_N-term'
        mass_observed = float(match.group(2)) * (1 if match.group(1) == '+' else -1)

        mod = lookup_modification(mass_observed, previous_aa, mod_annotated_df, ppm_tolerance=ppm_tolerance)
        if mod is not None:
            site_list.append(str(match_start-1-match_delta))
            mod_list.append(mod)
        
        else:
            error = True
            print(f'No modification found for mass {mass_observed} at position {match_start} with previous aa {previous_aa}')

        match_delta += (match_end - match_start)

    if error:
        return np.nan, np.nan
    else:
        return ';'.join(site_list), ';'.join(mod_list)
    
def get_annotated_mod_df():
    """ Annotates the modification dataframe with the location of the modification."""
    mod_annotated_df = MOD_DF.copy()
    mod_annotated_df['location'] = mod_annotated_df['mod_name'].str.split('@').str[1].str.split('^').str[0]
    mod_annotated_df = mod_annotated_df.sort_values(by='mass').reset_index(drop=True)
    mod_annotated_df['mod_name_stripped'] = mod_annotated_df['mod_name'].str.replace(' ','_')
    return mod_annotated_df

class SageReaderBase(PSMReaderBase):
    def __init__(self, *, 
        column_mapping: dict = None, 
        modification_mapping: dict = None,
        fdr = 0.01,
        keep_decoy=False, 
        rt_unit = 'second',
        **kwargs
    ):
        super().__init__(
            column_mapping=column_mapping, 
            modification_mapping=modification_mapping,
            fdr = fdr,
            keep_decoy=keep_decoy, 
            rt_unit = rt_unit,
            **kwargs
        )

    def _init_column_mapping(self):
        self.column_mapping = psm_reader_yaml[
            'sage'
        ]['column_mapping']
        
    def _init_modification_mapping(self):
        self.modification_mapping = {}

    def _load_file(self, filename):
        raise NotImplementedError

    def _transform_table(self, origin_df):
        self.psm_df['spec_idx'] = self.psm_df['scannr'].apply(
                sage_spec_idx_from_scannr
        )
        self.psm_df.drop(columns=['scannr'], inplace=True)

    def _translate_decoy(self, origin_df):
        if not self.keep_decoy:
            self._psm_df = self.psm_df[
                ~self.psm_df['decoy']
            ]

        self._psm_df = self.psm_df[self.psm_df['fdr'] <= self.keep_fdr]
        self._psm_df = self.psm_df[self.psm_df['peptide_fdr'] <= self.keep_fdr]
        self._psm_df = self.psm_df[self.psm_df['protein_fdr'] <= self.keep_fdr]

        # drop peptide_fdr, protein_fdr
        self._psm_df.drop(columns=['peptide_fdr', 'protein_fdr'], inplace=True)

    def _load_modifications(self, origin_df):
        pass

    def _translate_modifications(self):

        mod_annotated_df = get_annotated_mod_df()

        self._psm_df['mod_sites'], self._psm_df['mods'] = zip(*self.psm_df['modified_sequence'].apply(
            partial(
                capture_modifications, mod_annotated_df=mod_annotated_df, ppm_tolerance=10
                )
            ))
        # drop modified_sequence
        self._psm_df.drop(columns=['modified_sequence'], inplace=True)

class SageReaderTSV(SageReaderBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_file(self, filename):
        return pd.read_csv(filename, sep='\t')
    
class SageReaderParquet(SageReaderBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_file(self, filename):
        return pd.read_parquet(filename)


psm_reader_provider.register_reader('sage_tsv', SageReaderTSV)
psm_reader_provider.register_reader('sage_parquet', SageReaderParquet)