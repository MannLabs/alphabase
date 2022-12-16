import pandas as pd

from alphabase.spectral_library.base import (
    SpecLibBase
)
from alphabase.peptide.fragment import (
    flatten_fragments
)

from alphabase.io.hdf import HDF_File

import alphabase.peptide.precursor as precursor

class FlatSpecLib:
    """ 
    Flatten the spectral library (SpecLibBase) with `parse_base_library()`

    Parameters
    ----------
    min_fragment_intensity : float, optional
        minimal intensity to keep, by default 0.001

    keep_top_k_fragments : int, optional
        top k highest peaks to keep, by default 1000

    custom_fragment_df_columns : list, optional
        'mz' and 'intensity' columns are required. Others could be customized. 
        Defaults to ['type','number','position','charge','loss_type']
    """
    def __init__(self,
        min_fragment_intensity:float = 0.001,
        keep_top_k_fragments:int = 1000,
        custom_fragment_df_columns:list = [
            'type','number','position','charge','loss_type'
        ],
        **kwargs
    ):
        self.min_fragment_intensity = min_fragment_intensity
        self.keep_top_k_fragments = keep_top_k_fragments

        self.key_numeric_columns = [
            'ccs_pred', 'charge', 
            'decoy',
            'frag_end_idx', 'frag_start_idx',
            'isotope_m1_intensity', 'isotope_m1_mz',
            'isotope_apex_mz', 'isotope_apex_intensity',
            'isotope_apex_index',
            'isotope_right_most_mz', 'isotope_right_most_intensity',
            'isotope_right_most_index',
            'miss_cleavage', 'mobility_pred',
            'nAA', 
            'precursor_mz', 
            'rt_pred', 'rt_norm_pred'
        ]

        self.custom_fragment_df_columns = custom_fragment_df_columns

    @property
    def precursor_df(self)->pd.DataFrame:
        """: pd.DataFrame : precursor dataframe with columns
        'sequence', 'mods', 'mod_sites', 'charge', ...
        Identical to `self.peptide_df`.
        """
        return self._precursor_df

    @precursor_df.setter
    def precursor_df(self, df:pd.DataFrame):
        self._precursor_df = df
        precursor.refine_precursor_df(
            self._precursor_df,
            drop_frag_idx=False,
            ensure_data_validity=True,
        )

    @property
    def peptide_df(self)->pd.DataFrame:
        """Peptide dataframe with columns
        'sequence', 'mods', 'mod_sites', 'charge', ...
        Identical to `self.precursor_df`.
        """
        return self._precursor_df

    @peptide_df.setter
    def peptide_df(self, df:pd.DataFrame):
        self.precursor_df = df

    @property
    def fragment_df(self)->pd.DataFrame:
        """The fragment mz dataframe with 
        fragment types as columns (['b_z1', 'y_z2', ...])
        """
        return self._fragment_df

    def parse_base_library(self, library:SpecLibBase):
        """ Flatten a SpecLibBase object

        Parameters
        ----------
        library : SpecLibBase
            the library with fragment_mz_df and fragment_intensity_df
        """
        self._precursor_df, self._fragment_df = flatten_fragments(
            library.precursor_df, 
            library.fragment_mz_df, 
            library.fragment_intensity_df,
            min_fragment_intensity=self.min_fragment_intensity,
            keep_top_k_fragments=self.keep_top_k_fragments,
            custom_columns=self.custom_fragment_df_columns,
        )

    def save_hdf(self, hdf_file:str):
        """Save library dataframes into hdf_file.
        For `self.precursor_df`, this method will save it into two hdf groups:
        hdf_file: `flat_library/precursor_df` and `library/mod_seq_df`.

        `flat_library/precursor_df` contains all essential numberic columns those 
        can be loaded faster from hdf file into memory:
        `['precursor_mz', 'charge', 'mod_seq_hash', 'mod_seq_charge_hash',
        'frag_start_idx', 'frag_end_idx', 'decoy', 'rt_pred', 'ccs_pred',
        'mobility_pred', 'miss_cleave', 'nAA', 'isotope_mz_m1', 'isotope_intensity_m1', ...]`

        `flat_library/mod_seq_df` contains all string columns and the other 
        not essential columns:
        'sequence','mods','mod_sites', ['proteins', 'genes']...
        as well as 'mod_seq_hash', 'mod_seq_charge_hash' columns to map 
        back to `precursor_df`

        Parameters
        ----------
        hdf_file : str
            the hdf file path to save
            
        """
        _hdf = HDF_File(
            hdf_file, 
            read_only=False, 
            truncate=True,
            delete_existing=True
        )
        if 'mod_seq_charge_hash' not in self._precursor_df.columns:
            self.hash_precursor_df()

        key_columns = self.key_numeric_columns+[
            'mod_seq_hash', 'mod_seq_charge_hash'
        ]

        _hdf.flat_library = {
            'mod_seq_df': self._precursor_df[
                [
                    col for col in self._precursor_df.columns 
                    if col not in self.key_numeric_columns
                ]
            ],
            'precursor_df': self._precursor_df[
                [
                    col for col in self._precursor_df.columns 
                    if col in key_columns
                ]
            ],
            'fragment_df': self._fragment_df,
        }
        
    def load_hdf(self, hdf_file:str, load_mod_seq:bool=False):
        """Load the hdf library from hdf_file

        Parameters
        ----------
        hdf_file : str
            hdf library path to load

        load_mod_seq : bool, optional
            if also load mod_seq_df. 
            Defaults to False.
            
        """
        _hdf = HDF_File(
            hdf_file,
        )
        self._precursor_df:pd.DataFrame = _hdf.flat_library.precursor_df.values
        if load_mod_seq:
            key_columns = self.key_numeric_columns+[
                'mod_seq_hash', 'mod_seq_charge_hash'
            ]
            mod_seq_df = _hdf.flat_library.mod_seq_df.values
            cols = [
                col for col in mod_seq_df.columns 
                if col not in key_columns
            ]
            self._precursor_df[cols] = mod_seq_df[cols]
            
        self._fragment_df = _hdf.flat_library.fragment_df.values
        