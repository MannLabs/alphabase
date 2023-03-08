import pandas as pd

from alphabase.spectral_library.base import (
    SpecLibBase
)
from alphabase.peptide.fragment import (
    flatten_fragments
)

from alphabase.io.hdf import HDF_File

import alphabase.peptide.precursor as precursor

class SpecLibFlat(SpecLibBase):
    """ 
    Flatten the spectral library (SpecLibBase) by using :meth:`parse_base_library`.

    Attributes
    ----------
    custom_fragment_df_columns : list of str
        'mz' and 'intensity' columns are required in :attr:`fragment_df`, 
        others could be customized. 
        It can include ['type','number','position','charge','loss_type'].

    min_fragment_intensity : float
        minimal intensity to keep in :attr:`fragment_df`.

    keep_top_k_fragments : float
        top k highest peaks to keep in :attr:`fragment_df`.
    
    """

    key_numeric_columns = SpecLibBase.key_numeric_columns+[
        'flat_frag_start_idx','flat_frag_stop_idx'
    ]
    """ 
    :obj:`SpecLibBase.key_numeric_columns <alphabase.spectral_library.base.SpecLibBase.key_numeric_columns>` 
    + `['flat_frag_start_idx','flat_frag_stop_idx']`.
    """

    def __init__(self,
        charged_frag_types:list = ['b_z1','b_z2','y_z1','y_z2'],
        min_fragment_intensity:float = 0.001,
        keep_top_k_fragments:int = 1000,
        custom_fragment_df_columns:list = [
            'type','number','position','charge','loss_type'
        ],
        **kwargs
    ):
        """
        Parameters
        ----------
        min_fragment_intensity : float, optional
            minimal intensity to keep, by default 0.001

        keep_top_k_fragments : int, optional
            top k highest peaks to keep, by default 1000

        custom_fragment_df_columns : list, optional
            See :attr:`custom_fragment_df_columns`, 
            defaults to ['type','number','position','charge','loss_type']
        """
        super().__init__(charged_frag_types=charged_frag_types)
        self.min_fragment_intensity = min_fragment_intensity
        self.keep_top_k_fragments = keep_top_k_fragments

        self.custom_fragment_df_columns = custom_fragment_df_columns

    @property
    def fragment_df(self)->pd.DataFrame:
        """The flat fragment dataframe with columns
        (['mz', 'intensity'] + :attr:`custom_fragment_df_columns`.)
        """
        return self._fragment_df

    @property
    def protein_df(self)->pd.DataFrame:
        """ Protein dataframe """
        return self._protein_df

    def parse_base_library(self, 
        library:SpecLibBase,
        keep_original_frag_dfs:bool=True,
        copy_precursor_df:bool=False,
        **kwargs
    ):
        """ 
        Flatten an library object of SpecLibBase or its inherited class. 
        This method will generate :attr:`precursor_df` and :attr:`fragment_df`
        The fragments in fragment_df can be located by 
        `flat_frag_start_idx` and `flat_frag_stop_idx` in precursor_df. 

        Parameters
        ----------
        library : SpecLibBase
            A library object with attributes
            `precursor_df`, `fragment_mz_df` and `fragment_intensity_df`.
        
        keep_original_frag_dfs : bool, default True
            If `fragment_mz_df` and `fragment_intensity_df` are 
            kept in this library.

        copy_precursor_df : bool, default False
            If True, make a copy of `precursor_df` from `library`, 
            otherwise `flat_frag_start_idx` and `flat_frag_stop_idx` 
            columns will also append to the `library`.
        """
        self._precursor_df, self._fragment_df = flatten_fragments(
            library.precursor_df.copy() if copy_precursor_df else library.precursor_df, 
            library.fragment_mz_df, 
            library.fragment_intensity_df,
            min_fragment_intensity=self.min_fragment_intensity,
            keep_top_k_fragments=self.keep_top_k_fragments,
            custom_columns=self.custom_fragment_df_columns,
            **kwargs
        )
        
        if hasattr(library, 'protein_df'):
            self._protein_df = library.protein_df
        else:
            self._protein_df = pd.DataFrame()

        if keep_original_frag_dfs:
            self.charged_frag_types = library.fragment_mz_df.columns.values
            self._fragment_mz_df = library.fragment_mz_df
            self._fragment_intensity_df = library.fragment_intensity_df
        else:
            self._fragment_mz_df = pd.DataFrame()
            self._fragment_intensity_df = pd.DataFrame()

    def save_hdf(self, hdf_file:str):
        """Save library dataframes into hdf_file.
        For `self.precursor_df`, this method will save it into two hdf groups:
        hdf_file: `library/precursor_df` and `library/mod_seq_df`.

        `library/precursor_df` contains all essential numberic columns those 
        can be loaded faster from hdf file into memory:
        `['precursor_mz', 'charge', 'mod_seq_hash', 'mod_seq_charge_hash',
        'frag_start_idx', 'frag_stop_idx', 'flat_frag_start_idx', 'flat_frag_stop_idx', 
        'decoy', 'rt_pred', 'ccs_pred', 'mobility_pred', 'miss_cleave', 'nAA', 
        'isotope_mz_m1', 'isotope_intensity_m1', ...]`

        `library/mod_seq_df` contains all string columns and the other 
        not essential columns:
        'sequence','mods','mod_sites', ['proteins', 'genes']...
        as well as 'mod_seq_hash', 'mod_seq_charge_hash' columns to map 
        back to `precursor_df`

        Parameters
        ----------
        hdf_file : str
            the hdf file path to save
            
        """
        super().save_hdf(hdf_file)
        _hdf = HDF_File(
            hdf_file,
            read_only=False,
            truncate=True,
            delete_existing=False
        )
        _hdf.library.fragment_df = self.fragment_df
        _hdf.library.protein_df = self.protein_df
        _hdf.library.fragment_mz_df = self.fragment_mz_df
        _hdf.library.fragment_intensity_df = self.fragment_intensity_df
        
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
        super().load_hdf(hdf_file, load_mod_seq=load_mod_seq)
        _hdf = HDF_File(
            hdf_file,
        )
        self._fragment_df = _hdf.library.fragment_df.values
        self._protein_df = _hdf.library.protein_df.values
        self._fragment_mz_df = _hdf.library.fragment_mz_df.values
        self._fragment_intensity_df = _hdf.library.fragment_intensity_df.values
        
