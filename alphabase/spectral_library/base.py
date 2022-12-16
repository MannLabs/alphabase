import pandas as pd
import numpy as np
import typing

import alphabase.peptide.fragment as fragment
import alphabase.peptide.precursor as precursor
from alphabase.io.hdf import HDF_File

class SpecLibBase(object):
    def __init__(self,
        # ['b_z1','b_z2','y_z1','y_modloss_z1', ...]; 
        # 'b_z1': 'b' is the fragment type and 
        # 'z1' is the charge state z=1.
        charged_frag_types:typing.List[str] = [
            'b_z1','b_z2','y_z1', 'y_z2'
        ], 
        precursor_mz_min = 400, precursor_mz_max = 6000,
        decoy:str = None,
    ):
        """Base spectral library in alphabase and alphapeptdeep.

        Parameters
        ----------
        charged_frag_types : typing.List[str], optional
            fragment types with charge. 
            Defaults to [ 'b_z1','b_z2','y_z1', 'y_z2' ].

        precursor_mz_min : int, optional
            Use this to clip precursor df. 
            Defaults to 400.

        precursor_mz_max : int, optional
            Use this to clip precursor df. 
            Defaults to 6000.

        decoy : str, optional
            Decoy methods, could be "pseudo_reverse" or "diann".
            Defaults to None.

        Attributes
        ----------
        precursor_df : pd.DataFrame
            precursor dataframe.

        fragment_mz_df : pd.DataFrame
            fragment m/z dataframe.

        fragment_intensity_df : pd.DataFrame
            fragment intensity dataframe.

        charged_frag_types : list
            same as `charged_frag_types` in Args.

        min_precursor_mz : float
            same as `precursor_mz_min` in Args.

        max_precursor_mz : float
            same as `precursor_mz_max` in Args.

        decoy : str
            same as `decoy` in Args.

        key_numeric_columns : list of str
            key numeric columns to be saved 
            into library/precursor_df in the hdf file. Others will be saved into
            library/mod_seq_df instead.
            
        """
        self.charged_frag_types = charged_frag_types
        self._precursor_df = pd.DataFrame()
        self._fragment_intensity_df = pd.DataFrame()
        self._fragment_mz_df = pd.DataFrame()
        self.min_precursor_mz = precursor_mz_min
        self.max_precursor_mz = precursor_mz_max

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
        self.decoy = decoy
    
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
        """: pd.DataFrame : peptide dataframe with columns
        'sequence', 'mods', 'mod_sites', 'charge', ...
        Identical to `self.precursor_df`.
        """
        return self._precursor_df

    @peptide_df.setter
    def peptide_df(self, df:pd.DataFrame):
        self.precursor_df = df

    @property
    def fragment_mz_df(self)->pd.DataFrame:
        """: pd.DataFrame : The fragment mz dataframe with 
        fragment types as columns (['b_z1', 'y_z2', ...])
        """
        return self._fragment_mz_df

    @property
    def fragment_intensity_df(self)->pd.DataFrame:
        """: pd.DataFrame : The fragment intensity dataframe with 
        fragment types as columns (['b_z1', 'y_z2', ...])
        """
        return self._fragment_intensity_df

    def refine_df(self):
        """Sort nAA and reset_index for faster calculation (or prediction)
        """
        precursor.refine_precursor_df(
            self._precursor_df
        )

    def append_decoy_sequence(self):
        """Append decoy sequence into precursor_df.
        Decoy method is based on self.decoy(str).
        ```
        decoy_lib = (
            decoy_lib_provider.get_decoy_lib(
                self.decoy, self
            )
        )
        decoy_lib.decoy_sequence()
        ...
        ```
        """
        from alphabase.spectral_library.decoy import (
            decoy_lib_provider
        )
        decoy_lib = (
            decoy_lib_provider.get_decoy_lib(
                self.decoy, self
            )
        )
        if decoy_lib is None: return None
        decoy_lib.decoy_sequence()
        self._precursor_df['decoy'] = 0
        decoy_lib._precursor_df['decoy'] = 1
        self._precursor_df = pd.concat((
            self._precursor_df,
            decoy_lib._precursor_df
        ))
        self.refine_df()

    def clip_by_precursor_mz_(self):
        ''' 
        Clip self._precursor_df inplace by self.min_precursor_mz and self.max_precursor_mz
        '''
        self._precursor_df.drop(
            self._precursor_df.loc[
                (self._precursor_df['precursor_mz']<self.min_precursor_mz)|
                (self._precursor_df['precursor_mz']>self.max_precursor_mz)
            ].index, inplace=True
        )
        self._precursor_df.reset_index(drop=True, inplace=True)


    def flatten_fragment_data(
        self
    )->typing.Tuple[np.ndarray, np.ndarray]:
        '''
        Create flattened (1-D) np.ndarray for fragment mz and intensity 
        dataframes, respectively. The arrays are references to 
        original data, that means: 
          1. This method is fast; 
          2. Changing the array values will change the df values. 
        They can be unraveled back using:
          `array.reshape(len(self._fragment_mz_df.columns), -1)`

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            np.ndarray. 1-D flattened mz array (a reference to 
            original fragment mz df data)
            np.ndarray. 1-D flattened intensity array (a reference to 
            original fragment intensity df data)
        '''
        return (
            self._fragment_mz_df.values.reshape(-1),
            self._fragment_intensity_df.values.reshape(-1)
        )

    def calc_precursor_mz(self):
        """
        Calculate precursor mz for self._precursor_df,
        and clip the self._precursor_df using `self.clip_by_precursor_mz_`
        """
        fragment.update_precursor_mz(self._precursor_df)
        self.clip_by_precursor_mz_()

    def update_precursor_mz(self):
        """
        Calculate precursor mz for self._precursor_df,
        and clip the self._precursor_df using `self.clip_by_precursor_mz_`
        """
        self.calc_precursor_mz()
    
    def calc_precursor_isotope(self, 
        multiprocessing:bool=True,
        mp_process_num:int=8,
        mp_process_bar=None,
        min_precursor_num_to_run_mp:int=1000,
    ):
        """
        Append isotope columns into self.precursor_df.
        See `alphabase.peptide.precursor.calc_precursor_isotope` for details.
        """
        if 'precursor_mz' not in self._precursor_df.columns:
            self.calc_precursor_mz()
            self.clip_by_precursor_mz_()
        if multiprocessing and len(self.precursor_df)>min_precursor_num_to_run_mp:
            (
                self._precursor_df
            ) = precursor.calc_precursor_isotope_mp(
                self.precursor_df, 
                processes=mp_process_num,
                process_bar=mp_process_bar,
            )
        else:
            (
                self._precursor_df
            ) = precursor.calc_precursor_isotope(
                self.precursor_df
            )

    def calc_fragment_mz_df(self):
        """
        TODO: use multiprocessing here or in the
        `create_fragment_mz_dataframe` function.
        """
        if (
            self.charged_frag_types is not None 
            or len(self.charged_frag_types)
        ):
            (
                self._fragment_mz_df
            ) = fragment.create_fragment_mz_dataframe(
                self.precursor_df, self.charged_frag_types,
            )
        else:
            print('Skip fragment calculation as self.charged_frag_types is None or empty')

    def hash_precursor_df(self):
        """Insert hash codes for peptides and precursors"""
        precursor.hash_precursor_df(
            self._precursor_df
        )

    def _get_hdf_to_save(self, 
        hdf_file, 
        delete_existing=False
    ):
        """Internal function to get a HDF group to write"""
        _hdf = HDF_File(
            hdf_file, 
            read_only=False, 
            truncate=True,
            delete_existing=delete_existing
        )
        return _hdf.library

    def _get_hdf_to_load(self,
        hdf_file, 
    ):
        """Internal function to get a HDF group to read"""
        _hdf = HDF_File(
            hdf_file,
        )
        return _hdf.library

    def save_df_to_hdf(self, 
        hdf_file:str, 
        df_key: str,
        df: pd.DataFrame,
        delete_existing=False
    ):
        """Save a new HDF group or dataset into existing HDF file"""
        self._get_hdf_to_save(
            hdf_file, 
            delete_existing=delete_existing
        ).add_group(df_key, df)

    def load_df_from_hdf(self, 
        hdf_file:str, 
        df_name: str
    )->pd.DataFrame:
        """Load specific dataset (dataframe) from hdf_file.

        Parameters
        ----------
        hdf_file : str
            The hdf file name

        df_name : str
            The dataset/dataframe name in the hdf file

        Returns
        -------
        pd.DataFrame
            Loaded dataframe
        """
        return self._get_hdf_to_load(
            hdf_file
        ).__getattribute__(df_name).values

    def save_hdf(self, hdf_file:str):
        """Save library dataframes into hdf_file.
        For `self.precursor_df`, this method will save it into two hdf groups:
            hdf_file: `library/precursor_df` and `library/mod_seq_df`.

        `library/precursor_df` contains all essential numberic columns those 
        can be loaded faster from hdf file into memory:
            'precursor_mz', 'charge', 'mod_seq_hash', 'mod_seq_charge_hash',
            'frag_start_idx', 'frag_end_idx', 'decoy', 'rt_pred', 'ccs_pred',
            'mobility_pred', 'miss_cleave', 'nAA', 
            ['isotope_mz_m1', 'isotope_intensity_m1'], ...

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

        _hdf.library = {
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
            'fragment_mz_df': self._fragment_mz_df,
            'fragment_intensity_df': self._fragment_intensity_df,
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
        self._precursor_df:pd.DataFrame = _hdf.library.precursor_df.values
        if load_mod_seq:
            key_columns = self.key_numeric_columns+[
                'mod_seq_hash', 'mod_seq_charge_hash'
            ]
            mod_seq_df = _hdf.library.mod_seq_df.values
            cols = [
                col for col in mod_seq_df.columns 
                if col not in key_columns
            ]
            self._precursor_df[cols] = mod_seq_df[cols]
            
        self._fragment_mz_df = _hdf.library.fragment_mz_df.values
        self._fragment_intensity_df = _hdf.library.fragment_intensity_df.values

        self._fragment_mz_df = self._fragment_mz_df[
            [
                frag for frag in self.charged_frag_types 
                if frag in self._fragment_mz_df.columns
            ]
        ]
        self._fragment_intensity_df = self._fragment_intensity_df[
            [
                frag for frag in self.charged_frag_types 
                if frag in self._fragment_intensity_df.columns
            ]
        ]
        