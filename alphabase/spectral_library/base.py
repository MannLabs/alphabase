import pandas as pd
import numpy as np
import typing
import logging

import alphabase.peptide.fragment as fragment
import alphabase.peptide.precursor as precursor
from alphabase.io.hdf import HDF_File

class SpecLibBase(object):
    """
    Base spectral library in alphabase and alphapeptdeep.

    Attributes
    ----------
    charged_frag_types : list
        same as `charged_frag_types` in Parameters in :meth:`__init__`.

    min_precursor_mz : float
        same as `precursor_mz_min` in Parameters in :meth:`__init__`.

    max_precursor_mz : float
        same as `precursor_mz_max` in Parameters in :meth:`__init__`.

    decoy : str
        same as `decoy` in Parameters in :meth:`__init__`.
    """

    key_numeric_columns:list = [
        'ccs_pred', 'charge', 
        'decoy',
        'frag_stop_idx', 'frag_start_idx',
        'isotope_m1_intensity', 'isotope_m1_mz',
        'isotope_apex_mz', 'isotope_apex_intensity',
        'isotope_apex_offset',
        'isotope_right_most_mz', 'isotope_right_most_intensity',
        'isotope_right_most_offset',
        'miss_cleavage', 'mobility_pred',
        'nAA', 
        'precursor_mz', 
        'rt_pred', 'rt_norm_pred',
        'labeling_channel',
    ]
    """
    list of str: Key numeric columns to be saved 
    into library/precursor_df in the hdf file for fast loading, 
    others will be saved into library/mod_seq_df instead.
    """

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
        """
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
        """
        self.charged_frag_types = charged_frag_types
        self._precursor_df = pd.DataFrame()
        self._fragment_intensity_df = pd.DataFrame()
        self._fragment_mz_df = pd.DataFrame()
        self.min_precursor_mz = precursor_mz_min
        self.max_precursor_mz = precursor_mz_max

        self.decoy = decoy
    
    @property
    def precursor_df(self)->pd.DataFrame:
        """
        Precursor dataframe with columns
        'sequence', 'mods', 'mod_sites', 'charge', etc,
        identical to :attr:`peptide_df`.
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
        """
        Peptide dataframe with columns
        'sequence', 'mods', 'mod_sites', 'charge', etc,
        identical to :attr:`precursor_df`. 
        """
        return self._precursor_df

    @peptide_df.setter
    def peptide_df(self, df:pd.DataFrame):
        self.precursor_df = df

    @property
    def fragment_mz_df(self)->pd.DataFrame:
        """
        The fragment mz dataframe with 
        fragment types as columns (['b_z1', 'y_z2', ...])
        """
        return self._fragment_mz_df

    @property
    def fragment_intensity_df(self)->pd.DataFrame:
        """
        The fragment intensity dataframe with 
        fragment types as columns (['b_z1', 'y_z2', ...])
        """
        return self._fragment_intensity_df

    def refine_df(self):
        """
        Sort nAA and reset_index for faster calculation (or prediction)
        """
        precursor.refine_precursor_df(
            self._precursor_df
        )

    def append_decoy_sequence(self):
        """
        Append decoy sequence into precursor_df.
        Decoy method is based on self.decoy(str).
        ```
        >>> decoy_lib = (decoy_lib_provider.get_decoy_lib( self.decoy, self))
        >>> decoy_lib.decoy_sequence()
        >>> decoy_lib.append_to_target_lib()
        ...
        ```
        """
        from alphabase.spectral_library.decoy import (
            decoy_lib_provider
        )
        # register 'protein_reverse' to the decoy_lib_provider
        import alphabase.protein.protein_level_decoy

        decoy_lib = (
            decoy_lib_provider.get_decoy_lib(
                self.decoy, self
            )
        )
        if decoy_lib is None: return None
        decoy_lib.decoy_sequence()
        decoy_lib.append_to_target_lib()

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
    
    def annotate_fragments_from_speclib(self, 
        donor_speclib, 
        verbose = True
    ):
        """
        Annotate self.precursor_df with fragments from donor_speclib.
        The donor_speclib must have a fragment_mz_df and can optionally have a fragment_intensity_df.
        Fragment dataframes are updated inplace and overwritten.

        Parameters
        ----------
        donor_speclib : SpecLibBase
            The donor library to annotate fragments from.

        verbose : bool, optional
            Print progress, by default True, for example::
            
                2022-12-16 00:52:08> Speclib with 4 precursors will be reannotated with speclib with 12 precursors and 504 fragments
                2022-12-16 00:52:08> A total of 4 precursors were succesfully annotated, 0 precursors were not matched
            
            
        """
        self = annotate_fragments_from_speclib(
            self, donor_speclib, verbose = verbose
        )
    
    def remove_unused_fragments(self):
        """
        Remove unused fragments from self._fragment_mz_df and self._fragment_intensity_df.
        Fragment dataframes are updated inplace and overwritten.
        """


        if len(self._fragment_mz_df) > 0:
            
            # update both fragment mz and intensity df
            if len(self.fragment_intensity_df > 0):
                self._precursor_df,(self._fragment_mz_df, self._fragment_intensity_df) = fragment.remove_unused_fragments(
                    self._precursor_df,(self._fragment_mz_df, self._fragment_intensity_df)
                )
            # only update fragment mz df
            else:
                (self._precursor_df, (self._fragment_mz_df,)) = fragment.remove_unused_fragments(
                    self._precursor_df, (self._fragment_mz_df,)
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
        For `self.precursor_df`, this method will save it into two hdf groups in hdf_file:
        `library/precursor_df` and `library/mod_seq_df`.

        `library/precursor_df` contains all essential numberic columns those 
        can be loaded faster from hdf file into memory:

        'precursor_mz', 'charge', 'mod_seq_hash', 'mod_seq_charge_hash',
        'frag_start_idx', 'frag_stop_idx', 'decoy', 'rt_pred', 'ccs_pred',
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
        
def annotate_fragments_from_speclib(
    speclib: SpecLibBase, 
    fragment_speclib: SpecLibBase,
    verbose = True,
)->SpecLibBase:
    """Reannotate an SpecLibBase library with fragments from a different SpecLibBase.
    
    Parameters
    ----------
    speclib: alphabase.spectral_library.library_base.SpecLibBase
        Spectral library which contains the precursors to be annotated. All fragments mz and fragment intensities will be removed.

    fragment_speclib: alphabase.spectral_library.library_base.SpecLibBase
        Spectral library which contains the donor precursors whose fragments should be used.

    Returns
    -------

    alphabase.spectral_library.library_base.SpecLibBase
        newly annotated spectral library
 
    """
    if verbose:
        num_precursor_left = len(speclib.precursor_df)
        num_precursor_right = len(fragment_speclib.precursor_df)
        num_fragments_right = len(fragment_speclib.fragment_mz_df) * len(fragment_speclib.fragment_mz_df.columns)
        logging.info(f'Speclib with {num_precursor_left:,} precursors will be reannotated with speclib with {num_precursor_right:,} precursors and {num_fragments_right:,} fragments')

    # reannotation is based on mod_seq_hash column
    hash_column_name = 'mod_seq_hash'

    # create hash columns if missing
    if hash_column_name not in speclib.precursor_df.columns:
        speclib.hash_precursor_df()

    if fragment_speclib not in fragment_speclib.precursor_df.columns:
        fragment_speclib.hash_precursor_df()

    speclib_hash = speclib.precursor_df[hash_column_name].values
    fragment_speclib_hash = fragment_speclib.precursor_df[hash_column_name].values

    speclib_indices = fragment.join_left(speclib_hash, fragment_speclib_hash)

    matched_mask = (speclib_indices >= 0)

    if verbose:
        matched_count = np.sum(matched_mask)
        not_matched_count = np.sum(~matched_mask)
    
        logging.info(f'A total of {matched_count:,} precursors were succesfully annotated, {not_matched_count:,} precursors were not matched')


    frag_start_idx = fragment_speclib.precursor_df['frag_start_idx'].values[speclib_indices]
    frag_stop_idx = fragment_speclib.precursor_df['frag_stop_idx'].values[speclib_indices]
    
    speclib._precursor_df = speclib._precursor_df[matched_mask].copy()
    speclib._precursor_df['frag_start_idx'] = frag_start_idx[matched_mask]
    speclib._precursor_df['frag_stop_idx'] = frag_stop_idx[matched_mask]

    speclib._fragment_mz_df = fragment_speclib._fragment_mz_df.copy()
    speclib._fragment_intensity_df = fragment_speclib._fragment_intensity_df.copy()

    return speclib