import pandas as pd
import numpy as np
import warnings

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
    
    def available_dense_fragment_dfs(self):
        """Return the available dense fragment dataframes.
        This method is inherited from :class:`SpecLibBase` and will return an empty list for a flat library.
        """
        return []
    
    def remove_unused_fragments(self):
        """Remove unused fragments from fragment_df.
        This method is inherited from :class:`SpecLibBase` and has not been implemented for a flat library.
        """
        raise NotImplementedError("remove_unused_fragments is not implemented for a flat library")
    
    def parse_base_library(self, 
        library:SpecLibBase,
        keep_original_frag_dfs:bool=False,
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
            for dense_frag_df in library.available_dense_fragment_dfs():
                setattr(self, dense_frag_df, getattr(library, dense_frag_df))

            warnings.warn(
                "The SpecLibFlat object will have a strictly flat representation in the future. keep_original_frag_dfs=True will be deprecated.",
                DeprecationWarning
            )


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
        
    def get_full_charged_types(self,frag_df:pd.DataFrame) -> list:
        """
        Infer the full set of charged fragment types from the fragment dataframe
        by full we mean a complete set of fragment types for each charge
        so if we have a fragment b_z1 we should also have a fragment y_z1 and vice versa

        Parameters
        ----------
        frag_df : pd.DataFrame
            The fragment dataframe

        Returns
        -------
        charged_frag_types : list
            The full set of charged fragment types in the form of a list of strings such as ['a_z1','b_z1','c_z1','x_z1','y_z1','z_z1']

        """
        unique_charge_type_pairs = frag_df[['type','loss_type','charge']].drop_duplicates()
        #Fragtypes from ascii to char
        self.frag_types_as_char = {i: chr(i) for i in unique_charge_type_pairs['type'].unique()}

        charged_frag_types = set()
        # Now if we have a fragment type that is a,b,c we should have the corresponding x,y,z
        
        corresponding = {
            'a':'x',
            'b':'y',
            'c':'z',
            'x':'a',
            'y':'b',
            'z':'c'
        }
        loss_number_to_type = {0: '', 18: '_H2O', 17: '_NH3',98: '_modloss'}
        for type,loss,max_charge in unique_charge_type_pairs.values:
            for possible_charge  in range(1,max_charge+1):
                # Add the string for this pair
                charged_frag_types.add(f"{self.frag_types_as_char[type]}{loss_number_to_type[loss]}_z{possible_charge}")
                # Add the string for the corresponding pair
                charged_frag_types.add(f"{corresponding[self.frag_types_as_char[type]]}{loss_number_to_type[loss]}_z{possible_charge}")
        return list(charged_frag_types)
    def to_SpecLibBase(self) -> SpecLibBase:
        """
        Convert the flat library to SpecLibBase object.

        Returns:
        --------
        SpecLibBase
            A SpecLibBase object with `precursor_df`, `fragment_mz_df` and `fragment_intensity_df`, and 
            '_additional_fragment_columns_df' if there was more than mz and intensity in the original fragment_df.
        """
        # Check that fragment_df has the following columns ['mz', 'intensity', 'type', 'charge', 'position', 'loss_type']
        assert  set(['mz', 'intensity', 'type', 'charge', 'position', 'loss_type']).issubset(self._fragment_df.columns), f'fragment_df does not have the following columns: {set(["mz", "intensity", "type", "charge", "position", "loss_type"]) - set(self._fragment_df.columns)}'
        self.charged_frag_types = self.get_full_charged_types(self._fragment_df) # Infer the full set of charged fragment types from data
        # charged_frag_types = self.charged_frag_types #Use pre-defined charged_frag_types
        frag_type_to_col_dict = dict(zip(
            self.charged_frag_types,  
            range(len(self.charged_frag_types))
        ))
        loss_number_to_type = {0: '', 18: '_H2O', 17: '_NH3',98: '_modloss'}
    
        available_frag_types = self._fragment_df['type'].unique()
        self.frag_types_as_char = {i: chr(i) for i in available_frag_types}

        frag_types_z_charge = (
            self._fragment_df['type'].map(self.frag_types_as_char) + 
            self._fragment_df['loss_type'].map(loss_number_to_type) + 
            '_z' + 
            self._fragment_df['charge'].astype(str)
        )

        #Print number of nAA less 1
        accumlated_nAA = (self._precursor_df['nAA']-1).cumsum()
        # Define intensity and mz as a matrix of shape (accumlated_nAA[-1], len(self.charged_frag_types), 2) - 2 for mz and intensity
        intensity_and_mz = np.zeros((accumlated_nAA.iloc[-1], len(self.charged_frag_types), 2))

        # Start indices for each precursor is the accumlated nAA of the previous precursor and for the first precursor is 0
        start_indexes = accumlated_nAA.shift(1).fillna(0).astype(int)
        
        
        column_indices = frag_types_z_charge.map(frag_type_to_col_dict)

        # We need to calculate for each fragment the precursor_idx that maps a fragment to a precursor
        drop_precursor_idx = False
        if 'precursor_idx' not in self._fragment_df.columns:        
            drop_precursor_idx = True    
            # Sort precursor_df by 'flat_frag_start_idx'
            self._precursor_df = self._precursor_df.sort_values('flat_frag_start_idx')
            # Add precursor_idx to precursor_df as 0,1,2,3...
            self._precursor_df['precursor_idx'] = range(self._precursor_df.shape[0])

            # Add precursor_idx to fragment_df
            frag_precursor_idx = np.repeat(self._precursor_df['precursor_idx'], self._precursor_df['flat_frag_stop_idx'] - self._precursor_df['flat_frag_start_idx'])

            assert len(frag_precursor_idx) == self._fragment_df.shape[0], f'Number of fragments {len(frag_precursor_idx)} is not equal to the number of rows in fragment_df {self._fragment_df.shape[0]}'
        
            self._fragment_df['precursor_idx'] = frag_precursor_idx.values

        #Row indices of a fragment being the accumlated nAA of the precursor + fragment position -1
        precursor_idx_to_accumlated_nAA = dict(zip(self._precursor_df['precursor_idx'], start_indexes))
        row_indices = self._fragment_df['precursor_idx'].map(precursor_idx_to_accumlated_nAA,na_action = 'ignore') + self._fragment_df['position'] 
        
        # Drop elements were the column_indices is nan and drop them from both row_indices and column_indices
        nan_indices = column_indices.index[column_indices.isna()]
        row_indices = row_indices.drop(nan_indices)
        column_indices = column_indices.drop(nan_indices)

        assert row_indices.shape[0] == column_indices.shape[0], f'row_indices {row_indices.shape[0]} is not equal to column_indices {column_indices.shape[0]}'


        assert np.max(row_indices) <= intensity_and_mz.shape[0], f'row_indices {np.max(row_indices)} is greater than the number of fragments {intensity_and_mz.shape[0]}'
        
        # Assign the intensity and mz to the correct position in the matrix
        intensity_indices = np.array((row_indices, column_indices,np.zeros_like(row_indices)), dtype=int).tolist()
        mz_indices = np.array((row_indices, column_indices,np.ones_like(row_indices)), dtype=int).tolist()
        
        intensity_and_mz[tuple(intensity_indices)] = self._fragment_df['intensity']
        intensity_and_mz[tuple(mz_indices)] = self._fragment_df['mz']

        #Create fragment_mz_df and fragment_intensity_df
        fragment_mz_df = pd.DataFrame(intensity_and_mz[:,:,1], columns = self.charged_frag_types)
        fragment_intensity_df = pd.DataFrame(intensity_and_mz[:,:,0], columns = self.charged_frag_types)

        #Add columns frag_start_idx and frag_stop_idx to the precursor_df
        self._precursor_df['frag_start_idx'] = start_indexes
        self._precursor_df['frag_stop_idx'] = accumlated_nAA

        #Drop precursor Idx from both fragment_df and precursor_df
        if drop_precursor_idx:
            self._fragment_df = self._fragment_df.drop(columns = ['precursor_idx'])
            self._precursor_df = self._precursor_df.drop(columns = ['precursor_idx'])


        #Drop flat indices from precursor_df
        self._precursor_df = self._precursor_df.drop(columns = ['flat_frag_start_idx','flat_frag_stop_idx'])
        
         

        #Create SpecLibBase object
        spec_lib_base = SpecLibBase()
        spec_lib_base._precursor_df = self._precursor_df
        spec_lib_base._fragment_mz_df = fragment_mz_df
        spec_lib_base._fragment_intensity_df = fragment_intensity_df
        spec_lib_base.charged_frag_types = self.charged_frag_types
        
        #  Add additional columns from frag_df that were not mz and intensity
        additional_columns = set(self._fragment_df.columns) - set(['mz','intensity','type','charge','position','loss_type'])
        for col in additional_columns:
            additional_matrix = np.zeros((accumlated_nAA.iloc[-1], len(self.charged_frag_types)))
            data_indices = np.array((row_indices, column_indices), dtype=int).tolist()
            additional_matrix[tuple(data_indices)] = self._fragment_df[col]
            additional_df = pd.DataFrame(additional_matrix, columns = self.charged_frag_types)
            setattr(spec_lib_base,f'_fragment_{col}_df',additional_df)

        
        return spec_lib_base