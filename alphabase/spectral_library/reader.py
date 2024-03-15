import typing
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from alphabase.peptide.mobility import mobility_to_ccs_for_df
from alphabase.io.psm_reader.dia_search_reader import SpectronautReader
from alphabase.io.psm_reader.maxquant_reader import MaxQuantReader
from alphabase.spectral_library.base import SpecLibBase
from alphabase.psm_reader.psm_reader import psm_reader_yaml
from alphabase.psm_reader import psm_reader_provider

from alphabase.constants._const import CONST_FILE_FOLDER, PEAK_INTENSITY_DTYPE
from alphabase.yaml_utils import load_yaml

class LibraryReaderBase(MaxQuantReader, SpecLibBase):
    def __init__(self,
        charged_frag_types:typing.List[str] = [
            'b_z1','b_z2','y_z1', 'y_z2', 
            'b_modloss_z1','b_modloss_z2',
            'y_modloss_z1', 'y_modloss_z2'
        ],
        column_mapping:dict = None,
        modification_mapping:dict = None,
        fdr = 0.01,
        fixed_C57 = False,
        mod_seq_columns=psm_reader_yaml[
            'library_reader_base'
        ]['mod_seq_columns'],
        rt_unit='irt',
        precursor_mz_min:float = 400,
        precursor_mz_max:float = 2000,
        decoy:str = None,
        **kwargs
    ):
        """

        Base class for reading spectral libraries from long format csv files.

        Parameters
        ----------

        charged_frag_types: list of str
            List of fragment types to be used in the spectral library.
            The default is ['b_z1','b_z2','y_z1', 'y_z2', 'b_modloss_z1','b_modloss_z2','y_modloss_z1', 'y_modloss_z2']

        column_mapping: dict
            Dictionary mapping the column names in the csv file to the column names in the spectral library.
            The default is None, which uses the `library_reader_base` column mapping in `psm_reader.yaml` 

        modification_mapping: dict
            Dictionary mapping the modification names in the csv file to the modification names in the spectral library.

        fdr: float
            False discovery rate threshold for filtering the spectral library.
            default is 0.01

        fixed_C57: bool
    
        mod_seq_columns: list of str
            List of column names in the csv file containing the modified sequence.
            By default the mapping is taken from `psm_reader.yaml`

        rt_unit: str
            Unit of the retention time column in the csv file.
            The default is 'irt'

        precursor_mz_min: float
            Minimum precursor m/z value for filtering the spectral library.

        precursor_mz_max: float
            Maximum precursor m/z value for filtering the spectral library.

        decoy: str
            Decoy type for the spectral library.
            Can be either `pseudo_reverse` or `diann`

        """
        SpecLibBase.__init__(self,
            charged_frag_types = charged_frag_types,
            precursor_mz_min=precursor_mz_min,
            precursor_mz_max=precursor_mz_max,
            decoy=decoy
        )

        MaxQuantReader.__init__(self, 
            column_mapping = column_mapping,
            modification_mapping = modification_mapping,
            fdr = fdr,
            keep_decoy = False,
            fixed_C57 = fixed_C57,
            mod_seq_columns=mod_seq_columns,
            rt_unit=rt_unit,
        )

    def _init_column_mapping(self):
        """
        Initialize the column mapping from the `psm_reader.yaml` file.
        """
        self.column_mapping = psm_reader_yaml[
            'library_reader_base'
        ]['column_mapping']

    def _find_key_columns(self, lib_df:pd.DataFrame):

        """
        Find and create the key columns for the spectral library.

        Parameters
        ----------

        lib_df: pd.DataFrame
            Dataframe containing the spectral library.
        
        """
        if 'fragment_loss_type' not in lib_df.columns:
            lib_df['fragment_loss_type'] = ''

        lib_df.fillna({'fragment_loss_type':''}, inplace=True)
        lib_df.replace(
            {'fragment_loss_type':'noloss'},
            {'fragment_loss_type':''},inplace=True
        )

        if 'mods' not in lib_df.columns:
            lib_df['mods'] = ''

        if 'mod_sites' not in lib_df.columns:
            lib_df['mod_sites'] = ''

    def _get_fragment_intensity(self, lib_df:pd.DataFrame):
        """

        Create the self._fragment_intensity dataframe from a given spectral library.
        In the process, the input dataframe is converted from long format to a precursor dataframe and returned.

        Parameters
        ----------
        lib_df: pd.DataFrame
            Dataframe containing the spectral library.

        Returns
        -------
        precursor_df: pd.DataFrame
            Dataframe containing the fragment intensity.
        
        """
        frag_col_dict = dict(zip(
            self.charged_frag_types, 
            range(len(self.charged_frag_types))
        ))

        self._find_key_columns(lib_df)

        # drop all columns which are all NaN as they prohibit grouping
        lib_df = lib_df.dropna(axis=1, how='all')

        precursor_df_list = []

        frag_intens_list = []
        nAA_list = []

        fragment_columns = [
            'fragment_mz','fragment_type','fragment_charge','fragment_series','fragment_loss_type','fragment_intensity'
        ]

        # by default, all non-fragment columns are used to group the library
        non_fragment_columns = list(set(lib_df.columns) - set(fragment_columns))

        
        for keys, df_group in tqdm(lib_df.groupby(
            non_fragment_columns
        )):
            precursor_columns = dict(zip(non_fragment_columns, keys))

            nAA = len(precursor_columns['sequence'])

            intens = np.zeros(
                (nAA-1, len(self.charged_frag_types)),
                dtype=PEAK_INTENSITY_DTYPE,
            )
            for frag_type, frag_num, loss_type, frag_charge, inten in df_group[
                [
                    'fragment_type','fragment_series','fragment_loss_type',
                    'fragment_charge','fragment_intensity'
                ]
            ].values:
                if frag_type in 'abc':
                    frag_num -= 1
                elif frag_type in 'xyz':
                    frag_num = nAA-frag_num-1
                else:
                    continue
                
                if loss_type == '':
                    frag_type = f'{frag_type}_z{frag_charge}'
                elif loss_type == 'H3PO4':
                    frag_type = f'{frag_type}_modloss_z{frag_charge}'
                elif loss_type == 'H2O':
                    frag_type = f'{frag_type}_H2O_z{frag_charge}'
                elif loss_type == 'NH3':
                    frag_type = f'{frag_type}_NH3_z{frag_charge}'
                elif loss_type == 'unknown': # DiaNN+fragger
                    frag_type = f'{frag_type}_z{frag_charge}'
                else:
                    continue
                
                if frag_type not in frag_col_dict:
                    continue
                frag_col_idx = frag_col_dict[frag_type]
                intens[frag_num, frag_col_idx] = inten
            max_inten = np.max(intens)
            if max_inten <= 0: continue
            intens /= max_inten

            precursor_df_list.append(precursor_columns) 
            frag_intens_list.append(intens)
            nAA_list.append(nAA)

        df = pd.DataFrame(precursor_df_list)


        self._fragment_intensity_df = pd.DataFrame(
            np.concatenate(frag_intens_list),
            columns = self.charged_frag_types
        )

        indices = np.zeros(len(nAA_list)+1, dtype=np.int64)
        indices[1:] = np.array(nAA_list)-1
        indices = np.cumsum(indices)

        df['frag_start_idx'] = indices[:-1]
        df['frag_stop_idx'] = indices[1:]

        return df

    def _load_file(
            self, 
            filename:str
        ):
        """
        Load the spectral library from a csv file.
        Reimplementation of `PSMReaderBase._translate_columns`.
        """

        csv_sep = self._get_table_delimiter(filename)

        df = pd.read_csv(filename, sep=csv_sep,keep_default_na=False)
        self._find_mod_seq_column(df)
        
        return df

        
    def _post_process(
        self, 
        lib_df:pd.DataFrame,
    ):  
        """
        Process the spectral library and create the `fragment_intensity`, `fragment_mz`dataframe.
        Reimplementation of `PSMReaderBase._post_process`.
        """

        # identify unknown modifications
        len_before = len(self._psm_df)
        self._psm_df = self._psm_df[
            ~self._psm_df['mods'].isna()
        ]
        len_after = len(self._psm_df)

        if len_before != len_after:
            print(
                f'{len_before-len_after} Entries with unknown modifications are removed'
            ) 
        
        if 'nAA' not in self._psm_df.columns:
            self._psm_df['nAA'] = self._psm_df.sequence.str.len()

        self._psm_df = self._get_fragment_intensity(self._psm_df)
        
        self.normalize_rt_by_raw_name()
        
        if 'mobility' in self._psm_df.columns:
            self._psm_df['ccs'] = (
                mobility_to_ccs_for_df(
                    self._psm_df,
                    'mobility'
                )
            )

        self._psm_df.drop('modified_sequence', axis=1, inplace=True)
        self._precursor_df = self._psm_df

        self.calc_fragment_mz_df()

# legacy
SWATHLibraryReader = LibraryReaderBase

class LibraryReaderFromRawData(SpecLibBase):
    def __init__(self, 
        charged_frag_types:typing.List[str] = [
            'b_z1','b_z2','y_z1', 'y_z2', 
            'b_modloss_z1','b_modloss_z2',
            'y_modloss_z1', 'y_modloss_z2'
        ],
        precursor_mz_min:float = 400,
        precursor_mz_max:float = 2000,
        decoy:str = None,
        **kwargs
    ):
        super().__init__(
            charged_frag_types=charged_frag_types,
            precursor_mz_min=precursor_mz_min,
            precursor_mz_max=precursor_mz_max,
            decoy=decoy,
        )
    
    def import_psms(self, psm_files:list, psm_type:str):
        psm_reader = psm_reader_provider.get_reader(psm_type)
        if isinstance(psm_files, str):
            self._precursor_df = psm_reader.import_file(psm_files)
            self._psm_df = self._precursor_df
        else:
            psm_df_list = []
            for psm_file in psm_files:
                psm_df_list.append(psm_reader.import_file(psm_file))
            self._precursor_df = pd.concat(psm_df_list, ignore_index=True)
            self._psm_df = self._precursor_df

    def extract_fragments(self, raw_files:list):
        """ Include two steps:
            1. self.calc_fragment_mz_df() to generate self.fragment_mz_df
            2. Extract self.fragment_intensity_df from RAW files using AlphaRAW

        Parameters
        ----------
        raw_files : list
            RAW file paths
        """
        self.calc_fragment_mz_df()
        # TODO Use AlphaRAW to extract fragment intensities