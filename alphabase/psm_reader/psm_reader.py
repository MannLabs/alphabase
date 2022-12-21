import pandas as pd
import numpy as np
import alphabase.peptide.mobility as mobility
from alphabase.peptide.precursor import (
    update_precursor_mz, reset_precursor_df
)
from alphabase.constants._const import CONST_FILE_FOLDER

def translate_other_modification(
    mod_str: str, 
    mod_dict: dict
)->str:
    '''
    Translate modifications of `mod_str` to the AlphaBase 
    format mapped by mod_dict.
    
    Parameters
    ----------
        mod_str : str
            mod list in str format, seperated by ';', 
            e.g. ModA;ModB
        mod_dict : dict
            translate mod dict from others to AlphaBase, 
            e.g. for pFind, key=['Phospho[S]','Oxidation[M]'], 
            value=['Phospho@S','Oxidation@M']
    Returns
    -------
    str
        new mods in AlphaBase format seperated by ';'. if any
        modification is not in `mod_dict`, return pd.NA.
    '''
    if not mod_str: return ""
    ret_mods = []
    for mod in mod_str.split(';'):
        if mod in mod_dict:
            ret_mods.append(mod_dict[mod])
        else:
            return pd.NA
    return ";".join(ret_mods)

def keep_modifications(
    mod_str: str, 
    mod_set: set
)->str:
    '''
    Check if modifications of `mod_str` are in `mod_set`.

    Parameters
    ----------
    mod_str : str
        mod list in str format, seperated by ';', 
        e.g. Oxidation@M;Phospho@S.
    mod_set : set
        mod set to check
    Returns
    -------
    str
        original `mod_str` if all modifications are in mod_set 
        else pd.NA.
    '''
    if not mod_str: return ""
    for mod in mod_str.split(';'):
        if not mod in mod_set:
            return pd.NA
    return mod_str


from alphabase.yaml_utils import load_yaml
import os
import copy

psm_reader_yaml = load_yaml(
    os.path.join(
        CONST_FILE_FOLDER,
        'psm_reader.yaml'
    )
)

class PSMReaderBase(object):
    def __init__(self,
        *,
        column_mapping:dict = None,
        modification_mapping:dict = None,
        fdr = 0.01,
        keep_decoy = False,
        rt_unit:str = 'minute',
        **kwargs,
    ):
        """The Base class for all PSMReaders. The key of the sub-classes for different 
        search engine format is to re-define `column_mapping` and `modification_mapping`.
        
        Parameters
        ----------
        column_mapping : dict, optional
            A dict that maps alphabase's columns to other search engine's.
            The key of the column_mapping is alphabase's column name, and 
            the value could be the column name or a list of column names
            in other engine's result.
            If it is None, this dict will be init by 
            `self._init_column_mapping`. The dict values could be 
            either str or list, for exaplme:
            columns_mapping = {
            'sequence': 'NakedSequence', #str
            'charge': 'Charge', #str
            'proteins':['Proteins','UniprotIDs'], # list, this reader will automatically detect all of them.
            }
            Defaults to None.
        modification_mapping : dict, optional
            A dict that maps alphabase's modifications to other engine's.
            If it is None, this dict will be init by 
            `self._init_modification_mapping`. The dict values could be 
            either str or list, for exaplme:
            modification_mapping = {
            'Oxidation@M': 'Oxidation (M)', # str
            'Phospho@S': ['S(Phospho (STY))','S(ph)','pS'], # list, this reader will automatically detect all of them.
            }
            Defaults to None.
        fdr : float, optional
            FDR level to keep PSMs.
            Defaults to 0.01.
        keep_decoy : bool, optional
            If keep decoy PSMs in self.psm_df.
            Defautls to False.
        
        Attributes
        ----------
        column_mapping : dict
            Dict structure same as column_mapping in Args.
        modification_mapping : dict
            Dict structure same as modification_mapping in Args.
            We must use self.set_modification_mapping(new_mapping) to update it.
        _psm_df : pd.DataFrame
            the PSM DataFrame after loading from search engines.
        psm_df : pd.DataFrame
            the getter of self._psm_df
        keep_fdr : float
            The only PSMs with FDR<=keep_fdr were returned in self._psm_df. 
        keep_decoy : bool
            If keep decoy PSMs in self.psm_df.
        _min_max_rt_norm : bool
            if True, the 'rt_norm' values in self._psm_df 
            will be normalized by rt_norm = (self.psm_df.rt-rt_min)/(rt_max-rt_min).
            It is useful to normalize iRT values as they contain negative values.
            Defaults to False.
        """

        self.set_modification_mapping(modification_mapping)
        
        if column_mapping is not None:
            self.column_mapping = column_mapping
        else:
            self._init_column_mapping()

        self._psm_df = pd.DataFrame()
        self.keep_fdr = fdr
        self.keep_decoy = keep_decoy
        self._min_max_rt_norm = False
        self._engine_rt_unit = rt_unit

    @property
    def psm_df(self)->pd.DataFrame:
        return self._psm_df

    def add_modification_mapping(self, modification_mapping:dict):
        """
        Append additional modifications from other search engines

        Parameters
        ----------
        modification_mapping : dict
            The key of dict is a modification name in AlphaBase format; 
            the value could be a str or a list, see below
            ```
            add_modification_mapping({
            'Dimethyl@K': ['K(Dimethyl)'], # list
            'Dimethyl@Any N-term': '_(Dimethyl)', # str
            })
            ```
        """
        if (
            modification_mapping is None or
            len(modification_mapping) == 0
        ):
            return

        for key, val in list(modification_mapping.items()):
            if key in self.modification_mapping:
                if isinstance(val, str):
                    self.modification_mapping[key].append(val)
                else:
                    self.modification_mapping[key].extend(val)
            else:
                if isinstance(val, str):
                    self.modification_mapping[key] = [val]
                else:
                    self.modification_mapping[key] = val

        self.set_modification_mapping(self.modification_mapping)

    def set_modification_mapping(self, modification_mapping:dict):
        if modification_mapping is None:
            self._init_modification_mapping()
        elif isinstance(modification_mapping, str):
            if modification_mapping in psm_reader_yaml:
                self.modification_mapping = copy.deepcopy(
                    psm_reader_yaml[
                        modification_mapping
                    ]['modification_mapping']
                )
            else:
                raise ValueError(
                    f'Unknown modification mapping: {modification_mapping}'
                )
        else:
            self.modification_mapping = copy.deepcopy(
                modification_mapping
            )
        self._reverse_mod_mapping()

    def _init_modification_mapping(self):
        self.modification_mapping = {}
        
    def _reverse_mod_mapping(self):
        self.rev_mod_mapping = {}
        for (
            this_mod, other_mod
        ) in self.modification_mapping.items():
            if isinstance(other_mod, (list, tuple)):
                for _mod in other_mod:
                    if _mod in self.rev_mod_mapping:
                        if this_mod.endswith('Protein N-term'):
                            continue
                    self.rev_mod_mapping[_mod] = this_mod
            else:
                self.rev_mod_mapping[other_mod] = this_mod
                
    def _init_column_mapping(self):
        raise NotImplementedError(
            f'"{self.__class__}" must implement "_init_column_mapping()"'
        )
    
    def load(self, _file)->pd.DataFrame:
        """ Wrapper for import_file() """
        if isinstance(_file, list): 
            return self.import_files(_file)
        else: 
            return self.import_file(_file)

    def import_files(self, file_list:list):
        df_list = []
        for _file in file_list:
            df_list.append(self.import_file(_file))
        self._psm_df = pd.concat(df_list, ignore_index=True)
        return self._psm_df

    def import_file(self, _file:str)->pd.DataFrame:
        """
        This is the main entry function of PSM readers, 
        it imports the file with following steps:
        ```
        origin_df = self._load_file(_file)
        self._translate_columns(origin_df)
        self._translate_decoy(origin_df)
        self._translate_score(origin_df)
        self._load_modifications(origin_df)
        self._translate_modifications()
        self._post_process(origin_df)
        ```
        
        Parameters
        ----------
        _file: str
            file path or file stream (io).
        """
        origin_df = self._load_file(_file)
        if len(origin_df) == 0:
            self._psm_df = pd.DataFrame()
        else:
            self._translate_columns(origin_df)
            self._translate_decoy(origin_df)
            self._translate_score(origin_df)
            self._load_modifications(origin_df)
            self._translate_modifications()
            self._post_process(origin_df)
        return self._psm_df

    def _translate_decoy(
        self, 
        origin_df:pd.DataFrame=None
    ):
        pass

    def _translate_score(
        self, 
        origin_df:pd.DataFrame=None
    ):
        # some scores are evalue/pvalue, it should be translated
        # to -log(evalue), as score is the larger the better
        pass

    def normalize_rt(self):
        if 'rt' in self.psm_df.columns:
            if self._engine_rt_unit == 'second':
                # self.psm_df['rt_sec'] = self.psm_df.rt
                self.psm_df['rt'] = self.psm_df.rt/60
            # elif self._engine_rt_unit == 'minute':
                # self.psm_df['rt_sec'] = self.psm_df.rt*60
            min_rt = self.psm_df.rt.min()
            if not self._min_max_rt_norm or min_rt > 0:
                min_rt = 0
            self.psm_df['rt_norm'] = (
                self.psm_df.rt - min_rt
            ) / (self.psm_df.rt.max()-min_rt)

    def norm_rt(self):
        self.normalize_rt()

    def normalize_rt_by_raw_name(self):
        if not 'rt' in self.psm_df.columns:
            return
        if not 'rt_norm' in self.psm_df.columns:
            self.norm_rt()
        if not 'raw_name' in self.psm_df.columns:
            return
        for raw_name, df_group in self.psm_df.groupby('raw_name'):
            self.psm_df.loc[
                df_group.index,'rt_norm'
            ] = df_group.rt_norm / df_group.rt_norm.max()

    def _load_file(self, filename:str)->pd.DataFrame:
        """
        Load original dataframe from PSM filename. 
        Different search engines may store PSMs in different ways:
        tsv, csv, HDF, XML, ...

        Parameters
        ----------
        filename : str
            psm filename

        Raises
        ------
        NotImplementedError
            Subclasses must re-implement this method

        Returns:
        pd.DataFrame
            loaded dataframe
        """
        raise NotImplementedError(
            f'"{self.__class__}" must implement "_load_file()"'
        )

    def _translate_columns(self, origin_df:pd.DataFrame):
        """
        Translate the dataframe from other search engines 
        to AlphaBase format

        Parameters
        ----------
        origin_df : pd.DataFrame
            df of other search engines

        Returns
        -------
        None
            Add information inplace into self._psm_df
        """
        self._psm_df = pd.DataFrame()
        for col, map_col in self.column_mapping.items():
            if isinstance(map_col, str):
                if map_col in origin_df.columns:
                    self._psm_df[col] = origin_df[map_col]
            else:
                for other_col in map_col:
                    if other_col in origin_df.columns:
                        self._psm_df[col] = origin_df[other_col]
                        break
                    
        if (
            'scan_num' in self._psm_df.columns and 
            not 'spec_idx' in self._psm_df.columns
        ):
            self._psm_df['spec_idx'] = self._psm_df.scan_num - 1
    

    def _load_modifications(self, origin_df:pd.DataFrame):
        """Read modification information from 'origin_df'. 
        Some of search engines use modified_sequence, some of them
        use additional columns to store modifications and the sites.

        Parameters
        ----------
        origin_df : pd.DataFrame
            dataframe of original search engine.
        """
        raise NotImplementedError(
            f'"{self.__class__}" must implement "_load_modifications()"'
        )

    def _translate_modifications(self):
        '''
        Translate modifications to AlphaBase format.

        Raises
        ------
        KeyError
            if `mod` in `mod_names` is 
            not in `self.modification_mapping`
        '''
        self._psm_df.mods = self._psm_df.mods.apply(
            translate_other_modification, 
            mod_dict=self.rev_mod_mapping
        )

    def _post_process(self, 
        origin_df:pd.DataFrame
    ):
        """
        Set 'nAA' columns, remove unknown modifications 
        and perform other post processings, 
        e.g. get 'rt_norm', remove decoys, filter FDR...

        Parameters
        ----------
        origin_df : pd.DataFrame
            the loaded original df
        """
        self._psm_df['nAA'] = self._psm_df.sequence.str.len()

        self.normalize_rt_by_raw_name()

        self._psm_df = self._psm_df[
            ~self._psm_df['mods'].isna()
        ]

        keep_rows = np.ones(
            len(self._psm_df), dtype=bool
        )
        if 'fdr' in self._psm_df.columns:
            keep_rows &= (self._psm_df.fdr <= self.keep_fdr)
        if (
            'decoy' in self._psm_df.columns 
            and not self.keep_decoy
        ):
            keep_rows &= (self._psm_df.decoy == 0)

        self._psm_df = self._psm_df[keep_rows]
        
        reset_precursor_df(self._psm_df)
        
        if 'precursor_mz' not in self._psm_df:
            self._psm_df = update_precursor_mz(self._psm_df)

        if (
            'ccs' in self._psm_df.columns and 
            'mobility' not in self._psm_df.columns
        ):
            self._psm_df['mobility'] = (
                mobility.ccs_to_mobility_for_df(
                    self._psm_df,
                    'ccs'
                )
            )
        elif (
            'mobility' in self._psm_df.columns and
            'ccs' not in self._psm_df.columns
        ):
            self._psm_df['ccs'] = (
                mobility.mobility_to_ccs_for_df(
                    self._psm_df,
                    'mobility'
                )
            )

    def filter_psm_by_modifications(self, include_mod_set = set([
        'Oxidation@M','Phospho@S','Phospho@T',
        'Phospho@Y','Acetyl@Protein N-term'
    ])):
        '''
            Only keeps peptides with modifications in `include_mod_list`.
        '''
        self._psm_df.mods = self._psm_df.mods.apply(
            keep_modifications, mod_set=include_mod_set
        )
        
        self._psm_df.dropna(
            subset=['mods'], inplace=True
        )
        self._psm_df.reset_index(drop=True, inplace=True)


class PSMReaderProvider:
    def __init__(self):
        self.reader_dict = {}

    def register_reader(self, reader_type, reader_class):
        self.reader_dict[reader_type.lower()] = reader_class

    def get_reader(self, 
        reader_type:str,
        *,
        column_mapping:dict=None, 
        modification_mapping:dict=None,
        fdr=0.01, keep_decoy=False,
        **kwargs
    )->PSMReaderBase:
        return self.reader_dict[reader_type.lower()](
            column_mapping = column_mapping,
            modification_mapping=modification_mapping,
            fdr=fdr, keep_decoy=keep_decoy, **kwargs
        )

    def get_reader_by_yaml(self, 
        yaml_dict:dict,
    )->PSMReaderBase:
        return self.get_reader(
            **copy.deepcopy(yaml_dict)
        )

psm_reader_provider = PSMReaderProvider()
"""
A factory :class:`PSMReaderProvider` object to register and get readers for different PSM types.
"""
