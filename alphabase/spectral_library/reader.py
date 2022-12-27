import typing
import numpy as np
import pandas as pd

from alphabase.peptide.mobility import mobility_to_ccs_for_df
from alphabase.io.psm_reader.dia_search_reader import SpectronautReader
from alphabase.spectral_library.base import SpecLibBase
from alphabase.psm_reader.psm_reader import psm_reader_yaml
from alphabase.psm_reader import psm_reader_provider

class SWATHLibraryReader(SpectronautReader, SpecLibBase):
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
            'spectronaut'
        ]['mod_seq_columns'],
        csv_sep = '\t',
        rt_unit='irt',
        precursor_mz_min:float = 400,
        precursor_mz_max:float = 2000,
        decoy:str = None,
        **kwargs
    ):
        SpecLibBase.__init__(self,
            charged_frag_types = charged_frag_types,
            precursor_mz_min=precursor_mz_min,
            precursor_mz_max=precursor_mz_max,
            decoy=decoy
        )

        SpectronautReader.__init__(self, 
            column_mapping = column_mapping,
            modification_mapping = modification_mapping,
            fdr = fdr,
            keep_decoy = False,
            fixed_C57 = fixed_C57,
            mod_seq_columns=mod_seq_columns,
            csv_sep = csv_sep,
            rt_unit=rt_unit,
        )

        self._frag_type_columns = "FragmentType FragmentIonType ProductType ProductIonType".split(' ')
        self._frag_number_columns = "FragmentNumber FragmentSeriesNumber".split(' ')
        self._frag_charge_columns = "FragmentCharge FragmentIonCharge ProductCharge ProductIonCharge".split(' ')
        self._frag_loss_type_columns = "FragmentLossType FragmentIonLossType ProductLossType ProductIonLossType".split(' ')
        self._frag_inten_columns = "RelativeIntensity RelativeFragmentIntensity RelativeFragmentIonIntensity LibraryIntensity".split(' ')

    def _find_key_columns(self, lib_df:pd.DataFrame):
        def find_col(target_columns, df_columns):
            for col in target_columns:
                if col in df_columns:
                    return col
            return None
        self.mod_seq_col = find_col(self._mod_seq_columns, lib_df.columns)
        self.seq_col = find_col(self.column_mapping['sequence'], lib_df.columns)
        self.rt_col = find_col(self.column_mapping['rt'], lib_df.columns)
        self.mob_col = find_col(self.column_mapping['mobility'], lib_df.columns)
        self.raw_col = find_col(
            [self.column_mapping['raw_name']] 
            if isinstance(self.column_mapping['raw_name'],str)
            else self.column_mapping['raw_name'], 
            lib_df.columns
        )
        self.frag_type_col = find_col(self._frag_type_columns, lib_df.columns)
        self.frag_num_col = find_col(self._frag_number_columns, lib_df.columns)
        self.frag_charge_col = find_col(self._frag_charge_columns, lib_df.columns)
        self.frag_loss_type_col = find_col(self._frag_loss_type_columns, lib_df.columns)
        self.frag_inten_col = find_col(self._frag_inten_columns, lib_df.columns)

        if self.frag_loss_type_col is None:
            self.frag_loss_type_col = 'FragmentLossType'
            lib_df[self.frag_loss_type_col] = ''

    def _get_fragment_intensity(self, lib_df:pd.DataFrame):
        frag_col_dict = dict(zip(
            self.charged_frag_types, 
            range(len(self.charged_frag_types))
        ))

        self._find_key_columns(lib_df)
        lib_df[self.frag_loss_type_col].fillna('', inplace=True)
        lib_df[self.frag_loss_type_col].replace('noloss','',inplace=True)

        mod_seq_list = []
        seq_list = []
        charge_list = []
        rt_list = []
        mob_list = []
        frag_intens_list = []
        nAA_list = []
        raw_list = []

        group_cols = [self.mod_seq_col, self.seq_col, 'PrecursorCharge']

        if self.raw_col is not None:
            group_cols.append(self.raw_col)
        
        for keys, df_group in lib_df.groupby(
            group_cols
        ):
            if self.raw_col is None:
                mod_seq, seq, charge = keys
            else:
                mod_seq, seq, charge, raw = keys
            nAA = len(seq)
            intens = np.zeros(
                (nAA-1, len(self.charged_frag_types)),dtype=np.float32
            )
            for frag_type, frag_num, loss_type, frag_charge, inten in df_group[
                [
                    self.frag_type_col,self.frag_num_col,self.frag_loss_type_col,
                    self.frag_charge_col,self.frag_inten_col
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
                else:
                    continue
                
                if frag_type not in frag_col_dict:
                    continue
                frag_col_idx = frag_col_dict[frag_type]
                intens[frag_num, frag_col_idx] = inten
            max_inten = np.max(intens)
            if max_inten <= 0: continue
            intens /= max_inten

            mod_seq_list.append(mod_seq)
            seq_list.append(seq)
            charge_list.append(charge)
            rt_list.append(df_group[self.rt_col].values[0])
            if self.mob_col: 
                mob_list.append(df_group[self.mob_col].values[0])
            else:
                mob_list.append(0)
            frag_intens_list.append(intens)
            nAA_list.append(nAA)
            if self.raw_col is not None:
                raw_list.append(raw)
        
        df = pd.DataFrame({
            self.mod_seq_column: mod_seq_list,
            self.seq_col: seq_list,
            'PrecursorCharge': charge_list,
            self.rt_col: rt_list,
            self.mob_col: mob_list,
        })

        if self.raw_col is not None:
            df[self.raw_col] = raw_list

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

    def _load_file(self, filename):
        df = pd.read_csv(filename, sep=self.csv_sep)
        self._find_mod_seq_column(df)

        df = self._get_fragment_intensity(df)

        return df

    def _post_process(self, 
        lib_df
    ):  
        self._psm_df['nAA'] = self._psm_df.sequence.str.len()
        self._psm_df[
            ['frag_start_idx','frag_stop_idx']
        ] = lib_df[['frag_start_idx','frag_stop_idx']]

        self.normalize_rt_by_raw_name()

        if (
            'mobility' in self._psm_df.columns
        ):
            self._psm_df['ccs'] = (
                mobility_to_ccs_for_df(
                    self._psm_df,
                    'mobility'
                )
            )
        
        self._psm_df = self._psm_df[
            ~self._psm_df.mods.isna()
        ].reset_index(drop=True)

        self._precursor_df = self._psm_df

        self.calc_fragment_mz_df()


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