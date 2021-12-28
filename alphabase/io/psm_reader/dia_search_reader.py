# AUTOGENERATED! DO NOT EDIT! File to edit: nbdev_nbs/io/psm_reader/dia_search_reader.ipynb (unless otherwise specified).

__all__ = ['SpectronautReader', 'DiannReader']

# Cell
import pandas as pd
import numpy as np

from alphabase.io.psm_reader.psm_reader import (
    psm_reader_provider, psm_reader_yaml
)

from alphabase.io.psm_reader.maxquant_reader import (
    MaxQuantReader
)

class SpectronautReader(MaxQuantReader):
    def __init__(self,
        *,
        column_mapping:dict = None,
        modification_mapping:dict = None,
        fdr = 0.01,
        keep_decoy = False,
        mod_sep = '[]',
        underscore_for_ncterm=True,
        fixed_C57 = False,
        mod_seq_columns=[
            'ModifiedPeptide',
            'ModifiedSequence',
            'FullUniModPeptideName',
        ],
        csv_sep = '\t',
        **kwargs,
    ):
        super().__init__(
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            fdr=fdr, keep_decoy=keep_decoy,
            mod_sep=mod_sep,
            underscore_for_ncterm=underscore_for_ncterm,
            mod_seq_columns = mod_seq_columns,
            fixed_C57=fixed_C57
        )
        self.csv_sep = csv_sep

        self.mod_seq_column = 'ModifiedPeptide'

    def _init_column_mapping(self):
        self.column_mapping = psm_reader_yaml[
            'spectronaut'
        ]['column_mapping']

    def _load_file(self, filename):
        df = pd.read_csv(filename, sep=self.csv_sep)
        self._find_mod_seq_column(df)
        if 'ReferenceRun' in df.columns:
            df.drop_duplicates([
                'ReferenceRun',self.mod_seq_column, 'PrecursorCharge'
            ], inplace=True)
        else:
            df.drop_duplicates([
                self.mod_seq_column, 'PrecursorCharge'
            ], inplace=True)
        df.reset_index(drop=True, inplace=True)

        for rt_col in self.column_mapping['rt']:
            if rt_col not in df.columns: continue
            min_rt = df[rt_col].min()
            df['rt_norm'] = (
                df[rt_col] - min_rt
            )/(df[rt_col].max() - min_rt)
            break
        return df

class DiannReader(SpectronautReader):
    def __init__(self,
        *,
        column_mapping:dict = None,
        modification_mapping:dict = None,
        fdr = 0.01,
        keep_decoy = False,
        mod_sep = '()',
        underscore_for_ncterm=False,
        fixed_C57 = False,
        csv_sep = '\t',
        **kwargs,
    ):
        super().__init__(
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            fdr=fdr, keep_decoy=keep_decoy,
            mod_sep=mod_sep,
            underscore_for_ncterm=underscore_for_ncterm,
            fixed_C57=fixed_C57,
            csv_sep=csv_sep,
        )
        self.mod_seq_column = 'Modified.Sequence'

    def _init_column_mapping(self):
        self.column_mapping = psm_reader_yaml[
            'diann'
        ]['column_mapping']

    def _load_file(self, filename):
        df = pd.read_csv(filename, sep=self.csv_sep)

        # for rt_col in self.column_mapping['rt']:
            # if rt_col not in df.columns: continue
            # min_rt = df[rt_col].min()
            # df['rt_norm'] = (
            #     df[rt_col] - min_rt
            # )/(df[rt_col].max() - min_rt)
            # break
        return df

psm_reader_provider.register_reader(
    'spectronaut', SpectronautReader
)
psm_reader_provider.register_reader(
    'openswath', SpectronautReader
)
psm_reader_provider.register_reader(
    'diann', DiannReader
)