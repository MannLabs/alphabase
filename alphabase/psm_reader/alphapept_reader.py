import numba
import os
import pandas as pd
import numpy as np
import h5py

from alphabase.psm_reader.psm_reader import (
    PSMReaderBase, psm_reader_provider,
    psm_reader_yaml
)

@numba.njit
def parse_ap(precursor):
    """
    Parser to parse peptide strings
    """
    items = precursor.split('_')
    if len(items) == 3:
        decoy = 1
    else:
        decoy = 0
    modseq = items[0]
    charge = items[-1]

    parsed = []
    mods = []
    sites = []
    string = ""

    if modseq[0] == 'a':
        sites.append('0')
        mods.append('a')
        modseq = modseq[1:]
    elif modseq.startswith('tmt'):
        for l in range(3, len(modseq)):
            if modseq[l].isupper():
                break
        sites.append('0')
        mods.append(modseq[:l])
        modseq = modseq[l:]

    for i in modseq:
        string += i
        if i.isupper():
            parsed.append(i)
            if len(string) > 1:
                sites.append(str(len(parsed)))
                mods.append(string)
            string = ""

    return ''.join(parsed), ';'.join(mods), ';'.join(sites), charge, decoy

def get_x_tandem_score(df: pd.DataFrame) -> np.ndarray:
    b = df['hits_b'].astype('int').apply(lambda x: np.math.factorial(x)).values
    y = df['hits_y'].astype('int').apply(lambda x: np.math.factorial(x)).values
    x_tandem = np.log(b.astype('float')*y.astype('float')*df['fragments_matched_int_sum'].values)

    x_tandem[x_tandem==-np.inf] = 0

    return x_tandem

class AlphaPeptReader(PSMReaderBase):
    def __init__(self,
        *,
        column_mapping:dict = None,
        modification_mapping:dict = None,
        fdr = 0.01,
        keep_decoy = False,
        **kwargs,
    ):
        """
        Reading PSMs from alphapept's *.ms_data.hdf
        """
        super().__init__(
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            fdr = fdr,
            keep_decoy = keep_decoy,
            **kwargs,
        )
        if fdr <= 0.1:
            self.hdf_dataset = 'peptide_fdr'
        else:
            self.hdf_dataset = 'second_search'

    def _init_column_mapping(self):
        self.column_mapping = psm_reader_yaml[
            'alphapept'
        ]['column_mapping']

    def _init_modification_mapping(self):
        self.modification_mapping = psm_reader_yaml[
            'alphapept'
        ]['modification_mapping']

    def _load_file(self, filename):
        with h5py.File(filename, 'r') as _hdf:
            dataset = _hdf[self.hdf_dataset]
            df = pd.DataFrame({col:dataset[col] for col in dataset.keys()})
            df['raw_name'] = os.path.basename(filename)[:-len('.ms_data.hdf')]
            df['precursor'] = df['precursor'].str.decode('utf-8')
            #df['naked_sequence'] = df['naked_sequence'].str.decode('utf-8')
            if 'scan_no' in df.columns:
                df['scan_no'] = df['scan_no'].astype('int')
                df['raw_idx'] = df['scan_no']-1 # if thermo, use scan-1 as spec_idx
            df['charge'] = df['charge'].astype(int)

            if 'score' not in df.columns:
                df['score'] = get_x_tandem_score(df)
        return df
    
    def _load_modifications(self, df: pd.DataFrame):
        (
            self._psm_df['sequence'], self._psm_df['mods'],
            self._psm_df['mod_sites'], _charges,
            self._psm_df['decoy']
        ) = zip(*df['precursor'].apply(parse_ap))
        self._psm_df.decoy = self._psm_df.decoy.astype(np.int8)
    
psm_reader_provider.register_reader('alphapept', AlphaPeptReader)
