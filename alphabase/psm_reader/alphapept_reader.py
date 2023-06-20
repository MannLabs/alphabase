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

    for i in range(len(modseq)):
        if modseq[i].isupper():
            break
    if i > 0:
        sites.append('0')
        mods.append(modseq[:i])
        modseq = modseq[i:]

    for i in modseq:
        string += i
        if i.isupper():
            parsed.append(i)
            if len(string) > 1:
                sites.append(str(len(parsed)))
                mods.append(string)
            string = ""

    return ''.join(parsed), ';'.join(mods), ';'.join(sites), charge, decoy

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
        self.hdf_dataset = 'identifications'

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
        return df
    
    def _load_modifications(self, df: pd.DataFrame):
        if len(df) == 0: 
            self._psm_df['sequence'] = '' 
            self._psm_df['mods'] = ''
            self._psm_df['mod_sites'] = ''
            self._psm_df['decoy'] = 0
            return
            
        (
            self._psm_df['sequence'], self._psm_df['mods'],
            self._psm_df['mod_sites'], _charges,
            self._psm_df['decoy']
        ) = zip(*df['precursor'].apply(parse_ap))
        self._psm_df.decoy = self._psm_df.decoy.astype(np.int8)
    
psm_reader_provider.register_reader('alphapept', AlphaPeptReader)
