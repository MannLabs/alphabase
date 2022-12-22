import pandas as pd
import numpy as np

import alphabase.constants.modification as ap_mod

from alphabase.psm_reader.psm_reader import (
    PSMReaderBase, psm_reader_provider,
    psm_reader_yaml
)

def convert_one_pFind_mod(mod):
    if mod[-1] == ')':
        mod = mod[:(mod.find('(')-1)]
        idx = mod.rfind('[')
        name = mod[:idx]
        site = mod[(idx+1):]
    else:
        idx = mod.rfind('[')
        name = mod[:idx]
        site = mod[(idx+1):-1]
    if len(site) == 1:
        return name + '@' + site
    elif site == 'AnyN-term':
        return name + '@' + 'Any N-term'
    elif site == 'ProteinN-term':
        return name + '@' + 'Protein N-term'
    elif site.startswith('AnyN-term'):
        return name + '@' + site[-1] + '^Any N-term'
    elif site.startswith('ProteinN-term'):
        return name + '@' + site[-1] + '^Protein N-term'
    elif site == 'AnyC-term':
        return name + '@' + 'Any C-term'
    elif site == 'ProteinC-term':
        return name + '@' + 'Protein C-term'
    elif site.startswith('AnyC-term'):
        return name + '@' + site[-1] + '^Any C-term'
    elif site.startswith('ProteinC-term'):
        return name + '@' + site[-1] + '^Protein C-term'
    else:
        return None

def translate_pFind_mod(mod_str):
    if not mod_str: return ""
    ret_mods = []
    for mod in mod_str.split(';'):
        mod = convert_one_pFind_mod(mod)
        if not mod: return pd.NA
        elif mod not in ap_mod.MOD_INFO_DICT: return pd.NA
        else: ret_mods.append(mod)
    return ';'.join(ret_mods)

def get_pFind_mods(pfind_mod_str):
    pfind_mod_str = pfind_mod_str.strip(';')
    if not pfind_mod_str: return "", ""

    items = [
        item.split(',',3) 
        for item in pfind_mod_str.split(';')
    ]
    
    items = [
        ('-1',mod) if (mod.endswith('C-term]') 
        or mod[:-2].endswith('C-term'))
        #else ('0', mod) if mod.endswith('N-term]')
        else (site, mod) for site, mod in items
    ]
    items = list(zip(*items))
    return ';'.join(items[1]), ';'.join(items[0])

def parse_pfind_protein(protein, keep_reverse=True):
    proteins = protein.strip('/').split('/')
    return ';'.join(
        [
            protein for protein in proteins 
            if (
                not protein.startswith('REV_') 
                or keep_reverse
            )
        ]
    )


class pFindReader(PSMReaderBase):
    def __init__(self,
        *,
        column_mapping:dict = None,
        modification_mapping:dict = None,
        fdr = 0.01,
        keep_decoy = False,
        **kwargs,
    ):
        super().__init__(
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            fdr = fdr,
            keep_decoy = keep_decoy,
            **kwargs,
        )

    def _init_column_mapping(self):
        self.column_mapping = psm_reader_yaml[
            'pfind'
        ]['column_mapping']
        
    def _init_modification_mapping(self):
        self.modification_mapping = {}

    def _translate_modifications(self):
        pass

    def _load_file(self, filename):
        pfind_df = pd.read_csv(filename, index_col=False, sep='\t')
        pfind_df.fillna('', inplace=True)
        pfind_df = pfind_df[pfind_df.Sequence != '']
        pfind_df['raw_name'] = pfind_df[
            'File_Name'
        ].str.split('.').apply(lambda x: x[0])
        pfind_df['Proteins'] = pfind_df[
            'Proteins'
        ].apply(parse_pfind_protein)
        return pfind_df

    def _translate_decoy(self, origin_df=None):
        self._psm_df.decoy = (
            self._psm_df.decoy == 'decoy'
        ).astype(np.int8)
        
    def _translate_score(self, origin_df=None):
        self._psm_df.score = -np.log(
            self._psm_df.score.astype(float)+1e-100
        )

    def _load_modifications(self, pfind_df):
        (
            self._psm_df['mods'], self._psm_df['mod_sites']
        ) = zip(*pfind_df['Modification'].apply(get_pFind_mods))

        self._psm_df['mods'] = self._psm_df['mods'].apply(
            translate_pFind_mod
        )
        
psm_reader_provider.register_reader('pfind', pFindReader)
