import numpy as np
import pandas as pd

from alphabase.psm_reader.psm_reader import (
    PSMReaderBase, psm_reader_yaml,
    psm_reader_provider
)
from alphabase.psm_reader.maxquant_reader import MaxQuantReader
from alphabase.constants.aa import AA_ASCII_MASS
from alphabase.constants.modification import MOD_INFO_DICT as mod_info

#| export
try:
    import pyteomics.pepxml as pepxml
except:
    pepxml = None


def _is_fragger_decoy(proteins):
    for prot in proteins:
        if not prot.startswith('rev_'):
            return False
    return True

mass_mapped_mods = psm_reader_yaml['msfragger_pepxml']['mass_mapped_mods']
mod_mass_tol = psm_reader_yaml['msfragger_pepxml']['mod_mass_tol']

def _get_msf_mods(sequence, msf_aa_mods):
    mods = []
    mod_sites = []
    mod_deltas = []
    mod_delta_sites = []
    for mod in msf_aa_mods:
        mod_mass, site_str = mod.split('@')
        mod_mass = float(mod_mass)
        site = int(site_str)-1
        mod_mass = mod_mass - AA_ASCII_MASS[ord(sequence[site])]

        mod_considered = False
        for mod_name in mass_mapped_mods:
            if abs(mod_mass-mod_info[mod_name]['mass'])<mod_mass_tol:
                if site == 0 and mod_name.endswith('N-term'):
                    mods.append(mod_name)
                    mod_sites.append('0')
                    mod_considered = True
                    break
                _mod = mod_name.split('@')[0]+'@'+sequence[site]
                if _mod in mod_info:
                    mods.append(_mod)
                    mod_sites.append(site_str)
                    mod_considered = True
                    break
        if not mod_considered:
            mod_deltas.append(str(mod_mass))
            mod_delta_sites.append(site_str)
    return ';'.join(mods), ';'.join(mod_sites), ';'.join(mod_deltas), ';'.join(mod_delta_sites)


class MSFragger_PSM_TSV_Reader(PSMReaderBase):
    def __init__(self, *, 
        column_mapping: dict = None, 
        modification_mapping: dict = None, 
        fdr=0.01, 
        keep_decoy=False, 
        rt_unit = 'second',
        **kwargs
    ):
        raise NotImplementedError("MSFragger_PSM_TSV_Reader for psm.tsv")

psm_reader_provider.register_reader('msfragger_psm_tsv', MSFragger_PSM_TSV_Reader)
psm_reader_provider.register_reader('msfragger', MSFragger_PSM_TSV_Reader)

if pepxml is None:
    class MSFraggerPepXML:
        def __init__(self): raise NotImplementedError("")
else:
    class MSFraggerPepXML(PSMReaderBase):
        def __init__(self, *, 
            column_mapping: dict = None, 
            modification_mapping: dict = None,
            keep_decoy=True, 
            rt_unit = 'second',
            **kwargs
        ):
            """MSFragger is not fully supported as we can only access the pepxml file.
            """
            super().__init__(
                column_mapping=column_mapping, 
                modification_mapping=modification_mapping,
                keep_decoy=keep_decoy, 
                rt_unit = 'second',
                **kwargs
            )

        def _init_column_mapping(self):
            self.column_mapping = psm_reader_yaml[
                'msfragger_pepxml'
            ]['column_mapping']
            
        def _init_modification_mapping(self):
            self.modification_mapping = {}

        def _translate_modifications(self):
            pass

        def _load_file(self, filename):
            msf_df = pepxml.DataFrame(filename)
            msf_df.fillna('', inplace=True)
            msf_df.retention_time_sec /= 60
            msf_df['raw_name'] = msf_df[
                'spectrum'
            ].str.split('.').apply(lambda x: x[0])
            return msf_df

        def _translate_decoy(self, origin_df=None):
            self._psm_df['decoy'] = self._psm_df.proteins.apply(
                _is_fragger_decoy
            ).astype(np.int8)

            self._psm_df.proteins = self._psm_df.proteins.apply(
                lambda x: ';'.join(x)
            )
        def _translate_score(self, origin_df=None):
            if self.column_mapping['score'] == 'expect':
                # evalue score
                self._psm_df['score'] = -np.log(
                    self._psm_df['score']+1e-100
                )

        def _load_modifications(self, msf_df):
            if len(msf_df) == 0:
                self._psm_df['mods'] = ''
                self._psm_df['mod_sites'] = ''
                self._psm_df['mod_deltas'] = ''
                self._psm_df['mod_delta_sites'] = ''
                return

            (
                self._psm_df['mods'], self._psm_df['mod_sites'],
                self._psm_df['mod_deltas'], self._psm_df['mod_delta_sites'],
            ) = zip(*msf_df[['peptide','modifications']].apply(
                lambda x: _get_msf_mods(*x), axis=1)
            )

    psm_reader_provider.register_reader('msfragger_pepxml', MSFraggerPepXML)

