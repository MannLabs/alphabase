import numpy as np
import pandas as pd

from alphabase.psm_reader.psm_reader import (
    PSMReaderBase, psm_reader_yaml,
    psm_reader_provider
)
from alphabase.constants.aa import AA_ASCII_MASS
from alphabase.constants.atom import MASS_H, MASS_O, MASS_PROTON
from alphabase.constants.modification import MOD_MASS

try:
    import pyteomics.pepxml as pepxml
except:
    pepxml = None

def _is_fragger_decoy(proteins):
    for prot in proteins:
        if not prot.lower().startswith('rev_'):
            return False
    return True

mass_mapped_mods = psm_reader_yaml['msfragger_pepxml']['mass_mapped_mods']
mod_mass_tol = psm_reader_yaml['msfragger_pepxml']['mod_mass_tol']

def _get_mods_from_masses(sequence, msf_aa_mods):
    mods = []
    mod_sites = []
    aa_mass_diffs = []
    aa_mass_diff_sites = []
    for mod in msf_aa_mods:
        _mass_str, site_str = mod.split('@')
        mod_mass = float(_mass_str)
        site = int(site_str)
        cterm_position = len(sequence) + 1
        if site > 0:
            if site < cterm_position:
                mod_mass = mod_mass - AA_ASCII_MASS[ord(sequence[site-1])]
            else:
                mod_mass -= (2* MASS_H + MASS_O)
        else:
            mod_mass -= MASS_H

        mod_translated = False
        for mod_name in mass_mapped_mods:
            if abs(mod_mass-MOD_MASS[mod_name])<mod_mass_tol:
                if site==0:
                    _mod = mod_name.split('@')[0]+'@Any N-term'
                elif site==1:
                    if mod_name.endswith('^Any N-term'):
                        _mod = mod_name
                        site_str = '0'
                    else:
                        _mod = mod_name.split('@')[0]+'@'+sequence[0]
                elif site==cterm_position:
                    if mod_name.endswith('C-term'):
                        _mod = mod_name
                    else:
                        _mod = mod_name.split('@')[0]+'@Any C-term' #what if only Protein C-term is listed?
                    site_str = '-1'
                else:
                    _mod = mod_name.split('@')[0]+'@'+sequence[site-1]
                if _mod in MOD_MASS:
                    mods.append(_mod)
                    mod_sites.append(site_str)
                    mod_translated = True
                    break
        if not mod_translated:
            aa_mass_diffs.append(f'{mod_mass:.5f}')
            aa_mass_diff_sites.append(site_str)
    return ';'.join(mods), ';'.join(mod_sites), ';'.join(aa_mass_diffs), ';'.join(aa_mass_diff_sites)


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
            fdr = 0.001, # refers to E-value in the PepXML
            keep_decoy=False, 
            rt_unit = 'second',
            keep_unknown_aa_mass_diffs=False,
            **kwargs
        ):
            """MSFragger is not fully supported as we can only access the pepxml file.
            """
            super().__init__(
                column_mapping=column_mapping, 
                modification_mapping=modification_mapping,
                fdr = fdr,
                keep_decoy=keep_decoy, 
                rt_unit = rt_unit,
                **kwargs
            )
            self.keep_unknown_aa_mass_diffs = keep_unknown_aa_mass_diffs

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
            if 'ion_mobility' in msf_df.columns:
                msf_df['ion_mobility'] = msf_df.ion_mobility.astype(float)
            msf_df['raw_name'] = msf_df[
                'spectrum'
            ].str.split('.').apply(lambda x: x[0])
            msf_df['to_remove'] = 0
            self.column_mapping['to_remove'] = 'to_remove'
            return msf_df

        def _translate_decoy(self, origin_df=None):
            self._psm_df['decoy'] = self._psm_df.proteins.apply(
                _is_fragger_decoy
            ).astype(np.int8)

            self._psm_df.proteins = self._psm_df.proteins.apply(
                lambda x: ';'.join(x)
            )
            if not self.keep_decoy:
                self._psm_df['to_remove'] += (self._psm_df.decoy > 0)
        
        def _translate_score(self, origin_df=None):
            # evalue score
            self._psm_df['score'] = -np.log(
                self._psm_df['score']+1e-100
            )

        def _load_modifications(self, msf_df):
            if len(msf_df) == 0:
                self._psm_df['mods'] = ''
                self._psm_df['mod_sites'] = ''
                self._psm_df['aa_mass_diffs'] = ''
                self._psm_df['aa_mass_diff_sites'] = ''
                return

            (
                self._psm_df['mods'], self._psm_df['mod_sites'],
                self._psm_df['aa_mass_diffs'], self._psm_df['aa_mass_diff_sites'],
            ) = zip(*msf_df[['peptide','modifications']].apply(
                lambda x: _get_mods_from_masses(*x), axis=1)
            )
                
            if not self.keep_unknown_aa_mass_diffs:
                self._psm_df['to_remove'] += (self._psm_df.aa_mass_diffs != '')
                self._psm_df.drop(
                    columns=['aa_mass_diffs','aa_mass_diff_sites'], 
                    inplace=True
                )

        def _post_process(self, origin_df: pd.DataFrame):
            super()._post_process(origin_df)
            self._psm_df = self._psm_df.query(
                'to_remove==0'
            ).drop(columns='to_remove').reset_index(drop=True)

    psm_reader_provider.register_reader('msfragger_pepxml', MSFraggerPepXML)

