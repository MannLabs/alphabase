import pandas as pd

from alphabase.protein.fasta import SpecLibFasta
from alphabase.spectral_library.decoy import (
    decoy_lib_provider, SpecLibDecoy
)

class ProteinReverseDecoy(SpecLibDecoy):
    def __init__(self, target_lib:SpecLibFasta):
        self.target_lib = target_lib
        self._precursor_df:pd.DataFrame = pd.DataFrame()
        self.protein_df = pd.DataFrame()
        self.decoy_tag = "REV_"

    def _add_tag_to_a_column_in_protein_df(self, column:str):
        if column in self.protein_df.columns:
            self.protein_df[column] = self.decoy_tag+self.protein_df[column]

    
    def _make_empty_loc_for_target_protein_df(self):
        self.protein_df = pd.concat(
            [
                pd.DataFrame({'sequence':[""]*len(self.target_lib.protein_df)}),
                self.protein_df
            ], ignore_index=True
        ).fillna('')

    def _decoy_protein_df(self):
        self.protein_df = self.target_lib.protein_df.copy()
        self.protein_df['sequence'] = self.protein_df.sequence.str[::-1]
        self._add_tag_to_a_column_in_protein_df(
            'protein_id'
        )
        self._add_tag_to_a_column_in_protein_df(
            'full_name'
        )
        self._add_tag_to_a_column_in_protein_df(
            'gene_name'
        )
        self._make_empty_loc_for_target_protein_df()

    def _generate_decoy_sequences(self):
        _target_prot_df = self.target_lib.protein_df
        _target_pep_df = self.target_lib.precursor_df
        self.target_lib.protein_df = self.protein_df
        self.target_lib._get_peptides_from_protein_df()
        self._precursor_df = self.target_lib.precursor_df
        self.target_lib.protein_df = _target_prot_df
        self.target_lib._precursor_df = _target_pep_df

    def decoy_sequence(self):
        if (
            not hasattr(self.target_lib, 'protein_df')
            or len(self.target_lib.protein_df) == 0
        ): return
        
        self._decoy_protein_df()
        self._generate_decoy_sequences()
        self._remove_target_seqs()

    def append_to_target_lib(self):
        if (
            not hasattr(self.target_lib, 'protein_df')
            or len(self.target_lib.protein_df) == 0
        ): return
        super().append_to_target_lib()
        self._append_protein_df_to_target_lib()

    def _append_protein_df_to_target_lib(self):
        self.protein_df['decoy'] = 1
        self.target_lib.protein_df['decoy'] = 0
        self.target_lib.protein_df = pd.concat([
            self.target_lib.protein_df,
            self.protein_df.loc[len(self.target_lib.protein_df):]
        ])


decoy_lib_provider.register('protein_reverse', ProteinReverseDecoy)

        


