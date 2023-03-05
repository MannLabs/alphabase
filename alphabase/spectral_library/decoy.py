import copy
import pandas as pd
from alphabase.spectral_library.base import SpecLibBase
from alphabase.io.hdf import HDF_File

class SpecLibDecoy(SpecLibBase):
    """
    Pseudo-reverse peptide decoy generator.
    """
    def __init__(self, 
        target_lib:SpecLibBase,
        fix_C_term = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        target_lib : SpecLibBase
            Target library to decoy.

        fix_C_term : bool, optional
            If fix C-term AA when decoy. 
            Defaults to True.
        
        Attributes
        ----------
        target_lib : SpecLibBase
            same as 'target_lib' in Args.
        """
        self.__dict__ = copy.deepcopy(target_lib.__dict__)
        self.target_lib = target_lib
        self.fix_C_term = fix_C_term

    def translate_to_decoy(self):
        """
        Main entry of this class, it calls follows methods:
        - self.decoy_sequence()
        - self._decoy_mods()
        - self._decoy_meta()
        - self._decoy_frags()
        """
        self.decoy_sequence()
        self._decoy_mods()
        self._decoy_meta()
        self._decoy_frags()

    def append_to_target_lib(self):
        """
        A decoy method should define how to append itself to target_lib.
        Sub-classes should override this method when necessary. 
        """
        self._precursor_df['decoy'] = 1
        self.target_lib._precursor_df['decoy'] = 0
        self.target_lib._precursor_df = pd.concat((
            self.target_lib._precursor_df,
            self._precursor_df
        ), ignore_index=True)
        self.target_lib.refine_df()

    def decoy_sequence(self):
        """
        Generate decoy sequences from `self.target_lib`.
        Sub-classes should override this method when necessary. 
        """
        self._decoy_seq()
        self._remove_target_seqs()

    def append_decoy_sequence(self):
        pass

    def _decoy_seq(self):
        (
            self._precursor_df.sequence
        ) = self._precursor_df.sequence.apply(
            lambda x: (x[:-1][::-1]+x[-1])
             if self.fix_C_term else x[::-1]
        )

    def _remove_target_seqs(self):
        target_seqs = set(
            self.target_lib._precursor_df.sequence.values
        )
        self._precursor_df.drop(
            self._precursor_df.loc[
                self._precursor_df.sequence.isin(target_seqs)
            ].index, inplace=True
        )

    def _decoy_meta(self):
        """
        Decoy for CCS/RT or other meta data
        """
        pass

    def _decoy_mods(self):
        """
        Decoy for modifications and modification sites
        """
        pass

    def _decoy_frags(self):
        """
        Decoy for fragment masses and intensities
        """
        self._decoy_fragment_mz()
        self._decoy_fragment_intensity()
    
    def _decoy_fragment_mz(self):
        pass
        
    def _decoy_fragment_intensity(self):
        pass

    def _get_hdf_to_save(self, 
        hdf_file, 
        delete_existing=False
    ):
        _hdf = HDF_File(
            hdf_file, 
            read_only=False, 
            truncate=True,
            delete_existing=delete_existing
        )
        return _hdf.library.decoy

    def _get_hdf_to_load(self,
        hdf_file, 
    ):
        _hdf = HDF_File(
            hdf_file,
        )
        return _hdf.library.decoy

    def save_hdf(self, hdf_file):
        _hdf = HDF_File(
            hdf_file, 
            read_only=False, 
            truncate=True,
            delete_existing=False
        )
        _hdf.library.decoy = {
            'precursor_df': self._precursor_df,
            'fragment_mz_df': self._fragment_mz_df,
            'fragment_intensity_df': self._fragment_intensity_df,
        }

    def load_hdf(self, hdf_file):
        _hdf = HDF_File(
            hdf_file,
        )
        _hdf_lib = _hdf.library
        self._precursor_df = _hdf_lib.decoy.precursor_df.values
        self._fragment_mz_df = _hdf_lib.decoy.fragment_mz_df.values
        self._fragment_intensity_df = _hdf_lib.decoy.fragment_intensity_df.values

class SpecLibDecoyDiaNN(SpecLibDecoy):
    def __init__(self, 
        target_lib:SpecLibBase,
        raw_AAs:str = 'GAVLIFMPWSCTYHKRQENDBJOUXZ',
        mutated_AAs:str = 'LLLVVLLLLTSSSSLLNDQEVVVVVV', #DiaNN
        **kwargs,
    ):  
        """
        DiaNN-like decoy peptide generator

        Parameters
        ----------
        target_lib : SpecLibBase
            Target library object

        raw_AAs : str, optional
            AAs those DiaNN decoy from. 
            Defaults to 'GAVLIFMPWSCTYHKRQENDBJOUXZ'.

        mutated_AAs : str, optional
            AAs those DiaNN decoy to. 
            Defaults to 'LLLVVLLLLTSSSSLLNDQEVVVVVV'.
            
        """
        super().__init__(target_lib)
        self.raw_AAs = raw_AAs
        self.mutated_AAs = mutated_AAs

    def _decoy_seq(self):
        (
            self._precursor_df.sequence
        ) = self._precursor_df.sequence.apply(
            lambda x:
                x[0]+self.mutated_AAs[self.raw_AAs.index(x[1])]+
                x[2:-2]+self.mutated_AAs[self.raw_AAs.index(x[-2])]+x[-1]
        )

class SpecLibDecoyProvider(object):
    def __init__(self):
        self.decoy_dict = {}

    def register(self, name:str, decoy_class:SpecLibDecoy):
        """Register a new decoy class"""
        self.decoy_dict[name.lower()] = decoy_class

    def get_decoy_lib(self, name:str, 
        target_lib:SpecLibBase, **kwargs
    )->SpecLibDecoy:
        """Get an object of a subclass of `SpecLibDecoy` based on 
        registered name.

        Parameters
        ----------
        name : str
            Registered decoy class name
            
        target_lib : SpecLibBase
            Target library for decoy generation

        Returns
        -------
        SpecLibDecoy
            Decoy library object
        """
        if name is None: return None
        name = name.lower()
        if name in self.decoy_dict:
            return self.decoy_dict[name](
                target_lib, **kwargs
            )
        else:
            return None

decoy_lib_provider:SpecLibDecoyProvider = SpecLibDecoyProvider()
"""
Factory object of `SpecLibDecoyProvider` to 
register and get different types of decoy methods.
"""

decoy_lib_provider.register('pseudo_reverse', SpecLibDecoy)
decoy_lib_provider.register('diann', SpecLibDecoyDiaNN)
