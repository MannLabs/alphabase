import copy
from typing import Any
import pandas as pd
import multiprocessing as mp
from functools import partial
from alphabase.spectral_library.base import SpecLibBase
from alphabase.io.hdf import HDF_File

def _batchify_series(series, mp_batch_size):
    """Internal funciton for multiprocessing"""
    for i in range(0, len(series), mp_batch_size):
        yield series.iloc[i:i+mp_batch_size]
class BaseDecoyGenerator(object):
    """
    Base class for decoy generator.
    A class is used instead of a function to make as it needs to be pickled for multiprocessing.
    """
    def __call__(self, series: pd.Series) -> pd.Series:
        """
        Main entry of this class, it calls follows methods:
        - self._decoy()
        """

        return series.apply(self._decoy)

    def _decoy(self, sequence:str) -> str:
        raise NotImplementedError('Subclass should implement this method.')

class DIANNDecoyGenerator(BaseDecoyGenerator):
    def __init__(self, 
            raw_AAs:str = 'GAVLIFMPWSCTYHKRQENDBJOUXZ',
            mutated_AAs:str = 'LLLVVLLLLTSSSSLLNDQEVVVVVV'
        ):

        """
        DiaNN-like decoy peptide generator

        Parameters
        ----------

        raw_AAs : str, optional
            AAs those DiaNN decoy from. 
            Defaults to 'GAVLIFMPWSCTYHKRQENDBJOUXZ'.

        mutated_AAs : str, optional
            AAs those DiaNN decoy to. 
            Defaults to 'LLLVVLLLLTSSSSLLNDQEVVVVVV'.
            
        """
        self.raw_AAs = raw_AAs
        self.mutated_AAs = mutated_AAs


    def _decoy(self, sequence: str) -> str:
        return sequence[0]+ \
        self.mutated_AAs[self.raw_AAs.index(sequence[1])]+ \
        sequence[2:-2]+ \
        self.mutated_AAs[self.raw_AAs.index(sequence[-2])]+ \
        sequence[-1]

class PseudoReverseDecoyGenerator(BaseDecoyGenerator):
    def __init__(self, fix_C_term:bool=True):
        """
        Pseudo-reverse decoy generator.

        Parameters
        ----------

        fix_C_term : bool, optional
            If fix C-term AA when decoy. 
            Defaults to True.
        """

        self.fix_C_term = fix_C_term

    def _decoy(self, sequence: str) -> str:
        if self.fix_C_term:
            return (sequence[:-1][::-1] + sequence[-1])
        else:
            return sequence[::-1]

class SpecLibDecoy(SpecLibBase):
    """
    Pseudo-reverse peptide decoy generator.
    """

    def __init__(self, 
        target_lib:SpecLibBase,
        decoy_generator: Any = PseudoReverseDecoyGenerator,
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

        self.generator = decoy_generator(
            **kwargs
        )

    def translate_to_decoy(
            self, 
            multiprocessing : bool = True,
            mp_batch_size=10000, 
            mp_process_num: int = 8):
        """
        Main entry of this class, it calls follows methods:
        - self.decoy_sequence()

        Parameters
        ----------

        multiprocessing : bool, optional
            If true use multiprocessing.
            Defaults to True.

        mp_batch_size : int, optional
            Batch size for multiprocessing.
            Defaults to 10000.

        mp_process_num : int, optional
            Number of processes for multiprocessing.
            Defaults to 8.
            
        """
        self.decoy_sequence(
            multiprocessing=multiprocessing,
            mp_batch_size=mp_batch_size,
            mp_process_num=mp_process_num
        )

    def append_to_target_lib(self):
        """
        A decoy method should define how to append itself to target_lib.
        Sub-classes should override this method when necessary. 
        """
        self._remove_target_seqs()
        self._precursor_df['decoy'] = 1
        self.target_lib._precursor_df['decoy'] = 0
        self.target_lib._precursor_df = pd.concat((
            self.target_lib._precursor_df,
            self._precursor_df
        ), ignore_index=True)
        self.target_lib.refine_df()

    def decoy_sequence(
            self, 
            multiprocessing: bool = True, 
            mp_batch_size=10000, 
            mp_process_num: int = 8
        ):
        """
        Generate decoy sequences from `self.target_lib`.
        Sub-classes should override the `_decoy_seq` method when necessary.

        Parameters
        ----------

        multiprocessing : bool, optional
            If true use multiprocessing.
            Defaults to True.

        mp_batch_size : int, optional
            Batch size for multiprocessing.
            Defaults to 10000.

        mp_process_num : int, optional
            Number of processes for multiprocessing.
            Defaults to 8.
        """

        if not multiprocessing or self._precursor_df.shape[0] < mp_batch_size:
            self._precursor_df['sequence'] = self.generator(self._precursor_df['sequence'])
            self._remove_target_seqs()
            return
            
        sequence_batches = list(_batchify_series(
            self._precursor_df['sequence'], mp_batch_size
        ))

        series_list = []
        with mp.get_context("spawn").Pool(mp_process_num) as p:
            processing = p.imap(
                self.generator,
                sequence_batches
            )
            for df in processing:
                series_list.append(df)
        self._precursor_df['sequence'] = pd.concat(series_list)
        self._remove_target_seqs()

    def _remove_target_seqs(self):
        target_seqs = set(
            self.target_lib._precursor_df.sequence.values
        )
        self._precursor_df.drop(
            self._precursor_df.loc[
                self._precursor_df.sequence.isin(target_seqs)
            ].index, inplace=True
        )

class SpecLibDecoyProvider(object):
    def __init__(self):
        self.decoy_dict = {}

    def register(self, name:str, decoy_class:SpecLibDecoy):
        """Register a new decoy class"""
        self.decoy_dict[name.lower()] = decoy_class

    def get_decoy_lib(self, 
        name:str, 
        target_lib:SpecLibBase, 
        **kwargs
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
        if not name: return None
        name = name.lower()
        if name == "none" or name == "null":
            return None
        if name in self.decoy_dict:
            return SpecLibDecoy(
                target_lib,
                decoy_generator = self.decoy_dict[name],
                **kwargs
            )
        else:
            raise ValueError(f'Decoy method {name} not found.')

decoy_lib_provider:SpecLibDecoyProvider = SpecLibDecoyProvider()
"""
Factory object of `SpecLibDecoyProvider` to 
register and get different types of decoy methods.
"""

decoy_lib_provider.register('pseudo_reverse', PseudoReverseDecoyGenerator)
decoy_lib_provider.register('diann', DIANNDecoyGenerator)
