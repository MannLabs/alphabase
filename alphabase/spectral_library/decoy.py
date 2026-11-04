import copy
import multiprocessing as mp
from typing import Any

import numpy as np
import pandas as pd

from alphabase.spectral_library import near_mass_mutation
from alphabase.spectral_library.base import SpecLibBase

MIN_FALLBACK_LENGTH = 4
DROPPED_SEQUENCE = ""
# bounds the padded code arrays of one chunk to a few hundred MB
_NEAR_MASS_CHUNK_SIZE = 200_000


def _batchify_series(series, mp_batch_size):
    """Internal funciton for multiprocessing"""
    for i in range(0, len(series), mp_batch_size):
        yield series.iloc[i : i + mp_batch_size]


class BaseDecoyGenerator:
    """
    Base class for decoy generator.
    A class is used instead of a function to make as it needs to be pickled for multiprocessing.
    """

    # vectorised generators are faster in one process than after pickling batches to a pool
    supports_multiprocessing = True

    def __call__(self, series: pd.Series) -> pd.Series:
        """
        Main entry of this class, it calls follows methods:
        - self._decoy()
        """

        return series.apply(self._decoy)

    def fit(self, sequences: pd.Series) -> None:
        """Learn from the target sequences before decoying; most generators need nothing."""

    def _decoy(self, sequence: str) -> str:
        raise NotImplementedError("Subclass should implement this method.")


class DIANNDecoyGenerator(BaseDecoyGenerator):
    def __init__(
        self,
        raw_AAs: str = "GAVLIFMPWSCTYHKRQENDBJOUXZsty",
        mutated_AAs: str = "LLLVVLLLLTSSSSLLNDQEVVVVVVtss",
    ):
        """
        DiaNN-like decoy peptide generator

        Parameters
        ----------

        raw_AAs : str, optional
            AAs those DiaNN decoy from.
            Defaults to 'GAVLIFMPWSCTYHKRQENDBJOUXZsty'.

        mutated_AAs : str, optional
            AAs those DiaNN decoy to.
            Defaults to 'LLLVVLLLLTSSSSLLNDQEVVVVVVtss'.

        """
        self.raw_AAs = raw_AAs
        self.mutated_AAs = mutated_AAs

    def _decoy(self, sequence: str) -> str:
        return (
            sequence[0]
            + self.mutated_AAs[self.raw_AAs.index(sequence[1])]
            + sequence[2:-2]
            + self.mutated_AAs[self.raw_AAs.index(sequence[-2])]
            + sequence[-1]
        )


class PseudoReverseDecoyGenerator(BaseDecoyGenerator):
    def __init__(self, fix_C_term: bool = True):
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
            return sequence[:-1][::-1] + sequence[-1]
        else:
            return sequence[::-1]


class NearMassMutationDecoyGenerator(BaseDecoyGenerator):
    supports_multiprocessing = False

    def __init__(
        self,
        seed: int = 0,
        min_mass_delta: float = near_mass_mutation.MIN_MASS_DELTA,
        max_mass_delta: float = near_mass_mutation.MAX_MASS_DELTA,
        frozen_AAs: str = near_mass_mutation.FROZEN_AAS,
        excluded_mutations: tuple = near_mass_mutation.EXCLUDED_MUTATIONS,
    ):
        """
        Decoy generator replacing one inner residue of the target by a residue of similar mass.

        Unlike DIA-NN decoys, which send most residues to L at positions 2 and n-1, the replaced
        site and residue are spread like in the targets, so decoy fragments do not pile up on a
        few masses. The replacement is drawn from the targets' inner-residue composition (see
        `fit`), and the choice depends only on the seed and the sequence, so a sequence gets the
        same decoy in every batch and process.

        A proline at the second-to-last position is never replaced, and no proline is placed there.

        Peptides without a replaceable inner residue get the DIA-NN mutation if they have at least
        4 residues, otherwise an empty sequence, which `SpecLibDecoy` drops.

        The work is vectorised in `alphabase.spectral_library.near_mass_mutation`, and each unique
        sequence is decoyed once.

        Parameters
        ----------

        seed : int, optional
            Seed of the site and residue choice. Defaults to 0.

        min_mass_delta : float, optional
            Replacements must differ by more than this mass (Da). Defaults to 0.05.

        max_mass_delta : float, optional
            Replacements must differ by at most this mass (Da). Defaults to 25.

        frozen_AAs : str, optional
            Residues never replaced nor drawn. Defaults to 'CK'.

        excluded_mutations : tuple, optional
            Original and replacement residue pairs never drawn.
            Defaults to I to L, L to I, N to D, D to N, Q to E and E to Q.
        """
        self.seed = seed
        self.min_mass_delta = min_mass_delta
        self.max_mass_delta = max_mass_delta
        self.frozen_AAs = frozen_AAs
        self.excluded_mutations = tuple(excluded_mutations)
        self._fallback = DIANNDecoyGenerator()
        self._tables = None

    def fit(self, sequences: pd.Series) -> None:
        """Sets the replacement probabilities from the inner-residue composition of the targets."""
        codes = near_mass_mutation.encode(pd.unique(sequences.to_numpy()))
        self._tables = near_mass_mutation.build_replacement_tables(
            near_mass_mutation.inner_composition(codes),
            min_mass_delta=self.min_mass_delta,
            max_mass_delta=self.max_mass_delta,
            frozen_aas=self.frozen_AAs,
            excluded_mutations=self.excluded_mutations,
        )

    def __call__(self, series: pd.Series) -> pd.Series:
        if self._tables is None:
            raise RuntimeError("Call fit() with the target sequences before decoying.")

        unique_index, unique_sequences = pd.factorize(series.to_numpy())
        decoys = np.empty(len(unique_sequences), dtype=object)
        for start in range(0, len(unique_sequences), _NEAR_MASS_CHUNK_SIZE):
            chunk = slice(start, start + _NEAR_MASS_CHUNK_SIZE)
            decoys[chunk] = self._decoy_unique(unique_sequences[chunk])
        return pd.Series(decoys[unique_index], index=series.index, name=series.name)

    def _decoy(self, sequence: str) -> str:
        return self._decoy_unique(np.array([sequence], dtype=object))[0]

    def _decoy_unique(self, sequences: np.ndarray) -> np.ndarray:
        codes = near_mass_mutation.encode(sequences)
        mutated, has_site = near_mass_mutation.mutate(
            codes, near_mass_mutation.draw_uniforms(codes, self.seed), self._tables
        )
        decoys = near_mass_mutation.decode(mutated).astype(object)
        for row in np.flatnonzero(~has_site):
            decoys[row] = self._decoy_without_site(sequences[row])
        return decoys

    def _decoy_without_site(self, sequence: str) -> str:
        if len(sequence) >= MIN_FALLBACK_LENGTH:
            return self._fallback._decoy(sequence)
        return DROPPED_SEQUENCE


class SpecLibDecoy(SpecLibBase):
    """
    Pseudo-reverse peptide decoy generator.
    """

    def __init__(
        self,
        target_lib: SpecLibBase,
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

        self.generator = decoy_generator(**kwargs)

    def translate_to_decoy(
        self, multiprocessing: bool = True, mp_batch_size=10000, mp_process_num: int = 8
    ):
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
            mp_process_num=mp_process_num,
        )

    def append_to_target_lib(self):
        """
        A decoy method should define how to append itself to target_lib.
        Sub-classes should override this method when necessary.
        """
        self._remove_target_seqs()
        self._precursor_df["decoy"] = 1
        self.target_lib._precursor_df["decoy"] = 0
        self.target_lib._precursor_df = pd.concat(
            (self.target_lib._precursor_df, self._precursor_df), ignore_index=True
        )
        self.target_lib.refine_df()

    def decoy_sequence(
        self, multiprocessing: bool = True, mp_batch_size=10000, mp_process_num: int = 8
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
        self.generator.fit(self._precursor_df["sequence"])

        if (
            not multiprocessing
            or not self.generator.supports_multiprocessing
            or self._precursor_df.shape[0] < mp_batch_size
        ):
            self._precursor_df["sequence"] = self.generator(
                self._precursor_df["sequence"]
            )
            self._remove_target_seqs()
            return

        sequence_batches = list(
            _batchify_series(self._precursor_df["sequence"], mp_batch_size)
        )

        series_list = []
        with mp.get_context("spawn").Pool(mp_process_num) as p:
            processing = p.imap(self.generator, sequence_batches)
            for df in processing:
                series_list.append(df)
        self._precursor_df["sequence"] = pd.concat(series_list)
        self._remove_target_seqs()

    def _remove_target_seqs(self):
        """Drops decoys equal to a target up to I/L, which fragments cannot tell apart, and dropped sequences."""
        target_seqs = set(
            _equate_isoleucine_leucine(
                pd.unique(self.target_lib._precursor_df.sequence.to_numpy())
            )
        )
        decoy_index, decoy_seqs = pd.factorize(self._precursor_df.sequence.to_numpy())
        drop_unique = _equate_isoleucine_leucine(decoy_seqs).isin(target_seqs) | (
            decoy_seqs == DROPPED_SEQUENCE
        )
        self._precursor_df.drop(
            self._precursor_df.index[drop_unique.to_numpy()[decoy_index]],
            inplace=True,
        )


def _equate_isoleucine_leucine(sequences: np.ndarray) -> pd.Series:
    return pd.Series(sequences, dtype=object).str.replace("I", "L", regex=False)


class SpecLibDecoyProvider:
    def __init__(self):
        self.decoy_dict = {}

    def register(self, name: str, decoy_class: SpecLibDecoy):
        """Register a new decoy class"""
        self.decoy_dict[name.lower()] = decoy_class

    def get_decoy_lib(
        self, name: str, target_lib: SpecLibBase, **kwargs
    ) -> SpecLibDecoy:
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
        if not name:
            return None
        name = name.lower()
        if name == "none" or name == "null":
            return None
        if name in self.decoy_dict:
            return SpecLibDecoy(
                target_lib, decoy_generator=self.decoy_dict[name], **kwargs
            )
        else:
            raise ValueError(f"Decoy method {name} not found.")


decoy_lib_provider: SpecLibDecoyProvider = SpecLibDecoyProvider()
"""
Factory object of `SpecLibDecoyProvider` to
register and get different types of decoy methods.
"""

decoy_lib_provider.register("pseudo_reverse", PseudoReverseDecoyGenerator)
decoy_lib_provider.register("diann", DIANNDecoyGenerator)
decoy_lib_provider.register("near_mass_mutation", NearMassMutationDecoyGenerator)
