import copy
import hashlib
import multiprocessing as mp
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from alphabase.constants.aa import AA_ASCII_MASS
from alphabase.spectral_library.base import SpecLibBase

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
# A replacement this light is indistinguishable from the target at fragment level (I/L) or mimics a
# common modification (deamidation), so the decoy would be a plausible target rather than a false one
EXCLUDED_MUTATIONS = ("IL", "LI", "ND", "QE")
# Residues that carry fixed labels or define the digestion site keep the decoy's chemistry and cleavage
FROZEN_AAS = "CK"
MIN_MASS_DELTA = 0.05
MAX_MASS_DELTA = 25.0
MIN_FALLBACK_LENGTH = 4
DROPPED_SEQUENCE = ""
_UNIFORM_BYTES = 8


def _batchify_series(series, mp_batch_size):
    """Internal funciton for multiprocessing"""
    for i in range(0, len(series), mp_batch_size):
        yield series.iloc[i : i + mp_batch_size]


class BaseDecoyGenerator:
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
    def __init__(
        self,
        seed: int = 0,
        min_mass_delta: float = MIN_MASS_DELTA,
        max_mass_delta: float = MAX_MASS_DELTA,
        frozen_AAs: str = FROZEN_AAS,
        excluded_mutations: tuple = EXCLUDED_MUTATIONS,
    ):
        """
        Decoy generator replacing one inner residue of the target by a residue of similar mass.

        Unlike DIA-NN decoys, which send most residues to L at positions 2 and n-1, the replaced
        site and residue are spread like in the targets, so decoy fragments do not pile up on a
        few masses. The replacement is drawn from the targets' inner-residue composition (see
        `fit`), and the choice depends only on the seed and the sequence, so a sequence gets the
        same decoy in every batch and process.

        Peptides without a replaceable inner residue get the DIA-NN mutation if they have at least
        4 residues, otherwise an empty sequence, which `SpecLibDecoy` drops.

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
            Defaults to I to L, L to I, N to D and Q to E.
        """
        self.seed = seed
        self.min_mass_delta = min_mass_delta
        self.max_mass_delta = max_mass_delta
        self.frozen_AAs = frozen_AAs
        self.excluded_mutations = set(excluded_mutations)
        self._fallback = DIANNDecoyGenerator()
        # residue -> (replacement residues, their cumulative draw probabilities)
        self._replacements = None

    def fit(self, sequences: pd.Series) -> None:
        """Sets the replacement probabilities from the inner-residue composition of the targets."""
        composition = Counter()
        for sequence in sequences.unique():
            composition.update(sequence[1:-1])

        self._replacements = {}
        for original in STANDARD_AAS:
            if original in self.frozen_AAs:
                continue
            candidates = [
                aa
                for aa in STANDARD_AAS
                if composition[aa] > 0
                and aa not in self.frozen_AAs
                and original + aa not in self.excluded_mutations
                and self.min_mass_delta
                < abs(AA_ASCII_MASS[ord(aa)] - AA_ASCII_MASS[ord(original)])
                <= self.max_mass_delta
            ]
            if not candidates:
                continue
            weights = np.array([composition[aa] for aa in candidates], dtype=float)
            self._replacements[original] = (
                candidates,
                np.cumsum(weights) / weights.sum(),
            )

    def _uniforms(self, sequence: str) -> tuple:
        """Two uniform numbers in [0, 1) derived from the seed and the sequence only."""
        digest = hashlib.blake2b(
            f"{self.seed}:{sequence}".encode(), digest_size=2 * _UNIFORM_BYTES
        ).digest()
        scale = 2.0 ** (8 * _UNIFORM_BYTES)
        return (
            int.from_bytes(digest[:_UNIFORM_BYTES], "little") / scale,
            int.from_bytes(digest[_UNIFORM_BYTES:], "little") / scale,
        )

    def _decoy(self, sequence: str) -> str:
        if self._replacements is None:
            raise RuntimeError("Call fit() with the target sequences before decoying.")

        sites = [
            i for i in range(1, len(sequence) - 1) if sequence[i] in self._replacements
        ]
        if not sites:
            if len(sequence) >= MIN_FALLBACK_LENGTH:
                return self._fallback._decoy(sequence)
            return DROPPED_SEQUENCE

        site_draw, residue_draw = self._uniforms(sequence)
        site = sites[int(site_draw * len(sites))]
        candidates, cumulative = self._replacements[sequence[site]]
        index = min(
            int(np.searchsorted(cumulative, residue_draw, side="right")),
            len(candidates) - 1,
        )
        return sequence[:site] + candidates[index] + sequence[site + 1 :]


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

        if not multiprocessing or self._precursor_df.shape[0] < mp_batch_size:
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
            _equate_isoleucine_leucine(self.target_lib._precursor_df.sequence)
        )
        decoy_seqs = self._precursor_df.sequence
        self._precursor_df.drop(
            self._precursor_df.loc[
                _equate_isoleucine_leucine(decoy_seqs).isin(target_seqs)
                | (decoy_seqs == DROPPED_SEQUENCE)
            ].index,
            inplace=True,
        )


def _equate_isoleucine_leucine(sequences: pd.Series) -> pd.Series:
    return sequences.str.replace("I", "L", regex=False)


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
