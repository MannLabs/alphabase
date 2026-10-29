"""Tests for the decoy sequence generators."""

import numpy as np
import pandas as pd
import pytest

from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.decoy import (
    DIANNDecoyGenerator,
    DIANNKeepProlineDecoyGenerator,
    decoy_lib_provider,
)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def _random_sequences(n: int, min_length: int, max_length: int, seed: int = 0):
    """Random tryptic-like sequences, every second one ending in P[KR]."""
    rng = np.random.default_rng(seed)
    sequences = []
    for i in range(n):
        body = "".join(rng.choice(AMINO_ACIDS, rng.integers(min_length, max_length)))
        c_term = rng.choice(["K", "R"])
        sequences.append(body[:-1] + "P" + c_term if i % 2 else body + c_term)
    return sequences


def _decoy_library(name: str, sequences: list, multiprocessing: bool) -> list:
    target_lib = SpecLibBase()
    target_lib._precursor_df = pd.DataFrame({"sequence": sequences})
    decoy_lib = decoy_lib_provider.get_decoy_lib(name, target_lib)
    decoy_lib.decoy_sequence(
        multiprocessing=multiprocessing, mp_batch_size=10, mp_process_num=2
    )
    return decoy_lib.precursor_df["sequence"].tolist()


def test_keep_proline_decoy_mutates_the_residue_before_the_proline():
    assert DIANNKeepProlineDecoyGenerator()._decoy("AGLLDPK") == "ALLLEPK"


def test_keep_proline_decoy_mutates_a_proline_before_the_kept_proline():
    assert DIANNKeepProlineDecoyGenerator()._decoy("AGLLPPR") == "ALLLLPR"


def test_keep_proline_decoy_of_a_5_residue_sequence():
    assert DIANNKeepProlineDecoyGenerator()._decoy("AGLPK") == "ALVPK"


@pytest.mark.parametrize("sequence", ["AGLLDEK", "PEPTIDER", "APPLEPAR", "PAGLLDEK"])
def test_keep_proline_decoy_equals_diann_decoy_without_proline_before_the_c_terminus(
    sequence,
):
    assert DIANNKeepProlineDecoyGenerator()._decoy(
        sequence
    ) == DIANNDecoyGenerator()._decoy(sequence)


@pytest.mark.parametrize("sequence", ["AAPK", "PAPR"])
def test_keep_proline_decoy_of_sequences_shorter_than_5_is_the_diann_decoy(sequence):
    assert DIANNKeepProlineDecoyGenerator()._decoy(
        sequence
    ) == DIANNDecoyGenerator()._decoy(sequence)


def test_keep_proline_decoy_properties_on_random_sequences():
    generator = DIANNKeepProlineDecoyGenerator()
    for sequence in _random_sequences(2000, min_length=3, max_length=30):
        decoy = generator._decoy(sequence)

        assert len(decoy) == len(sequence)
        assert decoy[0] == sequence[0]
        assert decoy[-1] == sequence[-1]
        if len(sequence) >= 5 and sequence[-2] == "P":
            assert decoy[-2:] == sequence[-2:]
            assert sum(a != b for a, b in zip(decoy, sequence)) == 2
        else:
            assert decoy == DIANNDecoyGenerator()._decoy(sequence)


def test_keep_proline_decoy_is_registered():
    assert (
        decoy_lib_provider.decoy_dict["diann_keep_proline"]
        is DIANNKeepProlineDecoyGenerator
    )


@pytest.mark.parametrize("multiprocessing", [False, True])
def test_keep_proline_decoy_library_from_provider(multiprocessing):
    # given
    sequences = ["AGLLDPK", "AGLLDEK", *_random_sequences(40, 7, 20)]

    # when
    decoys = _decoy_library("diann_keep_proline", sequences, multiprocessing)

    # then
    generator = DIANNKeepProlineDecoyGenerator()
    expected = [generator._decoy(s) for s in sequences]
    assert decoys == [d for d in expected if d not in set(sequences)]
    assert decoys[:2] == ["ALLLEPK", "ALLLDDK"]
