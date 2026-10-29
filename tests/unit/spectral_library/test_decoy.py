"""Tests for the decoy sequence generators."""

import pandas as pd

from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.decoy import (
    DIANNDecoyGenerator,
    DIANNKeepProlineDecoyGenerator,
    decoy_lib_provider,
)


def test_keep_proline_decoy_mutates_the_residue_before_the_proline():
    assert DIANNKeepProlineDecoyGenerator()._decoy("AGLLDPK") == "ALLLEPK"


def test_keep_proline_decoy_equals_diann_decoy_without_proline_before_the_c_terminus():
    for sequence in ("AGLLDEK", "PEPTIDER", "APPLEPAR"):
        assert DIANNKeepProlineDecoyGenerator()._decoy(
            sequence
        ) == DIANNDecoyGenerator()._decoy(sequence)


def test_keep_proline_decoy_keeps_the_length_of_short_sequences():
    for sequence in ("AAPK", "AAAPK"):
        assert len(DIANNKeepProlineDecoyGenerator()._decoy(sequence)) == len(sequence)


def test_keep_proline_decoy_library_from_provider():
    # given
    target_lib = SpecLibBase()
    target_lib._precursor_df = pd.DataFrame({"sequence": ["AGLLDPK", "AGLLDEK"]})

    # when
    decoy_lib = decoy_lib_provider.get_decoy_lib("diann_keep_proline", target_lib)
    decoy_lib.decoy_sequence(multiprocessing=False)

    # then
    assert decoy_lib.precursor_df["sequence"].tolist() == ["ALLLEPK", "ALLLDDK"]
