"""Tests for the decoy sequence generators."""

import numpy as np
import pandas as pd
import pytest

from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.decoy import (
    DIANNDecoyGenerator,
    DIANNInnerDecoyGenerator,
    DIANNKeepProlineDecoyGenerator,
    NearMassMutationDecoyGenerator,
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


def test_inner_decoy_mutates_the_third_and_the_third_to_last_residue():
    assert DIANNInnerDecoyGenerator()._decoy("AGLLDEVK") == "AGVLDDVK"


def test_inner_decoy_keeps_b2_and_y2():
    decoy = DIANNInnerDecoyGenerator()._decoy("LGEHNIDVLEGNEQFINAAK")
    assert decoy[:2] == "LG"
    assert decoy[-2:] == "AK"


def test_inner_decoy_moves_a_mutation_off_a_proline():
    assert DIANNInnerDecoyGenerator()._decoy("PEPTIDEK") == "PEPSIEEK"
    assert DIANNInnerDecoyGenerator()._decoy("AAAAPGK") == "AALLPGK"


def test_inner_decoy_keeps_the_n_terminal_proline_without_room_to_move():
    # the C-terminal mutation already moved onto the residue after the proline
    assert DIANNInnerDecoyGenerator()._decoy("AAPAPGK") == "AALLPGK"


@pytest.mark.parametrize("sequence", ["AAPK", "AAAPK", "AGLDPK", "AGLDEK"])
def test_inner_decoy_of_sequences_shorter_than_7_is_the_keep_proline_decoy(
    sequence,
):
    assert DIANNInnerDecoyGenerator()._decoy(
        sequence
    ) == DIANNKeepProlineDecoyGenerator()._decoy(sequence)


def test_inner_decoy_properties_on_random_sequences():
    generator = DIANNInnerDecoyGenerator()
    for sequence in _random_sequences(2000, min_length=6, max_length=30):
        decoy = generator._decoy(sequence)

        assert len(decoy) == len(sequence)
        assert decoy[:2] == sequence[:2]
        assert decoy[-2:] == sequence[-2:]
        assert sum(a != b for a, b in zip(decoy, sequence)) == 2


def test_inner_decoy_is_registered():
    assert decoy_lib_provider.decoy_dict["diann_inner"] is DIANNInnerDecoyGenerator


@pytest.mark.parametrize("multiprocessing", [False, True])
def test_inner_decoy_library_from_provider(multiprocessing):
    # given
    sequences = ["AGLLDEVK", "PEPTIDEK", *_random_sequences(40, 7, 20)]

    # when
    decoys = _decoy_library("diann_inner", sequences, multiprocessing)

    # then
    generator = DIANNInnerDecoyGenerator()
    expected = [generator._decoy(s) for s in sequences]
    assert decoys == [d for d in expected if d not in set(sequences)]
    assert decoys[:2] == ["AGVLDDVK", "PEPSIEEK"]


TARGETS = ["PEPTIDEK", "LGANDQEWR", "SAMPLERFYK", "VHITGPR"]
CHARGE = 2


def _fitted_generator(targets: list, seed: int = 0) -> NearMassMutationDecoyGenerator:
    generator = NearMassMutationDecoyGenerator(seed=seed)
    generator.fit(pd.Series(targets))
    return generator


def _decoys(generator: NearMassMutationDecoyGenerator, sequences: list) -> list:
    return list(generator(pd.Series(sequences)))


def _target_lib(sequences: list) -> SpecLibBase:
    lib = SpecLibBase(["b_z1", "y_z1"])
    lib._precursor_df = pd.DataFrame(
        {
            "sequence": sequences,
            "mods": "",
            "mod_sites": "",
            "charge": CHARGE,
        }
    )
    lib.calc_precursor_mz()
    return lib


def test_near_mass_decoy_replaces_one_inner_residue_by_one_of_similar_mass():
    # given
    generator = _fitted_generator(TARGETS)

    # when
    decoys = _decoys(generator, TARGETS)

    # then I->D, Q->H, E->D and I->T, each within 25 Da
    assert decoys == ["PEPTDDEK", "LGANDHEWR", "SAMPLDRFYK", "VHTTGPR"]


def test_near_mass_decoy_depends_on_seed():
    # given
    generator = _fitted_generator(TARGETS, seed=1)

    # when
    decoys = _decoys(generator, TARGETS)

    # then
    assert decoys == ["PEPTDDEK", "LGAEDQEWR", "SAMPLLRFYK", "VHITAPR"]


def test_near_mass_decoy_of_a_sequence_does_not_depend_on_the_batch():
    # given
    generator = _fitted_generator(TARGETS)

    # when
    decoys = _decoys(generator, ["SAMPLERFYK"])

    # then same decoy as within the batch of all targets
    assert decoys == ["SAMPLDRFYK"]


@pytest.mark.parametrize(
    "targets, sequence",
    [
        (["ACKR"], "GCR"),
        (["ACKR"], "GKR"),
        (["ARR"], "GRR"),
        (["ARR"], "GFR"),
        (["ALIR"], "GIR"),
        (["ALIR"], "GLR"),
        (["ANDR"], "GNR"),
        (["ANDR"], "GDR"),
        (["AQER"], "GQR"),
        (["AQER"], "GER"),
    ],
)
def test_near_mass_decoy_never_draws_frozen_residues_or_excluded_mutations(
    targets, sequence
):
    # given the only other inner residue of the targets is frozen or an excluded replacement
    generator = _fitted_generator(targets)

    # when
    decoys = _decoys(generator, [sequence])

    # then nothing is replaceable and the short peptide is dropped
    assert decoys == [""]


def test_near_mass_decoy_draws_only_residues_present_in_targets():
    # given F is the only replacement of Y among the inner residues of the targets
    generator = _fitted_generator(["GAFR"])

    # when
    decoys = _decoys(generator, ["GYR"])

    # then
    assert decoys == ["GFR"]


def test_near_mass_decoy_keeps_proline_at_second_to_last_position():
    # given
    generator = _fitted_generator(["GASPR"])

    # when
    decoys = _decoys(generator, ["GGPR"])

    # then
    assert decoys == ["GAPR"]


def test_near_mass_decoy_does_not_place_proline_at_second_to_last_position():
    # given A and P are the only replacements of S among the inner residues of the targets,
    # and with this seed P would be drawn without the rule
    generator = _fitted_generator(["GSPAK"], seed=1)

    # when
    decoys = _decoys(generator, ["GSK"])

    # then
    assert decoys == ["GAK"]


def test_near_mass_decoy_falls_back_to_diann_mutation_without_replaceable_residue():
    # given
    generator = _fitted_generator(TARGETS)

    # when
    decoys = _decoys(generator, ["ACKCR"])

    # then
    assert decoys == ["ASKSR"]


def test_near_mass_decoy_drops_short_peptide_without_replaceable_residue():
    # given
    generator = _fitted_generator(TARGETS)

    # when
    decoys = _decoys(generator, ["ACR"])

    # then
    assert decoys == [""]


def test_near_mass_decoy_requires_fit():
    # given
    generator = NearMassMutationDecoyGenerator()

    # when / then
    with pytest.raises(RuntimeError):
        generator(pd.Series(["PEPTIDEK"]))


def test_decoy_lib_drops_decoys_equal_to_a_target_up_to_isoleucine_leucine():
    # given
    target_lib = _target_lib(["LAGR", "GAIR", "PEPTIDEK"])
    decoy_lib = decoy_lib_provider.get_decoy_lib("pseudo_reverse", target_lib)

    # when
    decoy_lib.decoy_sequence(multiprocessing=False)

    # then GALR and IAGR equal GAIR and LAGR up to I/L
    assert list(decoy_lib.precursor_df.sequence) == ["EDITPEPK"]


def test_near_mass_decoy_lib_recomputes_precursor_mz():
    # given
    target_lib = _target_lib(TARGETS)
    decoy_lib = decoy_lib_provider.get_decoy_lib("near_mass_mutation", target_lib)

    # when
    decoy_lib.decoy_sequence(multiprocessing=False)
    decoy_lib.calc_precursor_mz()

    # then
    assert list(decoy_lib.precursor_df.sequence) == [
        "VHTTGPR",
        "PEPTDDEK",
        "LGANDHEWR",
        "SAMPLDRFYK",
    ]
    assert list(decoy_lib.precursor_df.precursor_mz) == pytest.approx(
        [384.211569, 465.706180, 549.259779, 614.313166]
    )


def test_near_mass_decoy_lib_with_multiprocessing_matches_single_process():
    # given
    target_lib = _target_lib(TARGETS)
    decoy_lib = decoy_lib_provider.get_decoy_lib("near_mass_mutation", target_lib)

    # when
    decoy_lib.decoy_sequence(multiprocessing=True, mp_batch_size=2, mp_process_num=2)

    # then
    assert list(decoy_lib.precursor_df.sequence) == [
        "VHTTGPR",
        "PEPTDDEK",
        "LGANDHEWR",
        "SAMPLDRFYK",
    ]
