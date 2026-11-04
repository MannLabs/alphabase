import pandas as pd
import pytest

from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.decoy import (
    NearMassMutationDecoyGenerator,
    decoy_lib_provider,
)

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
    assert decoys == ["PEPTDDEK", "LGAEDQEWR", "SAMPLEYFYK", "VHITAPR"]


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
