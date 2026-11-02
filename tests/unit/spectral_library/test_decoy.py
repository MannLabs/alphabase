import numpy as np
import pandas as pd
import pytest

from alphabase.constants.aa import AA_ASCII_MASS
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.decoy import (
    NearMassMutationDecoyGenerator,
    decoy_lib_provider,
)

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
FORBIDDEN_MUTATIONS = {("I", "L"), ("L", "I"), ("N", "D"), ("Q", "E")}
MIN_MASS_DELTA = 0.05
MAX_MASS_DELTA = 25.0
CHARGE = 2


def _random_peptides(n: int, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(7, 25, size=n)
    return pd.Series(
        [
            "".join(rng.choice(list(STANDARD_AAS), size=length)) + "K"
            for length in lengths
        ]
    )


def _fitted_generator(sequences: pd.Series, **kwargs) -> NearMassMutationDecoyGenerator:
    generator = NearMassMutationDecoyGenerator(**kwargs)
    generator.fit(sequences)
    return generator


def _mutations(target: str, decoy: str) -> list:
    return [(i, a, b) for i, (a, b) in enumerate(zip(target, decoy)) if a != b]


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


def test_near_mass_decoy_replaces_one_inner_residue_within_mass_range():
    # given
    targets = _random_peptides(500)
    generator = _fitted_generator(targets)

    # when
    decoys = generator(targets)

    # then
    for target, decoy in zip(targets, decoys):
        assert len(decoy) == len(target)
        mutations = _mutations(target, decoy)
        assert len(mutations) == 1
        position, original, replacement = mutations[0]
        assert 0 < position < len(target) - 1
        mass_delta = abs(AA_ASCII_MASS[ord(replacement)] - AA_ASCII_MASS[ord(original)])
        assert MIN_MASS_DELTA < mass_delta <= MAX_MASS_DELTA


def test_near_mass_decoy_never_touches_frozen_or_forbidden_residues():
    # given
    targets = _random_peptides(2000)
    generator = _fitted_generator(targets)

    # when
    decoys = generator(targets)

    # then
    for target, decoy in zip(targets, decoys):
        ((_, original, replacement),) = _mutations(target, decoy)
        assert original not in "CK"
        assert replacement not in "CK"
        assert (original, replacement) not in FORBIDDEN_MUTATIONS


def test_near_mass_decoy_draws_only_residues_present_in_targets():
    # given
    targets = _random_peptides(500).str.replace("W", "F")
    generator = _fitted_generator(targets)

    # when
    decoys = generator(targets)

    # then
    assert not decoys.str.contains("W").any()


def test_near_mass_decoy_is_deterministic_and_seed_dependent():
    # given
    targets = _random_peptides(200)

    # when
    first = _fitted_generator(targets, seed=0)(targets)
    repeated = _fitted_generator(targets, seed=0)(targets)
    other_seed = _fitted_generator(targets, seed=1)(targets)

    # then
    pd.testing.assert_series_equal(first, repeated)
    assert (first != other_seed).any()


def test_near_mass_decoy_of_a_sequence_does_not_depend_on_the_batch():
    # given
    targets = _random_peptides(200)
    generator = _fitted_generator(targets)

    # when
    whole = generator(targets)
    in_batches = pd.concat([generator(targets.iloc[:50]), generator(targets.iloc[50:])])

    # then
    pd.testing.assert_series_equal(whole, in_batches)


def test_near_mass_decoy_falls_back_to_diann_mutation_without_replaceable_residue():
    # given
    targets = _random_peptides(100)
    generator = _fitted_generator(targets)

    # when
    decoy = generator(pd.Series(["ACKCR"])).iloc[0]

    # then
    assert decoy == "ASKSR"


def test_near_mass_decoy_drops_short_peptide_without_replaceable_residue():
    # given
    targets = _random_peptides(100)
    generator = _fitted_generator(targets)

    # when
    decoy = generator(pd.Series(["ACR"])).iloc[0]

    # then
    assert decoy == ""


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

    # then
    assert "GALR" not in decoy_lib.precursor_df.sequence.values
    assert "EDITPEPK" in decoy_lib.precursor_df.sequence.values


def test_near_mass_decoy_lib_recomputes_precursor_mz():
    # given
    target_lib = _target_lib(list(_random_peptides(50)))
    # carried into the decoy library, as calc_precursor_mz may reorder rows
    target_lib._precursor_df["target_sequence"] = target_lib.precursor_df.sequence
    target_lib._precursor_df["target_mz"] = target_lib.precursor_df.precursor_mz
    decoy_lib = decoy_lib_provider.get_decoy_lib("near_mass_mutation", target_lib)

    # when
    decoy_lib.decoy_sequence(multiprocessing=False)
    decoy_lib.calc_precursor_mz()

    # then
    assert len(decoy_lib.precursor_df) > 0
    for row in decoy_lib.precursor_df.itertuples():
        ((_, original, replacement),) = _mutations(row.target_sequence, row.sequence)
        mass_delta = AA_ASCII_MASS[ord(replacement)] - AA_ASCII_MASS[ord(original)]
        assert row.precursor_mz == pytest.approx(
            row.target_mz + mass_delta / CHARGE, abs=1e-6
        )


def test_near_mass_decoy_lib_multiprocessing_matches_single_process():
    # given
    target_lib = _target_lib(list(_random_peptides(40)))
    single = decoy_lib_provider.get_decoy_lib("near_mass_mutation", target_lib)
    parallel = decoy_lib_provider.get_decoy_lib("near_mass_mutation", target_lib)

    # when
    single.decoy_sequence(multiprocessing=False)
    parallel.decoy_sequence(multiprocessing=True, mp_batch_size=10, mp_process_num=2)

    # then
    pd.testing.assert_series_equal(
        single.precursor_df.sequence, parallel.precursor_df.sequence
    )
