import numpy as np
import pytest

from alphabase.spectral_library import near_mass_mutation

ALL_STANDARD_AAS_ONCE = "A" + near_mass_mutation.STANDARD_AAS + "A"


def _encode(sequences: list) -> np.ndarray:
    return near_mass_mutation.encode(np.array(sequences, dtype=object))


def _uniform_tables() -> near_mass_mutation.ReplacementTables:
    composition = near_mass_mutation.inner_composition(_encode([ALL_STANDARD_AAS_ONCE]))
    return near_mass_mutation.build_replacement_tables(composition)


def _candidates(tables, table: int, residue: str) -> str:
    n_candidates = tables.n_candidates[table, ord(residue)]
    return bytes(tables.candidates[table, ord(residue), :n_candidates]).decode()


def test_encode_pads_codes_with_zeros():
    # given
    sequences = ["AR", "PEPTIDEK"]

    # when
    codes = _encode(sequences)

    # then
    assert codes.tolist() == [
        [65, 82, 0, 0, 0, 0, 0, 0],
        [80, 69, 80, 84, 73, 68, 69, 75],
    ]


def test_decode_strips_padding():
    # given
    codes = _encode(["AR", "PEPTIDEK"])

    # when
    sequences = near_mass_mutation.decode(codes)

    # then
    assert sequences.tolist() == ["AR", "PEPTIDEK"]


def test_inner_composition_counts_all_but_terminal_residues():
    # given
    codes = _encode(["GASPR", "AK"])

    # when
    composition = near_mass_mutation.inner_composition(codes)

    # then
    assert {
        chr(code): int(count) for code, count in enumerate(composition) if count
    } == {
        "A": 1,
        "P": 1,
        "S": 1,
    }


@pytest.mark.parametrize(
    "table, residue, expected",
    [
        (near_mass_mutation.INNER_TABLE, "S", "APTV"),
        (near_mass_mutation.INNER_TABLE, "D", "EHILMPQTV"),
        (near_mass_mutation.INNER_TABLE, "P", "DILNSTV"),
        (near_mass_mutation.INNER_TABLE, "C", ""),
        (near_mass_mutation.INNER_TABLE, "R", ""),
        (near_mass_mutation.INNER_TABLE, "H", "DEFILMNQ"),
        (near_mass_mutation.PENULTIMATE_TABLE, "S", "ATV"),
        (near_mass_mutation.PENULTIMATE_TABLE, "D", "EHILMQTV"),
        (near_mass_mutation.PENULTIMATE_TABLE, "P", ""),
    ],
)
def test_replacement_tables_list_candidates_in_mass_window(table, residue, expected):
    # given
    tables = _uniform_tables()

    # when
    candidates = _candidates(tables, table, residue)

    # then
    assert candidates == expected


def test_replacement_tables_weight_candidates_by_composition():
    # given
    tables = _uniform_tables()

    # when
    cumulative = tables.cumulative[near_mass_mutation.INNER_TABLE, ord("S"), :4]

    # then
    assert cumulative.tolist() == [0.25, 0.5, 0.75, 1.0]


def test_draw_uniforms_depends_only_on_sequence_and_seed():
    # given
    codes = _encode(["AR", "PEPTIDEK"])

    # when
    uniforms = near_mass_mutation.draw_uniforms(codes, seed=0)

    # then the padded AR draws the same as AR alone
    assert uniforms.tolist() == [
        [0.63775358070047, 0.5631637724434956],
        [0.5470053053228655, 0.029938529866293306],
    ]
    assert near_mass_mutation.draw_uniforms(_encode(["AR"]), seed=0).tolist() == [
        [0.63775358070047, 0.5631637724434956]
    ]


@pytest.mark.parametrize(
    "sequence, uniforms, expected_sequence, expected_has_site",
    [
        ("GASPR", [0.0, 0.0], "GGSPR", True),
        ("GASPR", [0.99, 0.99], "GAVPR", True),
        ("GCKR", [0.5, 0.5], "GCKR", False),
    ],
)
def test_mutate_replaces_drawn_site_by_drawn_candidate(
    sequence, uniforms, expected_sequence, expected_has_site
):
    # given the first draw picks the site among A and S (P is second-to-last), the second the candidate
    codes = _encode([sequence])

    # when
    mutated, has_site = near_mass_mutation.mutate(
        codes, np.array([uniforms]), _uniform_tables()
    )

    # then
    assert near_mass_mutation.decode(mutated).tolist() == [expected_sequence]
    assert has_site.tolist() == [expected_has_site]
