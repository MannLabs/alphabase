"""Tests for the utility functions in the psm_reader module."""

import pandas as pd

from alphabase.psm_reader.utils import (
    get_extended_modifications,
    keep_modifications,
    translate_modifications,
)


def test_translate_other_modification_with_empty_mod_str():
    assert translate_modifications("", {"ModA": "ModA@X"}) == ("", [])


def test_translate_other_modification_with_all_mods_in_dict():
    assert translate_modifications(
        "ModA;ModB", {"ModA": "ModA@X", "ModB": "ModB@Y"}
    ) == ("ModA@X;ModB@Y", [])


def test_translate_other_modification_with_some_mods_not_in_dict():
    assert translate_modifications("ModA;UnknownMod", {"ModA": "ModA@X"}) == (
        pd.NA,
        ["UnknownMod"],
    )


def test_translate_other_modification_with_no_mods_in_dict():
    assert translate_modifications("UnknownMod1;UnknownMod2", {"ModA": "ModA@X"}) == (
        pd.NA,
        ["UnknownMod1", "UnknownMod2"],
    )


def test_translate_other_modification_with_empty_mod_dict():
    assert translate_modifications("ModA;ModB", {}) == (pd.NA, ["ModA", "ModB"])


def test_keep_modifications_with_empty_mod_str():
    assert keep_modifications("", {"Oxidation@M", "Phospho@S"}) == ""


def test_keep_modifications_with_all_mods_in_set():
    assert (
        keep_modifications("Oxidation@M;Phospho@S", {"Oxidation@M", "Phospho@S"})
        == "Oxidation@M;Phospho@S"
    )


def test_keep_modifications_with_some_mods_not_in_set():
    assert (
        keep_modifications("Oxidation@M;UnknownMod", {"Oxidation@M", "Phospho@S"})
        is pd.NA
    )


def test_keep_modifications_with_no_mods_in_set():
    assert (
        keep_modifications("UnknownMod1;UnknownMod2", {"Oxidation@M", "Phospho@S"})
        is pd.NA
    )


def test_keep_modifications_with_empty_mod_set():
    assert keep_modifications("Oxidation@M;Phospho@S", set()) is pd.NA


def test_get_extended_modifications_with_empty_list():
    assert get_extended_modifications([]) == []


def test_get_extended_modifications_with_single_modification():
    assert get_extended_modifications(["K(Acetyl)"]) == ["K(Acetyl)", "K[Acetyl]"]


def test_get_extended_modifications_with_multiple_modifications():
    assert get_extended_modifications(["K(Acetyl)", "(Phospho)"]) == [
        "(Phospho)",
        "K(Acetyl)",
        "K[Acetyl]",
        "[Phospho]",
        "_(Phospho)",
        "_[Phospho]",
    ]


def test_get_extended_modifications_with_modifications_starting_with_underscore():
    assert get_extended_modifications(["_Phospho"]) == ["Phospho", "_Phospho"]


def test_get_extended_modifications_with_modifications_starting_with_bracket():
    assert get_extended_modifications(["(Phospho)"]) == [
        "(Phospho)",
        "[Phospho]",
        "_(Phospho)",
        "_[Phospho]",
    ]


def test_get_extended_modifications_with_modifications_starting_with_square_bracket():
    assert get_extended_modifications(["[Phospho]"]) == [
        "(Phospho)",
        "[Phospho]",
        "_(Phospho)",
        "_[Phospho]",
    ]


def test_get_extended_modifications_with_modifications_starting_with_letter_square_bracket():
    assert get_extended_modifications(["K[Dimethyl]"]) == ["K(Dimethyl)", "K[Dimethyl]"]
