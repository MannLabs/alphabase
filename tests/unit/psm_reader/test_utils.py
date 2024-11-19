"""Tests for the utility functions in the psm_reader module."""

import pandas as pd

from alphabase.psm_reader.utils import get_mod_set, keep_modifications


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


def test_get_mod_set_with_empty_list():
    assert get_mod_set([]) == set()


def test_get_mod_set_with_single_modification():
    assert get_mod_set(["K(Acetyl)"]) == {"K(Acetyl)", "K[Acetyl]"}


def test_get_mod_set_with_multiple_modifications():
    assert get_mod_set(["K(Acetyl)", "(Phospho)"]) == {
        "K(Acetyl)",
        "K[Acetyl]",
        "(Phospho)",
        "_(Phospho)",
        "[Phospho]",
        "_[Phospho]",
    }


def test_get_mod_set_with_modifications_starting_with_underscore():
    assert get_mod_set(["_Phospho"]) == {"_Phospho", "Phospho"}


def test_get_mod_set_with_modifications_starting_with_bracket():
    assert get_mod_set(["(Phospho)"]) == {
        "(Phospho)",
        "_(Phospho)",
        "[Phospho]",
        "_[Phospho]",
    }


def test_get_mod_set_with_modifications_starting_with_square_bracket():
    assert get_mod_set(["[Phospho]"]) == {
        "_[Phospho]",
        "(Phospho)",
        "_(Phospho)",
        "[Phospho]",
    }


def test_get_mod_set_with_modifications_starting_with_underscore_square_bracket():
    assert get_mod_set(["_[Phospho]"]) == {"[Phospho]", "_[Phospho]", "_(Phospho)"}
