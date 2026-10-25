"""Tests for the modification lookup used when translating spectral libraries."""

import pytest

from alphabase.constants.modification import (
    MOD_DF,
    add_new_modifications,
    update_all_by_MOD_DF,
)
from alphabase.spectral_library.translate import mod_to_unimod_dict

TEST_MODS = ["LateUnimod@R", "LateNoUnimod@R", "Swapped1@R", "Swapped2@R"]


@pytest.fixture
def cleanup_test_mods():
    """Remove the test modifications after each test."""
    yield
    for mod in TEST_MODS:
        if mod in MOD_DF.index:
            MOD_DF.drop(mod, inplace=True)
    update_all_by_MOD_DF()


def test_mod_to_unimod_dict_contains_builtin_mods():
    """Built-in modifications map to their UniMod accession."""
    assert mod_to_unimod_dict["Carbamidomethyl@C"] == "UniMod:4"
    assert mod_to_unimod_dict["Oxidation@M"] == "UniMod:35"


def test_mod_to_unimod_dict_sees_mods_added_after_import(cleanup_test_mods):
    """A modification added after the module was imported is found."""
    # Given a modification added after translate was imported
    add_new_modifications({"LateUnimod@R": {"composition": "C(2)", "unimod_id": 21}})

    # When / Then
    assert mod_to_unimod_dict["LateUnimod@R"] == "UniMod:21"
    assert "LateUnimod@R" in mod_to_unimod_dict


def test_mod_to_unimod_dict_skips_mod_without_unimod_id(cleanup_test_mods):
    """A modification without a UniMod id is absent instead of exported as UniMod:0."""
    # Given a modification added without a UniMod id
    add_new_modifications({"LateNoUnimod@R": {"composition": "C(2)"}})

    # When / Then
    assert "LateNoUnimod@R" not in mod_to_unimod_dict
    with pytest.raises(KeyError):
        mod_to_unimod_dict["LateNoUnimod@R"]


def test_mod_to_unimod_dict_sees_swap_that_keeps_mod_count(cleanup_test_mods):
    """Removing one modification and adding another is picked up, although MOD_DF keeps its length."""
    # Given a modification that was looked up once
    add_new_modifications({"Swapped1@R": {"composition": "C(2)", "unimod_id": 21}})
    assert mod_to_unimod_dict["Swapped1@R"] == "UniMod:21"

    # When it is replaced by another one, leaving the number of modifications unchanged
    MOD_DF.drop("Swapped1@R", inplace=True)
    update_all_by_MOD_DF()
    add_new_modifications({"Swapped2@R": {"composition": "C(2)", "unimod_id": 35}})

    # Then
    assert "Swapped1@R" not in mod_to_unimod_dict
    assert mod_to_unimod_dict["Swapped2@R"] == "UniMod:35"


def test_mod_to_unimod_dict_sees_changed_unimod_id(cleanup_test_mods):
    """Changing the UniMod id of an existing modification is picked up."""
    # Given a modification that was looked up once
    add_new_modifications({"LateUnimod@R": {"composition": "C(2)", "unimod_id": 21}})
    assert mod_to_unimod_dict["LateUnimod@R"] == "UniMod:21"

    # When its UniMod id is edited in MOD_DF
    MOD_DF.loc["LateUnimod@R", "unimod_id"] = 35
    update_all_by_MOD_DF()

    # Then
    assert mod_to_unimod_dict["LateUnimod@R"] == "UniMod:35"
