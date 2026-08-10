import os

import numpy as np
import pytest

import alphabase.constants.modification as modification
from alphabase.constants._const import CONST_FILE_FOLDER
from alphabase.constants.modification import (
    MOD_DF,
    MOD_LOSS_IMPORTANCE,
    MOD_MASS,
    MOD_Composition,
    add_modifications_for_lower_case_AA,
    add_new_modifications,
    calc_modification_mass,
    calc_modloss_mass,
    load_mod_df,
    update_all_by_MOD_DF,
)

pytestmark = pytest.mark.requires_numba

# Store original modification state for restoration at the end of all tests
ORIGINAL_MOD_DF = MOD_DF.copy()


def setup_module(module):
    """Set up the module before any tests run."""
    # Nothing special needed for setup
    pass


def teardown_module(module):
    """Clean up after all tests in this module have run."""
    # Restore the original MOD_DF
    modification.MOD_DF = ORIGINAL_MOD_DF.copy()
    update_all_by_MOD_DF()


@pytest.fixture(scope="function")
def reset_modifications():
    """Reset modification state after each test."""
    # Store the original state
    original_df = MOD_DF.copy()

    yield

    # Restore original state
    modification.MOD_DF = original_df
    update_all_by_MOD_DF()


def test_calc_modification_mass(reset_modifications):
    """Test calculating modification mass."""
    seq = "AGHCEWQMK"
    mod_names = ["Acetyl@Protein_N-term", "Carbamidomethyl@C", "Oxidation@M"]
    mod_sites = [0, 4, 8]

    # Note: The indices are 0-based for sites, but for non-terminal mods,
    # the site is 1-based, so we subtract 1 for the array position
    expected = [42.01056468, 0, 0, 57.02146372, 0, 0, 0, 15.99491462, 0]
    actual = calc_modification_mass(len(seq), mod_names, mod_sites)

    assert np.allclose(actual, expected)


def test_calc_modloss_mass_level_0(reset_modifications):
    """Test calculating modification loss mass with importance level 0."""
    mod_names = ["Oxidation@M", "Phospho@S", "Carbamidomethyl@C"]
    mod_sites = [0, 4, 8]

    # Load with modloss_importance_level=0
    load_mod_df(modloss_importance_level=0)

    expected = [
        63.99828592,
        63.99828592,
        63.99828592,
        97.97689557,
        97.97689557,
        97.97689557,
        97.97689557,
        97.97689557,
        97.97689557,
    ]
    actual = calc_modloss_mass(10, mod_names, mod_sites, True)

    assert np.allclose(actual, expected)


def test_calc_modloss_mass_level_1(reset_modifications):
    """Test calculating modification loss mass with importance level 1."""
    mod_names = ["Oxidation@M", "Phospho@S", "Carbamidomethyl@C"]
    mod_sites = [0, 4, 8]

    # Load with modloss_importance_level=1
    load_mod_df(
        tsv=os.path.join(CONST_FILE_FOLDER, "modification.tsv"),
        modloss_importance_level=1,
    )

    expected = [
        0,
        0,
        0,
        97.97689557,
        97.97689557,
        97.97689557,
        97.97689557,
        97.97689557,
        97.97689557,
    ]
    actual = calc_modloss_mass(10, mod_names, mod_sites, True)

    assert np.allclose(actual, expected)


def test_calc_modloss_mass_backward(reset_modifications):
    """Test calculating modification loss mass backward."""
    mod_names = ["Oxidation@M", "Phospho@S", "Carbamidomethyl@C"]
    mod_sites = [0, 4, 8]

    expected = [97.97689557, 97.97689557, 97.97689557, 0, 0, 0, 0, 0, 0]
    actual = calc_modloss_mass(10, mod_names, mod_sites, False)

    assert np.allclose(actual, expected)


def test_add_modifications_for_lower_case_AA(reset_modifications):
    """Test adding modifications for lower case amino acids."""
    # Add lower case modifications
    add_modifications_for_lower_case_AA()

    # Check there are now modifications with lowercase AA
    lowercase_mods = modification.MOD_DF[
        modification.MOD_DF.index.str.contains("@[a-z]")
    ]
    assert not lowercase_mods.empty

    # Verify some specific lowercase modifications exist
    assert any(mod.endswith("@s") for mod in modification.MOD_DF.index)
    assert any(mod.endswith("@t") for mod in modification.MOD_DF.index)


def test_add_new_modifications_list(reset_modifications):
    """Test adding new modifications using a list."""
    # Remember the count of user-added modifications
    prev_value = (modification.MOD_DF.classification == "User-added").sum()

    # Add new modifications
    add_new_modifications([("Hello@S", "H(2)"), ("World@S", "O(10)", "O(3)")])

    # Check that two new modifications were added
    assert (modification.MOD_DF.classification == "User-added").sum() - prev_value == 2

    # Check that the modifications exist
    assert "Hello@S" in modification.MOD_DF.mod_name.values
    assert "Hello@S" in modification.MOD_DF.index
    assert "World@S" in modification.MOD_DF.mod_name.values
    assert "World@S" in modification.MOD_DF.index

    # Check modification loss
    assert modification.MOD_DF.loc["World@S", "modloss"] > 0
    assert modification.MOD_DF.loc["World@S", "modloss_importance"] > 0

    # Check composition dictionaries
    assert "Hello@S" in MOD_Composition
    assert "World@S" in MOD_MASS


def test_add_new_modifications_dict(reset_modifications):
    """Test adding new modifications using a dictionary."""
    # Remember the count of user-added modifications
    prev_value = (modification.MOD_DF.classification == "User-added").sum()

    # Add new modifications
    add_new_modifications(
        {
            "Hi@S": {"composition": "H(2)"},
            "AlphaX@S": {"composition": "O(10)", "modloss_composition": "O(3)"},
        }
    )

    # Check that two new modifications were added (on top of the ones from the previous test)
    assert (modification.MOD_DF.classification == "User-added").sum() - prev_value == 2

    # Check that the modifications exist
    assert "Hi@S" in modification.MOD_DF.mod_name.values
    assert "Hi@S" in modification.MOD_DF.index
    assert "AlphaX@S" in modification.MOD_DF.mod_name.values
    assert "AlphaX@S" in modification.MOD_DF.index

    # Check modification loss
    assert modification.MOD_DF.loc["AlphaX@S", "modloss"] > 0
    assert modification.MOD_DF.loc["AlphaX@S", "modloss_importance"] > 0

    # Check composition dictionaries
    assert "Hi@S" in MOD_Composition
    assert "AlphaX@S" in MOD_MASS
    assert "AlphaX@S" in MOD_LOSS_IMPORTANCE

    # Check unimod_mass is 0 for user-added modifications
    assert modification.MOD_DF.loc["AlphaX@S", "unimod_mass"] == 0
