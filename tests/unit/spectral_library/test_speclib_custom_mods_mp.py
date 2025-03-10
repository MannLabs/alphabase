from multiprocessing import cpu_count

import pandas as pd
import pytest

from alphabase.constants.modification import (
    MOD_DF,
    MOD_Composition,
    add_new_modifications,
    update_all_by_MOD_DF,
)
from alphabase.spectral_library.base import SpecLibBase


@pytest.fixture
def cleanup_test_mods():
    """Clean up test modifications after test."""
    # Keep track of test modifications
    test_mods = ["SpecLibMod@N-term", "SpecLibMod@T"]

    yield

    # Remove test modifications
    for mod in test_mods:
        if mod in MOD_DF.index:
            MOD_DF.drop(mod, inplace=True)
        if mod in MOD_Composition:
            del MOD_Composition[mod]

    # Update dictionaries
    update_all_by_MOD_DF()


@pytest.fixture
def sample_precursor_df_with_mods():
    """Create a sample precursor dataframe with modifications for testing."""
    data = {
        "sequence": ["PEPTIDE", "TESTPEPTIDE"],
        "mods": ["SpecLibMod@N-term", "SpecLibMod@N-term;SpecLibMod@T"],
        "mod_sites": ["0", "0;1"],
        "charge": [2, 3],
        "nAA": [7, 11],
        "precursor_mz": [400.2, 420.3],
    }
    return pd.DataFrame(data)


@pytest.fixture
def add_speclib_custom_modification(cleanup_test_mods):
    """Add a custom modification for testing SpecLibBase."""
    # Add a custom modification
    add_new_modifications(
        [
            ("SpecLibMod@N-term", "C(2)H(3)O(1)", "H(2)O(1)"),
            ("SpecLibMod@T", "C(2)H(3)O(1)", ""),
        ]
    )

    # Make sure MOD_Composition is updated
    update_all_by_MOD_DF()

    # Verify it was added
    assert "SpecLibMod@N-term" in MOD_DF.index
    assert "SpecLibMod@T" in MOD_DF.index
    assert "SpecLibMod@N-term" in MOD_Composition
    assert "SpecLibMod@T" in MOD_Composition


def test_speclib_with_custom_mods_mp(
    sample_precursor_df_with_mods, add_speclib_custom_modification
):
    """Test that SpecLibBase works with custom modifications in multiprocessing mode."""
    # Skip if only 1 CPU is available
    if cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Create a spectral library
    speclib = SpecLibBase()
    speclib.precursor_df = sample_precursor_df_with_mods.copy()

    # Calculate isotope intensities with multiprocessing
    # Use minimal isotope calculation (max_isotope=2)
    speclib.calc_precursor_isotope_intensity(max_isotope=2, mp_process_num=2)

    # Verify the calculation completed successfully with custom mods
    assert len(speclib.precursor_df) == len(sample_precursor_df_with_mods)
    assert "i_0" in speclib.precursor_df.columns
    assert "i_1" in speclib.precursor_df.columns

    # Verify results are reasonable (just basic sanity check)
    assert all(speclib.precursor_df["i_0"] > 0)
    assert all(speclib.precursor_df["i_1"] > 0)


def test_speclib_compare_single_vs_mp(
    sample_precursor_df_with_mods, add_speclib_custom_modification
):
    """Test that single and multiprocessing versions in SpecLibBase give consistent results with custom mods."""
    # Skip if only 1 CPU is available
    if cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Create two identical spectral libraries
    speclib_single = SpecLibBase()
    speclib_single.precursor_df = sample_precursor_df_with_mods.copy()

    speclib_mp = SpecLibBase()
    speclib_mp.precursor_df = sample_precursor_df_with_mods.copy()

    # Calculate isotope intensities with single process (minimal calculation)
    speclib_single.calc_precursor_isotope_intensity(
        max_isotope=2,
        mp_process_num=1,  # Force single process
    )

    # Calculate isotope intensities with multiprocessing (minimal calculation)
    speclib_mp.calc_precursor_isotope_intensity(max_isotope=2, mp_process_num=2)

    # Verify results are consistent between single and multi-process
    for col in ["i_0", "i_1"]:
        assert all(
            abs(speclib_single.precursor_df[col] - speclib_mp.precursor_df[col]) < 1e-5
        )
