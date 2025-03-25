from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import pytest

from alphabase.constants.modification import (
    MOD_DF,
    MOD_Composition,
    add_new_modifications,
    update_all_by_MOD_DF,
)
from alphabase.peptide.precursor import (
    calc_precursor_isotope_intensity,
    calc_precursor_isotope_intensity_mp,
)


@pytest.fixture
def sample_precursor_df_with_mods():
    """Create a sample precursor dataframe with modifications for testing."""
    data = {
        "sequence": ["PEPTIDE", "TESTPEPTIDE"],
        "mods": ["CustomMod@N-term", "CustomMod@N-term;CustomMod@T"],
        "mod_sites": ["0", "0;1"],
        "charge": [2, 3],
        "nAA": [7, 11],
        "precursor_mz": [400.2, 420.3],
    }
    return pd.DataFrame(data)


@pytest.fixture
def cleanup_test_mods():
    """Clean up test modifications after test."""
    # Keep track of test modifications
    test_mods = [
        "CustomMod@N-term",
        "CustomMod@T",
        "CustomMod1@N-term",
        "CustomMod2@S",
    ]

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
def add_custom_modification(cleanup_test_mods):
    """Add a custom modification for testing."""
    # Add a custom modification
    add_new_modifications(
        [
            ("CustomMod@N-term", "C(2)H(3)O(1)", "H(2)O(1)"),
            ("CustomMod@T", "C(2)H(3)O(1)", ""),
        ]
    )

    # Make sure MOD_Composition is updated
    update_all_by_MOD_DF()

    # Verify it was added
    assert "CustomMod@N-term" in MOD_DF.index
    assert "CustomMod@T" in MOD_DF.index
    assert "CustomMod@N-term" in MOD_Composition
    assert "CustomMod@T" in MOD_Composition


@pytest.fixture
def add_multiple_custom_modifications(cleanup_test_mods):
    """Add multiple custom modifications for testing."""
    # Add custom modifications
    add_new_modifications(
        [
            ("CustomMod1@N-term", "C(2)H(3)O(1)", "H(2)O(1)"),
            ("CustomMod2@S", "C(3)H(5)O(2)", ""),
        ]
    )

    # Make sure MOD_Composition is updated
    update_all_by_MOD_DF()

    # Verify they were added
    assert "CustomMod1@N-term" in MOD_DF.index
    assert "CustomMod2@S" in MOD_DF.index
    assert "CustomMod1@N-term" in MOD_Composition
    assert "CustomMod2@S" in MOD_Composition


def test_custom_mods_in_mp(sample_precursor_df_with_mods, add_custom_modification):
    """Test that custom modifications work in multiprocessing."""
    # Skip if only 1 CPU is available
    if cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Run the multi-process version with minimal isotope calculation
    result_df = calc_precursor_isotope_intensity_mp(
        sample_precursor_df_with_mods,
        max_isotope=2,  # Minimal isotope calculation
        mp_process_num=2,
        progress_bar=False,
    )

    # Verify the calculation completed successfully with custom mods
    assert len(result_df) == len(sample_precursor_df_with_mods)
    assert "i_0" in result_df.columns
    assert "i_1" in result_df.columns

    # Verify results are reasonable (just basic sanity check)
    assert all(result_df["i_0"] > 0)
    assert all(result_df["i_1"] > 0)


def test_compare_single_vs_mp_with_custom_mods(
    sample_precursor_df_with_mods, add_custom_modification
):
    """Test that single and multiprocessing versions give consistent results with custom mods."""
    # Skip if only 1 CPU is available
    if cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Run single-process version with minimal calculation
    single_result = calc_precursor_isotope_intensity(
        sample_precursor_df_with_mods.copy(), max_isotope=2
    )

    # Run multi-process version with minimal calculation
    mp_result = calc_precursor_isotope_intensity_mp(
        sample_precursor_df_with_mods.copy(),
        max_isotope=2,
        mp_process_num=2,
        progress_bar=False,
    )

    # Verify results are consistent between single and multi-process
    for col in ["i_0", "i_1"]:
        np.testing.assert_allclose(
            single_result[col].values, mp_result[col].values, rtol=1e-5
        )


def test_multiple_custom_mods_in_mp(add_multiple_custom_modifications):
    """Test that multiple different custom modifications work in multiprocessing."""
    # Skip if only 1 CPU is available
    if cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Create a minimal dataframe with multiple custom modifications
    data = {
        "sequence": ["PEPTIDES", "PEPTIDES"],
        "mods": ["CustomMod1@N-term", "CustomMod2@S"],
        "mod_sites": ["0", "7"],
        "charge": [2, 2],
        "nAA": [8, 8],
        "precursor_mz": [500.2, 500.2],
    }
    df = pd.DataFrame(data)

    # Run the multi-process version with minimal calculation
    result_df = calc_precursor_isotope_intensity_mp(
        df, max_isotope=2, mp_process_num=2, progress_bar=False
    )

    # Verify the calculation completed successfully with custom mods
    assert len(result_df) == len(df)
    assert "i_0" in result_df.columns
    assert "i_1" in result_df.columns

    # Verify results are reasonable (just basic sanity check)
    assert all(result_df["i_0"] > 0)
    assert all(result_df["i_1"] > 0)
