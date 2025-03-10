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
from alphabase.spectral_library.base import SpecLibBase


@pytest.fixture
def cleanup_test_mods():
    """Clean up test modifications after test."""
    # Keep track of test modifications
    test_mods = ["SpecLibMod@N-term", "SpecLibMod@T", "SpecLibMod@A"]

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
        "sequence": ["PEPTIDE", "TESTPEPTIDE", "ANOTHERPEPTIDE"],
        "mods": ["SpecLibMod@N-term", "SpecLibMod@N-term;SpecLibMod@T", "SpecLibMod@A"],
        "mod_sites": ["0", "0;1", "0"],
        "charge": [2, 3, 2],
        "nAA": [7, 11, 14],
        "precursor_mz": [400.2, 420.3, 750.4],
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
            ("SpecLibMod@A", "C(2)H(3)O(1)", ""),
        ]
    )

    # Make sure MOD_Composition is updated
    update_all_by_MOD_DF()

    # Verify it was added
    assert "SpecLibMod@N-term" in MOD_DF.index
    assert "SpecLibMod@T" in MOD_DF.index
    assert "SpecLibMod@A" in MOD_DF.index
    assert "SpecLibMod@N-term" in MOD_Composition
    assert "SpecLibMod@T" in MOD_Composition
    assert "SpecLibMod@A" in MOD_Composition


def test_speclib_isotope_intensity_mp(
    sample_precursor_df_with_mods, add_speclib_custom_modification
):
    """Test that SpecLibBase.calc_precursor_isotope_intensity works with custom modifications in multiprocessing mode."""
    # Skip if only 1 CPU is available
    if cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Create a spectral library
    speclib = SpecLibBase()
    speclib.precursor_df = sample_precursor_df_with_mods.copy()

    # Calculate isotope intensities with multiprocessing
    speclib.calc_precursor_isotope_intensity(max_isotope=3, mp_process_num=2)

    # Check that isotope intensity columns were added
    assert "i_0" in speclib.precursor_df.columns
    assert "i_1" in speclib.precursor_df.columns
    assert "i_2" in speclib.precursor_df.columns

    # Check that values are reasonable
    for i in range(len(speclib.precursor_df)):
        intensities = [speclib.precursor_df.iloc[i][f"i_{j}"] for j in range(3)]
        assert all(intensity > 0 for intensity in intensities)
        assert 0.99 <= sum(intensities) <= 1.01  # Sum should be close to 1


def test_speclib_isotope_info_mp(
    sample_precursor_df_with_mods, add_speclib_custom_modification
):
    """Test that SpecLibBase.calc_precursor_isotope_info works with custom modifications in multiprocessing mode."""
    # Skip if only 1 CPU is available
    if cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Create a spectral library
    speclib = SpecLibBase()
    speclib.precursor_df = sample_precursor_df_with_mods.copy()

    # Calculate isotope info with multiprocessing
    speclib.calc_precursor_isotope_info(mp_process_num=2)

    # Check that isotope info columns were added
    expected_columns = [
        "isotope_apex_offset",
        "isotope_apex_mz",
        "isotope_apex_intensity",
        "isotope_right_most_offset",
        "isotope_right_most_mz",
        "isotope_right_most_intensity",
        "isotope_m1_mz",
        "isotope_m1_intensity",
    ]

    for col in expected_columns:
        assert col in speclib.precursor_df.columns

    # Check that values are reasonable
    assert all(speclib.precursor_df["isotope_apex_offset"] >= 0)
    assert all(
        speclib.precursor_df["isotope_right_most_offset"]
        >= speclib.precursor_df["isotope_apex_offset"]
    )
    assert all(speclib.precursor_df["isotope_apex_intensity"] > 0)
    assert all(speclib.precursor_df["isotope_right_most_intensity"] > 0)


def test_speclib_compare_single_vs_mp(
    sample_precursor_df_with_mods, add_speclib_custom_modification
):
    """Test that single and multiprocessing versions in SpecLibBase give the same results with custom mods."""
    # Skip if only 1 CPU is available
    if cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Create two identical spectral libraries
    speclib_single = SpecLibBase()
    speclib_single.precursor_df = sample_precursor_df_with_mods.copy()

    speclib_mp = SpecLibBase()
    speclib_mp.precursor_df = sample_precursor_df_with_mods.copy()

    # Calculate isotope intensities with single process
    speclib_single.calc_precursor_isotope_intensity(
        max_isotope=3,
        mp_process_num=1,  # Force single process
    )

    # Calculate isotope intensities with multiprocessing
    speclib_mp.calc_precursor_isotope_intensity(max_isotope=3, mp_process_num=2)

    # Compare results (should be very close, allowing for minor floating-point differences)
    for i in range(3):  # For each isotope
        col = f"i_{i}"
        np.testing.assert_allclose(
            speclib_single.precursor_df[col].values,
            speclib_mp.precursor_df[col].values,
            rtol=1e-5,
        )


def test_speclib_large_batch(add_speclib_custom_modification):
    """Test SpecLibBase with a larger batch of data with custom modifications."""
    # Skip if only 1 CPU is available
    if cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Create a larger dataframe with custom modifications
    n_rows = 50  # Large enough to be split across processes but not too large for a unit test

    sequences = ["PEPTIDES", "TESTPEPTIDEK", "ANOTHERPEPTIDE"] * (n_rows // 3 + 1)
    mods = ["SpecLibMod@N-term"] * (n_rows // 3 + 1) * 3
    mod_sites = ["0"] * (n_rows // 3 + 1) * 3

    data = {
        "sequence": sequences[:n_rows],
        "mods": mods[:n_rows],
        "mod_sites": mod_sites[:n_rows],
        "charge": [2] * n_rows,
        "nAA": [len(seq) for seq in sequences[:n_rows]],
        "precursor_mz": [500.0 + i for i in range(n_rows)],
    }
    df = pd.DataFrame(data)

    # Create a spectral library
    speclib = SpecLibBase()
    speclib.precursor_df = df

    # Calculate isotope intensities with multiprocessing
    speclib.calc_precursor_isotope_intensity(
        max_isotope=3,
        mp_process_num=2,
        mp_batch_size=10,  # Small batch size to ensure multiple batches
    )

    # Check that all rows were processed
    assert len(speclib.precursor_df) == n_rows

    # Check that isotope intensity columns were added
    assert "i_0" in speclib.precursor_df.columns
    assert "i_1" in speclib.precursor_df.columns
    assert "i_2" in speclib.precursor_df.columns

    # Check that values are reasonable
    for i in range(len(speclib.precursor_df)):
        intensities = [speclib.precursor_df.iloc[i][f"i_{j}"] for j in range(3)]
        assert all(intensity > 0 for intensity in intensities)
        assert 0.99 <= sum(intensities) <= 1.01  # Sum should be close to 1
