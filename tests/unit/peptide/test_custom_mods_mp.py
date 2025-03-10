from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import pytest

from alphabase.constants.modification import (
    MOD_DF,
    MOD_Composition,
    add_new_modifications,
    get_custom_mods,
    update_all_by_MOD_DF,
)
from alphabase.peptide.precursor import (
    calc_precursor_isotope_info_mp,
    calc_precursor_isotope_intensity,
    calc_precursor_isotope_intensity_mp,
)
from alphabase.spectral_library.base import SpecLibBase


@pytest.fixture
def sample_precursor_df():
    """Create a sample precursor dataframe for testing."""
    data = {
        "sequence": ["PEPTIDE", "TESTPEPTIDE", "ANOTHERPEPTIDE"],
        "mods": ["", "", ""],
        "mod_sites": ["", "", ""],
        "charge": [2, 3, 2],
        "nAA": [7, 11, 14],
        "precursor_mz": [400.2, 420.3, 750.4],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_precursor_df_with_mods():
    """Create a sample precursor dataframe with modifications for testing."""
    data = {
        "sequence": ["PEPTIDE", "TESTPEPTIDE", "ANOTHERPEPTIDE"],
        "mods": ["CustomMod@N-term", "CustomMod@N-term;CustomMod@T", "CustomMod@A"],
        "mod_sites": ["0", "0;1", "0"],
        "charge": [2, 3, 2],
        "nAA": [7, 11, 14],
        "precursor_mz": [400.2, 420.3, 750.4],
    }
    return pd.DataFrame(data)


@pytest.fixture
def cleanup_test_mods():
    """Clean up test modifications after test."""
    # Keep track of test modifications
    test_mods = [
        "CustomMod@N-term",
        "CustomMod@T",
        "CustomMod@A",
        "CustomMod1@N-term",
        "CustomMod2@S",
        "CustomMod3@K",
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
            ("CustomMod@A", "C(2)H(3)O(1)", ""),
        ]
    )

    # Make sure MOD_Composition is updated
    update_all_by_MOD_DF()

    # Verify it was added
    assert "CustomMod@N-term" in MOD_DF.index
    assert "CustomMod@T" in MOD_DF.index
    assert "CustomMod@A" in MOD_DF.index
    assert "CustomMod@N-term" in MOD_Composition
    assert "CustomMod@T" in MOD_Composition
    assert "CustomMod@A" in MOD_Composition


@pytest.fixture
def add_multiple_custom_modifications(cleanup_test_mods):
    """Add multiple custom modifications for testing."""
    # Add custom modifications
    add_new_modifications(
        [
            ("CustomMod1@N-term", "C(2)H(3)O(1)", "H(2)O(1)"),
            ("CustomMod2@S", "C(3)H(5)O(2)", ""),
            ("CustomMod3@K", "C(4)H(7)N(1)", "N(1)H(1)"),
        ]
    )

    # Make sure MOD_Composition is updated
    update_all_by_MOD_DF()

    # Verify they were added
    assert "CustomMod1@N-term" in MOD_DF.index
    assert "CustomMod2@S" in MOD_DF.index
    assert "CustomMod3@K" in MOD_DF.index
    assert "CustomMod1@N-term" in MOD_Composition
    assert "CustomMod2@S" in MOD_Composition
    assert "CustomMod3@K" in MOD_Composition


def test_get_custom_mods(add_custom_modification):
    """Test that get_custom_mods returns the expected custom modifications."""
    custom_mods = get_custom_mods()

    assert "CustomMod@N-term" in custom_mods
    assert custom_mods["CustomMod@N-term"]["composition"] == "C(2)H(3)O(1)"
    assert custom_mods["CustomMod@N-term"]["modloss_composition"] == "H(2)O(1)"


def test_calc_precursor_isotope_intensity_with_custom_mods(
    sample_precursor_df_with_mods, add_custom_modification
):
    """Test that calc_precursor_isotope_intensity works with custom modifications."""
    # Run the single-process version
    result_df = calc_precursor_isotope_intensity(
        sample_precursor_df_with_mods, max_isotope=3
    )

    # Check that isotope intensity columns were added
    assert "i_0" in result_df.columns
    assert "i_1" in result_df.columns
    assert "i_2" in result_df.columns

    # Check that values are reasonable (non-zero, sum to approximately 1)
    for i in range(len(result_df)):
        intensities = [result_df.iloc[i][f"i_{j}"] for j in range(3)]
        assert all(intensity > 0 for intensity in intensities)
        assert 0.99 <= sum(intensities) <= 1.01  # Sum should be close to 1


def test_calc_precursor_isotope_intensity_mp_with_custom_mods(
    sample_precursor_df_with_mods, add_custom_modification
):
    """Test that calc_precursor_isotope_intensity_mp works with custom modifications."""
    # Skip if only 1 CPU is available
    if cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Run the multi-process version
    result_df = calc_precursor_isotope_intensity_mp(
        sample_precursor_df_with_mods,
        max_isotope=3,
        mp_process_num=2,
        progress_bar=False,
    )

    # Check that isotope intensity columns were added
    assert "i_0" in result_df.columns
    assert "i_1" in result_df.columns
    assert "i_2" in result_df.columns

    # Check that values are reasonable (non-zero, sum to approximately 1)
    for i in range(len(result_df)):
        intensities = [result_df.iloc[i][f"i_{j}"] for j in range(3)]
        assert all(intensity > 0 for intensity in intensities)
        assert 0.99 <= sum(intensities) <= 1.01  # Sum should be close to 1


def test_compare_single_vs_multiprocessing(
    sample_precursor_df_with_mods, add_custom_modification
):
    """Test that single and multiprocessing versions give the same results with custom mods."""
    # Skip if only 1 CPU is available
    if cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Run single-process version
    single_result = calc_precursor_isotope_intensity(
        sample_precursor_df_with_mods.copy(), max_isotope=3
    )

    # Run multi-process version
    mp_result = calc_precursor_isotope_intensity_mp(
        sample_precursor_df_with_mods.copy(),
        max_isotope=3,
        mp_process_num=2,
        progress_bar=False,
    )

    # Compare results (should be very close, allowing for minor floating-point differences)
    for i in range(3):  # For each isotope
        col = f"i_{i}"
        np.testing.assert_allclose(
            single_result[col].values, mp_result[col].values, rtol=1e-5
        )


def test_speclib_with_custom_mods(
    sample_precursor_df_with_mods, add_custom_modification
):
    """Test that SpecLibBase works with custom modifications in multiprocessing mode."""
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


def test_calc_precursor_isotope_info_mp_with_custom_mods(
    sample_precursor_df_with_mods, add_custom_modification
):
    """Test that calc_precursor_isotope_info_mp works with custom modifications."""
    # Skip if only 1 CPU is available
    if cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Run the multi-process version
    result_df = calc_precursor_isotope_info_mp(
        sample_precursor_df_with_mods, processes=2, progress_bar=None
    )

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
        assert col in result_df.columns

    # Check that values are reasonable
    assert all(result_df["isotope_apex_offset"] >= 0)
    assert all(
        result_df["isotope_right_most_offset"] >= result_df["isotope_apex_offset"]
    )
    assert all(result_df["isotope_apex_intensity"] > 0)
    assert all(result_df["isotope_right_most_intensity"] > 0)


def test_speclib_calc_precursor_isotope_info_with_custom_mods(
    sample_precursor_df_with_mods, add_custom_modification
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


def test_serialization_deserialization_of_custom_mods(
    add_multiple_custom_modifications,
):
    """Test the serialization and deserialization of custom modifications across processes."""
    # Skip if only 1 CPU is available
    if cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Get the custom modifications
    custom_mods = get_custom_mods()

    # Verify we have all the expected custom mods
    assert "CustomMod1@N-term" in custom_mods
    assert "CustomMod2@S" in custom_mods
    assert "CustomMod3@K" in custom_mods

    # Create a dataframe with all the custom modifications
    data = {
        "sequence": ["PEPTIDES", "TESTPEPTIDEK", "ANOTHERPEPTIDE"],
        "mods": [
            "CustomMod1@N-term;CustomMod2@S",
            "CustomMod1@N-term;CustomMod3@K",
            "CustomMod1@N-term;CustomMod2@S;CustomMod3@K",
        ],
        "mod_sites": ["0;7", "0;11", "0;7;14"],
        "charge": [2, 3, 2],
        "nAA": [8, 12, 14],
        "precursor_mz": [500.2, 600.3, 800.4],
    }
    df = pd.DataFrame(data)

    # Run the multi-process version
    result_df = calc_precursor_isotope_intensity_mp(
        df, max_isotope=3, mp_process_num=2, progress_bar=False
    )

    # Check that isotope intensity columns were added
    assert "i_0" in result_df.columns
    assert "i_1" in result_df.columns
    assert "i_2" in result_df.columns

    # Check that values are reasonable (non-zero, sum to approximately 1)
    for i in range(len(result_df)):
        intensities = [result_df.iloc[i][f"i_{j}"] for j in range(3)]
        assert all(intensity > 0 for intensity in intensities)
        assert 0.99 <= sum(intensities) <= 1.01  # Sum should be close to 1

    # Compare with single-process version to ensure consistency
    single_result = calc_precursor_isotope_intensity(df.copy(), max_isotope=3)

    # Results should be very close
    for i in range(3):  # For each isotope
        col = f"i_{i}"
        np.testing.assert_allclose(
            single_result[col].values, result_df[col].values, rtol=1e-5
        )


def test_large_batch_with_custom_mods(add_multiple_custom_modifications):
    """Test processing a larger batch of data with custom modifications."""
    # Skip if only 1 CPU is available
    if cpu_count() < 2:
        pytest.skip("Need at least 2 CPUs for multiprocessing test")

    # Create a larger dataframe with custom modifications
    n_rows = 50  # Large enough to be split across processes but not too large for a unit test

    sequences = ["PEPTIDES", "TESTPEPTIDEK", "ANOTHERPEPTIDE"] * (n_rows // 3 + 1)
    mods = [
        "CustomMod1@N-term;CustomMod2@S",
        "CustomMod1@N-term;CustomMod3@K",
        "CustomMod1@N-term;CustomMod2@S;CustomMod3@K",
    ] * (n_rows // 3 + 1)
    mod_sites = ["0;7", "0;11", "0;7;14"] * (n_rows // 3 + 1)

    data = {
        "sequence": sequences[:n_rows],
        "mods": mods[:n_rows],
        "mod_sites": mod_sites[:n_rows],
        "charge": [2] * n_rows,
        "nAA": [len(seq) for seq in sequences[:n_rows]],
        "precursor_mz": [500.0 + i for i in range(n_rows)],
    }
    df = pd.DataFrame(data)

    # Run with multiple processes
    result_df = calc_precursor_isotope_intensity_mp(
        df,
        max_isotope=3,
        mp_process_num=2,
        mp_batch_size=10,  # Small batch size to ensure multiple batches
        progress_bar=False,
    )

    # Check that all rows were processed
    assert len(result_df) == n_rows

    # Check that isotope intensity columns were added
    assert "i_0" in result_df.columns
    assert "i_1" in result_df.columns
    assert "i_2" in result_df.columns

    # Check that values are reasonable
    for i in range(len(result_df)):
        intensities = [result_df.iloc[i][f"i_{j}"] for j in range(3)]
        assert all(intensity > 0 for intensity in intensities)
        assert 0.99 <= sum(intensities) <= 1.01  # Sum should be close to 1
