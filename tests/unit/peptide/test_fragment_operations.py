"""
Tests for the peptide fragment operations.
Based on the tests in nbs_tests/peptide/fragment.ipynb
"""

import numpy as np
import pandas as pd
import pytest

from alphabase.peptide.fragment import (
    FRAGMENT_TYPES,
    FragmentType,
    create_fragment_mz_dataframe,
    create_fragment_mz_dataframe_by_sort_precursor,
    flatten_fragments,
    get_charged_frag_types,
    join_left,
    parse_charged_frag_type,
    remove_unused_fragments,
    sort_charged_frag_types,
)
from alphabase.peptide.precursor import update_precursor_mz


def test_fragment_types():
    """Test that fragment types are properly defined."""
    # All FragmentType instances in FRAGMENT_TYPES should be FragmentType objects
    for fragment_type in FRAGMENT_TYPES.values():
        assert isinstance(fragment_type, FragmentType)


def test_get_charged_frag_types():
    """Test generation of charged fragment types."""
    charged_types = get_charged_frag_types(["b", "b_modloss"], 2)
    expected = ["b_z1", "b_z2", "b_modloss_z1", "b_modloss_z2"]

    assert np.all(np.array(charged_types) == np.array(expected))


def test_sort_charged_frag_types():
    """Test sorting of charged fragment types."""
    unsorted = ["b_modloss_z1", "a_z1", "b_z1", "b_z2", "b_modloss_z2", "a_z2"]
    expected = ["a_z1", "a_z2", "b_z1", "b_z2", "b_modloss_z1", "b_modloss_z2"]

    result = sort_charged_frag_types(unsorted)
    assert result == expected


def test_parse_charged_frag_type():
    """Test parsing of charged fragment types."""
    assert parse_charged_frag_type("b_z2") == ("b", 2)
    assert parse_charged_frag_type("b_modloss_z2") == ("b_modloss", 2)


@pytest.fixture
def sample_precursor_df():
    """Create a sample precursor DataFrame for testing."""
    data = {
        "sequence": ["AAA", "PEPTIDE", "ACDKWLNSR"],
        "mods": ["", "Phospho@T", ""],
        "mod_sites": ["", "4", ""],
        "charge": [2, 3, 2],
        "nAA": [3, 7, 9],
    }
    df = pd.DataFrame(data)
    update_precursor_mz(df)
    return df


def test_fragment_mz_dataframe_by_sort(sample_precursor_df):
    """Test creation of fragment mz dataframe by sorting precursors."""
    # Define charged fragment types for testing
    charged_frag_types = ["b_z1", "b_z2", "y_z1", "y_z2"]

    # Create a copy to work with
    precursor_copy = sample_precursor_df.copy()

    # Create fragment mz dataframe
    fragment_mz_df = create_fragment_mz_dataframe_by_sort_precursor(
        precursor_copy, charged_frag_types
    )

    # Verify DataFrame properties
    assert precursor_copy.nAA.is_monotonic_increasing

    # After creation, the precursor_df should have frag_start_idx and frag_stop_idx
    assert "frag_start_idx" in precursor_copy.columns
    assert "frag_stop_idx" in precursor_copy.columns

    # Validate fragment masses for first peptide (AAA)
    frag_start = precursor_copy.iloc[0].frag_start_idx
    frag_end = precursor_copy.iloc[0].frag_stop_idx

    # b ions for 'AAA' should have specific values
    b_z1_values = fragment_mz_df.iloc[frag_start:frag_end]["b_z1"].values
    # These values are for amino acid 'A' (71.03711 + proton)
    expected_b_z1 = [72.04493882, 143.08210385]
    assert np.allclose(b_z1_values, expected_b_z1)

    # b_z2 values will not be zero as the fragment can have a charge of 2
    # For small peptides like 'AAA', all fragment ions are valid in the framework
    b_z2_values = fragment_mz_df.iloc[frag_start:frag_end]["b_z2"].values
    expected_b_z2 = [36.52586941, 72.04494693]  # (m+2H)/2
    assert np.allclose(b_z2_values, expected_b_z2, rtol=1e-3)


def test_fragment_mz_dataframe_without_sorting(sample_precursor_df):
    """Test creation of fragment mz dataframe without sorting precursors."""
    # Define charged fragment types for testing
    charged_frag_types = ["a_z1", "b_z1", "y_z1"]

    # Create a copy with unsorted nAA values
    unsorted_df = sample_precursor_df.copy().iloc[::-1].reset_index(drop=True)
    assert not unsorted_df.nAA.is_monotonic_increasing

    # Create fragment mz dataframe
    fragment_mz_df = create_fragment_mz_dataframe(unsorted_df, charged_frag_types)

    # Check that fragment masses are correct
    # a ions should be b ions minus CO (27.9949)
    assert np.allclose(
        fragment_mz_df.a_z1.values, fragment_mz_df.b_z1.values - 27.9949, atol=1e-4
    )


def test_join_left():
    """Test the join_left function for matching elements between arrays."""
    left = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    right = np.array([5, 3, 6, 9, 7])

    result = join_left(left, right)

    # Check that indices match correctly:
    # left[0]=3 should match right[1]=3 -> result[0]=1
    # left[4]=5 should match right[0]=5 -> result[4]=0
    # etc.
    expected = np.array([1, -1, -1, -1, 0, 3, -1, 2])
    assert np.all(result == expected)


def test_flatten_fragments(sample_precursor_df):
    """Test flattening fragment dataframes."""
    # First create fragment mz dataframe
    charged_frag_types = ["b_z1", "y_z1"]

    # Create a copy with fragment indices
    precursor_copy = sample_precursor_df.copy()
    fragment_mz_df = create_fragment_mz_dataframe_by_sort_precursor(
        precursor_copy, charged_frag_types
    )

    # Create intensity dataframe with same shape
    fragment_intensity_df = pd.DataFrame(
        np.random.random(fragment_mz_df.shape), columns=fragment_mz_df.columns
    )

    # Now we can flatten fragments
    precursor_df_flat, frag_df_flat = flatten_fragments(
        precursor_copy,  # This now has frag_start_idx and frag_stop_idx
        fragment_mz_df,
        fragment_intensity_df,
    )

    # Verify flattened structures
    assert "flat_frag_start_idx" in precursor_df_flat.columns
    assert "flat_frag_stop_idx" in precursor_df_flat.columns
    assert "mz" in frag_df_flat.columns
    assert "intensity" in frag_df_flat.columns
    assert "type" in frag_df_flat.columns
    assert "number" in frag_df_flat.columns

    # Verify total count of fragments
    # (will depend on filtering of zero values)
    assert len(frag_df_flat) > 0


def test_remove_unused_fragments(sample_precursor_df):
    """Test removing unused fragments."""
    # First create fragment mz dataframe
    charged_frag_types = ["b_z1", "y_z1"]

    # Create a copy with fragment indices
    precursor_copy = sample_precursor_df.copy()
    fragment_mz_df = create_fragment_mz_dataframe_by_sort_precursor(
        precursor_copy, charged_frag_types
    )

    # Create a filtered precursor_df with only some rows, keeping fragment indices
    filtered_df = precursor_copy.iloc[1:2].copy()

    # Now we can remove unused fragments
    new_precursor_df, (new_fragment_df,) = remove_unused_fragments(
        filtered_df, (fragment_mz_df,)
    )

    # Check that the new fragment df is smaller
    assert len(new_fragment_df) < len(fragment_mz_df)

    # Check that the fragment indices still align correctly
    frag_start = new_precursor_df.iloc[0].frag_start_idx
    frag_end = new_precursor_df.iloc[0].frag_stop_idx
    assert frag_start == 0  # Should start at 0 now
    assert frag_end - frag_start == filtered_df.iloc[0].nAA - 1
