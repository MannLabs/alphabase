import numpy as np
import pandas as pd
import pytest

from alphabase.peptide.fragment import (
    LOSS,
    SERIES,
    _calc_column_indices,
    _calc_row_indices,
    _start_stop_to_idx,
)


def test_calc_column_indices_unmapped_fragments():
    """Test handling of fragments that don't map to any charged_frag_types"""
    fragment_df = pd.DataFrame(
        {
            "type": [SERIES.B, SERIES.Y, SERIES.A],
            "loss_type": [LOSS.NONE] * 3,
            "charge": [1, 1, 1],
        }
    )

    charged_frag_types = ["b_z1", "y_z1"]  # No a_z1 in charged_frag_types

    indices = _calc_column_indices(fragment_df, charged_frag_types)

    assert len(indices) == 3
    assert indices[0] == 0  # b_z1
    assert indices[1] == 1  # y_z1
    assert indices[2] == -1  # a_z1 should be unmapped


def test_calc_column_indices_complex_mix():
    """Test complex mixture of fragment types, losses and charges"""
    fragment_df = pd.DataFrame(
        {
            "type": [SERIES.B, SERIES.Y, SERIES.B, SERIES.Y],
            "loss_type": [LOSS.NONE, LOSS.H2O, LOSS.NH3, LOSS.NONE],
            "charge": [1, 1, 2, 2],
        }
    )

    charged_frag_types = ["b_z1", "b_z2", "y_z1", "y_z2", "b_NH3_z2", "y_H2O_z1"]

    indices = _calc_column_indices(fragment_df, charged_frag_types)

    assert len(indices) == 4
    assert indices[0] == 0  # b_z1
    assert indices[1] == 5  # y_H2O_z1
    assert indices[2] == 4  # b_NH3_z2
    assert indices[3] == 3  # y_z2


def test_calc_column_indices_empty_input():
    """Test handling of empty input DataFrame"""
    fragment_df = pd.DataFrame({"type": [], "loss_type": [], "charge": []})

    charged_frag_types = ["b_z1", "y_z1"]

    indices = _calc_column_indices(fragment_df, charged_frag_types)

    assert len(indices) == 0
    assert isinstance(indices, np.ndarray)


def test_calc_row_indices_empty():
    """Test handling of empty input"""
    # Setup
    precursor_naa = np.array([])
    fragment_position = np.array([])
    precursor_df_idx = np.array([])
    fragment_df_idx = np.array([])

    # Execute
    row_indices, frag_start_idx, frag_stop_idx = _calc_row_indices(
        precursor_naa, fragment_position, precursor_df_idx, fragment_df_idx
    )

    # Assert
    assert len(row_indices) == 0
    assert len(frag_start_idx) == 0
    assert len(frag_stop_idx) == 0


def test_calc_row_indices_unordered_precursors():
    """Test that function handles unordered precursor indices correctly"""
    # Setup
    precursor_naa = np.array(
        [3, 4]
    )  # First precursor: 2 fragments, Second: 3 fragments
    fragment_position = np.array([0, 1, 0, 1, 2])
    precursor_df_idx = np.array([1, 0])  # Unordered precursor indices
    fragment_df_idx = np.array([1, 1, 0, 0, 0])

    # Execute
    row_indices, frag_start_idx, frag_stop_idx = _calc_row_indices(
        precursor_naa, fragment_position, precursor_df_idx, fragment_df_idx
    )

    # Assert
    np.testing.assert_array_equal(row_indices, np.array([0, 1, 2, 3, 4]))
    np.testing.assert_array_equal(frag_start_idx, np.array([0, 2]))
    np.testing.assert_array_equal(frag_stop_idx, np.array([2, 5]))


def test_calc_row_indices_mismatched_lengths():
    """Test that function raises error when input arrays have mismatched lengths"""
    # Setup
    precursor_naa = np.array([3, 4])
    fragment_position = np.array([0, 1])  # Too short
    precursor_df_idx = np.array([0, 1])
    fragment_df_idx = np.array([0, 0, 1])  # Different length than fragment_position

    # Execute and Assert
    with pytest.raises(ValueError):
        _calc_row_indices(
            precursor_naa, fragment_position, precursor_df_idx, fragment_df_idx
        )


def test_calc_row_indices_realistic_example():
    """Test with a realistic example including multiple fragment types"""
    # Setup
    precursor_naa = np.array([4, 5])  # Two peptides with 3 and 4 fragments respectively
    # Create positions for b and y ions for each peptide
    fragment_position = np.array(
        [
            # First peptide (3 fragments x 2 ion types)
            0,
            1,
            2,  # b ions
            0,
            1,
            2,  # y ions
            # Second peptide (4 fragments x 2 ion types)
            0,
            1,
            2,
            3,  # b ions
            0,
            1,
            2,
            3,  # y ions
        ]
    )
    precursor_df_idx = np.array([0, 1])
    fragment_df_idx = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,  # First peptide fragments
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,  # Second peptide fragments
        ]
    )

    # Execute
    row_indices, frag_start_idx, frag_stop_idx = _calc_row_indices(
        precursor_naa, fragment_position, precursor_df_idx, fragment_df_idx
    )

    # Assert
    expected_row_indices = np.array(
        [
            0,
            1,
            2,
            0,
            1,
            2,  # First peptide (3 positions x 2 ion types)
            3,
            4,
            5,
            6,
            3,
            4,
            5,
            6,  # Second peptide (4 positions x 2 ion types)
        ]
    )
    np.testing.assert_array_equal(row_indices, expected_row_indices)
    np.testing.assert_array_equal(frag_start_idx, np.array([0, 3]))
    np.testing.assert_array_equal(frag_stop_idx, np.array([3, 7]))


def test_start_stop_to_idx_unordered():
    """Test with unordered start indices"""
    precursor_df = pd.DataFrame(
        {"flat_frag_start_idx": [3, 0, 6], "flat_frag_stop_idx": [6, 3, 8]}
    )
    fragment_df = pd.DataFrame({"dummy": range(8)})

    precursor_df_idx, fragment_df_idx = _start_stop_to_idx(precursor_df, fragment_df)

    # Original order should be preserved in precursor_df_idx
    np.testing.assert_array_equal(precursor_df_idx, [1, 0, 2])

    # Fragment indices should still map correctly
    expected_fragment_idx = [0, 0, 0, 1, 1, 1, 2, 2]
    np.testing.assert_array_equal(fragment_df_idx, expected_fragment_idx)


def test_start_stop_to_idx_mismatch():
    """Test error handling when fragment counts don't match"""
    precursor_df = pd.DataFrame(
        {"flat_frag_start_idx": [0, 3], "flat_frag_stop_idx": [3, 6]}
    )
    fragment_df = pd.DataFrame({"dummy": range(5)})  # Only 5 fragments instead of 6

    with pytest.raises(
        ValueError,
        match="Number of fragments .* is not equal to the number of rows in fragment_df",
    ):
        _start_stop_to_idx(precursor_df, fragment_df)


def test_start_stop_to_idx_empty():
    """Test handling of empty DataFrames"""
    precursor_df = pd.DataFrame({"flat_frag_start_idx": [], "flat_frag_stop_idx": []})
    fragment_df = pd.DataFrame({"dummy": []})

    precursor_df_idx, fragment_df_idx = _start_stop_to_idx(precursor_df, fragment_df)

    assert len(precursor_df_idx) == 0
    assert len(fragment_df_idx) == 0
