import numpy as np
import pandas as pd
import pytest

from alphabase.peptide.fragment import (
    LOSS_MAPPING,
    SERIES_MAPPING,
    Loss,
    Series,
    _calc_column_indices,
    _calc_row_indices,
    _start_stop_to_idx,
    create_dense_matrices,
    get_charged_frag_types,
)
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.flat import SpecLibFlat


def test_calc_column_indices_unmapped_fragments():
    """Test handling of fragments that don't map to any charged_frag_types"""
    fragment_df = pd.DataFrame(
        {
            "type": [
                SERIES_MAPPING[Series.B],
                SERIES_MAPPING[Series.Y],
                SERIES_MAPPING[Series.A],
            ],
            "loss_type": [LOSS_MAPPING[Loss.NONE]] * 3,
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
            "type": [
                SERIES_MAPPING[Series.B],
                SERIES_MAPPING[Series.Y],
                SERIES_MAPPING[Series.B],
                SERIES_MAPPING[Series.Y],
            ],
            "loss_type": [
                LOSS_MAPPING[Loss.NONE],
                LOSS_MAPPING[Loss.H2O],
                LOSS_MAPPING[Loss.NH3],
                LOSS_MAPPING[Loss.NONE],
            ],
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


def test_calc_row_indices_no_start_stop():
    """Test with a realistic example including multiple fragment types"""
    # Setup
    precursor_naa = np.array(
        [4, 5, 3]
    )  # Two peptides with 3 and 4 fragments respectively
    precursor_df_idx = np.array([0, 1, 2])

    # fmt: off
    fragment_position = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    fragment_df_idx = np.array([1, 1, 1, 9, 9, 9, 0, 0, 0, 10, 10, 10, 2, 2, 2, 11, 11, 11])


    # Execute
    row_indices, frag_start_idx, frag_stop_idx = _calc_row_indices(
        precursor_naa, fragment_position, precursor_df_idx, fragment_df_idx
    )

    expected_mask = np.array([True, True, True, False, False, False, True, True, True, False, False, False, True, True, True, False, False, False])

    # fmt: on
    frag_start_idx_expected = np.array([0, 3, 7])
    frag_stop_idx_expected = np.array([3, 7, 9])

    mask = row_indices != -1
    np.testing.assert_array_equal(mask, expected_mask)

    np.testing.assert_array_equal(frag_start_idx, frag_start_idx_expected)
    np.testing.assert_array_equal(frag_stop_idx, frag_stop_idx_expected)


def test_calc_row_indices_with_start_stop():
    """Test with a realistic example including multiple fragment types"""
    # Setup
    precursor_naa = np.array(
        [4, 5, 3]
    )  # Two peptides with 3 and 4 fragments respectively
    precursor_df_idx = np.array([0, 1, 2])

    # fmt: off
    frag_start_idx = np.array([7, 0, 3])
    frag_stop_idx = np.array([9, 3, 7])

    fragment_position = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    fragment_df_idx = np.array([1, 1, 1, 9, 9, 9, 0, 0, 0, 10, 10, 10, 2, 2, 2, 11, 11, 11])
    # fmt: on

    # Execute
    row_indices, _frag_start_idx, _frag_stop_idx = _calc_row_indices(
        precursor_naa,
        fragment_position,
        precursor_df_idx,
        fragment_df_idx,
        frag_start_idx,
        frag_stop_idx,
    )

    # fmt: off
    expected_mask = np.array([True, True, True, False, False, False, True, True, True, False, False, False, True, True, True, False, False, False])
    expected_row_indices = np.array([0, 1, 2, -1, -1, -1, 7, 8, 9, -1, -1, -1, 3, 4, 5, -1, -1, -1])
    # fmt: on

    mask = row_indices != -1
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(row_indices, expected_row_indices)
    np.testing.assert_array_equal(frag_start_idx, _frag_start_idx)
    np.testing.assert_array_equal(frag_stop_idx, _frag_stop_idx)


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


def test_create_dense_matrices_with_precursor_idx():
    """Test create_dense_matrices with precursor_idx mapping"""
    # Setup test data
    precursor_df = pd.DataFrame(
        {
            "sequence": ["PEP", "PRO", "PEP"],
            "mods": ["", "", ""],
            "mod_sites": ["", "", ""],
            "charge": [2, 2, 2],
            "nAA": [3, 3, 3],
            "precursor_idx": [2, 0, 1],
        }
    )

    fragment_df = pd.DataFrame(
        {
            "type": [
                SERIES_MAPPING[Series.B],
                SERIES_MAPPING[Series.Y],
                SERIES_MAPPING[Series.B],
                SERIES_MAPPING[Series.Y],
                SERIES_MAPPING[Series.B],
                SERIES_MAPPING[Series.Y],
            ],
            "position": [0, 1, 0, 1, 0, 1],
            "charge": [1, 1, 1, 1, 1, 1],
            "loss_type": [LOSS_MAPPING[Loss.NONE]] * 6,
            "intensity": [100, 200, 300, 400, 500, 600],
            "precursor_idx": [0, 0, 1, 1, 2, 2],
        }
    )

    charged_frag_types = ["b_z1", "y_z1"]

    # Execute
    df_collection, frag_start_idx, frag_stop_idx = create_dense_matrices(
        precursor_df, fragment_df, charged_frag_types
    )

    # Assert
    assert set(df_collection.keys()) == {"mz", "intensity"}
    assert df_collection["intensity"].shape == (
        6,
        2,
    )  # 2 peptides * 6 positions * 2 ion types
    assert list(df_collection["intensity"].columns) == charged_frag_types

    print(precursor_df["precursor_idx"].values)
    print(frag_start_idx)
    print(frag_stop_idx)
    print(df_collection["intensity"])

    # Check intensity values are in correct positions
    np.testing.assert_array_equal(
        df_collection["intensity"].values,
        np.array(
            [
                [500, 0],
                [0, 600],
                [100, 0],
                [0, 200],
                [300, 0],
                [0, 400],
            ]
        ),
    )


def test_create_dense_matrices_with_frag_start_idx():
    """Test create_dense_matrices with fragment start/stop indices"""
    # Setup test data
    precursor_df = pd.DataFrame(
        {
            "sequence": ["PEP", "PRO", "PEP"],
            "mods": ["", "", ""],
            "mod_sites": ["", "", ""],
            "charge": [2, 2, 2],
            "nAA": [3, 3, 3],
            "flat_frag_start_idx": [0, 2, 4],  # Each precursor has 2 fragments
            "flat_frag_stop_idx": [2, 4, 6],
        }
    )

    fragment_df = pd.DataFrame(
        {
            "type": [
                SERIES_MAPPING[Series.B],
                SERIES_MAPPING[Series.Y],
                SERIES_MAPPING[Series.B],
                SERIES_MAPPING[Series.Y],
                SERIES_MAPPING[Series.B],
                SERIES_MAPPING[Series.Y],
            ],
            "position": [0, 1, 0, 1, 0, 1],
            "charge": [1, 1, 1, 1, 1, 1],
            "loss_type": [LOSS_MAPPING[Loss.NONE]] * 6,
            "intensity": [100, 200, 300, 400, 500, 600],
        }
    )

    charged_frag_types = ["b_z1", "y_z1"]

    # Execute
    df_collection, frag_start_idx, frag_stop_idx = create_dense_matrices(
        precursor_df, fragment_df, charged_frag_types
    )

    # Assert
    assert set(df_collection.keys()) == {"mz", "intensity"}
    assert df_collection["intensity"].shape == (
        6,
        2,
    )  # 3 peptides * 2 positions * 2 ion types
    assert list(df_collection["intensity"].columns) == charged_frag_types

    # Check intensity values are in correct positions
    np.testing.assert_array_equal(
        df_collection["intensity"].values,
        np.array(
            [
                [100, 0],
                [0, 200],
                [300, 0],
                [0, 400],
                [500, 0],
                [0, 600],
            ]
        ),
    )

    # Check fragment start/stop indices
    np.testing.assert_array_equal(frag_start_idx, [0, 2, 4])
    np.testing.assert_array_equal(frag_stop_idx, [2, 4, 6])


def test_speclib_base_to_flat_conversion():
    """Test conversion from SpecLibBase to SpecLibFlat and back to dense matrices,
    including handling of missing fragment types"""

    # Create test precursor data
    repeat = 10
    precursor_df = pd.DataFrame(
        {
            "sequence": ["PEPTIDE", "PROTEIN", "MREPEPTIDES", "MDPEPTIDE"] * repeat,
            "mods": ["", "Acetyl@Any_N-term", "", "Oxidation@M"] * repeat,
            "mod_sites": ["", "0", "", "1"] * repeat,
            "charge": [2, 3, 2, 3] * repeat,
        }
    )
    precursor_df["nAA"] = precursor_df["sequence"].str.len()

    # Initialize SpecLibBase with only b and y ions
    base_frag_types = [
        "b",
        "y",
        "b_H2O",
        "y_H2O",
        "b_NH3",
        "y_NH3",
        "b_modloss",
        "y_modloss",
    ]
    charged_frag_types = get_charged_frag_types(base_frag_types, 2)
    speclib_base = SpecLibBase(charged_frag_types=charged_frag_types)
    speclib_base.precursor_df = precursor_df

    # Calculate fragment m/z values
    speclib_base.calc_fragment_mz_df()

    # Create random intensities for fragments
    speclib_base._fragment_intensity_df = pd.DataFrame(
        np.random.rand(*speclib_base.fragment_mz_df.shape),
        columns=speclib_base.charged_frag_types,
    )

    # Convert to flat representation
    speclib_flat = SpecLibFlat()
    speclib_flat.parse_base_library(speclib_base)

    # Store original m/z values
    speclib_flat.fragment_df["mz_old"] = speclib_flat.fragment_df["mz"]

    # Convert back to dense matrices, including a and x ions that weren't in original data
    dense_frag_types = ["a", "b", "x", "y"]
    df_collection, frag_start_idx, frag_stop_idx = create_dense_matrices(
        speclib_flat.precursor_df,
        speclib_flat.fragment_df,
        get_charged_frag_types(dense_frag_types, 2),
        flat_columns=["intensity", "mz_old"],
    )

    # Verify the conversion
    assert "mz" in df_collection
    assert "mz_old" in df_collection

    # Get the column names for each ion type
    a_cols = [col for col in df_collection["mz"].columns if col.startswith("a_")]
    b_cols = [col for col in df_collection["mz"].columns if col.startswith("b_")]
    x_cols = [col for col in df_collection["mz"].columns if col.startswith("x_")]
    y_cols = [col for col in df_collection["mz"].columns if col.startswith("y_")]

    # Verify b and y ions have values
    assert not df_collection["mz_old"][b_cols].isna().all().all()
    assert not df_collection["mz_old"][y_cols].isna().all().all()

    # Verify a and x ions are empty (all zeros or NaN)
    assert (df_collection["mz_old"][a_cols] == 0).all().all()
    assert (df_collection["mz_old"][x_cols] == 0).all().all()

    # Compare original and reconstructed m/z values for b and y ions
    for col_type in [b_cols, y_cols]:
        mz_mask = df_collection["mz_old"][col_type].values != 0

        # Check that non-zero m/z values match within tolerance
        np.testing.assert_allclose(
            df_collection["mz"][col_type].values[mz_mask],
            df_collection["mz_old"][col_type].values[mz_mask],
            rtol=1e-6,
        )

    # Verify structure of output
    assert isinstance(frag_start_idx, np.ndarray)
    assert isinstance(frag_stop_idx, np.ndarray)
    assert len(frag_start_idx) == len(precursor_df)
    assert len(frag_stop_idx) == len(precursor_df)
    assert all(stop > start for start, stop in zip(frag_start_idx, frag_stop_idx))

    # Verify dimensions
    expected_rows = sum(
        (stop - start) for start, stop in zip(frag_start_idx, frag_stop_idx)
    )
    expected_cols = len(get_charged_frag_types(dense_frag_types, 2))
    assert df_collection["mz"].shape == (expected_rows, expected_cols)


def test_calc_dense_fragments():
    """Test calc_dense_fragments method of SpecLibFlat"""
    # Create test precursor data
    precursor_df = pd.DataFrame(
        {
            "sequence": ["PEPTIDE", "PROTEIN"],
            "mods": ["", "Acetyl@Any_N-term"],
            "mod_sites": ["", "0"],
            "charge": [2, 3],
            "nAA": [7, 7],
            "flat_frag_start_idx": [0, 4],
            "flat_frag_stop_idx": [4, 8],
        }
    )

    # Create test fragment data
    fragment_df = pd.DataFrame(
        {
            "type": [
                SERIES_MAPPING[Series.B],
                SERIES_MAPPING[Series.Y],
                SERIES_MAPPING[Series.B],
                SERIES_MAPPING[Series.Y],
            ]
            * 2,
            "position": [0, 0, 1, 1] * 2,
            "charge": [1, 1, 1, 1] * 2,
            "loss_type": [LOSS_MAPPING[Loss.NONE]] * 8,
            "intensity": [100, 200, 300, 400, 500, 600, 700, 800],
            "mz": [800, 700, 600, 500, 400, 300, 200, 100],
            "correlation": [10] * 8,
        }
    )

    # Initialize SpecLibFlat
    speclib_flat = SpecLibFlat()
    speclib_flat._precursor_df = precursor_df
    speclib_flat._fragment_df = fragment_df

    # Call calc_dense_fragments with additional columns
    speclib_flat.calc_dense_fragments(
        additional_columns=["intensity", "mz", "correlation"],
        charged_frag_types=["b_z1", "y_z1", "b_H2O_z1"],
    )

    # Verify the dense matrices were created
    assert hasattr(speclib_flat, "_fragment_mz_df")
    assert hasattr(speclib_flat, "_fragment_intensity_df")

    # Check dimensions of created matrices
    expected_rows = (
        12  # Total number of fragment positions (4 per peptide * 2 peptides)
    )
    expected_cols = 3  # Number of fragment types (b_z1, y_z1)
    assert speclib_flat._fragment_mz_df.shape == (expected_rows, expected_cols)
    assert speclib_flat._fragment_intensity_df.shape == (expected_rows, expected_cols)

    # Verify fragment indices were updated in precursor_df
    assert "frag_start_idx" in speclib_flat.precursor_df.columns
    assert "frag_stop_idx" in speclib_flat.precursor_df.columns

    np.testing.assert_array_equal(
        speclib_flat.precursor_df["frag_start_idx"].values, [0, 6]
    )
    np.testing.assert_array_equal(
        speclib_flat.precursor_df["frag_stop_idx"].values, [6, 12]
    )

    # Check intensity values are in correct positions
    expected_intensities = np.array(
        [
            [100, 200, 0],
            [300, 400, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [500, 600, 0],
            [700, 800, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(
        speclib_flat._fragment_intensity_df.values, expected_intensities
    )

    # After the intensity test, add m/z test
    expected_mz = np.array(
        [
            [800, 700, 0],
            [600, 500, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [400, 300, 0],
            [200, 100, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(speclib_flat._fragment_mz_df.values, expected_mz)

    # Add correlation test
    expected_correlation = np.array(
        [
            [10, 10, 0],
            [10, 10, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [10, 10, 0],
            [10, 10, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(
        speclib_flat._fragment_correlation_df.values, expected_correlation
    )
