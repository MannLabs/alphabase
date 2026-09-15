from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alphabase.peptide.fragment import (
    PEAK_INTENSITY_DTYPE,
    PEAK_MZ_DTYPE,
    filter_valid_charged_frag_types,
    flatten_fragments,
    parse_charged_frag_type,
)


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("b_z1", ("b", 1)),
        ("b_modloss_z2", ("b_modloss", 2)),
    ],
)
def test_parse_charged_frag_type_with_valid_input(input_str, expected):
    """Test parse_charged_frag_type with valid input."""
    result = parse_charged_frag_type(input_str)
    assert result == (expected[0], expected[1])


@pytest.mark.parametrize(
    "input_str, match",
    [
        ("b_z1_z2", "Only charged fragment types are supported"),
        ("b_z1.5", "Charge state must be a positive integer"),
        ("b_z0", "Charge state must be a positive integer"),
        ("b_z-1", "Charge state must be a positive integer"),
        ("unsupported_z1", "Fragment type unsupported is currently not supported"),
    ],
)
def test_parse_charged_frag_type_with_exceptions(input_str, match):
    """Test parse_charged_frag_type handles errors correctly."""
    with pytest.raises(ValueError, match=match):
        parse_charged_frag_type(input_str)


@patch("alphabase.peptide.fragment.parse_charged_frag_type")
def test_filter_valid_charged_frag_types(mock_parse):
    """Test filter_valid_charged_frag_types handles errors correctly."""
    mock_parse.side_effect = [("b", 1), ValueError, ("y", 2)]
    with pytest.warns(UserWarning) as recorded_warnings:
        result = filter_valid_charged_frag_types(
            [
                "b_z1",
                "unsupported_z1",
                "y_z2",
            ]
        )
    assert result == ["b_z1", "y_z2"]
    assert len(recorded_warnings) == 1  # Should have 2 warning messages


# one reverse series without loss and one forward series with loss, so that direction,
# series and loss all vary over two columns
CHARGED_FRAG_TYPES = ["y_z1", "b_modloss_z1"]
N_PRECURSORS = 2
ROWS_PER_PRECURSOR = 2
# mirrors the max_frag_per_peptide default of fill_in_indices
MAX_FRAG_PER_PEPTIDE = 300
# each mz encodes its own slot as 100 + row * 10 + column
MZ = [
    # y_z1, b_modloss_z1
    [100.0, 0.0],  # precursor 0, row 0, unmodified: no modloss fragment
    [110.0, 0.0],  # precursor 0, row 1, unmodified: no modloss fragment
    [120.0, 121.0],  # precursor 1, row 0
    [0.0, 131.0],  # precursor 1, row 1, y_z1 outside the mz range
]
# distinct over the whole library, so the top k selection is unambiguous
INTENSITY = [
    [0.11, 0.01],
    [0.82, 0.02],
    [0.93, 0.48],
    [0.03, 0.61],
]
# slots kept when only the mz == 0 padding is dropped
KEEP_UNFILTERED = [
    [1, 0],
    [1, 0],
    [1, 1],
    [0, 1],
]
# keep_top_k_fragments, min_fragment_intensity, kept slots per dense row
FILTER_CASES = [
    pytest.param(1000, -1, KEEP_UNFILTERED, id="padding_only"),
    pytest.param(
        1000,
        0.3,
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ],
        id="min_intensity",
    ),
    pytest.param(
        1,
        -1,
        [
            [0, 0],
            [1, 0],
            [1, 0],
            [0, 0],
        ],
        id="top_1_per_precursor",
    ),
]


@pytest.fixture
def library():
    """Give MZ and INTENSITY as fragment frames with the matching precursor pointers."""
    frag_start_idx = np.arange(N_PRECURSORS) * ROWS_PER_PRECURSOR
    precursor_df = pd.DataFrame(
        {
            "frag_start_idx": frag_start_idx,
            "frag_stop_idx": frag_start_idx + ROWS_PER_PRECURSOR,
        }
    )
    return (
        precursor_df,
        pd.DataFrame(np.array(MZ, dtype=PEAK_MZ_DTYPE), columns=CHARGED_FRAG_TYPES),
        pd.DataFrame(
            np.array(INTENSITY, dtype=PEAK_INTENSITY_DTYPE), columns=CHARGED_FRAG_TYPES
        ),
    )


def _flat(values, dtype):
    """Flatten a dense grid to the slot order of the flat fragment dataframe."""
    return np.array(values, dtype=dtype).reshape(-1)


@pytest.mark.requires_numba
@pytest.mark.parametrize(
    "keep_top_k_fragments, min_fragment_intensity, expected_keep", FILTER_CASES
)
def test_flatten_fragments_retains_expected_fragments(
    library, keep_top_k_fragments, min_fragment_intensity, expected_keep
):
    """The flat library keeps only the slots listed in FILTER_CASES."""
    # Given
    precursor_df, mz_df, intensity_df = library
    keep = _flat(expected_keep, bool)

    # When
    _, frag_df = flatten_fragments(
        precursor_df,
        mz_df,
        intensity_df,
        min_fragment_intensity=min_fragment_intensity,
        keep_top_k_fragments=keep_top_k_fragments,
    )

    # Then
    expected_df = pd.DataFrame(
        {
            "mz": _flat(MZ, PEAK_MZ_DTYPE)[keep],
            "intensity": _flat(INTENSITY, PEAK_INTENSITY_DTYPE)[keep],
        }
    )
    pd.testing.assert_frame_equal(frag_df[["mz", "intensity"]], expected_df)


@pytest.mark.requires_numba
def test_flatten_fragments_annotates_retained_fragments(library):
    """Each annotation column describes the dense slot of its fragment."""
    # Given
    precursor_df, mz_df, intensity_df = library
    # position is the dense row within the precursor, number counts the ion series:
    # b_modloss forward from position 0, y backward over ROWS_PER_PRECURSOR rows
    expected_df = pd.DataFrame(
        {
            "mz": np.array([100.0, 110.0, 120.0, 121.0, 131.0], dtype=PEAK_MZ_DTYPE),
            "intensity": np.array(
                [0.11, 0.82, 0.93, 0.48, 0.61], dtype=PEAK_INTENSITY_DTYPE
            ),
            "type": np.array([121, 121, 121, 98, 98], dtype=np.int8),
            "loss_type": np.array([0, 0, 0, 98, 98], dtype=np.int16),
            "charge": np.array([1, 1, 1, 1, 1], dtype=np.int8),
            "number": np.array([2, 1, 2, 1, 2], dtype=np.uint32),
            "position": np.array([0, 1, 0, 0, 1], dtype=np.uint32),
        }
    )

    # When
    _, frag_df = flatten_fragments(precursor_df, mz_df, intensity_df)

    # Then
    pd.testing.assert_frame_equal(frag_df, expected_df)


@pytest.mark.requires_numba
@pytest.mark.parametrize(
    "keep_top_k_fragments, min_fragment_intensity, expected_keep", FILTER_CASES
)
def test_flatten_fragments_reannotates_precursor_pointers(
    library, keep_top_k_fragments, min_fragment_intensity, expected_keep
):
    """The flat pointers of a precursor address only its own fragments."""
    # Given
    precursor_df, mz_df, intensity_df = library
    n_types = len(CHARGED_FRAG_TYPES)
    mz = _flat(MZ, PEAK_MZ_DTYPE)
    keep = _flat(expected_keep, bool)

    # When
    precursor_df, frag_df = flatten_fragments(
        precursor_df,
        mz_df,
        intensity_df,
        min_fragment_intensity=min_fragment_intensity,
        keep_top_k_fragments=keep_top_k_fragments,
    )

    # Then
    for row in precursor_df.itertuples():
        block = slice(row.frag_start_idx * n_types, row.frag_stop_idx * n_types)
        np.testing.assert_array_equal(
            frag_df["mz"].values[row.flat_frag_start_idx : row.flat_frag_stop_idx],
            mz[block][keep[block]],
        )

    # the pointers must cover the fragment dataframe with no gap and no overlap
    assert precursor_df.flat_frag_start_idx.iloc[0] == 0
    assert precursor_df.flat_frag_stop_idx.iloc[-1] == len(frag_df)
    np.testing.assert_array_equal(
        precursor_df.flat_frag_start_idx.values[1:],
        precursor_df.flat_frag_stop_idx.values[:-1],
    )


@pytest.mark.requires_numba
def test_flatten_fragments_filters_custom_df_columns(library):
    """flatten_fragments filters a custom_df column like the mz column."""
    # Given
    precursor_df, mz_df, intensity_df = library
    cardinality = np.arange(mz_df.size, dtype=np.uint8).reshape(mz_df.shape)
    cardinality_df = pd.DataFrame(cardinality, columns=CHARGED_FRAG_TYPES)

    # When
    _, frag_df = flatten_fragments(
        precursor_df,
        mz_df,
        intensity_df,
        custom_df={"cardinality": cardinality_df},
    )

    # Then
    np.testing.assert_array_equal(
        frag_df["cardinality"].values,
        cardinality.reshape(-1)[_flat(KEEP_UNFILTERED, bool)],
    )


@pytest.mark.requires_numba
def test_flatten_fragments_without_intensity(library):
    """Without intensities, flatten_fragments removes only the mz == 0 padding."""
    # Given
    precursor_df, mz_df, _ = library

    # When
    _, frag_df = flatten_fragments(precursor_df, mz_df, pd.DataFrame())

    # Then
    assert "intensity" not in frag_df.columns
    np.testing.assert_array_equal(
        frag_df["mz"].values, _flat(MZ, PEAK_MZ_DTYPE)[_flat(KEEP_UNFILTERED, bool)]
    )


@pytest.mark.requires_numba
def test_flatten_fragments_selects_custom_columns(library):
    """flatten_fragments creates only the requested annotation columns."""
    # Given
    precursor_df, mz_df, intensity_df = library

    # When
    _, frag_df = flatten_fragments(
        precursor_df, mz_df, intensity_df, custom_columns=["number", "charge"]
    )

    # Then
    assert list(frag_df.columns) == ["mz", "intensity", "charge", "number"]


@pytest.mark.requires_numba
def test_flatten_fragments_empty_precursor_df(library):
    """An empty library gives an empty fragment dataframe."""
    # Given
    _, mz_df, intensity_df = library

    # When
    precursor_df, frag_df = flatten_fragments(
        pd.DataFrame({"frag_start_idx": [], "frag_stop_idx": []}), mz_df, intensity_df
    )

    # Then
    assert len(precursor_df) == 0
    assert len(frag_df) == 0


@pytest.mark.requires_numba
def test_flatten_fragments_long_precursor():
    """A long precursor keeps fragment numbers above 255, and the columns stay uint32."""
    # position (n_rows - 1) and number (n_rows) must both exceed the uint8 range
    n_rows = np.iinfo(np.uint8).max + 2
    assert (
        n_rows <= MAX_FRAG_PER_PEPTIDE
    ), "fill_in_indices cannot index a precursor this long"
    n_types = len(CHARGED_FRAG_TYPES)
    rng = np.random.default_rng(0)
    mz_df = pd.DataFrame(
        (rng.random((n_rows, n_types)) * 1000 + 100).astype(PEAK_MZ_DTYPE),
        columns=CHARGED_FRAG_TYPES,
    )
    intensity_df = pd.DataFrame(
        rng.random((n_rows, n_types)).astype(PEAK_INTENSITY_DTYPE),
        columns=CHARGED_FRAG_TYPES,
    )
    precursor_df = pd.DataFrame({"frag_start_idx": [0], "frag_stop_idx": [n_rows]})

    _, frag_df = flatten_fragments(
        precursor_df, mz_df, intensity_df, keep_top_k_fragments=n_rows * n_types
    )

    assert frag_df["position"].max() == n_rows - 1
    assert frag_df["number"].max() == n_rows
    assert frag_df["position"].dtype == np.uint32
    assert frag_df["number"].dtype == np.uint32
