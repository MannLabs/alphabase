from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alphabase.peptide.fragment import (
    FRAGMENT_TYPES,
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


CHARGED_FRAG_TYPES = [
    "b_z1",
    "b_z2",
    "y_z1",
    "y_z2",
    "b_modloss_z1",
    "y_modloss_z1",
]
ROWS_PER_PRECURSOR = 4
N_PRECURSORS = 5


def _dense_library():
    """Make dense fragment frames with padding, and the matching precursor pointers."""
    rng = np.random.default_rng(0)
    n_rows = N_PRECURSORS * ROWS_PER_PRECURSOR
    n_types = len(CHARGED_FRAG_TYPES)

    mz = (rng.random((n_rows, n_types)) * 1000 + 100).astype(PEAK_MZ_DTYPE)
    # unmodified precursors have no modloss fragments, and some fragments fall
    # outside the mz range. Both cases give mz == 0 padding.
    mz[: 2 * ROWS_PER_PRECURSOR, 4:] = 0
    mz[3, 0] = 0
    # distinct intensities make the top-k selection unambiguous
    intensity = rng.permutation(n_rows * n_types).reshape(n_rows, n_types)
    intensity = (intensity / intensity.max()).astype(PEAK_INTENSITY_DTYPE)

    frag_start_idx = np.arange(N_PRECURSORS) * ROWS_PER_PRECURSOR
    precursor_df = pd.DataFrame(
        {
            "frag_start_idx": frag_start_idx,
            "frag_stop_idx": frag_start_idx + ROWS_PER_PRECURSOR,
        }
    )
    return (
        precursor_df,
        pd.DataFrame(mz, columns=CHARGED_FRAG_TYPES),
        pd.DataFrame(intensity, columns=CHARGED_FRAG_TYPES),
    )


def _expected_keep_mask(
    precursor_df, mz_df, intensity_df, keep_top_k_fragments, min_fragment_intensity
):
    """Give the keep mask over all dense slots. This does not use flatten_fragments."""
    n_types = mz_df.shape[1]
    mz = mz_df.values.reshape(-1)
    intensity = None if len(intensity_df) == 0 else intensity_df.values.reshape(-1)

    mask = np.zeros(mz.size, dtype=bool)
    for start, stop in zip(precursor_df.frag_start_idx, precursor_df.frag_stop_idx):
        block = slice(start * n_types, stop * n_types)
        keep = mz[block] != 0
        if intensity is not None:
            keep &= intensity[block] >= min_fragment_intensity
            n_slots = (stop - start) * n_types
            if keep_top_k_fragments < n_slots:
                in_top_k = np.zeros(n_slots, dtype=bool)
                in_top_k[np.argsort(intensity[block])[-keep_top_k_fragments:]] = True
                keep &= in_top_k
        mask[block] = keep
    return mask


@pytest.mark.requires_numba
@pytest.mark.parametrize(
    "keep_top_k_fragments, min_fragment_intensity",
    [(1000, -1), (1000, 0.5), (8, -1), (5, 0.2), (1, -1)],
)
def test_flatten_fragments_retains_expected_fragments(
    keep_top_k_fragments, min_fragment_intensity
):
    """The flat library keeps only the slots that pass all filters."""
    precursor_df, mz_df, intensity_df = _dense_library()
    expected_mask = _expected_keep_mask(
        precursor_df,
        mz_df,
        intensity_df,
        keep_top_k_fragments,
        min_fragment_intensity,
    )

    _, frag_df = flatten_fragments(
        precursor_df,
        mz_df,
        intensity_df,
        min_fragment_intensity=min_fragment_intensity,
        keep_top_k_fragments=keep_top_k_fragments,
    )

    assert len(frag_df) == expected_mask.sum()
    np.testing.assert_array_equal(
        frag_df["mz"].values, mz_df.values.reshape(-1)[expected_mask]
    )
    np.testing.assert_array_equal(
        frag_df["intensity"].values, intensity_df.values.reshape(-1)[expected_mask]
    )


@pytest.mark.requires_numba
@pytest.mark.parametrize("keep_top_k_fragments", [1000, 5])
def test_flatten_fragments_annotates_retained_fragments(keep_top_k_fragments):
    """Each annotation column describes the dense slot of its fragment."""
    precursor_df, mz_df, intensity_df = _dense_library()
    n_types = mz_df.shape[1]
    expected_mask = _expected_keep_mask(
        precursor_df, mz_df, intensity_df, keep_top_k_fragments, -1
    )
    kept_slots = np.flatnonzero(expected_mask)
    kept_columns = kept_slots % n_types
    kept_rows = kept_slots // n_types

    _, frag_df = flatten_fragments(
        precursor_df, mz_df, intensity_df, keep_top_k_fragments=keep_top_k_fragments
    )

    frag_types = [parse_charged_frag_type(col) for col in CHARGED_FRAG_TYPES]
    np.testing.assert_array_equal(
        frag_df["charge"].values,
        np.array([charge for _, charge in frag_types])[kept_columns],
    )
    np.testing.assert_array_equal(
        frag_df["type"].values,
        np.array([FRAGMENT_TYPES[name].series_id for name, _ in frag_types])[
            kept_columns
        ],
    )
    np.testing.assert_array_equal(
        frag_df["loss_type"].values,
        np.array([FRAGMENT_TYPES[name].loss_id for name, _ in frag_types])[
            kept_columns
        ],
    )

    # position counts the fragment rows of a precursor. number counts the ion
    # series in the direction of the fragment type.
    expected_position = kept_rows % ROWS_PER_PRECURSOR
    np.testing.assert_array_equal(frag_df["position"].values, expected_position)
    directions = np.array(
        [FRAGMENT_TYPES[name].direction_id for name, _ in frag_types]
    )[kept_columns]
    expected_number = np.where(
        directions == 1, expected_position + 1, ROWS_PER_PRECURSOR - expected_position
    )
    np.testing.assert_array_equal(frag_df["number"].values, expected_number)


@pytest.mark.requires_numba
@pytest.mark.parametrize("keep_top_k_fragments", [1000, 5])
def test_flatten_fragments_reannotates_precursor_pointers(keep_top_k_fragments):
    """The flat pointers of a precursor address only its own fragments."""
    precursor_df, mz_df, intensity_df = _dense_library()
    n_types = mz_df.shape[1]
    mz = mz_df.values.reshape(-1)
    expected_mask = _expected_keep_mask(
        precursor_df, mz_df, intensity_df, keep_top_k_fragments, -1
    )

    precursor_df, frag_df = flatten_fragments(
        precursor_df, mz_df, intensity_df, keep_top_k_fragments=keep_top_k_fragments
    )

    for row in precursor_df.itertuples():
        block = slice(row.frag_start_idx * n_types, row.frag_stop_idx * n_types)
        np.testing.assert_array_equal(
            frag_df["mz"].values[row.flat_frag_start_idx : row.flat_frag_stop_idx],
            mz[block][expected_mask[block]],
        )

    # the pointers must cover the fragment dataframe with no gap and no overlap
    assert precursor_df.flat_frag_start_idx.iloc[0] == 0
    assert precursor_df.flat_frag_stop_idx.iloc[-1] == len(frag_df)
    np.testing.assert_array_equal(
        precursor_df.flat_frag_start_idx.values[1:],
        precursor_df.flat_frag_stop_idx.values[:-1],
    )


@pytest.mark.requires_numba
def test_flatten_fragments_filters_custom_df_columns():
    """flatten_fragments filters a custom_df column like the mz column."""
    precursor_df, mz_df, intensity_df = _dense_library()
    cardinality_df = pd.DataFrame(
        np.arange(mz_df.size, dtype=np.uint8).reshape(mz_df.shape),
        columns=CHARGED_FRAG_TYPES,
    )
    expected_mask = _expected_keep_mask(precursor_df, mz_df, intensity_df, 5, -1)

    _, frag_df = flatten_fragments(
        precursor_df,
        mz_df,
        intensity_df,
        keep_top_k_fragments=5,
        custom_df={"cardinality": cardinality_df},
    )

    assert "cardinality" in frag_df.columns
    np.testing.assert_array_equal(
        frag_df["cardinality"].values,
        cardinality_df.values.reshape(-1)[expected_mask],
    )


@pytest.mark.requires_numba
def test_flatten_fragments_without_intensity():
    """Without intensities, flatten_fragments removes only the mz == 0 padding."""
    precursor_df, mz_df, _ = _dense_library()
    expected_mask = _expected_keep_mask(precursor_df, mz_df, pd.DataFrame(), 1000, -1)

    _, frag_df = flatten_fragments(precursor_df, mz_df, pd.DataFrame())

    assert "intensity" not in frag_df.columns
    np.testing.assert_array_equal(
        frag_df["mz"].values, mz_df.values.reshape(-1)[expected_mask]
    )


@pytest.mark.requires_numba
def test_flatten_fragments_selects_custom_columns():
    """flatten_fragments creates only the requested annotation columns."""
    precursor_df, mz_df, intensity_df = _dense_library()

    _, frag_df = flatten_fragments(
        precursor_df, mz_df, intensity_df, custom_columns=["number", "charge"]
    )

    assert list(frag_df.columns) == ["mz", "intensity", "charge", "number"]


@pytest.mark.requires_numba
def test_flatten_fragments_empty_precursor_df():
    """An empty library gives an empty fragment dataframe."""
    _, mz_df, intensity_df = _dense_library()

    precursor_df, frag_df = flatten_fragments(
        pd.DataFrame({"frag_start_idx": [], "frag_stop_idx": []}), mz_df, intensity_df
    )

    assert len(precursor_df) == 0
    assert len(frag_df) == 0
