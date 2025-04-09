from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from alpharaw.mzml import MzMLReader

from alphabase.spectral_library.annotate import (
    _add_frag_column_annotation,
    _get_dense_column,
    _mass_accuracy_metric,
    add_dense_lib,
    annotate_fragments_flat,
    annotate_precursors_flat,
    annotate_spectrum,
    calculate_pif,
)
from alphabase.spectral_library.flat import SpecLibFlat


@pytest.fixture
def sample_data():
    return {
        "spectrum_mz": np.array(
            [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 500.0]
        ),
        "fragment_mz": np.array([100.0, 200.001, 300.001, 500.0]),
        "fragment_type": np.array([1, 2, 3, 4], dtype=np.uint8),
        "fragment_loss_type": np.array([10, 20, 30, 40], dtype=np.uint8),
        "fragment_charge": np.array([1, 1, 2, 2], dtype=np.uint8),
        "fragment_number": np.array([1, 2, 3, 4], dtype=np.uint8),
        "fragment_position": np.array([10, 20, 30, 40], dtype=np.uint8),
    }


def test_annotate_spectrum_basic(sample_data):
    result = annotate_spectrum(**sample_data, mass_error_ppm=20.0)
    assert len(result) == 6
    assert all(len(arr) == len(sample_data["spectrum_mz"]) for arr in result)
    assert result[0].dtype == np.uint8
    assert result[5].dtype == np.float32


def test_annotate_spectrum_matching(sample_data):
    result = annotate_spectrum(**sample_data, mass_error_ppm=20.0)
    # Check matching peaks
    expected_types = np.array([255, 1, 255, 2, 255, 3, 255, 4], dtype=np.uint8)
    assert np.array_equal(result[0], expected_types)

    expected_loss_types = np.array([255, 10, 255, 20, 255, 30, 255, 40], dtype=np.uint8)
    assert np.array_equal(result[1], expected_loss_types)

    expected_charges = np.array([255, 1, 255, 1, 255, 2, 255, 2], dtype=np.uint8)
    assert np.array_equal(result[2], expected_charges)

    expected_numbers = np.array([255, 1, 255, 2, 255, 3, 255, 4], dtype=np.uint8)
    assert np.array_equal(result[3], expected_numbers)

    expected_positions = np.array([255, 10, 255, 20, 255, 30, 255, 40], dtype=np.uint8)
    assert np.array_equal(result[4], expected_positions)

    # Check errors
    assert np.all(
        result[5][[1, 3, 5, 7]] < 20.0
    )  # Errors for matching peaks should be less than 20 ppm
    assert np.all(
        np.isinf(result[5][[0, 2, 4, 6]])
    )  # Errors for non-matching peaks should be inf


def test_annotate_spectrum_no_match(sample_data):
    sample_data["spectrum_mz"] = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    result = annotate_spectrum(**sample_data, mass_error_ppm=20.0)
    assert np.all(result[0] == 255)
    assert np.all(np.isinf(result[5]))


def test_annotate_spectrum_mass_error_threshold(sample_data):
    # Test with 1 ppm (should match only the exact 100.0 m/z)
    result = annotate_spectrum(**sample_data, mass_error_ppm=1.0)
    expected_types = np.array([255, 1, 255, 255, 255, 255, 255, 4], dtype=np.uint8)
    assert np.array_equal(result[0], expected_types)
    assert np.all(np.isinf(result[5][[0, 2, 3, 4, 5, 6]]))
    assert np.all(result[5][[1, 7]] < 1.0)


def test_annotate_spectrum_empty_input():
    empty_data = {
        key: np.array([])
        for key in [
            "spectrum_mz",
            "fragment_mz",
            "fragment_type",
            "fragment_loss_type",
            "fragment_charge",
            "fragment_number",
            "fragment_position",
        ]
    }
    result = annotate_spectrum(**empty_data, mass_error_ppm=20.0)
    assert all(len(arr) == 0 for arr in result)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {"intensity": [100, 200, 300, 400, 500], "type": [1, 2, 255, 3, 255]}
    )


def test_calculate_pif_normal_case(sample_df):
    result = calculate_pif(sample_df)
    expected = (100 + 200 + 400) / (100 + 200 + 300 + 400 + 500)
    assert np.isclose(result, expected)


def test_calculate_pif_all_type_255():
    df = pd.DataFrame({"intensity": [100, 200, 300], "type": [255, 255, 255]})
    assert calculate_pif(df) == 0


def test_calculate_pif_no_type_255():
    df = pd.DataFrame({"intensity": [100, 200, 300], "type": [1, 2, 3]})
    assert calculate_pif(df) == 1


def test_calculate_pif_empty_dataframe():
    df = pd.DataFrame({"intensity": [], "type": []})
    assert calculate_pif(df) == 0


def test_calculate_pif_all_zero_intensity():
    df = pd.DataFrame({"intensity": [0, 0, 0], "type": [1, 2, 3]})
    assert calculate_pif(df) == 0


@pytest.fixture
def mock_raw_data():
    raw_data = MzMLReader()
    raw_data.spectrum_df = pd.DataFrame(
        {
            "spec_idx": [0, 1],
            "peak_start_idx": [0, 100],
            "peak_stop_idx": [100, 200],
            "rt": [10.5, 15.7],
            "precursor_mz": [500.5, 600.6],
        }
    )
    raw_data.peak_df = pd.DataFrame(
        {"mz": np.linspace(100, 1000, 200), "intensity": np.random.rand(200)}
    )
    return raw_data


@pytest.fixture
def mock_speclib_flat():
    speclib_flat = SpecLibFlat()
    speclib_flat.precursor_df = pd.DataFrame(
        {
            "sequence": ["PEPTIDE", "SEQUENCE"],
            "charge": [2, 3],
            "raw_name": ["file1.raw", "file2.raw"],
            "score": [0.9, 0.8],
            "proteins": ["ProtA", "ProtB"],
            "fdr": [0.01, 0.02],
            "spec_idx": [0, 1],
            "mod_sites": ["", ""],
            "mods": ["", ""],
            "mod_seq_hash": ["hash1", "hash2"],
            "mod_seq_charge_hash": ["hash1_2", "hash2_3"],
            "flat_frag_start_idx": [0, 10],
            "flat_frag_stop_idx": [10, 20],
            "precursor_mz": [500.5, 600.6],
        }
    )
    speclib_flat._fragment_df = pd.DataFrame(
        {
            "mz": np.linspace(100, 1000, 20),
            "type": np.random.randint(0, 5, 20),
            "loss_type": np.random.randint(0, 3, 20),
            "charge": np.random.randint(1, 4, 20),
            "number": np.random.randint(1, 10, 20),
            "position": np.random.randint(0, 7, 20),
        }
    )
    return speclib_flat


# Tests for annotate_precursors_flat


def test_annotate_precursors_flat_success(mock_speclib_flat, mock_raw_data):
    result = annotate_precursors_flat(mock_speclib_flat, mock_raw_data)
    assert isinstance(result, SpecLibFlat)
    assert "peak_start_idx" in result.precursor_df.columns
    assert "peak_stop_idx" in result.precursor_df.columns
    assert "rt" in result.precursor_df.columns
    assert len(result.precursor_df) == 2


def test_annotate_precursors_flat_spec_idx_offset(mock_speclib_flat, mock_raw_data):
    mock_speclib_flat.precursor_df["spec_idx"] = [1, 2]
    result = annotate_precursors_flat(
        mock_speclib_flat, mock_raw_data, spec_idx_offset=1
    )
    assert len(result.precursor_df) == 2
    assert np.all(
        result.precursor_df["spec_idx"] == mock_raw_data.spectrum_df["spec_idx"] + 1
    )


def test_annotate_precursors_flat_spec_idx_no_offset(mock_speclib_flat, mock_raw_data):
    # Use a spec_idx offset that doesn't match the raw data
    with pytest.raises(ValueError):
        annotate_precursors_flat(mock_speclib_flat, mock_raw_data, spec_idx_offset=1)


def test_annotate_precursors_flat_missing_columns(mock_speclib_flat, mock_raw_data):
    mock_speclib_flat.precursor_df = mock_speclib_flat.precursor_df.drop(
        columns=["sequence"]
    )
    with pytest.raises(ValueError):
        annotate_precursors_flat(mock_speclib_flat, mock_raw_data)


def test_annotate_precursors_flat_precursor_mz_mismatch(
    mock_speclib_flat, mock_raw_data
):
    mock_raw_data.spectrum_df["precursor_mz"] = [
        510.5,
        610.6,
    ]  # Deviate by more than 5 m/z
    with pytest.raises(
        ValueError,
        match="precursor_mz and precursor_mz_observed deviate by more than 5 m/z units",
    ):
        annotate_precursors_flat(mock_speclib_flat, mock_raw_data)


# Tests for annotate_fragments_flat


def test_annotate_fragments_flat_success(mock_speclib_flat, mock_raw_data):
    with (
        patch(
            "alphabase.spectral_library.annotate.annotate_spectrum"
        ) as mock_annotate_spectrum,
        patch(
            "alphabase.spectral_library.annotate.calculate_pif"
        ) as mock_calculate_pif,
    ):
        mock_annotate_spectrum.return_value = (
            np.zeros(100, dtype=np.uint8),
            np.zeros(100, dtype=np.uint8),
            np.zeros(100, dtype=np.uint8),
            np.zeros(100, dtype=np.uint8),
            np.zeros(100, dtype=np.uint8),
            np.zeros(100, dtype=np.float32),
        )
        mock_calculate_pif.return_value = 0.8

        annotated_speclib = annotate_precursors_flat(mock_speclib_flat, mock_raw_data)
        result = annotate_fragments_flat(annotated_speclib, mock_raw_data)

        assert isinstance(result, SpecLibFlat)
        assert "pif" in result.precursor_df.columns
        assert len(result._fragment_df) == 200  # 100 peaks per spectrum, 2 spectra
        assert all(result.precursor_df["pif"] == 0.8)
        assert mock_annotate_spectrum.call_count == 2
        assert mock_calculate_pif.call_count == 2


def test_annotate_fragments_flat_missing_columns(mock_speclib_flat, mock_raw_data):
    annotated_speclib = annotate_precursors_flat(mock_speclib_flat, mock_raw_data)
    annotated_speclib.precursor_df = annotated_speclib.precursor_df.drop(
        columns=["peak_start_idx"]
    )
    with pytest.raises(ValueError):
        annotate_fragments_flat(annotated_speclib, mock_raw_data)


def test_annotate_fragments_flat_empty_spectra(mock_speclib_flat, mock_raw_data):
    mock_raw_data.peak_df = pd.DataFrame(columns=["mz", "intensity"], dtype=np.float32)
    annotated_speclib = annotate_precursors_flat(mock_speclib_flat, mock_raw_data)
    result = annotate_fragments_flat(annotated_speclib, mock_raw_data)
    assert len(result._fragment_df) == 0
    assert all(result.precursor_df["pif"] == 0)


@pytest.fixture
def mappings():
    return {
        "frag_type_mapping": {98: "b", 121: "y"},
        "loss_type_mapping": {17: "NH3", 18: "H2O"},
        "charge_type_mapping": {1: "z1", 2: "z2"},
    }


@pytest.mark.parametrize(
    "type, loss_type, charge, expected",
    [
        (98, 17, 2, "b_NH3_z2"),  # All parameters present
        (121, 0, 1, "y_z1"),  # No loss type
        (99, 18, 2, "H2O_z2"),  # Unknown fragment type
        (0, 0, 0, ""),  # All unknown parameters
        (98, 0, 0, "b"),  # Only fragment type
        (121, 0, 2, "y_z2"),  # Fragment and charge, no loss type
    ],
)
def test_get_dense_column(type, loss_type, charge, expected, mappings):
    result = _get_dense_column(type, loss_type, charge, **mappings)
    assert result == expected


@pytest.fixture
def mock_speclib_annotation_flat():
    speclib_flat = SpecLibFlat()
    speclib_flat.precursor_df = pd.DataFrame(
        {
            "sequence": ["PEPTI", "SEQUE"],
            "charge": [2, 3],
            "raw_name": ["file1.raw", "file2.raw"],
            "score": [0.9, 0.8],
            "proteins": ["ProtA", "ProtB"],
            "fdr": [0.01, 0.02],
            "spec_idx": [1, 2],
            "mod_sites": ["", ""],
            "mods": ["", ""],
            "mod_seq_hash": ["hash1", "hash2"],
            "mod_seq_charge_hash": ["hash1_2", "hash2_3"],
            "flat_frag_start_idx": [0, 4],
            "flat_frag_stop_idx": [4, 8],
        }
    )
    speclib_flat._fragment_df = pd.DataFrame(
        {
            "mz": [100, 200, 300, 400, 500, 600, 700, 800],
            "type": [98, 121, 98, 121, 98, 121, 98, 121],
            "loss_type": [0, 0, 17, 17, 0, 0, 18, 18],
            "charge": [1, 1, 1, 1, 1, 1, 1, 1],
            "number": [1, 2, 3, 4, 5, 6, 7, 8],
            "intensity": [100, 200, 300, 400, 500, 600, 700, 800],
            "position": [0, 1, 2, 3, 0, 1, 2, 3],
            "error": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    return speclib_flat


def test_frag_column_annotation(mock_speclib_annotation_flat):
    _add_frag_column_annotation(
        mock_speclib_annotation_flat,
        charged_frag_types=[
            "b_z1",
            "y_z1",
            "b_NH3_z1",
            "y_NH3_z1",
            "b_H2O_z1",
            "y_H2O_z1",
        ],
    )
    assert mock_speclib_annotation_flat.fragment_df["frag_column"].tolist() == [
        "b_z1",
        "y_z1",
        "b_NH3_z1",
        "y_NH3_z1",
        "b_z1",
        "y_z1",
        "b_H2O_z1",
        "y_H2O_z1",
    ]


def test_add_dense_lib(mock_speclib_annotation_flat):
    outlib = add_dense_lib(
        mock_speclib_annotation_flat,
        charged_frag_types=[
            "b_z1",
            "y_z1",
            "b_NH3_z1",
            "y_NH3_z1",
            "b_H2O_z1",
            "y_H2O_z1",
        ],
    )

    assert np.all(
        outlib._fragment_intensity_df.sum(axis=1) == outlib.fragment_df["intensity"]
    )
    assert np.all(outlib.precursor_df["sequence_coverage"] == [1.0, 1.0])


def test_mass_accuracy_metric_normal_case():
    speclib_flat = SpecLibFlat()
    speclib_flat.precursor_df = pd.DataFrame(
        {
            "flat_frag_start_idx": [0, 3, 6],
            "flat_frag_stop_idx": [3, 6, 9],
            "sequence": ["PEP", "PEPT", "PEPTI"],
        }
    )
    speclib_flat._fragment_df = pd.DataFrame(
        {
            "error": [
                1.0,
                2.0,
                3.0,  # First precursor
                4.0,
                5.0,
                6.0,  # Second precursor
                7.0,
                8.0,
                9.0,
            ]  # Third precursor
        }
    )
    print(speclib_flat.precursor_df)

    _mass_accuracy_metric(speclib_flat)
    expected = np.array([2.0, 5.0, 8.0])  # Medians for each group
    assert np.allclose(speclib_flat.precursor_df["mass_accuracy"], expected)


def test_mass_accuracy_metric_with_nans():
    speclib_flat = SpecLibFlat()
    speclib_flat.precursor_df = pd.DataFrame(
        {
            "flat_frag_start_idx": [0, 3],
            "flat_frag_stop_idx": [3, 6],
            "sequence": ["PEPTIDE", "SEQUENCE"],
        }
    )
    speclib_flat._fragment_df = pd.DataFrame(
        {
            "error": [
                1.0,
                np.nan,
                3.0,  # First precursor
                np.nan,
                np.nan,
                np.nan,
            ]  # Second precursor
        }
    )

    _mass_accuracy_metric(speclib_flat)
    expected = np.array([2.0, np.inf])  # Median of [1.0, 3.0] and inf for all NaNs
    assert np.allclose(
        speclib_flat.precursor_df["mass_accuracy"], expected, equal_nan=True
    )


def test_mass_accuracy_metric_with_infs():
    speclib_flat = SpecLibFlat()
    speclib_flat.precursor_df = pd.DataFrame(
        {
            "flat_frag_start_idx": [0, 3],
            "flat_frag_stop_idx": [3, 6],
            "sequence": ["PEPTIDE", "SEQUENCE"],
        }
    )
    speclib_flat._fragment_df = pd.DataFrame(
        {
            "error": [
                1.0,
                np.inf,
                3.0,  # First precursor
                np.inf,
                np.inf,
                np.inf,
            ]  # Second precursor
        }
    )

    _mass_accuracy_metric(speclib_flat)
    expected = np.array([2.0, np.inf])  # Median of [1.0, 3.0] and inf for all infs
    assert np.allclose(
        speclib_flat.precursor_df["mass_accuracy"], expected, equal_nan=True
    )


def test_mass_accuracy_metric_empty_fragments():
    speclib_flat = SpecLibFlat()
    speclib_flat.precursor_df = pd.DataFrame(
        {
            "flat_frag_start_idx": [0, 0],
            "flat_frag_stop_idx": [0, 0],
            "sequence": ["PEPTIDE", "SEQUENCE"],
        }
    )
    speclib_flat._fragment_df = pd.DataFrame({"error": []})

    _mass_accuracy_metric(speclib_flat)
    expected = np.array([np.inf, np.inf])  # inf for empty fragments
    assert np.allclose(
        speclib_flat.precursor_df["mass_accuracy"], expected, equal_nan=True
    )
