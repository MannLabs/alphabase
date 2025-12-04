from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alphabase.spectral_library.annotate import (
    add_dense_lib,
    annotate_fragments_flat,
    annotate_precursors_flat,
    annotate_spectrum,
)
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.flat import SpecLibFlat
from alphabase.spectral_library.metrics import (
    _mass_accuracy_metric,
    _pif_metric,
    calculate_pif,
)


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
    result = calculate_pif(
        sample_df["intensity"].to_numpy(), sample_df["type"].to_numpy()
    )
    expected = (100 + 200 + 400) / (100 + 200 + 300 + 400 + 500)
    assert np.isclose(result, expected)


def test_calculate_pif_all_type_255():
    intensities = np.array([100, 200, 300])
    types = np.array([255, 255, 255], dtype=np.uint8)
    assert calculate_pif(intensities, types) == 0


def test_calculate_pif_no_type_255():
    intensities = np.array([100, 200, 300])
    types = np.array([1, 2, 3], dtype=np.uint8)
    assert calculate_pif(intensities, types) == 1


def test_calculate_pif_empty_input():
    intensities = np.array([], dtype=np.float64)
    types = np.array([], dtype=np.uint8)
    assert calculate_pif(intensities, types) == 0


def test_calculate_pif_all_zero_intensity():
    intensities = np.array([0, 0, 0])
    types = np.array([1, 2, 3], dtype=np.uint8)
    assert calculate_pif(intensities, types) == 0


def test_pif_metric_updates_precursor_df(pif_speclib_flat):
    _pif_metric(pif_speclib_flat)
    expected = np.array([0.5, 4 / 9])
    assert np.allclose(pif_speclib_flat.precursor_df["pif"], expected)


@pytest.fixture
def pif_speclib_flat():
    flat = SpecLibFlat()
    flat.precursor_df = pd.DataFrame(
        {
            "sequence": ["PEP", "PEPT"],
            "mods": ["", ""],
            "mod_sites": ["", ""],
            "charge": [2, 2],
            "flat_frag_start_idx": [0, 3],
            "flat_frag_stop_idx": [3, 5],
        }
    )
    flat._fragment_df = pd.DataFrame(
        {
            "intensity": [100.0, 200.0, 300.0, 400.0, 500.0],
            "type": np.array([1, 2, 255, 3, 255], dtype=np.uint8),
        }
    )
    return flat


@pytest.fixture
def mock_raw_data():
    raw_data_spectrum_df = pd.DataFrame(
        {
            "spec_idx": [0, 1],
            "peak_start_idx": [0, 100],
            "peak_stop_idx": [100, 200],
            "rt": [10.5, 15.7],
            "precursor_mz": [500.5, 600.6],
        }
    )
    raw_data_peak_df = pd.DataFrame(
        {"mz": np.linspace(100, 1000, 200), "intensity": np.random.rand(200)}
    )
    return raw_data_spectrum_df, raw_data_peak_df


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
    mock_raw_data_spectrum_df, _mock_raw_data_peak_df = mock_raw_data
    result = annotate_precursors_flat(mock_speclib_flat, mock_raw_data_spectrum_df)
    assert isinstance(result, SpecLibFlat)
    assert "peak_start_idx" in result.precursor_df.columns
    assert "peak_stop_idx" in result.precursor_df.columns
    assert "rt" in result.precursor_df.columns
    assert len(result.precursor_df) == 2


def test_annotate_precursors_flat_spec_idx_offset(mock_speclib_flat, mock_raw_data):
    mock_raw_data_spectrum_df, _mock_raw_data_peak_df = mock_raw_data
    mock_speclib_flat.precursor_df["spec_idx"] = [1, 2]
    result = annotate_precursors_flat(
        mock_speclib_flat, mock_raw_data_spectrum_df, spec_idx_offset=1
    )
    assert len(result.precursor_df) == 2
    assert np.all(
        result.precursor_df["spec_idx"] == mock_raw_data_spectrum_df["spec_idx"] + 1
    )


def test_annotate_precursors_flat_spec_idx_no_offset(mock_speclib_flat, mock_raw_data):
    mock_raw_data_spectrum_df, _mock_raw_data_peak_df = mock_raw_data
    # Use a spec_idx offset that doesn't match the raw data
    with pytest.raises(ValueError):
        annotate_precursors_flat(
            mock_speclib_flat, mock_raw_data_spectrum_df, spec_idx_offset=1
        )


def test_annotate_precursors_flat_missing_columns(mock_speclib_flat, mock_raw_data):
    mock_raw_data_spectrum_df, _mock_raw_data_peak_df = mock_raw_data
    mock_speclib_flat.precursor_df = mock_speclib_flat.precursor_df.drop(
        columns=["sequence"]
    )
    with pytest.raises(ValueError):
        annotate_precursors_flat(mock_speclib_flat, mock_raw_data_spectrum_df)


def test_annotate_precursors_flat_precursor_mz_mismatch(
    mock_speclib_flat, mock_raw_data
):
    mock_raw_data_spectrum_df, _mock_raw_data_peak_df = mock_raw_data
    mock_raw_data_spectrum_df["precursor_mz"] = [
        510.5,
        610.6,
    ]  # Deviate by more than 5 m/z
    with pytest.raises(
        ValueError,
        match="precursor_mz and precursor_mz_observed deviate by more than 5 m/z units",
    ):
        annotate_precursors_flat(mock_speclib_flat, mock_raw_data_spectrum_df)


# Tests for annotate_fragments_flat


def test_annotate_fragments_flat_success(mock_speclib_flat, mock_raw_data):
    mock_raw_data_spectrum_df, mock_raw_data_peak_df = mock_raw_data
    with patch(
        "alphabase.spectral_library.annotate.annotate_spectrum"
    ) as mock_annotate_spectrum:
        mock_annotate_spectrum.return_value = (
            np.zeros(100, dtype=np.uint8),
            np.zeros(100, dtype=np.uint8),
            np.zeros(100, dtype=np.uint8),
            np.zeros(100, dtype=np.uint8),
            np.zeros(100, dtype=np.uint8),
            np.zeros(100, dtype=np.float32),
        )

        annotated_speclib = annotate_precursors_flat(
            mock_speclib_flat, mock_raw_data_spectrum_df
        )
        result = annotate_fragments_flat(annotated_speclib, mock_raw_data_peak_df)

        assert isinstance(result, SpecLibFlat)
        assert len(result._fragment_df) == 200  # 100 peaks per spectrum, 2 spectra
        assert mock_annotate_spectrum.call_count == 2


def test_annotate_fragments_flat_missing_columns(mock_speclib_flat, mock_raw_data):
    mock_raw_data_spectrum_df, mock_raw_data_peak_df = mock_raw_data
    annotated_speclib = annotate_precursors_flat(
        mock_speclib_flat, mock_raw_data_spectrum_df
    )
    annotated_speclib.precursor_df = annotated_speclib.precursor_df.drop(
        columns=["peak_start_idx"]
    )
    with pytest.raises(ValueError):
        annotate_fragments_flat(annotated_speclib, mock_raw_data_peak_df)


def test_annotate_fragments_flat_empty_spectra(mock_speclib_flat, mock_raw_data):
    mock_raw_data_spectrum_df, mock_raw_data_peak_df = mock_raw_data
    mock_raw_data_peak_df = pd.DataFrame(columns=["mz", "intensity"], dtype=np.float32)
    annotated_speclib = annotate_precursors_flat(
        mock_speclib_flat, mock_raw_data_spectrum_df
    )
    result = annotate_fragments_flat(annotated_speclib, mock_raw_data_peak_df)
    assert len(result._fragment_df) == 0


@pytest.fixture
def mock_speclib_annotation_flat():
    charged_frag_types = [
        "b_z1",
        "y_z1",
        "b_NH3_z1",
        "y_NH3_z1",
        "b_H2O_z1",
        "y_H2O_z1",
    ]

    speclib_base = SpecLibBase(charged_frag_types=charged_frag_types)
    speclib_base.precursor_df = pd.DataFrame(
        {
            "sequence": ["PEPTI", "SEQUE"],
            "mods": ["", ""],
            "mod_sites": ["", ""],
            "charge": [2, 3],
            "spec_idx": [1, 2],
            "precursor_mz": [500.5, 600.6],
            "rt": [10.0, 20.0],
            "nce": [30.0, 32.0],
            "activation": ["HCD", "HCD"],
        }
    )
    speclib_base.calc_fragment_mz_df()
    speclib_base._fragment_intensity_df = speclib_base.fragment_mz_df.copy()

    speclib_flat = SpecLibFlat(charged_frag_types=charged_frag_types)
    speclib_flat.parse_base_library(speclib_base)

    return speclib_flat


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

    assert np.isclose(
        outlib._fragment_intensity_df.values.sum(),
        outlib.fragment_df["intensity"].sum(),
    )
    assert np.all(outlib.precursor_df["sequence_coverage"] == [1.0, 1.0])
    assert np.allclose(outlib.precursor_df["pif"], 1.0)


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


def test_mass_accuracy_metric_index_mismatch():
    """Verify that mass_accuracy uses correct indices (flat vs dense).

    This test creates a scenario where flat and dense indices differ significantly.
    If _mass_accuracy_metric incorrectly uses dense indices (frag_start_idx) to
    slice the flat _fragment_df, the results will be wrong.
    """
    speclib_flat = SpecLibFlat()
    speclib_flat.precursor_df = pd.DataFrame(
        {
            "frag_start_idx": [0, 4],
            "frag_stop_idx": [4, 8],
            "flat_frag_start_idx": [0, 100],
            "flat_frag_stop_idx": [100, 200],
            "sequence": ["PEPTIDE", "SEQUENCE"],
        }
    )

    # Flat fragment_df has 200 rows total
    # First 100 (precursor 0's peaks) have error=1.0
    # Next 100 (precursor 1's peaks) have error=2.0
    speclib_flat._fragment_df = pd.DataFrame({"error": [1.0] * 100 + [2.0] * 100})

    _mass_accuracy_metric(speclib_flat)

    expected = np.array([1.0, 2.0])
    actual = speclib_flat.precursor_df["mass_accuracy"].to_numpy()
    assert np.allclose(actual, expected), (
        f"mass_accuracy uses wrong indices! Expected {expected}, got {actual}. "
        "Should use flat_frag_start_idx/flat_frag_stop_idx, not frag_start_idx/frag_stop_idx."
    )
