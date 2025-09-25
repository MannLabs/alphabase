"""Module for annotating spectral libraries with raw mass spectrometry data."""

from typing import List, Tuple, Union

import numba as nb
import numpy as np
import pandas as pd

from alphabase.peptide.fragment import (
    create_dense_matrices,
    filter_valid_charged_frag_types,
)
from alphabase.spectral_library.flat import SpecLibFlat

UNANNOTATED_TYPE = 255

REQUIRED_PSM_COLUMNS = [
    "sequence",
    "charge",
    "raw_name",
    "score",
    "proteins",
    "spec_idx",
    "mod_sites",
    "mods",
    "mod_seq_hash",
    "mod_seq_charge_hash",
    "flat_frag_start_idx",
    "flat_frag_stop_idx",
    "precursor_mz",
]

REQUIRED_RAW_COLUMNS = [
    "_spec_idx",
    "peak_start_idx",
    "peak_stop_idx",
    "precursor_mz",
    "nce",
    "rt",
    "rt_max",
    "rt_norm",
    "activation",
]


def annotate_precursors_flat(
    speclib_flat: SpecLibFlat,
    raw_data_spectrum_df: pd.DataFrame,
    spec_idx_offset: int = 0,
) -> SpecLibFlat:
    """Annotate precursor information in a spectral library flat with raw data.

    This function merges precursor information from the spectral library flat
    with raw data, performs necessary data cleaning, and updates the precursor
    DataFrame in the spectral library flat.

    Parameters
    ----------
    speclib_flat : SpecLibFlat
        A spectral library flat object containing precursor information.
    raw_data_spectrum_df : pd.DataFrame
        Raw data object containing spectrum information.
    spec_idx_offset : int, optional
        Offset to apply to spec_idx in raw_data, by default 0.

    Returns
    -------
    SpecLibFlat
        An updated SpecLibFlat object with annotated precursor information.

    """
    # Define required columns
    if "activation" not in raw_data_spectrum_df.columns:
        raw_data_spectrum_df["activation"] = "Any"

    if "nce" not in raw_data_spectrum_df.columns:
        raw_data_spectrum_df["nce"] = 0

    raw_data_spectrum_df["rt_max"] = raw_data_spectrum_df["rt"].max()
    raw_data_spectrum_df["rt_norm"] = (
        raw_data_spectrum_df["rt"] / raw_data_spectrum_df["rt_max"]
    )

    raw_data_spectrum_df["_spec_idx"] = (
        raw_data_spectrum_df["spec_idx"] + spec_idx_offset
    )
    # Check if all required columns are present
    if not all(
        col in speclib_flat.precursor_df.columns for col in REQUIRED_PSM_COLUMNS
    ):
        raise ValueError(
            f"The following columns are missing from speclib_flat.precursor_df: {set(REQUIRED_PSM_COLUMNS) - set(speclib_flat.precursor_df.columns)}"
        )
    if not all(col in raw_data_spectrum_df.columns for col in REQUIRED_RAW_COLUMNS):
        raise ValueError(
            f"The following columns are missing from raw_data_spectrum_df: {set(REQUIRED_RAW_COLUMNS) - set(raw_data_spectrum_df.columns)}"
        )

    # Merge precursor data with raw data
    merged_precursor_df = speclib_flat.precursor_df[REQUIRED_PSM_COLUMNS].merge(
        raw_data_spectrum_df[REQUIRED_RAW_COLUMNS],
        left_on="spec_idx",
        right_on="_spec_idx",
        how="left",
        suffixes=("", "_observed"),
    )

    # sanity check precursor_mz
    if not np.all(
        np.isclose(
            merged_precursor_df["precursor_mz"],
            merged_precursor_df["precursor_mz_observed"],
            atol=5,
        )
    ):
        raise ValueError(
            "precursor_mz and precursor_mz_observed deviate by more than 5 m/z units. Please check the input data."
        )

    # Clean up the merged data
    merged_precursor_df = merged_precursor_df.dropna(
        subset=["peak_start_idx", "peak_stop_idx"]
    )
    merged_precursor_df["peak_start_idx"] = merged_precursor_df[
        "peak_start_idx"
    ].astype(int)
    merged_precursor_df["peak_stop_idx"] = merged_precursor_df["peak_stop_idx"].astype(
        int
    )

    # Create a new SpecLibFlat object with the updated precursor DataFrame
    annotated_speclib_flat = speclib_flat.copy()
    annotated_speclib_flat.precursor_df = merged_precursor_df

    return annotated_speclib_flat


def annotate_fragments_flat(
    speclib_flat: SpecLibFlat,
    raw_data_peak_df: pd.DataFrame,
    mass_error_ppm: float = 20,
) -> SpecLibFlat:
    """Annotate a spectral library flat with raw data and calculate PIF.

    This function matches observed spectrum peaks to theoretical fragments,
    calculates the precursor ion fraction (PIF), and organizes the results
    into DataFrames.

    Parameters
    ----------
    speclib_flat : SpecLibFlat
        A spectral library flat object containing precursor and fragment information.
    raw_data_peak_df : pd.DataFrame
        Raw data object containing peak information.
    mass_error_ppm : float, optional
        Mass error tolerance in parts per million (default: 20.0).

    Returns
    -------
    SpecLibFlat
        An updated SpecLibFlat object with annotated fragment information.

    """
    if not all(
        col in speclib_flat.precursor_df.columns
        for col in ["peak_start_idx", "peak_stop_idx"]
    ):
        raise ValueError(
            "peak_start_idx and peak_stop_idx must be present in speclib_flat.precursor_df. Please run annotate_precursors_flat() first."
        )

    fragment_df_list = []
    matched_precursor_df = speclib_flat.precursor_df.copy()
    matched_precursor_df["pif"] = 0.0

    start_index = 0
    for i, (peak_start, peak_stop, flat_frag_start, flat_frag_stop) in enumerate(
        zip(
            matched_precursor_df["peak_start_idx"],
            matched_precursor_df["peak_stop_idx"],
            matched_precursor_df["flat_frag_start_idx"],
            matched_precursor_df["flat_frag_stop_idx"],
        )
    ):
        spectrum_peak_df = raw_data_peak_df.iloc[peak_start:peak_stop].copy()
        fragment_df = speclib_flat.fragment_df.iloc[flat_frag_start:flat_frag_stop]

        (
            spectrum_peak_df["type"],
            spectrum_peak_df["loss_type"],
            spectrum_peak_df["charge"],
            spectrum_peak_df["number"],
            spectrum_peak_df["position"],
            spectrum_peak_df["error"],
        ) = annotate_spectrum(
            spectrum_peak_df["mz"].to_numpy(),
            fragment_df["mz"].to_numpy(),
            fragment_df["type"].to_numpy(),
            fragment_df["loss_type"].to_numpy(),
            fragment_df["charge"].to_numpy(),
            fragment_df["number"].to_numpy(),
            fragment_df["position"].to_numpy(),
            mass_error_ppm=mass_error_ppm,
        )

        matched_precursor_df.loc[i, "pif"] = calculate_pif(spectrum_peak_df)
        matched_precursor_df.loc[i, "flat_frag_start_idx"] = start_index
        matched_precursor_df.loc[i, "flat_frag_stop_idx"] = start_index + len(
            spectrum_peak_df
        )
        start_index += len(spectrum_peak_df)

        fragment_df_list.append(spectrum_peak_df)

    outlib_flat = SpecLibFlat()
    outlib_flat.precursor_df = matched_precursor_df.drop(
        columns=["peak_start_idx", "peak_stop_idx"]
    )
    outlib_flat._fragment_df = pd.concat(fragment_df_list, ignore_index=True)  # noqa: SLF001

    return outlib_flat


def calculate_pif(spectrum_peak_df: pd.DataFrame) -> float:
    """Calculate the precursor ion fraction (PIF) of a spectrum.

    This function computes the ratio of the sum of intensities for non-255 type
    peaks to the sum of all intensities. It handles edge cases such as empty
    DataFrames or arrays, and cases where all intensities are zero.

    Parameters
    ----------
    spectrum_peak_df : pandas.DataFrame
        A DataFrame containing at least two columns:
        - 'intensity': numeric values representing peak intensities
        - 'type': integer values where 255 represents a special peak type

    Returns
    -------
    float
        The calculated Peak Integral Fraction (PIF).
        Returns 0 if the DataFrame is empty, if either 'intensity' or 'type'
        arrays are empty, or if the sum of all intensities is zero.

    """
    if spectrum_peak_df.empty:
        return 0

    intensities = spectrum_peak_df["intensity"].to_numpy()
    types = spectrum_peak_df["type"].to_numpy()

    if len(intensities) == 0 or len(types) == 0:
        return 0

    numerator = np.sum(intensities[types != UNANNOTATED_TYPE])
    denominator = np.sum(intensities)

    if denominator == 0:
        return 0

    return numerator / denominator


@nb.njit()
def annotate_spectrum(  # noqa: PLR0913
    spectrum_mz: np.ndarray,
    fragment_mz: np.ndarray,
    fragment_type: np.ndarray,
    fragment_loss_type: np.ndarray,
    fragment_charge: np.ndarray,
    fragment_number: np.ndarray,
    fragment_position: np.ndarray,
    mass_error_ppm: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Annotate a spectrum with theoretical fragment information.

    This function matches observed spectrum peaks to theoretical fragments
    based on m/z values within a specified mass error tolerance.

    Parameters
    ----------
    spectrum_mz : np.ndarray
        M/Z values of the observed spectrum peaks. Shape: (n_peaks,)
    fragment_mz : np.ndarray
        M/Z values of the theoretical fragments. Shape: (n_fragments,)
    fragment_type : np.ndarray
        Types of the theoretical fragments. Shape: (n_fragments,)
    fragment_loss_type : np.ndarray
        Loss types of the theoretical fragments. Shape: (n_fragments,)
    fragment_charge : np.ndarray
        Charges of the theoretical fragments. Shape: (n_fragments,)
    fragment_number : np.ndarray
        Numbers of the theoretical fragments. Shape: (n_fragments,)
    fragment_position : np.ndarray
        Positions of the theoretical fragments. Shape: (n_fragments,)
    mass_error_ppm : float, optional
        Mass error tolerance in parts per million (default: 20.0).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing six numpy arrays, each with shape (n_peaks,):
        1. Annotated fragment types
        2. Annotated fragment loss types
        3. Annotated fragment charges
        4. Annotated fragment numbers
        5. Annotated fragment positions
        6. Annotated mass errors in ppm

        Each array corresponds to the input spectrum peaks. Peaks without
        a match are annotated with 255 (for integer arrays) or inf (for float array).

    """
    n_peaks = len(spectrum_mz)
    annotated_type = np.full(n_peaks, 255, dtype=np.uint8)
    annotated_loss_type = np.full(n_peaks, 255, dtype=np.uint8)
    annotated_charge = np.full(n_peaks, 255, dtype=np.uint8)
    annotated_number = np.full(n_peaks, 255, dtype=np.uint8)
    annotated_position = np.full(n_peaks, 255, dtype=np.uint8)
    annotated_error = np.full(n_peaks, np.inf, dtype=np.float32)

    # Sort fragment arrays based on m/z values
    sort_idx = np.argsort(fragment_mz)
    fragment_mz_sorted = fragment_mz[sort_idx]
    fragment_type_sorted = fragment_type[sort_idx]
    fragment_loss_type_sorted = fragment_loss_type[sort_idx]
    fragment_charge_sorted = fragment_charge[sort_idx]
    fragment_number_sorted = fragment_number[sort_idx]
    fragment_position_sorted = fragment_position[sort_idx]

    for i, mz in enumerate(spectrum_mz):
        # Find the index of the closest theoretical fragment
        idx = np.searchsorted(fragment_mz_sorted, mz)

        # Check the error for this index and the previous one (if it exists)
        error = np.inf
        best_idx = -1

        if idx < len(fragment_mz_sorted):
            error_right = (
                abs(mz - fragment_mz_sorted[idx]) / fragment_mz_sorted[idx] * 1e6
            )
            if error_right <= mass_error_ppm:
                error = error_right
                best_idx = idx

        if idx > 0:
            error_left = (
                abs(mz - fragment_mz_sorted[idx - 1])
                / fragment_mz_sorted[idx - 1]
                * 1e6
            )
            if error_left <= mass_error_ppm and error_left < error:
                error = error_left
                best_idx = idx - 1

        if best_idx != -1:
            annotated_type[i] = fragment_type_sorted[best_idx]
            annotated_loss_type[i] = fragment_loss_type_sorted[best_idx]
            annotated_charge[i] = fragment_charge_sorted[best_idx]
            annotated_number[i] = fragment_number_sorted[best_idx]
            annotated_position[i] = fragment_position_sorted[best_idx]
            annotated_error[i] = error

    return (
        annotated_type,
        annotated_loss_type,
        annotated_charge,
        annotated_number,
        annotated_position,
        annotated_error,
    )


def _sequence_coverage_metric(flatlib: SpecLibFlat) -> None:
    """Calculate the sequence coverage metric for the precursors.

    Important
    ---------
    This function operates in place and modifies the flatlib object

    Parameters
    ----------
    flatlib : SpecLibFlat
        The flatlib object to be modified

    Returns
    -------
    None

    """
    fragment_coverage = (
        (flatlib._fragment_intensity_df.sum(axis=1) > 0).astype(int).to_numpy()  # noqa: SLF001
    )

    # Vectorized operation to calculate coverage for all precursors at once
    start_indices = flatlib.precursor_df["frag_start_idx"].to_numpy()
    stop_indices = flatlib.precursor_df["frag_stop_idx"].to_numpy()

    flatlib.precursor_df["sequence_coverage"] = np.array(
        [
            fragment_coverage[start:stop].mean()
            for start, stop in zip(start_indices, stop_indices)
        ]
    )


def _sequence_gini_metric(flatlib: SpecLibFlat) -> None:
    """Calculate the sequence gini coefficient metric for the precursors.

    Important
    ---------
    This function operates in place and modifies the flatlib object.
    The Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).
    For fragment intensities, a lower Gini coefficient indicates more evenly
    distributed fragment intensities.

    Parameters
    ----------
    flatlib : SpecLibFlat
        The flatlib object to be modified

    Returns
    -------
    None

    """
    # Vectorized operation to calculate Gini coefficient for all precursors
    start_indices = flatlib.precursor_df["frag_start_idx"].to_numpy()
    stop_indices = flatlib.precursor_df["frag_stop_idx"].to_numpy()

    def gini(intensities: np.ndarray) -> float:
        # Handle empty or zero cases
        if len(intensities) == 0 or np.sum(intensities) == 0:
            return 1.0

        # Sort intensities in ascending order
        sorted_intensities = np.sort(intensities)
        n = len(sorted_intensities)
        index = np.arange(1, n + 1)

        # Create intermediate values to improve readability and reduce in-place operations
        intensity_sum = np.sum(sorted_intensities)
        weighted_sum = np.sum(index * sorted_intensities)

        return (2 * weighted_sum) / (n * intensity_sum) - ((n + 1) / n)

    flatlib.precursor_df["sequence_gini"] = np.array(
        [
            gini(flatlib._fragment_intensity_df.to_numpy()[start:stop].flatten())  # noqa: SLF001
            for start, stop in zip(start_indices, stop_indices)
        ]
    )


def _normalized_count_metric(flatlib: SpecLibFlat) -> None:
    """Calculate the number of fragments per precursor normalized by the precursor length.

    Important:
    ---------
    This function operates in place and modifies the flatlib object.

    """
    # Get all values at once to avoid repeated access
    fragment_values = flatlib._fragment_intensity_df.to_numpy()  # noqa: SLF001
    start_indices = flatlib.precursor_df["frag_start_idx"].to_numpy()
    stop_indices = flatlib.precursor_df["frag_stop_idx"].to_numpy()

    # Calculate lengths once
    lengths = stop_indices - start_indices

    # Calculate counts using vectorized operations
    counts = np.array(
        [
            np.sum(fragment_values[start:stop].flatten() > 0) / length
            for start, stop, length in zip(start_indices, stop_indices, lengths)
        ]
    )

    flatlib.precursor_df["normalized_count"] = counts


def _nan_median(series: np.ndarray) -> float:
    """Calculate the median of a numpy array, ignoring NaNs and infs.

    defaults to inf if there are no valid values
    """
    valid_values = series[~np.isnan(series) & ~np.isinf(series)]
    if len(valid_values) == 0:
        return np.inf
    return np.median(valid_values)


def _mass_accuracy_metric(flatlib: SpecLibFlat) -> None:
    """Calculate the mass accuracy metric for the precursors.

    Important:
    ---------
    This funmction operates in place and modifies the flatlib object

    """
    # Vectorized operation to calculate coverage for all precursors at once
    start_indices = flatlib.precursor_df["frag_start_idx"].to_numpy()
    stop_indices = flatlib.precursor_df["frag_stop_idx"].to_numpy()

    flatlib.precursor_df["mass_accuracy"] = np.array(
        [
            _nan_median(flatlib._fragment_df["error"].to_numpy()[start:stop])  # noqa: SLF001
            for start, stop in zip(start_indices, stop_indices)
        ]
    )


def add_dense_lib(
    flatlib: SpecLibFlat, charged_frag_types: Union[List[str], Tuple[str, ...]]
) -> SpecLibFlat:
    """Add dense format to the flatlib object.

    Parameters
    ----------
    flatlib : SpecLibFlat
        The flatlib object to be modified

    charged_frag_types : Union[List[str], Tuple[str, ...]]
        A list of fragment types that should be considered as charged. for example: `get_charged_frag_types(["b", "y", "b_NH3", "y_NH3","b_H2O", "y_H2O"], 2)`

    Returns
    -------
    SpecLibFlat
        The modified flatlib object

    """
    outlib_flat = flatlib.copy()

    valid_charged_frag_types = filter_valid_charged_frag_types(charged_frag_types)

    outlib_flat.charged_frag_types = valid_charged_frag_types

    df_collection, frag_start_idx, frag_stop_idx = create_dense_matrices(
        outlib_flat.precursor_df,
        outlib_flat.fragment_df,
        valid_charged_frag_types,
        flat_columns=["intensity", "error"],
    )

    outlib_flat._fragment_mz_df = df_collection["mz"]  # noqa: SLF001
    outlib_flat._fragment_intensity_df = df_collection["intensity"]  # noqa: SLF001

    outlib_flat.precursor_df["frag_start_idx"] = frag_start_idx
    outlib_flat.precursor_df["frag_stop_idx"] = frag_stop_idx

    _sequence_coverage_metric(outlib_flat)
    _sequence_gini_metric(outlib_flat)
    _normalized_count_metric(outlib_flat)
    _mass_accuracy_metric(outlib_flat)

    return outlib_flat
