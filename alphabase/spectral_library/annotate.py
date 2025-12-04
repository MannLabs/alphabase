"""Module for annotating spectral libraries with raw mass spectrometry data."""

from typing import List, Tuple, Union

import numba as nb
import numpy as np
import pandas as pd

from alphabase.peptide.fragment import (
    UNANNOTATED_TYPE,
    create_dense_matrices,
    create_fragment_mz_dataframe,
    filter_valid_charged_frag_types,
    flatten_fragments,
    init_fragment_by_precursor_dataframe,
)
from alphabase.spectral_library.flat import SpecLibFlat
from alphabase.spectral_library.metrics import apply_precursor_metrics

REQUIRED_PSM_COLUMNS = [
    "sequence",
    "charge",
    "spec_idx",
    "mod_sites",
    "mods",
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
    """Annotate a spectral library flat with raw data.

    This function matches observed spectrum peaks to theoretical fragments
    and organizes the results into DataFrames.

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
        a match are annotated with ``UNANNOTATED_TYPE`` (for integer arrays)
        or ``np.inf`` (for the float array).

    """
    n_peaks = len(spectrum_mz)
    annotated_type = np.full(n_peaks, UNANNOTATED_TYPE, dtype=np.uint8)
    annotated_loss_type = np.full(n_peaks, UNANNOTATED_TYPE, dtype=np.uint8)
    annotated_charge = np.full(n_peaks, UNANNOTATED_TYPE, dtype=np.uint8)
    annotated_number = np.full(n_peaks, UNANNOTATED_TYPE, dtype=np.uint8)
    annotated_position = np.full(n_peaks, UNANNOTATED_TYPE, dtype=np.uint8)
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


def _build_theoretical_flatlib(
    template: SpecLibFlat, charged_frag_types: List[str]
) -> SpecLibFlat:
    """Generate a flat spectral library containing theoretical fragments."""
    theoretical_flat = SpecLibFlat(
        charged_frag_types=charged_frag_types,
    )

    precursor_df = template.precursor_df.copy()
    fragment_mz_df = init_fragment_by_precursor_dataframe(
        precursor_df, charged_frag_types
    )
    fragment_mz_df = create_fragment_mz_dataframe(
        precursor_df,
        charged_frag_types,
        reference_fragment_df=fragment_mz_df,
        inplace_in_reference=True,
    )

    precursor_df, fragment_df = flatten_fragments(
        precursor_df,
        fragment_mz_df,
        pd.DataFrame(),
    )

    theoretical_flat._precursor_df = precursor_df  # noqa: SLF001
    theoretical_flat._fragment_df = fragment_df  # noqa: SLF001

    return theoretical_flat


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
    original_spec_idx = flatlib.precursor_df["spec_idx"].copy()
    flatlib.precursor_df["spec_idx"] = np.arange(len(flatlib.precursor_df))
    flatlib.precursor_df["peak_start_idx"] = flatlib.precursor_df["flat_frag_start_idx"]
    flatlib.precursor_df["peak_stop_idx"] = flatlib.precursor_df["flat_frag_stop_idx"]

    valid_charged_frag_types = filter_valid_charged_frag_types(charged_frag_types)

    theoretical_flat = _build_theoretical_flatlib(flatlib, valid_charged_frag_types)
    raw_precursor_df = flatlib.precursor_df.copy()
    raw_fragment_df = flatlib.fragment_df.copy()

    outlib_flat = annotate_precursors_flat(
        theoretical_flat,
        raw_precursor_df,
        spec_idx_offset=0,
    )
    outlib_flat = annotate_fragments_flat(
        outlib_flat,
        raw_fragment_df,
        mass_error_ppm=20,
    )

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

    apply_precursor_metrics(outlib_flat)

    outlib_flat.precursor_df["spec_idx"] = original_spec_idx.iloc[
        outlib_flat.precursor_df["spec_idx"].to_numpy()
    ].to_numpy()

    return outlib_flat
