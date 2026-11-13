"""Metrics utilities for spectral library precursor annotation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alphabase.numba_wrapper import numba_njit_or_python
from alphabase.peptide.fragment import UNANNOTATED_TYPE

if TYPE_CHECKING:
    from alphabase.spectral_library.base import SpecLibBase
    from alphabase.spectral_library.flat import SpecLibFlat


@numba_njit_or_python
def calculate_pif(intensities: np.ndarray, types: np.ndarray) -> float:
    """Calculate the precursor ion fraction (PIF) of a spectrum.

    This function computes the ratio of the sum of intensities for annotated
    fragment peaks to the sum of all intensities. It handles edge cases such as
    empty arrays and cases where the sum of all intensities is zero.

    Parameters
    ----------
    intensities : np.ndarray
        Array of peak intensities.
    types : np.ndarray
        Array of fragment types where ``UNANNOTATED_TYPE`` represents a special peak type.

    Returns
    -------
    float
        The calculated Peak Integral Fraction (PIF).
        Returns 0 if the arrays are empty or if the sum of all intensities is zero.

    """
    if len(intensities) == 0 or len(types) == 0:
        return 0.0

    numerator = np.sum(intensities[types != UNANNOTATED_TYPE])
    denominator = np.sum(intensities)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _pif_metric(flatlib: SpecLibFlat) -> None:
    """Calculate the precursor ion fraction (PIF) metric for the precursors.

    Important
    ---------
    This function operates in place and modifies the flatlib object.
    The PIF is the ratio of annotated fragment intensity to total intensity.

    Parameters
    ----------
    flatlib : SpecLibFlat
        The flatlib object to be modified

    Returns
    -------
    None

    """
    start_indices = flatlib.precursor_df["flat_frag_start_idx"].to_numpy()
    stop_indices = flatlib.precursor_df["flat_frag_stop_idx"].to_numpy()

    intensity_array = flatlib._fragment_df["intensity"].to_numpy()  # noqa: SLF001
    type_array = flatlib._fragment_df["type"].to_numpy()  # noqa: SLF001

    flatlib.precursor_df["pif"] = np.array(
        [
            calculate_pif(
                intensity_array[start:stop],
                type_array[start:stop],
            )
            for start, stop in zip(start_indices, stop_indices)
        ]
    )


def _sequence_coverage_metric(speclib: SpecLibBase) -> None:
    """Calculate the sequence coverage metric for the precursors.

    Important
    ---------
    This function operates in place and modifies the speclib object

    Parameters
    ----------
    speclib : SpecLibBase
        The speclib object to be modified

    Returns
    -------
    None

    """
    fragment_coverage = (
        (speclib._fragment_intensity_df.sum(axis=1) > 0).astype(int).to_numpy()  # noqa: SLF001
    )

    start_indices = speclib.precursor_df["frag_start_idx"].to_numpy()
    stop_indices = speclib.precursor_df["frag_stop_idx"].to_numpy()

    speclib.precursor_df["sequence_coverage"] = np.array(
        [
            fragment_coverage[start:stop].mean()
            for start, stop in zip(start_indices, stop_indices)
        ]
    )


def _sequence_gini_metric(speclib: SpecLibBase) -> None:
    """Calculate the sequence gini coefficient metric for the precursors.

    Important
    ---------
    This function operates in place and modifies the speclib object.
    The Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).
    For fragment intensities, a lower Gini coefficient indicates more evenly
    distributed fragment intensities.

    Parameters
    ----------
    speclib : SpecLibBase
        The speclib object to be modified

    Returns
    -------
    None

    """
    start_indices = speclib.precursor_df["frag_start_idx"].to_numpy()
    stop_indices = speclib.precursor_df["frag_stop_idx"].to_numpy()

    def gini(intensities: np.ndarray) -> float:
        if len(intensities) == 0 or np.sum(intensities) == 0:
            return 1.0

        sorted_intensities = np.sort(intensities)
        n = len(sorted_intensities)
        index = np.arange(1, n + 1)

        intensity_sum = np.sum(sorted_intensities)
        weighted_sum = np.sum(index * sorted_intensities)

        return (2 * weighted_sum) / (n * intensity_sum) - ((n + 1) / n)

    speclib.precursor_df["sequence_gini"] = np.array(
        [
            gini(speclib._fragment_intensity_df.to_numpy()[start:stop].flatten())  # noqa: SLF001
            for start, stop in zip(start_indices, stop_indices)
        ]
    )


def _normalized_count_metric(speclib: SpecLibBase) -> None:
    """Calculate the number of fragments per precursor normalized by the precursor length.

    Important:
    ---------
    This function operates in place and modifies the speclib object.

    """
    fragment_values = speclib._fragment_intensity_df.to_numpy()  # noqa: SLF001
    start_indices = speclib.precursor_df["frag_start_idx"].to_numpy()
    stop_indices = speclib.precursor_df["frag_stop_idx"].to_numpy()

    lengths = stop_indices - start_indices

    counts = np.array(
        [
            np.sum(fragment_values[start:stop].flatten() > 0) / length
            for start, stop, length in zip(start_indices, stop_indices, lengths)
        ]
    )

    speclib.precursor_df["normalized_count"] = counts


@numba_njit_or_python
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
    This function operates in place and modifies the flatlib object

    """
    start_indices = flatlib.precursor_df["flat_frag_start_idx"].to_numpy()
    stop_indices = flatlib.precursor_df["flat_frag_stop_idx"].to_numpy()

    error_array = flatlib._fragment_df["error"].to_numpy()  # noqa: SLF001

    flatlib.precursor_df["mass_accuracy"] = np.array(
        [
            _nan_median(error_array[start:stop])
            for start, stop in zip(start_indices, stop_indices)
        ]
    )


def apply_dense_precursor_metrics(speclib: SpecLibBase) -> SpecLibBase:
    """Populate the precursor metrics derived from the dense fragment intensities.

    These are ``sequence_coverage``, ``sequence_gini`` and ``normalized_count``.
    """
    _sequence_coverage_metric(speclib)
    _sequence_gini_metric(speclib)
    _normalized_count_metric(speclib)
    return speclib


def apply_precursor_metrics(flatlib: SpecLibFlat) -> SpecLibFlat:
    """Populate derived precursor metrics on the provided spectral library."""
    _pif_metric(flatlib)
    apply_dense_precursor_metrics(flatlib)
    _mass_accuracy_metric(flatlib)
    return flatlib


__all__ = [
    "UNANNOTATED_TYPE",
    "apply_dense_precursor_metrics",
    "apply_precursor_metrics",
    "calculate_pif",
]
