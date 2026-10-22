"""Metrics utilities for spectral library precursor annotation."""

import numpy as np
import pandas as pd

from alphabase.peptide.fragment import UNANNOTATED_TYPE
from alphabase.spectral_library.flat import SpecLibFlat

PRECURSOR_METRICS = (
    "pif",
    "sequence_coverage",
    "sequence_gini",
    "normalized_count",
    "mass_accuracy",
)


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


def calculate_precursor_metrics(
    intensities: np.ndarray,
    types: np.ndarray,
    errors: np.ndarray,
    dense_intensities: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Calculate coverage and precursor metrics from one annotated spectrum.

    Parameters
    ----------
    intensities, types, errors : np.ndarray
        Observed peak intensities, annotated types and ppm errors. Unannotated
        peaks have type ``UNANNOTATED_TYPE`` and infinite error.
    dense_intensities : np.ndarray
        Intensities indexed by cleavage site and charged fragment type.

    Returns
    -------
    np.ndarray
        Boolean cleavage-site coverage: any fragment with positive intensity.
    dict[str, float]
        PIF, sequence coverage, Gini, normalized count and median finite error.
        With no cleavage sites, coverage and normalized count are zero. With
        no intensity, Gini is one; with no finite errors, mass accuracy is infinite.
    """
    coverage = (dense_intensities > 0).any(axis=1)
    ordered = np.sort(dense_intensities, axis=None)
    total = ordered.sum(dtype=np.float64)
    gini = 1.0
    if ordered.size and total:
        weighted = np.sum(np.arange(1, ordered.size + 1) * ordered)
        gini = 2 * weighted / (ordered.size * total) - (ordered.size + 1) / ordered.size
    finite_errors = errors[np.isfinite(errors)]
    return coverage, {
        "pif": float(calculate_pif(intensities, types)),
        "sequence_coverage": float(coverage.mean()) if len(coverage) else 0.0,
        "sequence_gini": float(gini),
        "normalized_count": float(
            np.count_nonzero(dense_intensities > 0) / len(coverage)
        )
        if len(coverage)
        else 0.0,
        "mass_accuracy": float(np.median(finite_errors))
        if len(finite_errors)
        else np.inf,
    }


def apply_precursor_metrics(flatlib: SpecLibFlat) -> SpecLibFlat:
    """Populate precursor metrics using the same calculations as array annotation.

    Parameters
    ----------
    flatlib : SpecLibFlat
        Annotated library containing flat peaks and dense fragment intensities.

    Returns
    -------
    SpecLibFlat
        The input library with its precursor metrics updated in place.
    """
    peaks = flatlib.fragment_df
    intensity = peaks["intensity"].to_numpy()
    types = peaks["type"].to_numpy()
    errors = peaks["error"].to_numpy()
    dense = flatlib.fragment_intensity_df.to_numpy()
    metrics = []
    for precursor in flatlib.precursor_df.itertuples(index=False):
        peak_range = slice(precursor.flat_frag_start_idx, precursor.flat_frag_stop_idx)
        _, values = calculate_precursor_metrics(
            intensity[peak_range],
            types[peak_range],
            errors[peak_range],
            dense[precursor.frag_start_idx : precursor.frag_stop_idx],
        )
        metrics.append(values)
    flatlib.precursor_df[list(PRECURSOR_METRICS)] = pd.DataFrame(
        metrics,
        index=flatlib.precursor_df.index,
        columns=PRECURSOR_METRICS,
    )
    return flatlib


__all__ = [
    "UNANNOTATED_TYPE",
    "PRECURSOR_METRICS",
    "apply_precursor_metrics",
    "calculate_pif",
    "calculate_precursor_metrics",
]
