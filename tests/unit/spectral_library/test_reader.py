"""Unit tests for :class:`alphabase.spectral_library.reader.LibraryReaderBase`."""

import io
import warnings

import numpy as np
import pandas as pd
import pytest

from alphabase.constants.aa import AA_ASCII_MASS
from alphabase.constants.atom import MASS_H2O, MASS_PROTON
from alphabase.constants.modification import MOD_MASS
from alphabase.spectral_library.reader import LibraryReaderBase

# Fragment m/z calculation for modified peptides uses `_calc_modloss` (numba)
pytestmark = pytest.mark.requires_numba


def _expected_y_mz_z1(sequence: str, y_num: int, mod_mass_sum: float = 0.0) -> float:
    """Compute theoretical monoisotopic y-ion m/z (charge +1) from element tables."""
    suffix = sequence[len(sequence) - y_num :]
    neutral = sum(AA_ASCII_MASS[ord(aa)] for aa in suffix) + MASS_H2O + mod_mass_sum
    return neutral + MASS_PROTON


def _expected_b_mz_z1(sequence: str, b_num: int, mod_mass_sum: float = 0.0) -> float:
    """Compute theoretical monoisotopic b-ion m/z (charge +1) from element tables."""
    prefix = sequence[:b_num]
    neutral = sum(AA_ASCII_MASS[ord(aa)] for aa in prefix) + mod_mass_sum
    return neutral + MASS_PROTON


def _sample_partial_null_lib_df() -> pd.DataFrame:
    """Build a 3-precursor (20 dense fragment rows) library with partial metadata nulls."""
    return pd.DataFrame(
        {
            "Modified.Sequence": [
                "AC(UniMod:4)DEFGHK",
                "AC(UniMod:4)DEFGHK",
                "PEPTIDEK",
                "PEPTIDEK",
                "M(UniMod:35)ELODYR",
                "M(UniMod:35)ELODYR",
            ],
            "Stripped.Sequence": [
                "ACDEFGHK",
                "ACDEFGHK",
                "PEPTIDEK",
                "PEPTIDEK",
                "MELODYR",
                "MELODYR",
            ],
            "Precursor.Charge": [2, 2, 2, 2, 2, 2],
            "Precursor.Mz": [
                474.2055,
                474.2055,
                464.7347,
                464.7347,
                486.2162,
                486.2162,
            ],
            "RT": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0],
            "IM": [0.85, 0.85, 0.90, 0.90, np.nan, np.nan],
            "Genes": ["ACTB", "ACTB", None, None, "MYH9", "MYH9"],
            "Protein.Group": [
                "P60709",
                "P60709",
                "CON__KRT1",
                "CON__KRT1",
                "P35579",
                "P35579",
            ],
            "Product.Mz": [284.1717, 347.1020, 276.1554, 324.1554, 338.1823, 390.1693],
            "Relative.Intensity": [100.0, 50.0, 100.0, 40.0, 80.0, 100.0],
            "Fragment.Type": ["y", "b", "y", "b", "y", "b"],
            "Fragment.Charge": [1, 1, 1, 1, 1, 1],
            "Fragment.Series.Number": [2, 3, 2, 3, 2, 3],
            "Fragment.Loss.Type": ["", "", None, None, "noloss", "noloss"],
        }
    )


def test_parquet_partial_missing_metadata_keeps_all_precursors(tmp_path) -> None:
    """Precursors with partial None/NaN in metadata columns must not be dropped."""
    lib_df = _sample_partial_null_lib_df()
    parquet_path = tmp_path / "diann_partial_nulls.parquet"
    lib_df.to_parquet(parquet_path)

    reader = LibraryReaderBase()
    reader.import_file(parquet_path)

    precursor_df = reader.precursor_df
    assert len(precursor_df) == 3
    assert set(precursor_df["sequence"]) == {"ACDEFGHK", "PEPTIDEK", "MELODYR"}
    # Dense fragment rows: (8-1) + (8-1) + (7-1) = 20
    assert len(reader.fragment_mz_df) == 20
    assert len(reader.fragment_intensity_df) == 20

    by_seq = precursor_df.set_index("sequence")
    assert by_seq.loc["PEPTIDEK", "genes"] == ""
    assert by_seq.loc["MELODYR", "genes"] == "MYH9"
    assert by_seq.loc["MELODYR", "ccs"] == pytest.approx(0.0)
    assert by_seq.loc["PEPTIDEK", "ccs"] > 0.0

    # Verify fragment m/z and normalized intensities for recovered precursors
    pep_row = by_seq.loc["PEPTIDEK"]
    pep_mz = reader.fragment_mz_df.iloc[
        pep_row["frag_start_idx"] : pep_row["frag_stop_idx"]
    ]
    pep_int = reader.fragment_intensity_df.iloc[
        pep_row["frag_start_idx"] : pep_row["frag_stop_idx"]
    ]
    assert pep_mz.iloc[5]["y_z1"] == pytest.approx(
        _expected_y_mz_z1("PEPTIDEK", 2), abs=1e-4
    )
    assert pep_mz.iloc[2]["b_z1"] == pytest.approx(
        _expected_b_mz_z1("PEPTIDEK", 3), abs=1e-4
    )
    assert pep_int.iloc[5]["y_z1"] == pytest.approx(1.0)
    assert pep_int.iloc[2]["b_z1"] == pytest.approx(0.4)

    mel_row = by_seq.loc["MELODYR"]
    mel_mz = reader.fragment_mz_df.iloc[
        mel_row["frag_start_idx"] : mel_row["frag_stop_idx"]
    ]
    mel_int = reader.fragment_intensity_df.iloc[
        mel_row["frag_start_idx"] : mel_row["frag_stop_idx"]
    ]
    assert mel_mz.iloc[4]["y_z1"] == pytest.approx(
        _expected_y_mz_z1("MELODYR", 2), abs=1e-4
    )
    assert mel_mz.iloc[2]["b_z1"] == pytest.approx(
        _expected_b_mz_z1("MELODYR", 3, MOD_MASS["Oxidation@M"]), abs=1e-4
    )
    assert mel_int.iloc[4]["y_z1"] == pytest.approx(0.8)
    assert mel_int.iloc[2]["b_z1"] == pytest.approx(1.0)


def test_tsv_partial_na_metadata_keeps_all_precursors() -> None:
    """TSV libraries with NA/NaN in partially populated metadata columns keep all precursors."""
    tsv = (
        "ModifiedPeptideSequence\tPeptideSequence\tPrecursorCharge\tPrecursorMz\tiRT\t"
        "IonMobility\tGenes\tProductMz\tLibraryIntensity\tFragmentType\tFragmentCharge\t"
        "FragmentSeriesNumber\tFragmentLossType\n"
        "_AC(UniMod:4)DEFGHK_\tACDEFGHK\t2\t474.2055\t10.0\t0.85\tACTB\t284.1717\t100.0\ty\t1\t2\tnoloss\n"
        "_PEPTIDEK_\tPEPTIDEK\t2\t464.7347\t20.0\t0.90\tNA\t276.1554\t100.0\ty\t1\t2\tnoloss\n"
        "_M(UniMod:35)ELODYR_\tMELODYR\t2\t486.2162\t30.0\tNaN\tMYH9\t338.1823\t100.0\ty\t1\t2\tnoloss\n"
    )
    reader = LibraryReaderBase()
    reader.import_file(io.StringIO(tsv))

    assert len(reader.precursor_df) == 3
    assert set(reader.precursor_df["sequence"]) == {"ACDEFGHK", "PEPTIDEK", "MELODYR"}
    by_seq = reader.precursor_df.set_index("sequence")
    assert by_seq.loc["PEPTIDEK", "genes"] == ""
    assert by_seq.loc["MELODYR", "ccs"] == pytest.approx(0.0)


def test_every_precursor_having_partial_missing_column_does_not_crash(tmp_path) -> None:
    """When every precursor has a missing value in a different metadata column, import still succeeds."""
    lib_df = pd.DataFrame(
        {
            "Modified.Sequence": [
                "PEPTIDEK",
                "PEPTIDEK",
                "M(UniMod:35)ELODYR",
                "M(UniMod:35)ELODYR",
            ],
            "Stripped.Sequence": ["PEPTIDEK", "PEPTIDEK", "MELODYR", "MELODYR"],
            "Precursor.Charge": [2, 2, 2, 2],
            "Precursor.Mz": [464.7347, 464.7347, 486.2162, 486.2162],
            "RT": [20.0, 20.0, 30.0, 30.0],
            "IM": [0.90, 0.90, np.nan, np.nan],
            "Genes": [None, None, "MYH9", "MYH9"],
            "Product.Mz": [276.1554, 324.1554, 338.1823, 390.1693],
            "Relative.Intensity": [100.0, 50.0, 80.0, 100.0],
            "Fragment.Type": ["y", "b", "y", "b"],
            "Fragment.Charge": [1, 1, 1, 1],
            "Fragment.Series.Number": [2, 3, 2, 3],
            "Fragment.Loss.Type": ["", "", "", ""],
        }
    )
    parquet_path = tmp_path / "all_rows_partial_null.parquet"
    lib_df.to_parquet(parquet_path)

    reader = LibraryReaderBase()
    reader.import_file(parquet_path)

    assert len(reader.precursor_df) == 2
    assert set(reader.precursor_df["sequence"]) == {"PEPTIDEK", "MELODYR"}
    assert len(reader.fragment_mz_df) == 13


def test_all_nan_optional_column_is_still_dropped() -> None:
    """Completely empty numeric metadata columns (e.g. all-NaN IM) remain dropped."""
    tsv = (
        "ModifiedPeptideSequence\tPeptideSequence\tPrecursorCharge\tPrecursorMz\tiRT\t"
        "IonMobility\tGenes\tProductMz\tLibraryIntensity\tFragmentType\tFragmentCharge\t"
        "FragmentSeriesNumber\tFragmentLossType\n"
        "_PEPTIDEK_\tPEPTIDEK\t2\t464.7347\t20.0\tNaN\tACTB\t276.1554\t100.0\ty\t1\t2\tnoloss\n"
    )
    reader = LibraryReaderBase()
    reader.import_file(io.StringIO(tsv))

    assert len(reader.precursor_df) == 1
    assert "mobility" not in reader.precursor_df.columns
    assert "ccs" not in reader.precursor_df.columns


def test_parquet_complete_categorical_genes_matches_string_genes(tmp_path) -> None:
    """Complete categorical Genes matches plain string Genes with no TypeError or FutureWarning."""
    str_df = _sample_partial_null_lib_df()
    str_df["Genes"] = ["ACTB", "ACTB", "KRT1", "KRT1", "MYH9", "MYH9"]
    str_df["IM"] = [0.85, 0.85, 0.90, 0.90, 0.95, 0.95]
    str_df["Fragment.Loss.Type"] = ["", "", "", "", "noloss", "noloss"]

    cat_df = str_df.copy()
    cat_df["Genes"] = pd.Categorical(
        cat_df["Genes"], categories=["ACTB", "KRT1", "MYH9", "UNUSED_GENE"]
    )

    str_path = tmp_path / "complete_str.parquet"
    cat_path = tmp_path / "complete_cat.parquet"
    str_df.to_parquet(str_path)
    cat_df.to_parquet(cat_path)

    str_reader = LibraryReaderBase()
    str_reader.import_file(str_path)

    cat_reader = LibraryReaderBase()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cat_reader.import_file(cat_path)

    future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert future_warnings == []
    pd.testing.assert_frame_equal(cat_reader.precursor_df, str_reader.precursor_df)
    pd.testing.assert_frame_equal(cat_reader.fragment_mz_df, str_reader.fragment_mz_df)
    pd.testing.assert_frame_equal(
        cat_reader.fragment_intensity_df, str_reader.fragment_intensity_df
    )


def test_parquet_partial_missing_categorical_genes_keeps_all_precursors(
    tmp_path,
) -> None:
    """Partially missing categorical Genes maps nulls to '' and preserves all precursors."""
    lib_df = _sample_partial_null_lib_df()
    lib_df["Genes"] = pd.Categorical(
        lib_df["Genes"], categories=["ACTB", "MYH9", "UNUSED_GENE"]
    )
    parquet_path = tmp_path / "partial_cat.parquet"
    lib_df.to_parquet(parquet_path)

    reader = LibraryReaderBase()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reader.import_file(parquet_path)

    future_warnings = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert future_warnings == []

    precursor_df = reader.precursor_df
    assert len(precursor_df) == 3
    assert set(precursor_df["sequence"]) == {"ACDEFGHK", "PEPTIDEK", "MELODYR"}
    assert len(reader.fragment_mz_df) == 20
    assert len(reader.fragment_intensity_df) == 20

    by_seq = precursor_df.set_index("sequence")
    assert by_seq.loc["ACDEFGHK", "genes"] == "ACTB"
    assert by_seq.loc["PEPTIDEK", "genes"] == ""
    assert by_seq.loc["MELODYR", "genes"] == "MYH9"
    assert by_seq.loc["MELODYR", "ccs"] == pytest.approx(0.0)
    assert by_seq.loc["PEPTIDEK", "ccs"] > 0.0
