"""Tests for translating spectral libraries to external formats."""

import numpy as np
import pandas as pd
import pytest

from alphabase.peptide.fragment import get_charged_frag_types
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.reader import LibraryReaderBase
from alphabase.spectral_library.translate_diann import (
    DIANN_PARQUET_COLUMN_ORDER,
    DIANN_PARQUET_SCHEMA,
    translate_to_diann_df,
    translate_to_parquet,
)

# every test builds a library with modified peptides, whose fragment m/z calculation
# needs `_calc_modloss` from numba
pytestmark = pytest.mark.requires_numba


def _build_speclib() -> SpecLibBase:
    """Build a small SpecLibBase with a few (modified) precursors and fragments."""
    precursor_df = pd.DataFrame(
        {
            "sequence": ["PEPTIDEK", "ACDEFGHIK", "MSEQUENCEK", "AAAAAK"],
            "mods": ["", "Carbamidomethyl@C", "Oxidation@M", "Acetyl@Any_N-term"],
            "mod_sites": ["", "2", "1", "0"],
            "charge": [2, 3, 2, 2],
            "rt": [0.1, 0.5, 0.7, 0.9],
            "proteins": ["PROT1", "PROT2;PROT9", "PROT3", "PROT4"],
            "uniprot_ids": ["P1", "P2;P9", "P3", "P4"],
            "genes": ["G1", "G2", "G3", "G4"],
            "is_prot_nterm": [False, False, False, True],
            "is_prot_cterm": [False, True, False, False],
        }
    )
    precursor_df["nAA"] = precursor_df["sequence"].str.len()

    charged_frag_types = get_charged_frag_types(["b", "y"], 2)
    speclib = SpecLibBase(charged_frag_types=charged_frag_types)
    speclib.precursor_df = precursor_df
    speclib.calc_fragment_mz_df()

    rng = np.random.default_rng(42)
    speclib._fragment_intensity_df = pd.DataFrame(
        rng.random(speclib.fragment_mz_df.shape),
        columns=speclib.charged_frag_types,
    )
    return speclib


def test_translate_to_diann_df_columns_and_mod_format() -> None:
    """The DIA-NN dataframe matches DIA-NN's schema, dtypes and (UniMod:N) sequences."""
    speclib = _build_speclib()

    df = translate_to_diann_df(
        speclib, min_frag_mz=0, max_frag_mz=0, min_frag_intensity=0.0, verbose=False
    )

    # exact DIA-NN column set/order, and `Signature` must NOT be present
    assert list(df.columns) == DIANN_PARQUET_COLUMN_ORDER
    assert "Signature" not in df.columns
    assert "Protein.Ids" in df.columns

    # DIA-NN requires INT64 / FLOAT(float32) numeric columns
    expected_dtype = {"int": np.int64, "float": np.float32}
    for col, dtype in DIANN_PARQUET_SCHEMA:
        if dtype in expected_dtype:
            assert df[col].dtype == expected_dtype[dtype], col

    # DIA-NN modified sequence format: (UniMod:N) inline / N-term prefix
    mod_seqs = set(df["Modified.Sequence"])
    assert "PEPTIDEK" in mod_seqs
    assert "AC(UniMod:4)DEFGHIK" in mod_seqs
    assert "M(UniMod:35)SEQUENCEK" in mod_seqs
    assert "(UniMod:1)AAAAAK" in mod_seqs

    # fragment types are single-letter, no-loss fragments are labelled "noloss"
    assert set(df["Fragment.Type"]).issubset({"a", "b", "c", "x", "y", "z"})
    assert "noloss" in set(df["Fragment.Loss.Type"])

    # protein-terminus flags come from is_prot_nterm / is_prot_cterm (not from mods)
    assert set(df.loc[df["N.Term"] == 1, "Stripped.Sequence"]) == {"AAAAAK"}
    assert set(df.loc[df["C.Term"] == 1, "Stripped.Sequence"]) == {"ACDEFGHIK"}

    # PTM.Site.Confidence is 1.0 (site is exact in a predicted library)
    assert (df["PTM.Site.Confidence"] == 1.0).all()

    # Proteotypic: 0 for the peptide shared across proteins (P2;P9), 1 otherwise
    assert set(df.loc[df["Proteotypic"] == 0, "Stripped.Sequence"]) == {"ACDEFGHIK"}


@pytest.mark.parametrize(
    "rt_column", ["irt_pred", "rt_pred", "rt_norm_pred", "rt", "irt", "rt_norm"]
)
def test_translate_to_diann_df_accepts_rt_columns(rt_column: str) -> None:
    """Any of the recognised retention time columns provides RT.

    The candidate columns are shared with the transition list format, so both
    formats accept `rt_norm_pred` as peptdeep writes it.
    """
    speclib = _build_speclib()
    speclib.precursor_df.rename(columns={"rt": rt_column}, inplace=True)  # noqa: PD002

    df = translate_to_diann_df(
        speclib, min_frag_mz=0, max_frag_mz=0, min_frag_intensity=0.0, verbose=False
    )

    assert df["RT"].notna().all()


def test_translate_to_diann_df_flags() -> None:
    """DIA-NN `Flags`: bit 0 on every fragment, bit 4 on one base peak per precursor."""
    speclib = _build_speclib()

    df = translate_to_diann_df(
        speclib, min_frag_mz=0, max_frag_mz=0, min_frag_intensity=0.0, verbose=False
    )

    # every row has the base bit (1 << 0)
    assert (df["Flags"] & (1 << 0) == (1 << 0)).all()
    # exactly one base-peak fragment (1 << 4) per precursor
    base_peak = df.groupby("Precursor.Id")["Flags"].apply(
        lambda s: int((s & (1 << 4) > 0).sum())
    )
    assert (base_peak == 1).all()
    # and it is the highest-intensity fragment of that precursor
    for _, group in df.groupby("Precursor.Id"):
        top = group.loc[group["Relative.Intensity"].idxmax()]
        assert top["Flags"] & (1 << 4) > 0


def test_translate_to_diann_df_flags_one_base_peak_per_precursor_row() -> None:
    """Precursor rows that share a peptidoform and charge each get their own base peak.

    The base peak used to be found by grouping on `Precursor.Id`, which is only the
    modified sequence and the charge, so two such rows shared one bit between them.
    """
    precursor_df = pd.DataFrame(
        {
            # the first two rows are the same peptidoform at the same charge
            "sequence": ["PEPTIDEK", "PEPTIDEK", "ACDEFGHIK"],
            "mods": ["", "", ""],
            "mod_sites": ["", "", ""],
            "charge": [2, 2, 2],
            "rt": [0.1, 0.2, 0.3],
            "proteins": ["PROT1", "PROT1", "PROT2"],
        }
    )
    precursor_df["nAA"] = precursor_df["sequence"].str.len()

    speclib = SpecLibBase(charged_frag_types=get_charged_frag_types(["b", "y"], 2))
    speclib.precursor_df = precursor_df
    speclib.calc_fragment_mz_df()
    rng = np.random.default_rng(0)
    speclib._fragment_intensity_df = pd.DataFrame(
        rng.random(speclib.fragment_mz_df.shape), columns=speclib.charged_frag_types
    )

    df = translate_to_diann_df(
        speclib, min_frag_mz=0, max_frag_mz=0, min_frag_intensity=0.0, verbose=False
    )

    assert df["Precursor.Id"].nunique() == 2  # noqa: PLR2004  two distinct peptidoforms
    # one base peak per precursor row, so three across two Precursor.Ids
    is_base_peak = (df["Flags"] & (1 << 4)) > 0
    assert is_base_peak.sum() == len(speclib.precursor_df)
    # and each is the most intense fragment of its own precursor
    assert (df.loc[is_base_peak, "Relative.Intensity"] == 1.0).all()


def test_speclib_to_diann_df_is_a_deprecated_alias() -> None:
    """The old name still works, and says what to use instead."""
    from alphabase.spectral_library.translate_diann import speclib_to_diann_df

    with pytest.warns(FutureWarning, match="translate_to_diann_df"):
        deprecated = speclib_to_diann_df(_build_speclib(), verbose=False)

    pd.testing.assert_frame_equal(
        deprecated, translate_to_diann_df(_build_speclib(), verbose=False)
    )


def test_translate_to_parquet_roundtrip(tmp_path) -> None:
    """SpecLibBase -> DIA-NN parquet -> LibraryReaderBase preserves precursors/fragments."""
    speclib = _build_speclib()
    n_precursors = len(speclib.precursor_df)

    out_path = str(tmp_path / "lib.parquet")
    translate_to_parquet(
        speclib, out_path, min_frag_mz=0, max_frag_mz=0, min_frag_intensity=0.0
    )

    exported = pd.read_parquet(out_path)
    assert "Modified.Sequence" in exported.columns
    assert "Product.Mz" in exported.columns

    reader = LibraryReaderBase()
    reader.import_file(out_path)

    # all precursors survive the round trip (compare on sequence/mods/mod_sites/charge)
    def _keys(df: pd.DataFrame) -> set:
        return set(
            zip(df["sequence"], df["mods"], df["mod_sites"], df["charge"].astype(int))
        )

    assert len(reader.precursor_df) == n_precursors
    assert _keys(reader.precursor_df) == _keys(speclib.precursor_df)
