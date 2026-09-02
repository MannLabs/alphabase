"""Characterization tests for `alphabase.spectral_library.translate`.

Tests only. No production change.

`translate.py` had no unit test; its only coverage was
`nbs_tests/spectral_library/translate.ipynb`. These black-box tests pin the
behaviour of the module as it is today, so the refactor that follows can be
checked step by step.

Tests named `..._characterization` pin behaviour that is a bug. They record
what the module does today, not what it should do, and the commit that fixes
each one is expected to change them.
"""

import io
import warnings

import numpy as np
import pandas as pd
import pytest

from alphabase.peptide.fragment import get_charged_frag_types
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.translate import (
    SWATH_FRAGMENT_COLUMNS,
    create_modified_sequence,
    mod_to_unimod_dict,
    speclib_to_swath_df,
    translate_to_transition_df,
    translate_to_tsv,
)
from alphabase.spectral_library.translate_core import (
    FragmentFilter,
    explode_top_fragments,
)

# the libraries hold modified peptides, whose fragment m/z calculation needs
# `_calc_modloss` from numba
pytestmark = pytest.mark.requires_numba

_CHARGED_FRAG_TYPES = get_charged_frag_types(["b", "y", "y_modloss"], 2)


def _build_speclib(*, rt_column: str = "rt") -> SpecLibBase:
    """Build a small SpecLibBase with (modified) precursors and fragment intensities."""
    precursor_df = pd.DataFrame(
        {
            "sequence": ["PEPTIDEK", "ACDEFGHIK", "MSEQUENCEK", "SAAGHISK"],
            "mods": [
                "",
                "Carbamidomethyl@C",
                "Oxidation@M",
                "Phospho@S",
            ],
            "mod_sites": ["", "2", "1", "1"],
            "charge": [2, 3, 2, 2],
            "proteins": ["PROT1", "PROT2;PROT9", "PROT3", "PROT4"],
            "uniprot_ids": ["P1", "P2;P9", "P3", "P4"],
            "genes": ["G1", "G2", "G3", "G4"],
            "decoy": [0, 0, 1, 1],
        }
    )
    precursor_df[rt_column] = [0.1, 0.5, 0.7, 0.9]
    precursor_df["nAA"] = precursor_df["sequence"].str.len()

    speclib = SpecLibBase(charged_frag_types=_CHARGED_FRAG_TYPES)
    speclib.precursor_df = precursor_df
    speclib.calc_fragment_mz_df()

    rng = np.random.default_rng(42)
    speclib._fragment_intensity_df = pd.DataFrame(
        rng.random(speclib.fragment_mz_df.shape),
        columns=speclib.charged_frag_types,
    )
    return speclib


# --------------------------------------------------------------------------------------
# create_modified_sequence
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seq_mods_sites", "kwargs", "expected"),
    [
        # no modification: only the terminus markers are added
        (("PEPTIDEK", "", ""), {}, "_PEPTIDEK_"),
        # a single side-chain modification, inserted after its residue
        (("ACDEFGHIK", "Carbamidomethyl@C", "2"), {}, "_AC[Carbamidomethyl]DEFGHIK_"),
        # site 0 is the peptide N-term, site -1 the C-term
        (("PEPTIDEK", "Acetyl@Any_N-term", "0"), {}, "_[Acetyl]PEPTIDEK_"),
        (("PEPTIDEK", "Amidated@Any_C-term", "-1"), {}, "_PEPTIDEK_[Amidated]"),
        # several modifications are applied from the C-term inwards, so the
        # earlier sites keep their original offsets
        (
            ("ACDEFGHIK", "Carbamidomethyl@C;Oxidation@M", "2;5"),
            {},
            "_AC[Carbamidomethyl]DEF[Oxidation]GHIK_",
        ),
        # mod_sep and the terminus markers are configurable
        (
            ("ACDEFGHIK", "Carbamidomethyl@C", "2"),
            {"mod_sep": "()", "nterm": "", "cterm": ""},
            "AC(Carbamidomethyl)DEFGHIK",
        ),
        # translate_mod_dict replaces the alphabase names
        (
            ("ACDEFGHIK", "Carbamidomethyl@C", "2"),
            {"translate_mod_dict": {"Carbamidomethyl@C": "UniMod:4"}},
            "_AC[UniMod:4]DEFGHIK_",
        ),
    ],
)
def test_create_modified_sequence(seq_mods_sites, kwargs, expected) -> None:
    """`create_modified_sequence` renders (sequence, mods, mod_sites) as a mod sequence."""
    assert create_modified_sequence(seq_mods_sites, **kwargs) == expected


def test_create_modified_sequence_unimod_dict() -> None:
    """`mod_to_unimod_dict` maps alphabase mod names onto UniMod ids."""
    assert mod_to_unimod_dict["Carbamidomethyl@C"] == "UniMod:4"
    assert mod_to_unimod_dict["Oxidation@M"] == "UniMod:35"


# --------------------------------------------------------------------------------------
# explode_top_fragments
# --------------------------------------------------------------------------------------


def test_explode_top_fragments_keeps_k_highest_per_precursor() -> None:
    """Each precursor keeps its `keep_k_highest` most intense fragments, most intense first."""
    speclib = _build_speclib()
    top_n = 5

    df = explode_top_fragments(
        speclib.precursor_df,
        speclib.fragment_mz_df,
        speclib.fragment_intensity_df,
        columns=SWATH_FRAGMENT_COLUMNS,
        fragment_filter=FragmentFilter(
            keep_k_highest=top_n, min_mz=0, max_mz=0, min_intensity=-1
        ),
        verbose=False,
    )

    assert len(df) == top_n * len(speclib.precursor_df)
    for _, group in df.groupby(level=0, sort=False):
        intensities = group["RelativeIntensity"].astype(float).to_numpy()
        assert len(intensities) == top_n
        # rows of a precursor come out in descending intensity order
        assert np.all(np.diff(intensities) <= 0)
        # intensities are relative to the most intense fragment of the precursor
        assert intensities[0] == pytest.approx(1.0)


def test_explode_top_fragments_annotates_fragments() -> None:
    """Fragment type, charge, loss type and series number describe the dense column."""
    speclib = _build_speclib()

    df = explode_top_fragments(
        speclib.precursor_df,
        speclib.fragment_mz_df,
        speclib.fragment_intensity_df,
        columns=SWATH_FRAGMENT_COLUMNS,
        fragment_filter=FragmentFilter(
            keep_k_highest=100, min_mz=0, max_mz=0, min_intensity=-1
        ),
        modloss_label="modloss",
        verbose=False,
    )

    assert set(df["FragmentType"]) <= {"b", "y"}
    assert set(df["FragmentCharge"]) == {"1", "2"}
    assert set(df["FragmentLossType"]) == {"noloss", "modloss"}
    # the fragment pointers are not carried into the result
    assert "frag_start_idx" not in df.columns

    n_aa = speclib.precursor_df["nAA"]
    for precursor_idx, group in df.groupby(level=0, sort=False):
        frag_len = n_aa.loc[precursor_idx] - 1
        numbers = group["FragmentNumber"].astype(int)
        # b ions are numbered from the N-term, y ions from the C-term; both run 1..frag_len
        assert numbers.min() >= 1
        assert numbers.max() <= frag_len


# --------------------------------------------------------------------------------------
# translate_to_transition_df
# --------------------------------------------------------------------------------------


def test_translate_to_transition_df_columns() -> None:
    """The SWATH-like dataframe carries one row per fragment and its precursor columns."""
    speclib = _build_speclib()

    df = translate_to_transition_df(speclib, verbose=False)

    assert list(df.columns) == [
        "ModifiedPeptide",
        "PrecursorCharge",
        "RT",
        "StrippedPeptide",
        "PrecursorMz",
        "ProteinID",
        "Genes",
        "Decoy",
        "FragmentType",
        "FragmentMz",
        "RelativeIntensity",
        "FragmentCharge",
        "FragmentNumber",
        "FragmentLossType",
    ]
    # the fragment pointers are dropped from the output
    assert "frag_start_idx" not in df.columns
    assert "frag_stop_idx" not in df.columns

    precursor_df = speclib.precursor_df
    assert set(df["StrippedPeptide"]) == set(precursor_df["sequence"])
    # ProteinID comes from uniprot_ids, which takes precedence over proteins
    assert set(df["ProteinID"]) <= set(precursor_df["uniprot_ids"])
    assert set(df["Decoy"]) <= set(precursor_df["decoy"])


def test_translate_to_transition_df_modified_peptide_uses_alphabase_mod_names() -> None:
    """Without a translate_mod_dict the mod name is used, stripped of its site."""
    speclib = _build_speclib()

    df = translate_to_transition_df(speclib, verbose=False)

    modified = set(df["ModifiedPeptide"])
    assert "_PEPTIDEK_" in modified
    assert "_AC[Carbamidomethyl]DEFGHIK_" in modified


def test_translate_to_transition_df_translate_mod_dict() -> None:
    """A translate_mod_dict renames the modifications, here onto UniMod ids."""
    speclib = _build_speclib()

    df = translate_to_transition_df(
        speclib, translate_mod_dict=mod_to_unimod_dict, verbose=False
    )

    assert "_AC[UniMod:4]DEFGHIK_" in set(df["ModifiedPeptide"])


def test_translate_to_transition_df_filters_by_fragment_mz() -> None:
    """Fragments outside [min_frag_mz, max_frag_mz] are dropped."""
    speclib = _build_speclib()

    df = translate_to_transition_df(
        speclib, min_frag_mz=300, max_frag_mz=800, verbose=False
    )

    assert len(df) > 0
    assert df["FragmentMz"].astype(float).min() >= 300
    assert df["FragmentMz"].astype(float).max() <= 800


def test_translate_to_transition_df_filters_by_intensity() -> None:
    """Fragments at or below min_frag_intensity are dropped."""
    speclib = _build_speclib()

    df = translate_to_transition_df(speclib, min_frag_intensity=0.5, verbose=False)

    assert len(df) > 0
    assert df["RelativeIntensity"].astype(float).min() > 0.5


def test_translate_to_transition_df_min_frag_nAA_masks_shortest_fragments() -> None:
    """min_frag_nAA masks the b/y fragments below that series number."""
    speclib = _build_speclib()

    df = translate_to_transition_df(
        speclib, min_frag_mz=0, min_frag_intensity=0.0, min_frag_nAA=3, verbose=False
    )

    assert len(df) > 0
    assert df["FragmentNumber"].astype(int).min() >= 3


def test_translate_to_transition_df_labels_modloss() -> None:
    """The `modloss` loss type is renamed to the given molecule."""
    speclib = _build_speclib()

    df = translate_to_transition_df(
        speclib, min_frag_mz=0, min_frag_intensity=0.0, modloss="H3PO4", verbose=False
    )

    assert "H3PO4" in set(df["FragmentLossType"])
    assert "modloss" not in set(df["FragmentLossType"])


@pytest.mark.parametrize(
    "rt_column", ["irt_pred", "rt_pred", "rt_norm_pred", "rt", "irt", "rt_norm"]
)
def test_translate_to_transition_df_accepts_rt_columns(rt_column: str) -> None:
    """Any of the recognised retention time columns provides RT."""
    speclib = _build_speclib(rt_column=rt_column)

    df = translate_to_transition_df(speclib, verbose=False)

    assert df["RT"].notna().all()


def test_translate_to_transition_df_requires_an_rt_column() -> None:
    """Without a recognised retention time column the conversion fails."""
    speclib = _build_speclib()
    speclib._precursor_df = speclib._precursor_df.drop(columns=["rt"])

    with pytest.raises(ValueError, match="RT"):
        translate_to_transition_df(speclib, verbose=False)


def test_translate_to_transition_df_rt_norm_pred_is_an_rt_column() -> None:
    """`rt_norm_pred` provides RT on its own, and defers to `rt_pred` when both exist.

    peptdeep writes `rt_norm_pred` alongside `rt_pred`, so a library predicted by
    peptdeep can carry either.
    """
    speclib = _build_speclib(rt_column="rt_norm_pred")
    df = translate_to_transition_df(speclib, min_frag_intensity=0.0, verbose=False)
    assert df["RT"].notna().all()

    speclib = _build_speclib(rt_column="rt_norm_pred")
    speclib.precursor_df["rt_pred"] = speclib.precursor_df["rt_norm_pred"] + 10
    df = translate_to_transition_df(speclib, min_frag_intensity=0.0, verbose=False)
    assert df["RT"].min() >= 10


def test_translate_to_transition_df_rt_column_precedence() -> None:
    """`irt_pred` wins over `rt_pred`, which wins over `rt`."""
    speclib = _build_speclib()
    precursor_df = speclib.precursor_df
    precursor_df["rt_pred"] = precursor_df["rt"] + 10
    precursor_df["irt_pred"] = precursor_df["rt"] + 20

    df = translate_to_transition_df(speclib, min_frag_intensity=0.0, verbose=False)
    assert df["RT"].min() >= 20

    speclib._precursor_df = precursor_df.drop(columns=["irt_pred"])
    df = translate_to_transition_df(speclib, min_frag_intensity=0.0, verbose=False)
    assert 10 <= df["RT"].min() < 20


def test_translate_to_transition_df_mobility_and_ccs() -> None:
    """IonMobility and CCS are taken from the precursor dataframe when present."""
    speclib = _build_speclib()
    speclib.precursor_df["mobility_pred"] = 1.0
    speclib.precursor_df["ccs_pred"] = 2.0

    df = translate_to_transition_df(speclib, verbose=False)

    assert (df["IonMobility"] == 1.0).all()
    assert (df["CCS"] == 2.0).all()


def test_translate_to_transition_df_calculates_missing_precursor_mz() -> None:
    """precursor_mz is calculated when the precursor dataframe does not carry it."""
    speclib = _build_speclib()
    assert "precursor_mz" not in speclib.precursor_df.columns

    df = translate_to_transition_df(speclib, verbose=False)

    assert (df["PrecursorMz"] > 0).all()


# --------------------------------------------------------------------------------------
# translate_to_tsv
# --------------------------------------------------------------------------------------


def _read_tsv(buffer: io.StringIO) -> pd.DataFrame:
    buffer.seek(0)
    return pd.read_csv(buffer, sep="\t")


def test_translate_to_tsv_matches_translate_to_transition_df() -> None:
    """The streamed tsv holds the same rows as the in-memory conversion."""
    expected = translate_to_transition_df(
        _build_speclib(), translate_mod_dict=mod_to_unimod_dict, verbose=False
    )

    buffer = io.StringIO()
    translate_to_tsv(
        _build_speclib(),
        buffer,
        translate_mod_dict=mod_to_unimod_dict,
        multiprocessing=False,
    )
    written = _read_tsv(buffer)

    assert len(written) == len(expected)
    assert list(written.columns) == list(expected.columns)
    np.testing.assert_allclose(
        written["FragmentMz"].to_numpy(),
        expected["FragmentMz"].astype(float).to_numpy(),
    )
    assert (
        written["ModifiedPeptide"].to_numpy() == expected["ModifiedPeptide"].to_numpy()
    ).all()


def test_translate_to_tsv_accepts_a_path(tmp_path) -> None:
    """A str path is truncated and written; a file object is appended to."""
    tsv_path = tmp_path / "library.tsv"
    tsv_path.write_text("stale content that must be overwritten\n")

    translate_to_tsv(_build_speclib(), str(tsv_path), multiprocessing=False)

    written = pd.read_csv(tsv_path, sep="\t")
    assert len(written) > 0
    assert "stale" not in tsv_path.read_text()


def test_translate_to_tsv_multiprocessing_matches_single_process(tmp_path) -> None:
    """The forked writing process produces the same file as writing inline."""
    inline_path = tmp_path / "inline.tsv"
    translate_to_tsv(_build_speclib(), str(inline_path), multiprocessing=False)

    mp_path = tmp_path / "mp.tsv"
    translate_to_tsv(_build_speclib(), str(mp_path), multiprocessing=True)

    pd.testing.assert_frame_equal(
        pd.read_csv(inline_path, sep="\t"), pd.read_csv(mp_path, sep="\t")
    )


def test_translate_to_tsv_can_be_read_back_by_the_library_reader(tmp_path) -> None:
    """The written tsv is a library alphabase can read back.

    `LibraryReaderBase` recalculates the fragment m/z from the sequence and takes
    only the intensities from the file, so the round trip restores both frames.
    """
    from alphabase.spectral_library.reader import LibraryReaderBase

    speclib = _build_speclib()
    tsv_path = tmp_path / "library.tsv"
    translate_to_tsv(
        speclib,
        str(tsv_path),
        translate_mod_dict=mod_to_unimod_dict,
        multiprocessing=False,
    )

    reader = LibraryReaderBase()
    reader.import_file(str(tsv_path))

    assert len(reader.precursor_df) == len(speclib.precursor_df)
    assert set(reader.precursor_df["sequence"]) == set(speclib.precursor_df["sequence"])
    assert not (reader.fragment_mz_df.to_numpy() == 0).all()
    assert not (reader.fragment_intensity_df.to_numpy() == 0).all()


def test_translate_to_tsv_batches_do_not_change_the_output() -> None:
    """Batching splits the precursors only; the rows are unchanged."""
    buffer_whole = io.StringIO()
    translate_to_tsv(
        _build_speclib(), buffer_whole, batch_size=1000, multiprocessing=False
    )

    buffer_batched = io.StringIO()
    translate_to_tsv(
        _build_speclib(), buffer_batched, batch_size=2, multiprocessing=False
    )

    pd.testing.assert_frame_equal(_read_tsv(buffer_whole), _read_tsv(buffer_batched))


# --------------------------------------------------------------------------------------
# the library is not modified by being exported
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("min_frag_nAA", [0, 3])
def test_translate_to_transition_df_does_not_modify_the_library(
    min_frag_nAA: int,
) -> None:
    """Exporting leaves the library it is given untouched.

    The fragment filters used to be applied to `speclib._fragment_intensity_df` in
    place, which zeroed the intensities of the caller's library, and calculating
    `precursor_mz` added a column to its precursor dataframe.
    """
    speclib = _build_speclib()
    intensities_before = speclib.fragment_intensity_df.to_numpy().copy()
    mz_before = speclib.fragment_mz_df.to_numpy().copy()
    precursor_columns_before = list(speclib.precursor_df.columns)

    df = translate_to_transition_df(
        speclib,
        min_frag_mz=200,
        max_frag_mz=2000,
        min_frag_nAA=min_frag_nAA,
        verbose=False,
    )

    assert len(df) > 0
    np.testing.assert_array_equal(
        intensities_before, speclib.fragment_intensity_df.to_numpy()
    )
    np.testing.assert_array_equal(mz_before, speclib.fragment_mz_df.to_numpy())
    assert list(speclib.precursor_df.columns) == precursor_columns_before


def test_translate_to_tsv_does_not_modify_the_library() -> None:
    """The streamed export leaves the library untouched, and warns about nothing.

    The batch slice used to be assigned to a stand-in `SpecLibBase`, and writing
    `precursor_mz` into that slice raised `SettingWithCopyWarning` once per batch.
    """
    speclib = _build_speclib()
    intensities_before = speclib.fragment_intensity_df.to_numpy().copy()
    precursor_columns_before = list(speclib.precursor_df.columns)

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.SettingWithCopyWarning)
        translate_to_tsv(speclib, io.StringIO(), batch_size=2, multiprocessing=False)

    np.testing.assert_array_equal(
        intensities_before, speclib.fragment_intensity_df.to_numpy()
    )
    assert list(speclib.precursor_df.columns) == precursor_columns_before


def test_repeated_exports_give_the_same_result() -> None:
    """Exporting the same library twice gives the same rows both times.

    The in-place masking made the second export see already-zeroed intensities.
    """
    speclib = _build_speclib()

    first = translate_to_transition_df(
        speclib, min_frag_mz=300, max_frag_mz=1800, verbose=False
    )
    second = translate_to_transition_df(
        speclib, min_frag_mz=300, max_frag_mz=1800, verbose=False
    )

    pd.testing.assert_frame_equal(first, second)

    # and a narrower window first does not shrink what a wider one can find
    speclib = _build_speclib()
    translate_to_transition_df(
        speclib, min_frag_mz=1000, max_frag_mz=1100, verbose=False
    )
    wide = translate_to_transition_df(
        speclib, min_frag_mz=300, max_frag_mz=1800, verbose=False
    )
    assert len(wide) == len(first)


# --------------------------------------------------------------------------------------
# fragment filters
# --------------------------------------------------------------------------------------


def test_translate_to_tsv_disabled_mz_range_keeps_every_fragment() -> None:
    """`min_frag_mz=0, max_frag_mz=0` disables the m/z filter, as it is documented to.

    `translate_to_tsv` used to mask unconditionally, so with both bounds at 0 the
    mask became `(mz > 0) | (mz < 0)` and only the m/z 0 padding slots came through.
    """
    buffer = io.StringIO()
    translate_to_tsv(
        _build_speclib(), buffer, min_frag_mz=0, max_frag_mz=0, multiprocessing=False
    )
    written = _read_tsv(buffer)

    in_memory = translate_to_transition_df(
        _build_speclib(), min_frag_mz=0, max_frag_mz=0, verbose=False
    )

    assert (written["FragmentMz"] > 0).any()
    assert len(written) == len(in_memory)
    np.testing.assert_allclose(
        written["FragmentMz"].to_numpy(),
        in_memory["FragmentMz"].astype(float).to_numpy(),
    )


def test_min_frag_nAA_is_bounded_by_the_precursor_length() -> None:
    """A mask window longer than a precursor masks only that precursor's fragments.

    The window used to be applied as `frag_start_idx + i` without checking that the
    row still belonged to the precursor, so it reached into the next precursor's
    fragments and, at the end of the library, off the end of the array.
    """
    speclib = _build_speclib()
    n_fragment_rows = len(speclib.fragment_mz_df)

    # a window longer than the whole library masks everything and raises nothing
    df = translate_to_transition_df(
        speclib,
        min_frag_nAA=n_fragment_rows + 2,
        min_frag_mz=0,
        max_frag_mz=0,
        min_frag_intensity=-1,
        verbose=False,
    )
    assert (df["RelativeIntensity"].astype(float) == 0).all()

    # a window of 3 leaves the fragments numbered 3 and up, for every precursor
    df = translate_to_transition_df(
        speclib, min_frag_nAA=3, min_frag_mz=0, max_frag_mz=0, verbose=False
    )
    assert df["FragmentNumber"].astype(int).min() >= 3
    assert set(df["StrippedPeptide"]) == set(speclib.precursor_df["sequence"])


def test_speclib_to_single_df_is_a_deprecated_alias() -> None:
    """The old name still works, and says what to use instead."""
    from alphabase.spectral_library.translate import speclib_to_single_df

    with pytest.warns(FutureWarning, match="translate_to_transition_df"):
        deprecated = speclib_to_single_df(_build_speclib(), verbose=False)

    pd.testing.assert_frame_equal(
        deprecated, translate_to_transition_df(_build_speclib(), verbose=False)
    )


def test_speclib_to_swath_df_returns_none_characterization() -> None:
    """`speclib_to_swath_df` never returns its result.

    The function has been missing its `return` since translate.py moved into
    alphabase, so it is annotated `-> pd.DataFrame` but always gives None.
    """
    assert speclib_to_swath_df(_build_speclib()) is None
