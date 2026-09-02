"""Tests for `alphabase.spectral_library.translate_core`.

The shared machinery the format modules are built on: modified sequences, precursor
column lookup, and flattening the dense fragment dataframes.
"""

import numpy as np
import pandas as pd
import pytest

from alphabase.peptide.fragment import get_charged_frag_types
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.translate_core import (
    CCS_COLUMNS,
    MOBILITY_COLUMNS,
    RT_COLUMNS,
    FragmentColumns,
    FragmentFilter,
    create_modified_sequence,
    explode_top_fragments,
    first_present_column,
    is_nterm_frag,
    mod_to_unimod_dict,
    precursor_mz_series,
    translate_in_batches,
)

# the libraries hold modified peptides, whose fragment m/z calculation needs
# `_calc_modloss` from numba
pytestmark = pytest.mark.requires_numba

_CHARGED_FRAG_TYPES = get_charged_frag_types(["b", "y", "y_modloss"], 2)

# names for the fragment columns of the tests, one word each
_COLUMNS = FragmentColumns(
    frag_type="type",
    mz="mz",
    intensity="intensity",
    charge="charge",
    series_number="number",
    loss_type="loss",
)

# keeps every fragment, so a test opts in to each filter it is about
_KEEP_ALL = FragmentFilter(
    keep_k_highest=1000, min_mz=0, max_mz=0, min_intensity=-1, min_nAA=0
)


def _build_speclib() -> SpecLibBase:
    """Build a minimal SpecLibBase: (modified) precursors and fragment intensities."""
    precursor_df = pd.DataFrame(
        {
            "sequence": ["PEPTIDEK", "ACDEFGHIK", "SAAGHISK"],
            "mods": ["", "Carbamidomethyl@C", "Phospho@S"],
            "mod_sites": ["", "2", "1"],
            "charge": [2, 3, 2],
        }
    )
    precursor_df["nAA"] = precursor_df["sequence"].str.len()

    speclib = SpecLibBase(charged_frag_types=_CHARGED_FRAG_TYPES)
    speclib.precursor_df = precursor_df
    speclib.calc_fragment_mz_df()

    rng = np.random.default_rng(0)
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


def test_mod_to_unimod_dict() -> None:
    """`mod_to_unimod_dict` maps alphabase mod names onto UniMod ids."""
    assert mod_to_unimod_dict["Carbamidomethyl@C"] == "UniMod:4"
    assert mod_to_unimod_dict["Oxidation@M"] == "UniMod:35"


# --------------------------------------------------------------------------------------
# precursor columns
# --------------------------------------------------------------------------------------


def test_first_present_column_prefers_the_earlier_candidate() -> None:
    """The first candidate that the dataframe holds wins."""
    df = pd.DataFrame({"b": [1, 2], "c": [3, 4]})

    assert first_present_column(df, ("a", "b", "c")).tolist() == [1, 2]
    assert first_present_column(df, ("c", "b")).tolist() == [3, 4]


def test_first_present_column_falls_back_to_the_default() -> None:
    """A dataframe with none of the candidates gives the default, None by default."""
    df = pd.DataFrame({"b": [1, 2]})

    assert first_present_column(df, ("x", "y")) is None
    assert first_present_column(df, ("x",), 0.0) == 0.0


@pytest.mark.parametrize(
    ("candidates", "expected_first"),
    [
        (RT_COLUMNS, "irt_pred"),
        (MOBILITY_COLUMNS, "mobility_pred"),
        (CCS_COLUMNS, "ccs_pred"),
    ],
)
def test_candidate_columns_prefer_predictions(candidates, expected_first: str) -> None:
    """A predicted value takes precedence over a measured one."""
    assert candidates[0] == expected_first


def test_rt_columns_holds_both_peptdeep_prediction_names() -> None:
    """peptdeep writes rt_pred and rt_norm_pred, so both are recognised."""
    assert "rt_pred" in RT_COLUMNS
    assert "rt_norm_pred" in RT_COLUMNS


def test_precursor_mz_series_does_not_modify_the_dataframe() -> None:
    """The m/z is calculated on a copy, so no column is added to the caller."""
    precursor_df = _build_speclib().precursor_df
    columns_before = list(precursor_df.columns)

    mz = precursor_mz_series(precursor_df)

    assert list(precursor_df.columns) == columns_before
    assert (mz > 0).all()
    assert mz.index.equals(precursor_df.index)


def test_precursor_mz_series_keeps_an_existing_column() -> None:
    """An already calculated precursor_mz is given back as it is."""
    precursor_df = _build_speclib().precursor_df.copy()
    precursor_df["precursor_mz"] = [1.0, 2.0, 3.0]

    assert precursor_mz_series(precursor_df).tolist() == [1.0, 2.0, 3.0]


def test_precursor_mz_series_keeps_the_row_order_of_an_unsorted_frame() -> None:
    """Rows are not reordered, even though the calculation groups by peptide length."""
    precursor_df = _build_speclib().precursor_df
    unsorted_df = precursor_df.iloc[::-1]

    mz = precursor_mz_series(unsorted_df)

    assert mz.index.tolist() == unsorted_df.index.tolist()
    np.testing.assert_allclose(
        mz.to_numpy(), precursor_mz_series(precursor_df).to_numpy()[::-1]
    )


# --------------------------------------------------------------------------------------
# FragmentFilter
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("min_mz", "max_mz", "expected"),
    [(0, 0, False), (200, 2000, True), (0, 2000, True), (200, 0, True)],
)
def test_fragment_filter_limits_mz(
    min_mz: float, max_mz: float, expected: bool
) -> None:
    """Both bounds at zero is the documented way to keep every m/z."""
    assert FragmentFilter(min_mz=min_mz, max_mz=max_mz).limits_mz is expected


@pytest.mark.parametrize(("min_nAA", "expected"), [(0, 0), (1, 0), (3, 2)])
def test_fragment_filter_masked_frag_nAA(min_nAA: int, expected: int) -> None:
    """Keeping fragments numbered `min_nAA` and up masks `min_nAA - 1` at each end."""
    assert FragmentFilter(min_nAA=min_nAA).masked_frag_nAA == expected


# --------------------------------------------------------------------------------------
# explode_top_fragments
# --------------------------------------------------------------------------------------


def _explode(speclib: SpecLibBase, fragment_filter: FragmentFilter) -> pd.DataFrame:
    return explode_top_fragments(
        speclib.precursor_df,
        speclib.fragment_mz_df,
        speclib.fragment_intensity_df,
        columns=_COLUMNS,
        fragment_filter=fragment_filter,
        verbose=False,
    )


def test_explode_top_fragments_gives_one_row_per_kept_fragment() -> None:
    """The precursor columns are repeated once per fragment, the pointers dropped."""
    speclib = _build_speclib()

    df = _explode(speclib, _KEEP_ALL)

    n_dense_slots = len(speclib.fragment_mz_df) * len(speclib.fragment_mz_df.columns)
    assert len(df) == n_dense_slots
    assert set(_COLUMNS.as_list()) <= set(df.columns)
    assert "frag_start_idx" not in df.columns
    assert "frag_stop_idx" not in df.columns
    # the precursor columns are carried through
    assert set(df["sequence"]) == set(speclib.precursor_df["sequence"])


def test_explode_top_fragments_keeps_k_highest_in_descending_order() -> None:
    """Each precursor keeps its k most intense fragments, most intense first."""
    speclib = _build_speclib()
    top_n = 5

    df = _explode(
        speclib,
        FragmentFilter(keep_k_highest=top_n, min_mz=0, max_mz=0, min_intensity=-1),
    )

    assert len(df) == top_n * len(speclib.precursor_df)
    for _, group in df.groupby(level=0, sort=False):
        intensities = group["intensity"].astype(float).to_numpy()
        assert len(intensities) == top_n
        assert np.all(np.diff(intensities) <= 0)
        # normalized against the most intense fragment of that precursor
        assert intensities[0] == pytest.approx(1.0)


def test_explode_top_fragments_annotates_the_dense_column() -> None:
    """Type, charge, loss type and series number describe the fragment's dense column."""
    speclib = _build_speclib()

    df = _explode(speclib, _KEEP_ALL)

    assert set(df["type"]) <= {"b", "y"}
    assert set(df["charge"]) == {"1", "2"}
    assert set(df["loss"]) == {"noloss", "H3PO4"}  # modloss is relabelled by default

    n_aa = speclib.precursor_df["nAA"]
    for precursor_idx, group in df.groupby(level=0, sort=False):
        numbers = group["number"].astype(int)
        assert numbers.min() >= 1
        assert numbers.max() <= n_aa.loc[precursor_idx] - 1


def test_explode_top_fragments_labels_modloss() -> None:
    """The `modloss` loss type is renamed to the molecule that is lost."""
    df = _explode(_build_speclib(), _KEEP_ALL)
    assert "modloss" not in set(df["loss"])

    df = explode_top_fragments(
        _build_speclib().precursor_df,
        _build_speclib().fragment_mz_df,
        _build_speclib().fragment_intensity_df,
        columns=_COLUMNS,
        fragment_filter=_KEEP_ALL,
        modloss_label="modloss",
        verbose=False,
    )
    assert "modloss" in set(df["loss"])


def test_explode_top_fragments_zeroes_the_intensity_outside_the_mz_range() -> None:
    """The m/z range zeroes an intensity; the intensity threshold is what drops the row.

    So an out-of-range fragment cannot win a top-k slot from an in-range one, and with
    the threshold switched off it is still reported, at intensity 0.
    """
    speclib = _build_speclib()
    mz_range = FragmentFilter(
        min_mz=300, max_mz=800, keep_k_highest=1000, min_intensity=-1
    )

    kept = _explode(speclib, mz_range)
    out_of_range = ~kept["mz"].astype(float).between(300, 800)
    assert out_of_range.any()
    assert (kept.loc[out_of_range, "intensity"].astype(float) == 0).all()

    # with a threshold above zero, those rows are gone
    filtered = _explode(
        speclib,
        FragmentFilter(min_mz=300, max_mz=800, keep_k_highest=1000, min_intensity=0),
    )
    assert len(filtered) > 0
    assert filtered["mz"].astype(float).between(300, 800).all()


def test_explode_top_fragments_filters_by_relative_intensity() -> None:
    """The intensity threshold is relative to each precursor's most intense fragment."""
    speclib = _build_speclib()

    df = _explode(
        speclib,
        FragmentFilter(keep_k_highest=1000, min_mz=0, max_mz=0, min_intensity=0.5),
    )

    assert len(df) > 0
    assert df["intensity"].astype(float).min() > 0.5
    # every precursor still keeps its own most intense fragment
    assert df.groupby(level=0)["intensity"].max().eq(1.0).all()


def test_explode_top_fragments_does_not_modify_the_fragment_frames() -> None:
    """The filters are applied to a copy, so the library is left alone."""
    speclib = _build_speclib()
    intensities_before = speclib.fragment_intensity_df.to_numpy().copy()
    mz_before = speclib.fragment_mz_df.to_numpy().copy()

    _explode(speclib, FragmentFilter(min_mz=300, max_mz=800, min_nAA=3))

    np.testing.assert_array_equal(
        intensities_before, speclib.fragment_intensity_df.to_numpy()
    )
    np.testing.assert_array_equal(mz_before, speclib.fragment_mz_df.to_numpy())


def test_explode_top_fragments_masks_the_shortest_fragments_of_each_precursor() -> None:
    """`min_nAA` drops the b/y fragments numbered below it, per precursor."""
    speclib = _build_speclib()

    df = _explode(
        speclib,
        FragmentFilter(
            keep_k_highest=1000, min_mz=0, max_mz=0, min_intensity=0, min_nAA=3
        ),
    )

    assert df["number"].astype(int).min() >= 3
    # and no precursor lost all of its fragments to its neighbour's window
    assert df.index.nunique() == len(speclib.precursor_df)


# --------------------------------------------------------------------------------------
# translate_in_batches
# --------------------------------------------------------------------------------------


def test_translate_in_batches_covers_every_precursor_once() -> None:
    """Each batch holds the next `batch_size` precursors, and the fragments stay whole."""
    speclib = _build_speclib()
    seen = []

    def convert(precursor_df, fragment_mz_df, fragment_intensity_df) -> pd.DataFrame:
        # the fragment frames are passed whole, because the pointers are absolute
        assert len(fragment_mz_df) == len(speclib.fragment_mz_df)
        assert len(fragment_intensity_df) == len(speclib.fragment_intensity_df)
        return precursor_df

    def write(df: pd.DataFrame, batch_start: int) -> None:
        seen.append((batch_start, list(df["sequence"])))

    translate_in_batches(
        speclib.precursor_df,
        speclib.fragment_mz_df,
        speclib.fragment_intensity_df,
        convert,
        write,
        batch_size=2,
        progress=False,
    )

    assert [start for start, _ in seen] == [0, 2]
    assert [seq for _, batch in seen for seq in batch] == list(
        speclib.precursor_df["sequence"]
    )


def test_translate_in_batches_writes_nothing_for_an_empty_library() -> None:
    """A library with no precursors produces no batches."""
    calls = []

    translate_in_batches(
        pd.DataFrame(columns=["sequence"]),
        pd.DataFrame(),
        pd.DataFrame(),
        lambda p, _mz, _inten: p,
        lambda df, start: calls.append((df, start)),
        batch_size=10,
        progress=False,
    )

    assert calls == []


# --------------------------------------------------------------------------------------
# fragment annotation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("frag_type", "expected"),
    [("a", True), ("b", True), ("c", True), ("x", False), ("y", False), ("z", False)],
)
def test_is_nterm_frag(frag_type: str, expected: bool) -> None:
    """The a/b/c series are numbered from the N-terminus, x/y/z from the C-terminus."""
    assert is_nterm_frag(frag_type) is expected
    assert is_nterm_frag(f"{frag_type}_modloss_z2") is expected
