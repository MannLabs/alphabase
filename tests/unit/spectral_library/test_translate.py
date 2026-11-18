"""Characterization tests for the SWATH/Spectronaut transition-list export.

These tests pin the behaviour of `alphabase.spectral_library.translate` as it is
today, so that the upcoming restructuring can be shown to change nothing.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from alphabase.peptide.fragment import get_charged_frag_types
from alphabase.spectral_library import translate
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.reader import LibraryReaderBase
from alphabase.spectral_library.translate import (
    mod_to_unimod_dict,
    speclib_to_single_df,
    speclib_to_swath_df,
    translate_to_tsv,
)

# every test builds a library with modified peptides, whose fragment m/z calculation
# needs `_calc_modloss` from numba
pytestmark = pytest.mark.requires_numba

# the export's own defaults, spelled out where a test needs to reason about them
DEFAULT_KEEP_K = 12
DEFAULT_MIN_FRAG_MZ = 200
DEFAULT_MAX_FRAG_MZ = 2000


def _build_speclib(**extra_columns) -> SpecLibBase:
    """Build a SpecLibBase covering N-term, C-term, internal and loss-bearing mods.

    `b`/`y`/`b_modloss`/`y_modloss` at charges 1-2 give the library both real and
    empty (m/z 0) fragment slots, which is what the m/z filtering behaves on.
    """
    precursor_df = pd.DataFrame(
        {
            "sequence": [
                "PEPTIDEK",
                "ACDEFGHIK",
                "MSEQUENCEK",
                "AAAAAK",
                "SVIVSPYSTGAK",
                "LHDSTPPPYK",
                "GLSDGEWQK",
            ],
            "mods": [
                "",
                "Carbamidomethyl@C",
                "Oxidation@M",
                "Acetyl@Any_N-term",
                "Phospho@S",
                "Phospho@Y",
                "Amidated@Any_C-term",
            ],
            "mod_sites": ["", "2", "1", "0", "1", "9", "-1"],
            "charge": [2, 3, 2, 2, 2, 3, 2],
            "rt_pred": [0.1, 0.5, 0.7, 0.9, 0.3, 0.4, 0.6],
            "proteins": [
                "PROT1",
                "PROT2;PROT9",
                "PROT3",
                "PROT4",
                "PROT5",
                "PROT6",
                "PROT7",
            ],
            "genes": ["G1", "G2", "G3", "G4", "G5", "G6", "G7"],
            **extra_columns,
        }
    )
    precursor_df["nAA"] = precursor_df["sequence"].str.len()

    speclib = SpecLibBase(
        charged_frag_types=get_charged_frag_types(
            ["b", "y", "b_modloss", "y_modloss"], 2
        )
    )
    speclib.precursor_df = precursor_df
    speclib.calc_fragment_mz_df()
    rng = np.random.default_rng(42)
    speclib._fragment_intensity_df = pd.DataFrame(
        rng.random(speclib.fragment_mz_df.shape),
        columns=speclib.charged_frag_types,
    )
    return speclib


def _export(speclib: SpecLibBase, **kwargs) -> pd.DataFrame:
    """Run the in-memory export with the progress bar off."""
    return speclib_to_swath_df(speclib, verbose=False, **kwargs)


def _unfiltered(speclib: SpecLibBase, **kwargs) -> pd.DataFrame:
    """Export with every fragment filter disabled, to see the raw selection."""
    return _export(
        speclib, min_frag_mz=0, max_frag_mz=np.inf, min_frag_intensity=0.0, **kwargs
    )


def test_columns_and_order() -> None:
    """The transition list has SWATH column names, in a fixed order."""
    df = _export(_build_speclib())

    assert list(df.columns) == [
        "ModifiedPeptide",
        "PrecursorCharge",
        "RT",
        "StrippedPeptide",
        "PrecursorMz",
        "ProteinID",
        "Genes",
        "FragmentType",
        "FragmentMz",
        "RelativeIntensity",
        "FragmentCharge",
        "FragmentNumber",
        "FragmentLossType",
    ]
    # the fragment index columns are internal and must not reach the output
    assert "frag_start_idx" not in df.columns
    assert "frag_stop_idx" not in df.columns


def test_modified_sequence_rendering() -> None:
    """Mods render as `_A[Mod]B_`, with site 0 on the N-term and -1 on the C-term."""
    df = _export(_build_speclib())
    mod_seqs = set(df["ModifiedPeptide"])

    assert "_PEPTIDEK_" in mod_seqs  # unmodified: bare underscores
    assert "_AC[Carbamidomethyl]DEFGHIK_" in mod_seqs  # internal, after the residue
    assert "_M[Oxidation]SEQUENCEK_" in mod_seqs  # internal, first residue
    assert "_[Acetyl]AAAAAK_" in mod_seqs  # site 0 -> inside the N-term underscore
    assert (
        "_GLSDGEWQK_[Amidated]" in mod_seqs
    )  # site -1 -> after the closing underscore, as Spectronaut expects

    # the stripped sequence keeps no mod annotation
    assert set(df["StrippedPeptide"]) == {
        "PEPTIDEK",
        "ACDEFGHIK",
        "MSEQUENCEK",
        "AAAAAK",
        "SVIVSPYSTGAK",
        "LHDSTPPPYK",
        "GLSDGEWQK",
    }


def test_modified_sequence_uses_translate_mod_dict() -> None:
    """`translate_mod_dict` replaces the alphabase mod names, e.g. with UniMod ids."""
    df = _export(_build_speclib(), translate_mod_dict=mod_to_unimod_dict)

    assert "_AC[UniMod:4]DEFGHIK_" in set(df["ModifiedPeptide"])


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        (["irt_pred", "rt_pred", "rt_norm_pred", "rt", "irt", "rt_norm"], "irt_pred"),
        (["rt_pred", "rt_norm_pred", "rt", "irt", "rt_norm"], "rt_pred"),
        (["rt_norm_pred", "rt", "irt", "rt_norm"], "rt_norm_pred"),
        (["rt", "irt", "rt_norm"], "rt"),
        (["irt", "rt_norm"], "irt"),
        (["rt_norm"], "rt_norm"),
    ],
)
def test_rt_column_precedence(present: list, expected: str) -> None:
    """RT comes from the first present candidate, predictions before measurements."""
    speclib = _build_speclib()
    # give each candidate column a distinct value, so `RT` identifies its source
    values = {name: float(i + 1) for i, name in enumerate(present)}
    speclib._precursor_df = speclib._precursor_df.drop(columns=["rt_pred"]).assign(
        **values
    )

    assert _export(speclib)["RT"].unique().tolist() == [values[expected]]


def test_rt_norm_pred_is_accepted() -> None:
    """A library carrying only `rt_norm_pred` exports, rather than being rejected.

    peptdeep writes `rt_norm_pred` alongside `rt_pred`, so this matters for a
    library whose `rt_pred` was dropped.
    """
    speclib = _build_speclib()
    speclib._precursor_df = speclib._precursor_df.rename(
        columns={"rt_pred": "rt_norm_pred"}
    )

    assert _export(speclib)["RT"].notna().all()


def test_export_without_any_rt_column_is_rejected() -> None:
    """A library with no retention time at all is still an error."""
    speclib = _build_speclib()
    speclib._precursor_df = speclib._precursor_df.drop(columns=["rt_pred"])

    with pytest.raises(ValueError, match="must contain the RT columns"):
        _export(speclib)


def test_optional_columns_follow_the_source_library() -> None:
    """IonMobility, CCS, Genes and Decoy appear only if the library has them."""
    without = _export(_build_speclib())
    assert "IonMobility" not in without.columns
    assert "CCS" not in without.columns
    assert "Decoy" not in without.columns
    assert "Genes" in without.columns  # the fixture always carries genes

    n_precursors = 7
    with_all = _export(
        _build_speclib(
            mobility_pred=np.linspace(0.5, 1.1, n_precursors),
            ccs_pred=np.linspace(300.0, 400.0, n_precursors),
            decoy=[0, 1, 0, 1, 0, 1, 0],
        )
    )
    assert "IonMobility" in with_all.columns
    assert "CCS" in with_all.columns
    assert "Decoy" in with_all.columns

    # mobility_pred / ccs_pred take precedence over mobility / ccs
    preferred = _export(
        _build_speclib(
            mobility_pred=np.full(n_precursors, 1.0),
            mobility=np.full(n_precursors, 2.0),
            ccs_pred=np.full(n_precursors, 3.0),
            ccs=np.full(n_precursors, 4.0),
        )
    )
    assert preferred["IonMobility"].unique().tolist() == [1.0]
    assert preferred["CCS"].unique().tolist() == [3.0]


def test_protein_id_prefers_uniprot_ids() -> None:
    """`ProteinID` comes from uniprot_ids when present, else from proteins."""
    from_proteins = _export(_build_speclib())
    assert set(from_proteins["ProteinID"]) >= {"PROT1", "PROT2;PROT9"}

    with_uniprot = _export(_build_speclib(uniprot_ids=[f"P{i}" for i in range(7)]))
    assert set(with_uniprot["ProteinID"]) == {f"P{i}" for i in range(7)}


def test_top_k_selection_and_normalization() -> None:
    """At most k fragments per precursor, intensity-normalized and descending."""
    df = _export(_build_speclib())

    per_precursor = df.groupby("ModifiedPeptide", sort=False)
    assert (per_precursor.size() <= DEFAULT_KEEP_K).all()
    # the most intense kept fragment of each precursor is scaled to 1.0
    assert per_precursor["RelativeIntensity"].max().eq(1.0).all()
    # and fragments are emitted in descending intensity within a precursor
    for _, group in per_precursor:
        assert group["RelativeIntensity"].is_monotonic_decreasing

    fewer = _export(_build_speclib(), keep_k_highest_fragments=3)
    assert (fewer.groupby("ModifiedPeptide").size() <= 3).all()


def test_min_frag_intensity_is_exclusive() -> None:
    """`min_frag_intensity` drops fragments at or below the threshold."""
    df = _export(_build_speclib(), min_frag_intensity=0.5)

    assert (df["RelativeIntensity"] > 0.5).all()


def test_mz_window_selects_fragments_inside_it() -> None:
    """Only fragments inside [min_frag_mz, max_frag_mz] reach the output."""
    df = _export(_build_speclib(), min_frag_mz=600, max_frag_mz=900)

    assert df["FragmentMz"].between(600, 900).all()


def test_unbounded_mz_window_spellings_agree() -> None:
    """0 and `np.inf` mean no bound; the old 0/0 sentinel warns but still works."""
    unbounded = _export(_build_speclib(), min_frag_mz=0, max_frag_mz=np.inf)
    pd.testing.assert_frame_equal(
        unbounded, _export(_build_speclib(), min_frag_mz=-np.inf, max_frag_mz=np.inf)
    )

    with pytest.warns(FutureWarning, match="max_frag_mz=np.inf"):
        deprecated = _export(_build_speclib(), min_frag_mz=0, max_frag_mz=0)
    pd.testing.assert_frame_equal(unbounded, deprecated)

    # a bound of 0 on its own now means what it says, rather than "no bound"
    assert len(_export(_build_speclib(), max_frag_mz=0)) == 0


def test_min_frag_nAA_masks_the_smallest_fragments() -> None:
    """`min_frag_nAA=n` removes the n-1 smallest b and y fragments per terminus."""
    for min_frag_nAA in (2, 3, 4):
        df = _export(_build_speclib(), min_frag_nAA=min_frag_nAA)
        for series in ("b", "y"):
            numbers = df.loc[df["FragmentType"] == series, "FragmentNumber"]
            assert numbers.min() == min_frag_nAA, (series, min_frag_nAA)


def test_min_frag_nAA_wider_than_any_precursor() -> None:
    """Regression guard: a mask wider than the block covers all of it, not part.

    A `min_frag_nAA` larger than any precursor's fragment count is not a request
    the export defines an answer to -- it is pinned only because the masking
    works on row offsets, where an unclamped negative start silently masks the
    wrong end. This is what main does too.
    """
    assert len(_export(_build_speclib(), min_frag_nAA=12)) == 0


def test_modloss_label() -> None:
    """The `modloss` loss label is replaced by the `modloss` argument's value."""
    default = _export(_build_speclib())
    assert "modloss" not in set(default["FragmentLossType"])
    assert set(default["FragmentLossType"]) <= {"noloss", "H3PO4"}

    custom = _export(_build_speclib(), modloss="H2O")
    assert set(custom["FragmentLossType"]) <= {"noloss", "H2O"}


def test_export_leaves_the_source_library_untouched() -> None:
    """The export reads the library and writes nothing back to it."""
    speclib = _build_speclib()
    intensities_before = speclib.fragment_intensity_df.to_numpy(copy=True)
    mz_before = speclib.fragment_mz_df.to_numpy(copy=True)
    columns_before = set(speclib.precursor_df.columns)

    _export(speclib)

    np.testing.assert_array_equal(
        speclib.fragment_intensity_df.to_numpy(), intensities_before
    )
    np.testing.assert_array_equal(speclib.fragment_mz_df.to_numpy(), mz_before)
    # `precursor_mz` in particular is computed on a copy, not onto the library
    assert set(speclib.precursor_df.columns) == columns_before


def test_second_export_of_the_same_library_matches_a_fresh_one() -> None:
    """Exporting twice at different m/z windows is not order-dependent.

    A narrow window followed by a wider one used to be unable to recover the
    fragments the first export had zeroed in the library.
    """
    reused = _build_speclib()
    _export(reused, min_frag_mz=DEFAULT_MIN_FRAG_MZ, max_frag_mz=DEFAULT_MAX_FRAG_MZ)
    second = _export(reused, min_frag_mz=100, max_frag_mz=3000)
    fresh = _export(_build_speclib(), min_frag_mz=100, max_frag_mz=3000)

    pd.testing.assert_frame_equal(second, fresh)


def test_disabled_mz_window_skips_empty_fragment_slots() -> None:
    """Empty fragment slots are never exported, m/z window or not.

    A precursor's `*_modloss` slots carry m/z 0 unless it has a loss-bearing mod.
    The m/z window used to be the only thing removing them, so disabling it
    emitted them as if they were real fragments -- and they took top-k slots away
    from fragments that do exist.
    """
    df = _unfiltered(_build_speclib())

    assert (df["FragmentMz"] > 0).all()
    # the freed slots go to real fragments, so disabling the window cannot yield
    # fewer of them than the default window does
    assert len(df) >= len(_export(_build_speclib()))
    # only the loss-bearing precursors carry a loss fragment
    with_loss = set(df.loc[df["FragmentLossType"] == "H3PO4", "StrippedPeptide"])
    assert with_loss <= {"SVIVSPYSTGAK", "LHDSTPPPYK"}


def test_fragment_columns_are_typed() -> None:
    """The fragment columns are numbers, and m/z and intensity keep their own dtype.

    The fixture builds a float32 m/z frame and a float64 intensity frame, so the
    two carry through independently rather than being cast to one width.
    """
    speclib = _build_speclib()
    df = _export(speclib)

    assert df["FragmentMz"].dtype == speclib.fragment_mz_df.to_numpy().dtype
    assert (
        df["RelativeIntensity"].dtype == speclib.fragment_intensity_df.to_numpy().dtype
    )
    assert df["FragmentCharge"].dtype.kind == "i"
    assert df["FragmentNumber"].dtype.kind == "i"
    # the two label columns stay strings
    assert df["FragmentType"].dtype == object
    assert df["FragmentLossType"].dtype == object

    assert set(df["FragmentCharge"]) == {1, 2}


def test_speclib_to_single_df_is_a_deprecated_alias() -> None:
    """The old name still works, warns, and returns what the new one returns."""
    with pytest.warns(FutureWarning, match="use speclib_to_swath_df"):
        aliased = speclib_to_single_df(_build_speclib(), verbose=False)

    pd.testing.assert_frame_equal(aliased, _export(_build_speclib()))


def test_translate_to_tsv_matches_the_in_memory_export(tmp_path) -> None:
    """The tsv holds exactly the in-memory transition list, with one header."""
    tsv = str(tmp_path / "lib.tsv")
    translate_to_tsv(_build_speclib(), tsv, multiprocessing=False)

    with open(tsv) as f:
        lines = f.read().splitlines()
    expected = _export(_build_speclib())

    assert sum(line.startswith("ModifiedPeptide") for line in lines) == 1
    assert len(lines) == len(expected) + 1
    written = pd.read_csv(tsv, sep="\t")
    assert list(written.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(
        written.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


def test_translate_to_tsv_batching_does_not_change_the_file(tmp_path) -> None:
    """Batch size affects only memory use, not the written file."""
    digests = []
    for batch_size in (100000, 2, 1):
        tsv = str(tmp_path / f"lib_{batch_size}.tsv")
        translate_to_tsv(
            _build_speclib(), tsv, batch_size=batch_size, multiprocessing=False
        )
        with open(tsv, "rb") as f:
            digests.append(hashlib.sha256(f.read()).hexdigest())

    assert len(set(digests)) == 1


def test_translate_to_tsv_multiprocessing_matches_single_process(tmp_path) -> None:
    """The forked writer produces a byte-identical file."""
    digests = []
    for multiprocessing in (False, True):
        tsv = str(tmp_path / f"lib_{multiprocessing}.tsv")
        translate_to_tsv(_build_speclib(), tsv, multiprocessing=multiprocessing)
        with open(tsv, "rb") as f:
            digests.append(hashlib.sha256(f.read()).hexdigest())

    assert len(set(digests)) == 1


def test_translate_to_tsv_disabled_mz_window_matches_the_in_memory_export(
    tmp_path,
) -> None:
    """`min_frag_mz=0, max_frag_mz=np.inf` disables the filter in both entry points.

    `translate_to_tsv` used to mask by m/z without checking whether the window
    was disabled, so 0/0 -- the documented way to turn the filter off -- zeroed
    every real fragment and wrote a file of nothing but empty slots, while
    `speclib_to_swath_df` given the same arguments kept the real ones.
    """
    tsv = str(tmp_path / "lib.tsv")
    translate_to_tsv(
        _build_speclib(),
        tsv,
        min_frag_mz=0,
        max_frag_mz=np.inf,
        min_frag_intensity=0.0,
        multiprocessing=False,
    )

    written = pd.read_csv(tsv, sep="\t")
    assert len(written) > 0
    assert (written["FragmentMz"] > 0).all()

    expected = _unfiltered(_build_speclib())
    pd.testing.assert_frame_equal(
        written.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


def test_translate_to_tsv_writes_a_readable_library(tmp_path) -> None:
    """Every precursor survives a tsv export and re-import.

    The assertions above check the shape of the output; this one checks that it is
    consumable, which they cannot.

    UniMod names are needed because the default alphabase short names (`S[Phospho]`,
    `_[Acetyl]`) are not in the reader's mapping and those precursors would be
    dropped on import -- a reader gap, not an export one.
    """
    speclib = _build_speclib()
    tsv = str(tmp_path / "lib.tsv")
    translate_to_tsv(
        speclib, tsv, translate_mod_dict=mod_to_unimod_dict, multiprocessing=False
    )

    reader = LibraryReaderBase()
    reader.import_file(tsv)

    def keys(df: pd.DataFrame) -> set:
        return set(zip(df["sequence"], df["mod_sites"], df["charge"].astype(int)))

    assert keys(reader.precursor_df) == keys(speclib.precursor_df)


class _DeadWriter:
    """Stands in for a `WritingProcess` whose child died without writing."""

    exitcode = 1

    def __init__(self, task_queue, tsv) -> None:
        pass

    def start(self) -> None:
        pass

    def join(self) -> None:
        pass


def test_translate_to_tsv_raises_when_the_writing_process_dies(
    tmp_path, monkeypatch
) -> None:
    """A writer that died is an error, not a quietly truncated file.

    `multiprocessing=True` is the default, and the writer process dies before it
    writes anything if the calling script has no `if __name__ == "__main__":`
    guard -- the norm on the spawn platforms, macOS and Windows. Nothing checked
    on it, so the export printed its success message and returned normally,
    leaving a 0-byte tsv behind.
    """
    monkeypatch.setattr(translate, "WritingProcess", _DeadWriter)

    with pytest.raises(RuntimeError, match="exited with code 1"):
        translate_to_tsv(_build_speclib(), str(tmp_path / "lib.tsv"))
