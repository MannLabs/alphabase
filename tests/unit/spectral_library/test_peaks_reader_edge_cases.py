"""Edge-case tests for PEAKSLibraryReader against a hand-crafted synthetic library.

`input_data/peaks_edge_cases.tsv` is a 28-row PEAKS-format TSV built specifically
to exercise the parsing branches that the real 52,672-row MPIB example library
happens not to contain (e.g. a fragment charge+loss combined, a fragment
number that exceeds what's chemically possible, an N-terminal mod together
with a side-chain mod). Three of its rows are *expected* to be dropped (not
silently mis-parsed) - see `test_edge_case_file_drops_only_expected_rows`.

Four more files (`peaks_missing_*.tsv`, `peaks_impossible_fragment.tsv`,
`peaks_empty_library.tsv`) are single-purpose fixtures for behavior that
can't coexist with a successful import in the same file (hard errors, or a
header-only file).
"""

from pathlib import Path

import pandas as pd
import pytest

from alphabase.spectral_library.flat import SpecLibFlat
from alphabase.spectral_library.peaks_reader import PEAKSLibraryReader

_DATA_DIR = Path(__file__).parent / "input_data"


def _frags_for(speclib: SpecLibFlat, precursor_mz: float):
    row = speclib.precursor_df[speclib.precursor_df["precursor_mz"] == precursor_mz].iloc[0]
    return speclib.fragment_df.iloc[row["flat_frag_start_idx"] : row["flat_frag_stop_idx"]]


@pytest.fixture(scope="module")
def edge_case_speclib() -> SpecLibFlat:
    with pytest.warns(UserWarning, match="unmapped modifications"):
        return PEAKSLibraryReader().import_file(str(_DATA_DIR / "peaks_edge_cases.tsv"))


@pytest.fixture(scope="module")
def edge_case_precursor_df(edge_case_speclib):
    return edge_case_speclib.precursor_df


# --- file-level: exactly the expected rows survive, nothing more/less ---


def test_edge_case_file_drops_only_expected_rows(edge_case_precursor_df):
    """28 rows in; 3 are expected soft-drops (2 bad sequences, 1 unmapped mod)."""
    assert len(edge_case_precursor_df) == 28 - 3


def test_full_standard_aa_alphabet_is_covered(edge_case_precursor_df):
    """Fixture-design check: every standard AA appears somewhere, so a typo'd
    mass table entry elsewhere in AlphaBase wouldn't slip past this fixture unnoticed.
    """
    all_chars = set("".join(edge_case_precursor_df["sequence"]))
    assert set("ACDEFGHIKLMNPQRSTVWY") <= all_chars


# --- Modifications column ---


def test_no_modification_gives_empty_strings_not_nan_text(edge_case_precursor_df):
    row = edge_case_precursor_df[edge_case_precursor_df["precursor_mz"] == 500.0].iloc[0]
    assert row["mods"] == ""
    assert row["mod_sites"] == ""
    assert "nan" not in row["mods"].lower()


def test_single_modification(edge_case_precursor_df):
    row = edge_case_precursor_df[edge_case_precursor_df["precursor_mz"] == 510.0].iloc[0]
    assert row["mods"] == "Carboxymethyl@C"
    assert row["mod_sites"] == "2"


def test_three_modifications_on_one_peptide(edge_case_precursor_df):
    row = edge_case_precursor_df[edge_case_precursor_df["precursor_mz"] == 520.0].iloc[0]
    assert row["mods"] == "Carboxymethyl@C;Oxidation@M;Carboxymethyl@C"
    assert row["mod_sites"] == "1;2;7"


def test_n_terminal_modification_gets_fixed_site_zero(edge_case_precursor_df):
    row = edge_case_precursor_df[edge_case_precursor_df["precursor_mz"] == 430.0].iloc[0]
    assert row["mods"] == "Acetyl@Protein_N-term"
    assert row["mod_sites"] == "0"


def test_modification_on_last_residue_site_equals_sequence_length(edge_case_precursor_df):
    """Off-by-one guard: PEAKS 0-based index 5 on a 6-residue peptide -> site "6"
    (== len(sequence)), not 5 and not 7.
    """
    row = edge_case_precursor_df[edge_case_precursor_df["precursor_mz"] == 440.0].iloc[0]
    assert len(row["sequence"]) == 6
    assert row["mod_sites"] == "6"


def test_two_modifications_at_the_same_position_are_both_kept(edge_case_precursor_df):
    """Malformed input (two mods claiming the same residue), but a defined,
    tested behavior: both are kept verbatim, no dedup/error.
    """
    row = edge_case_precursor_df[edge_case_precursor_df["precursor_mz"] == 450.0].iloc[0]
    assert row["mods"] == "Carboxymethyl@C;Oxidation@M"
    assert row["mod_sites"] == "2;2"


def test_modifications_out_of_position_order_preserve_input_order(edge_case_precursor_df):
    """PEAKS listed the higher position (8) before the lower one (2) - AlphaBase's
    mods/mod_sites preserve that same (unsorted) order, they don't get re-sorted by position.
    """
    row = edge_case_precursor_df[edge_case_precursor_df["precursor_mz"] == 460.0].iloc[0]
    assert row["mods"] == "Carboxymethyl@C;Oxidation@M"
    assert row["mod_sites"] == "9;3"


def test_trailing_stray_delimiter_is_ignored(edge_case_precursor_df):
    row = edge_case_precursor_df[edge_case_precursor_df["precursor_mz"] == 470.0].iloc[0]
    assert row["mods"] == "Oxidation@M"
    assert row["mod_sites"] == "4"


def test_unmapped_modification_drops_precursor_with_warning():
    """Covered by the module-scoped fixture's pytest.warns - this test just
    confirms the specific precursor (m/z 290.0) is actually gone.
    """
    with pytest.warns(UserWarning, match="unmapped modifications"):
        speclib = PEAKSLibraryReader().import_file(str(_DATA_DIR / "peaks_edge_cases.tsv"))
    assert (speclib.precursor_df["precursor_mz"] == 290.0).sum() == 0


# --- Sequence ---


def test_short_and_long_peptide_boundaries(edge_case_precursor_df):
    short_row = edge_case_precursor_df[edge_case_precursor_df["precursor_mz"] == 300.0].iloc[0]
    long_row = edge_case_precursor_df[edge_case_precursor_df["precursor_mz"] == 900.0].iloc[0]
    assert short_row["nAA"] == 5
    assert long_row["nAA"] == 30


def test_invalid_sequence_characters_are_dropped_with_warning(caplog):
    """Lowercase letters and digits can't come from a real PEAKS export -
    dropped (with a logged warning), not silently kept to produce garbage downstream.
    """
    with pytest.warns(UserWarning, match="unmapped modifications"):
        speclib = PEAKSLibraryReader().import_file(str(_DATA_DIR / "peaks_edge_cases.tsv"))
    sequences = set(speclib.precursor_df["sequence"])
    assert "ACdEFGH" not in sequences  # lowercase 'd'
    assert "AC3EFGH" not in sequences  # digit


# --- Charge ---


def test_charge_range_one_through_six(edge_case_precursor_df):
    assert set(edge_case_precursor_df["charge"].unique()) >= {1, 2, 3, 6}


def test_same_peptide_different_charges_stay_separate_rows(edge_case_precursor_df):
    rows = edge_case_precursor_df[edge_case_precursor_df["sequence"] == "GAVLIKMSTR"]
    assert sorted(rows["charge"]) == [3, 4, 6]
    assert rows["flat_frag_start_idx"].nunique() == 3  # each has its own fragment slice


# --- Peaks List / fragment parsing ---


def test_bare_ion_no_loss_no_charge(edge_case_speclib):
    frags = _frags_for(edge_case_speclib, 280.0)
    assert len(frags) == 1
    f = frags.iloc[0]
    assert f["charge"] == 1
    assert f["loss_type"] == 0


def test_charge_only_suffix(edge_case_speclib):
    from alphabase.peptide.fragment import SERIES_MAPPING

    frags = _frags_for(edge_case_speclib, 281.0)
    f = frags.iloc[0]
    assert f["type"] == SERIES_MAPPING["y"]
    assert f["number"] == 13
    assert f["charge"] == 2
    assert f["loss_type"] == 0


def test_loss_only_suffix(edge_case_speclib):
    from alphabase.peptide.fragment import LOSS_MAPPING

    frags = _frags_for(edge_case_speclib, 282.0)
    f = frags.iloc[0]
    assert f["loss_type"] == LOSS_MAPPING["H2O"]
    assert f["charge"] == 1


def test_loss_and_charge_combined_suffix(edge_case_speclib):
    """"y5-NH3[2+]" - loss and charge together in one label. Not observed in the
    real file, so this is exactly the kind of case that needed a hand-built row.
    """
    from alphabase.peptide.fragment import LOSS_MAPPING, SERIES_MAPPING

    frags = _frags_for(edge_case_speclib, 283.0)
    f = frags.iloc[0]
    assert f["type"] == SERIES_MAPPING["y"]
    assert f["number"] == 5
    assert f["charge"] == 2
    assert f["loss_type"] == LOSS_MAPPING["NH3"]


def test_empty_peaks_list_gives_zero_fragments_not_a_crash(edge_case_speclib):
    frags = _frags_for(edge_case_speclib, 284.0)
    assert len(frags) == 0


def test_single_fragment_peaks_list(edge_case_speclib):
    frags = _frags_for(edge_case_speclib, 285.0)
    assert len(frags) == 1


def test_duplicate_fragment_entries_are_both_kept(edge_case_speclib):
    """Same ion listed twice in one Peaks List - documented behavior: kept, not deduped."""
    frags = _frags_for(edge_case_speclib, 286.0)
    assert len(frags) == 2
    assert frags.iloc[0]["mz"] == frags.iloc[1]["mz"]


def test_intensity_extremes_preserved_exactly(edge_case_speclib):
    frags = _frags_for(edge_case_speclib, 287.0)
    assert sorted(frags["intensity"]) == [0.0, 1.0]


def test_unparseable_ion_label_is_skipped_not_fatal(edge_case_speclib):
    """Row 293.0 has 12 good b/y tokens plus 2 garbled ones appended. The
    garbled tokens are dropped (with a warning); the precursor and its 12
    good fragments survive - unlike an unmapped modification (whole
    precursor dropped) or a chemically-impossible fragment number (hard
    raise), a single garbled *label* shouldn't cost the rest of the spectrum.
    """
    assert (edge_case_speclib.precursor_df["precursor_mz"] == 293.0).any()
    frags = _frags_for(edge_case_speclib, 293.0)
    assert len(frags) == 12  # GAVLIKM: 7 residues -> 6 b + 6 y


def test_fragment_number_exceeding_peptide_length_raises_clear_error():
    """"y10" on a 5-residue peptide would compute a negative backbone position
    (uint32 in fragment_df, so it would otherwise silently wrap to a huge
    number instead of erroring) - must raise, not corrupt the library.
    """
    with pytest.raises(ValueError, match="not chemically possible"):
        PEAKSLibraryReader().import_file(str(_DATA_DIR / "peaks_impossible_fragment.tsv"))


# --- Retention time ---


def test_rt_zero_is_not_treated_as_missing(edge_case_precursor_df):
    row = edge_case_precursor_df[edge_case_precursor_df["precursor_mz"] == 288.0].iloc[0]
    assert row["rt"] == 0.0
    assert not pd.isna(row["rt"])


# --- File / row level ---


def test_exact_duplicate_rows_are_both_kept(edge_case_precursor_df):
    """Same sequence/charge/mz/fragments appearing twice - documented
    behavior: no automatic deduplication.
    """
    rows = edge_case_precursor_df[edge_case_precursor_df["precursor_mz"] == 289.0]
    assert len(rows) == 2
    assert rows["flat_frag_start_idx"].nunique() == 2  # each keeps its own fragment slice


def test_missing_precursor_mz_raises_informative_error():
    with pytest.raises(ValueError, match="precursor_mz"):
        PEAKSLibraryReader().import_file(str(_DATA_DIR / "peaks_missing_precursor_mz.tsv"))


def test_missing_charge_raises_informative_error():
    with pytest.raises(ValueError, match="charge"):
        PEAKSLibraryReader().import_file(str(_DATA_DIR / "peaks_missing_charge.tsv"))


def test_empty_library_returns_valid_empty_speclib():
    speclib = PEAKSLibraryReader().import_file(str(_DATA_DIR / "peaks_empty_library.tsv"))
    assert isinstance(speclib, SpecLibFlat)
    assert len(speclib.precursor_df) == 0
    assert len(speclib.fragment_df) == 0


# --- Structural: 0/1/many fragments, frag_start/stop_idx line up exactly ---


def test_frag_start_stop_idx_line_up_for_zero_one_and_many_fragments(edge_case_speclib):
    """The single easiest place for an off-by-one to hide: after concatenating
    fragments from many precursors with 0, 1, or many fragments each, every
    precursor's [flat_frag_start_idx, flat_frag_stop_idx) slice must exactly
    match its own fragments - no overlap, no gap, no drift.
    """
    pdf, fdf = edge_case_speclib.precursor_df, edge_case_speclib.fragment_df

    # explicit 0/1/many trio from the fixture
    zero_frag = pdf[pdf["precursor_mz"] == 284.0].iloc[0]
    one_frag = pdf[pdf["precursor_mz"] == 285.0].iloc[0]
    many_frag = pdf[pdf["precursor_mz"] == 900.0].iloc[0]  # 30-mer, many b/y ions

    assert zero_frag["flat_frag_stop_idx"] - zero_frag["flat_frag_start_idx"] == 0
    assert one_frag["flat_frag_stop_idx"] - one_frag["flat_frag_start_idx"] == 1
    assert many_frag["flat_frag_stop_idx"] - many_frag["flat_frag_start_idx"] == 58  # 30-mer: 29 b + 29 y

    # global invariant: ranges are sorted, non-overlapping, and exactly cover fragment_df
    sorted_pdf = pdf.sort_values("flat_frag_start_idx")
    assert (sorted_pdf["flat_frag_start_idx"].to_numpy()[1:] == sorted_pdf["flat_frag_stop_idx"].to_numpy()[:-1]).all()
    assert sorted_pdf["flat_frag_start_idx"].iloc[0] == 0
    assert sorted_pdf["flat_frag_stop_idx"].iloc[-1] == len(fdf)
