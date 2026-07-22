"""Unit tests for PEAKSLibraryReader.

Fixture rows below are copied verbatim from the MPIB example library
(dia_db_library.tsv, https://datashare.biochem.mpg.de/s/GxW5M9wmYbJW6Zz), with
two exceptions clearly marked as synthetic (built to isolate a single parsing
branch that no single real row exercises in isolation): the multi-modification
row and the charge-2-fragment row.

The exact position-indexing convention (PEAKS reports 0-based sequence
indices for side-chain mods) and the full modification vocabulary in
peaks_reader.py were both verified against all 52,672 rows of the real file
before writing the assertions below - see peaks_reader.py's module docstring
and comments for that verification.
"""

import io

import pytest

from alphabase.spectral_library.flat import SpecLibFlat
from alphabase.spectral_library.peaks_reader import PEAKSLibraryReader

_HEADER = (
    "m/z\tz\trt (seconds)\tActivation Mode\tSequence (backbone)\t"
    "Modifications\tPeaks Count\tPeaks List\tEngine"
)

# unmodified peptide, single charge fragments only
_ROW_NO_MOD = (
    '"478.77979"\t"2"\t"58.2"\t"CID, CAD(y and b ions)"\t"AAAAAAALQAK"\t""\t"16"\t'
    '"214.11861:1.0000:b3;218.14990:0.1106:y2;285.15570:0.6207:b4;346.20847:0.1872:y3;'
    "356.19281:0.4090:b5;427.22992:0.1952:b6;459.29254:0.2122:y4;498.26703:0.1207:b7;"
    "530.32965:0.3947:y5;601.36676:0.7234:y6;611.35107:0.0564:b8;672.40387:0.9511:y7;"
    '739.40967:0.0261:b9;743.44098:0.4926:y8;814.47809:0.3936:y9;885.51520:0.0250:y10"\t'
    '"DB_SEARCH"'
)

# single side-chain modification (Carboxymethyl), includes two neutral-loss fragments
_ROW_ONE_MOD = (
    '"763.32721"\t"2"\t"65.5"\t"CID, CAD(y and b ions)"\t"AAAGEFADDPCSSVK"\t'
    '"10-Carboxymethyl-(58.01)"\t"20"\t'
    '"214.11861:0.4500:b3;246.18120:0.0999:y2;271.14005:0.4500:b4;333.21323:0.1499:y3;'
    "400.18265:0.4498:b5;420.24524:0.1000:y4;547.25104:0.0500:b6;581.25995:0.1499:y5;"
    "618.28815:0.2998:b7;678.31268:0.7499:y6;716.28815:0.0500:b8-NH3;775.32904:0.0500:y7-H2O;"
    "776.31262:0.0500:y7-NH3;848.34204:0.1000:b9;908.36658:0.4000:y8;979.40369:1.0000:y9;"
    '1126.47205:0.9001:y10;1237.50415:0.0500:y11-H2O;1312.53613:0.3999:y12;1383.57324:0.0500:y13"\t'
    '"DB_SEARCH"'
)

# SYNTHETIC: real multi-mod rows in the example file don't happen to include
# an N-terminal mod alongside a side-chain one on a short, easy-to-verify
# sequence, so this one is hand-built (chemically nonsensical - Carboxymethyl
# normally targets Cys, not Glu - but that's irrelevant for parsing logic).
_ROW_MULTI_MOD = (
    '"500.00000"\t"2"\t"70.0"\t"CID, CAD(y and b ions)"\t"AAAGEFADDPCSSVK"\t'
    '"0-Acetylation (Protein N-term)-(42.01);4-Carboxymethyl-(58.01)"\t"2"\t'
    '"214.11861:1.0000:b3;218.11353:0.5000:y2"\t"DB_SEARCH"'
)

# SYNTHETIC: minimal row isolating the "[n+]" fragment-charge suffix, taken
# from the real ion label "b8[2+]" seen in the example file but attached to a
# short made-up peptide/spectrum for a self-contained test.
_ROW_CHARGED_FRAGMENT = (
    '"400.00000"\t"2"\t"50.0"\t"CID, CAD(y and b ions)"\t"AAAAAAALQAK"\t""\t"2"\t'
    '"214.11861:1.0000:b3;285.15570:0.7745:b4[2+]"\t"DB_SEARCH"'
)


def _make_library_tsv(*rows: str) -> io.StringIO:
    return io.StringIO("\n".join([_HEADER, *rows]))


@pytest.fixture
def peaks_reader() -> PEAKSLibraryReader:
    return PEAKSLibraryReader()


def test_import_file_returns_speclib_flat(peaks_reader):
    """import_file() should return a SpecLibFlat with non-empty precursor_df/fragment_df."""
    speclib = peaks_reader.import_file(_make_library_tsv(_ROW_NO_MOD))

    assert isinstance(speclib, SpecLibFlat)
    assert len(speclib.precursor_df) == 1
    assert len(speclib.fragment_df) == 16


def test_precursor_columns_are_harmonized(peaks_reader):
    """sequence/charge/precursor_mz/rt should be read and renamed correctly."""
    speclib = peaks_reader.import_file(_make_library_tsv(_ROW_NO_MOD))
    row = speclib.precursor_df.iloc[0]

    assert row["sequence"] == "AAAAAAALQAK"
    assert row["charge"] == 2
    assert row["precursor_mz"] == pytest.approx(478.77979)
    # AlphaBase's internal `rt` convention is minutes; PEAKS reports seconds.
    assert row["rt"] == pytest.approx(58.2 / 60)
    assert row["nAA"] == 11


def test_unmodified_peptide_has_empty_mods(peaks_reader):
    speclib = peaks_reader.import_file(_make_library_tsv(_ROW_NO_MOD))
    row = speclib.precursor_df.iloc[0]

    assert row["mods"] == ""
    assert row["mod_sites"] == ""


def test_single_modification_harmonized_to_unimod(peaks_reader):
    """ "10-Carboxymethyl-(58.01)" on "AAAGEFADDPCSSVK" should become
    "Carboxymethyl@C" at 1-based site 11 (PEAKS 0-based index 10 -> AlphaBase
    1-based site 10 + 1), matching the 'C' at that position in the sequence.
    """
    speclib = peaks_reader.import_file(_make_library_tsv(_ROW_ONE_MOD))
    row = speclib.precursor_df.iloc[0]

    assert row["sequence"][10] == "C"  # sanity check on the fixture itself
    assert row["mods"] == "Carboxymethyl@C"
    assert row["mod_sites"] == "11"


def test_multiple_modifications_preserve_order(peaks_reader):
    """Multiple ';'-separated PEAKS mods should map to ';'-joined mods/mod_sites,
    in the same order, with the N-terminal mod mapped to the fixed site "0"
    rather than position + 1.
    """
    speclib = peaks_reader.import_file(_make_library_tsv(_ROW_MULTI_MOD))
    row = speclib.precursor_df.iloc[0]

    assert row["mods"] == "Acetyl@Protein_N-term;Carboxymethyl@C"
    assert row["mod_sites"] == "0;5"


def test_unknown_modification_drops_precursor(peaks_reader):
    """A PEAKS modification name with no entry in the mapping should cause the
    precursor to be dropped (with a warning), matching the convention in
    PSMReaderBase._translate_modifications.
    """
    row_with_unknown_mod = _ROW_ONE_MOD.replace("Carboxymethyl", "TotallyMadeUpMod")

    with pytest.warns(UserWarning, match="unmapped modifications"):
        speclib = peaks_reader.import_file(_make_library_tsv(row_with_unknown_mod))

    assert len(speclib.precursor_df) == 0


def test_fragments_are_exploded_one_row_per_ion(peaks_reader):
    """Each ';'-separated token in "Peaks List" should become one fragment_df row,
    linked back to its precursor via flat_frag_start_idx/flat_frag_stop_idx,
    with 'position' following the N-term-to-C-term 0-based convention used by
    alphabase.peptide.fragment.flatten_fragments.
    """
    speclib = peaks_reader.import_file(_make_library_tsv(_ROW_ONE_MOD))
    precursor = speclib.precursor_df.iloc[0]

    assert precursor["flat_frag_stop_idx"] - precursor["flat_frag_start_idx"] == 20

    fragments = speclib.fragment_df.iloc[
        precursor["flat_frag_start_idx"] : precursor["flat_frag_stop_idx"]
    ].reset_index(drop=True)

    # first token: "214.11861:0.4500:b3" on a 15-residue peptide
    first = fragments.iloc[0]
    assert first["mz"] == pytest.approx(214.11861)
    assert first["intensity"] == pytest.approx(0.4500)
    assert first["type"] == ord("b")
    assert first["number"] == 3
    assert first["position"] == 3 - 1
    assert first["charge"] == 1

    # second token: "246.18120:0.0999:y2" on the same 15-residue peptide
    second = fragments.iloc[1]
    assert second["type"] == ord("y")
    assert second["number"] == 2
    assert second["position"] == 15 - 2 - 1


def test_fragment_charge_from_bracket_suffix(peaks_reader):
    """ "b4[2+]" should get fragment charge 2; a plain "b3" (no suffix) should
    default to charge 1.
    """
    speclib = peaks_reader.import_file(_make_library_tsv(_ROW_CHARGED_FRAGMENT))
    fragments = speclib.fragment_df

    assert fragments.iloc[0]["charge"] == 1  # "b3"
    assert fragments.iloc[1]["charge"] == 2  # "b4[2+]"


def test_neutral_loss_fragments_get_loss_type(peaks_reader):
    """ "b8-NH3" / "y7-H2O" style ion labels (present in _ROW_ONE_MOD, tokens
    11 and 12) should set fragment_df's integer loss_type using AlphaBase's
    LOSS_MAPPING (NH3=17, H2O=18, none=0) - see alphabase.peptide.fragment.
    """
    from alphabase.peptide.fragment import LOSS_MAPPING

    speclib = peaks_reader.import_file(_make_library_tsv(_ROW_ONE_MOD))
    fragments = speclib.fragment_df.reset_index(drop=True)

    assert fragments.iloc[0]["loss_type"] == LOSS_MAPPING[""]  # "b3", no loss
    assert fragments.iloc[10]["loss_type"] == LOSS_MAPPING["NH3"]  # "b8-NH3"
    assert fragments.iloc[11]["loss_type"] == LOSS_MAPPING["H2O"]  # "y7-H2O"
