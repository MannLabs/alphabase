"""Unit tests for the PEAKS DIA spectral library reader.

Every test builds its input inline from a few hand-written PEAKS rows and writes
it to a pytest ``tmp_path`` file. Nothing here reads the real example library, so
there is no committed data-file dependency.
"""

import numpy as np
import pandas as pd
import pytest

from alphabase.psm_reader.keys import PsmDfCols
from alphabase.spectral_library.peaks_reader import (
    DEFAULT_PEAKS_CHARGED_FRAG_TYPES,
    PeaksLibraryReader,
    PeaksModificationTranslator,
    _parse_fragment_annotation,
)

# The 9 columns of a PEAKS DIA library export. The reader only consumes
# 'Sequence (backbone)', 'z', 'm/z', 'rt (seconds)', 'Modifications' and
# 'Peaks List'; the other three are filler to keep the shape realistic.
_COLUMNS = [
    "m/z",
    "z",
    "rt (seconds)",
    "Activation Mode",
    "Sequence (backbone)",
    "Modifications",
    "Peaks Count",
    "Peaks List",
    "Engine",
]


def _row(sequence, mods, peaks, *, charge="2", mz="500.0", rt="83.0"):
    """Build one PEAKS library row as a list of cell strings.

    Parameters
    ----------
    sequence : str
        Backbone sequence, e.g. "PEPTIDEK".
    mods : str
        The PEAKS 'Modifications' cell, ';'-separated '<pos>-<name>-(<mass>)'
        entries (empty string for an unmodified precursor).
    peaks : str
        The 'Peaks List' cell: ';'-separated 'mz:intensity:annotation' triplets,
        e.g. "200.1:50:b2;350.1:100:y3".

    """
    peak_count = str(len(peaks.split(";"))) if peaks else "0"
    return [
        mz,
        charge,
        rt,
        "CID, CAD(y and b ions)",
        sequence,
        mods,
        peak_count,
        peaks,
        "DB_SEARCH",
    ]


def _write_library(tmp_path, rows):
    """Write a PEAKS-format TSV (header + rows) to tmp_path, return its path."""
    lines = ["\t".join(_COLUMNS)] + ["\t".join(row) for row in rows]
    path = tmp_path / "sample_peaks_library.tsv"
    path.write_text("\n".join(lines))
    return str(path)


@pytest.fixture
def sample_library(tmp_path):
    """A 3-precursor sample covering the features the reader must handle.

    - PEPTIDEK: no mods; fragments include an H2O loss, an NH3 loss and a 2+ ion.
    - AMPEPTK:  N-terminal acetylation + a positional Met oxidation.
    - ACPEPTK:  a positional Carboxymethyl on Cys.
    """
    return _write_library(
        tmp_path,
        [
            _row(
                "PEPTIDEK",
                "",
                "200.1:50:b2;400.2:30:b4-H2O;350.1:100:y3;500.2:40:y5[2+];250.1:20:y2-NH3",
                mz="456.7",
                rt="83.0",
            ),
            _row(
                "AMPEPTK",
                "0-Acetylation (Protein N-term)-(42.01);1-Oxidation (M)-(15.99)",
                "100.1:100:b1;250.1:80:y2;300.2:50:y4[2+]",
                mz="400.2",
                rt="90.0",
            ),
            _row(
                "ACPEPTK",
                "1-Carboxymethyl-(58.01)",
                "180.1:100:b2;320.1:60:y3",
                mz="410.2",
                rt="95.0",
            ),
        ],
    )


class TestFragmentAnnotationParsing:
    """Tests for _parse_fragment_annotation (annotation string -> components)."""

    @pytest.mark.parametrize(
        ("annotation", "expected"),
        [
            ("b6", ("b", 6, 1, "")),
            ("y13[2+]", ("y", 13, 2, "")),
            ("b12-H2O", ("b", 12, 1, "H2O")),
            ("y5-NH3", ("y", 5, 1, "NH3")),
            ("y5-NH3[3+]", ("y", 5, 3, "NH3")),
        ],
    )
    def test_valid_annotations(self, annotation, expected):
        assert _parse_fragment_annotation(annotation) == expected

    @pytest.mark.parametrize("annotation", ["", "garbage", "b", "z+"])
    def test_invalid_annotations(self, annotation):
        assert _parse_fragment_annotation(annotation) is None


class TestModificationMatching:
    """Tests for PeaksModificationTranslator._match_mod_by_mass (mass + residue -> name)."""

    _CANDIDATES = ["Carboxymethyl@C", "Oxidation@M", "Acetyl@Protein_N-term"]

    @pytest.fixture
    def translator(self):
        return PeaksModificationTranslator(self._CANDIDATES, mod_mass_tol=0.1)

    def test_positional_match_uses_residue(self, translator):
        assert translator._match_mod_by_mass(15.99, "M") == "Oxidation@M"
        assert translator._match_mod_by_mass(58.01, "C") == "Carboxymethyl@C"

    def test_terminal_match(self, translator):
        assert (
            translator._match_mod_by_mass(42.01, "Any_N-term")
            == "Acetyl@Protein_N-term"
        )

    def test_unknown_mass_returns_none(self, translator):
        assert translator._match_mod_by_mass(79.97, "S") is None

    def test_wrong_residue_returns_none(self, translator):
        # 58.01 exists as Carboxymethyl@C but not on residue 'A'
        assert translator._match_mod_by_mass(58.01, "A") is None


class TestReaderBasics:
    """Tests for reader construction and configuration."""

    def test_initialization(self):
        reader = PeaksLibraryReader()
        assert reader._reader_type == "peaks_library"
        assert reader.charged_frag_types == DEFAULT_PEAKS_CHARGED_FRAG_TYPES
        assert "Oxidation@M" in reader._mod_translator._mass_mapped_mods
        assert reader._mod_translator._mod_mass_tol == 0.1
        assert reader.column_mapping[PsmDfCols.SEQUENCE] == "Sequence (backbone)"


class TestEndToEnd:
    """Tests for the full import pipeline on a written sample library."""

    def test_modifications_are_parsed(self, sample_library):
        reader = PeaksLibraryReader()
        reader.import_file(sample_library)

        # sequence/mods/mod_sites/charge for every precursor, in one comparison.
        # - PEPTIDEK: unmodified -> empty mods/sites
        # - AMPEPTK:  N-terminal acetyl (site 0) + Met oxidation (0-based pos 1 -> site 2)
        # - ACPEPTK:  Carboxymethyl on Cys (0-based pos 1 -> site 2)
        expected = pd.DataFrame(
            {
                "sequence": ["ACPEPTK", "AMPEPTK", "PEPTIDEK"],
                "mods": ["Carboxymethyl@C", "Acetyl@Protein_N-term;Oxidation@M", ""],
                "mod_sites": ["2", "0;2", ""],
                "charge": np.array(
                    [2, 2, 2], dtype=reader.precursor_df["charge"].dtype
                ),
            }
        )
        actual = reader.precursor_df.sort_values("sequence").reset_index(drop=True)[
            expected.columns
        ]
        pd.testing.assert_frame_equal(actual, expected)

    def test_fragment_tables(self, sample_library):
        reader = PeaksLibraryReader()
        reader.import_file(sample_library)

        # one dense fragment row per (nAA - 1) per precursor: 7 + 6 + 6 = 19
        expected_rows = sum(reader.precursor_df["nAA"] - 1)
        assert expected_rows == 19
        assert len(reader.fragment_intensity_df) == expected_rows
        assert reader.fragment_intensity_df.shape == reader.fragment_mz_df.shape
        assert (
            list(reader.fragment_intensity_df.columns)
            == DEFAULT_PEAKS_CHARGED_FRAG_TYPES
        )

        # every precursor's intensities are normalized to a max of 1.0
        intensities = reader.fragment_intensity_df.to_numpy()
        for start, stop in zip(
            reader.precursor_df["frag_start_idx"],
            reader.precursor_df["frag_stop_idx"],
        ):
            assert np.isclose(intensities[start:stop].max(), 1.0)

    def test_neutral_loss_and_charge_fragments_are_kept(self, sample_library):
        reader = PeaksLibraryReader()
        reader.import_file(sample_library)

        peptidek = reader.precursor_df.set_index("sequence").loc["PEPTIDEK"]
        frags = reader.fragment_intensity_df.iloc[
            peptidek["frag_start_idx"] : peptidek["frag_stop_idx"]
        ]
        # H2O loss, NH3 loss and a 2+ fragment all survived the import
        assert frags["b_H2O_z1"].sum() > 0
        assert frags["y_NH3_z1"].sum() > 0
        assert frags["y_z2"].sum() > 0

    def test_unmapped_modification_warns_and_drops_precursor(self, tmp_path):
        """An unresolved modification drops its precursor (and warns), keeping the rest."""
        library = _write_library(
            tmp_path,
            [
                _row("PEPTIDEK", "", "200.1:50:b2;350.1:100:y3"),
                # Phospho is not in mass_mapped_mods -> unresolved -> dropped
                _row("SPEPTIK", "0-Phospho-(79.97)", "180.1:60:b2;300.1:100:y3"),
            ],
        )

        reader = PeaksLibraryReader()
        with pytest.warns(UserWarning, match="Unknown PEAKS modifications"):
            reader.import_file(library)

        sequences = set(reader.precursor_df["sequence"])
        assert sequences == {"PEPTIDEK"}  # good precursor survives, bad one dropped
