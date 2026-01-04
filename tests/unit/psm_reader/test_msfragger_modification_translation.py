"""Unit tests for MSFragger modification translation."""

import pandas as pd
import pytest

from alphabase.constants.modification import MOD_MASS
from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.msfragger_reader import MSFraggerModificationTranslator
from alphabase.psm_reader.psm_reader import psm_reader_yaml


@pytest.fixture
def translator():
    """Fixture to create translator instance with default config."""
    mass_mapped_mods = psm_reader_yaml.get("msfragger_psm_tsv", {}).get(
        "mass_mapped_mods", []
    )
    return MSFraggerModificationTranslator(
        mass_mapped_mods=mass_mapped_mods,
        mod_mass_tol=0.1,
        rev_mod_mapping={},
    )


@pytest.fixture
def test_psm_df():
    """Fixture for test DataFrame with modifications."""
    return pd.DataFrame(
        {
            "Peptide": ["PEPTIDE", "SEQUENCE", "ANOTHER"],
            PsmDfCols.TMP_MODS: [
                "5S(79.9663), N-term(304.2071)",
                "",
                "6M(15.9949)",
            ],
            "Charge": [2, 3, 2],
        }
    )


@pytest.mark.parametrize(
    "mass_mapped_mods,mod_mass_tol,expected_mods,expected_tol",
    [
        (["Phospho@S", "Oxidation@M"], 0.1, ["Phospho@S", "Oxidation@M"], 0.1),
        (["TMT@K"], 0.2, ["TMT@K"], 0.2),
    ],
)
def test_translator_initialization(
    mass_mapped_mods, mod_mass_tol, expected_mods, expected_tol
):
    """Test translator initialization with various parameters."""
    translator = MSFraggerModificationTranslator(
        mass_mapped_mods=mass_mapped_mods,
        mod_mass_tol=mod_mass_tol,
        rev_mod_mapping={},
    )
    assert translator._mass_mapped_mods == expected_mods
    assert translator._mod_mass_tol == expected_tol


@pytest.mark.parametrize(
    "assigned_mods,expected_mods,expected_sites",
    [
        ("", "", ""),
        ("5S(79.9663)", "Phospho@S", "5"),
        ("6M(15.9949)", "Oxidation@M", "6"),
        ("N-term(304.2071)", "TMTpro@Any_N-term", "0"),
        (
            "5S(79.9663), 7T(79.9663), 9M(15.9949)",
            "Phospho@S;Phospho@T;Oxidation@M",
            "5;7;9",
        ),
    ],
)
def test_parse_modifications(translator, assigned_mods, expected_mods, expected_sites):
    """Test parsing various modification formats with complete results."""
    mods, sites = translator._parse_assigned_modifications(assigned_mods)
    assert mods == expected_mods
    assert sites == expected_sites


@pytest.mark.parametrize(
    "malformed_entry,error_match",
    [
        ("5S79.9663", "could not parse amino acid and mass"),  # Missing parentheses
        ("S(79.9663)", "expected format"),  # Missing position
    ],
)
def test_malformed_modification_raises(translator, malformed_entry, error_match):
    """Test that malformed modification entries raise ValueError."""
    with pytest.raises(ValueError, match=error_match):
        translator._parse_assigned_modifications(malformed_entry)


@pytest.mark.parametrize(
    "mass,aa,expected",
    [
        (79.9663, "S", "Phospho@S"),
        (15.9949, "M", "Oxidation@M"),
        (304.2071, "K", "TMTpro@K"),
        (79.96, "S", "Phospho@S"),  # Within tolerance
    ],
)
def test_mass_matching(translator, mass, aa, expected):
    """Test mass matching for known modifications returns exact match."""
    assert translator._match_mod_by_mass(mass, aa) == expected


@pytest.mark.parametrize("mass,aa", [(79.5, "S"), (999.9999, "X")])
def test_unknown_modification_raises(translator, mass, aa):
    """Test that unknown modifications raise ValueError."""
    with pytest.raises(ValueError, match="Unknown modification"):
        translator._match_mod_by_mass(mass, aa)


def test_closest_mass_match():
    """Test that closest mass match is returned when multiple mods are within tolerance."""
    # Formyl@K: 27.9949, Dimethyl@K: 28.0313 - both within 0.1 Da of each other
    translator = MSFraggerModificationTranslator(
        mass_mapped_mods=["Formyl@K", "Dimethyl@K"],
        mod_mass_tol=0.1,
        rev_mod_mapping={},
    )
    # Exact Formyl mass should return Formyl
    assert translator._match_mod_by_mass(27.9949, "K") == "Formyl@K"
    # Exact Dimethyl mass should return Dimethyl
    assert translator._match_mod_by_mass(28.0313, "K") == "Dimethyl@K"
    # Mass closer to Formyl should return Formyl
    assert translator._match_mod_by_mass(28.00, "K") == "Formyl@K"
    # Mass closer to Dimethyl should return Dimethyl
    assert translator._match_mod_by_mass(28.02, "K") == "Dimethyl@K"


def test_dataframe_translation(translator, test_psm_df):
    """Test end-to-end translation preserves original columns and adds complete mods."""
    result_df = translator.translate(test_psm_df)

    # Check original columns preserved with exact values
    assert result_df["Peptide"].tolist() == ["PEPTIDE", "SEQUENCE", "ANOTHER"]
    assert result_df["Charge"].tolist() == [2, 3, 2]
    assert len(result_df) == 3

    # Check mod columns added
    assert PsmDfCols.MODS in result_df.columns
    assert PsmDfCols.MOD_SITES in result_df.columns

    # Check complete modification results for all rows
    assert result_df[PsmDfCols.MODS].iloc[0] == "Phospho@S;TMTpro@Any_N-term"
    assert result_df[PsmDfCols.MOD_SITES].iloc[0] == "5;0"
    assert result_df[PsmDfCols.MODS].iloc[1] == ""
    assert result_df[PsmDfCols.MOD_SITES].iloc[1] == ""
    assert result_df[PsmDfCols.MODS].iloc[2] == "Oxidation@M"
    assert result_df[PsmDfCols.MOD_SITES].iloc[2] == "6"


def test_database_integration(translator):
    """Test configured modifications exist in database."""
    for mod_name in translator._mass_mapped_mods:
        assert mod_name in MOD_MASS
        mod_mass = MOD_MASS[mod_name]
        aa = mod_name.split("@")[1]
        if aa not in ["Any_N-term", "Any_C-term"] and len(aa) == 1:
            assert translator._match_mod_by_mass(mod_mass, aa) == mod_name


class TestRevModMapping:
    """Tests for rev_mod_mapping functionality."""

    def test_rev_mapping_used_over_mass_matching(self):
        """Test that rev_mod_mapping takes precedence over mass-based matching."""
        translator = MSFraggerModificationTranslator(
            mass_mapped_mods=["Phospho@S", "Oxidation@M"],
            mod_mass_tol=0.1,
            rev_mod_mapping={"S(79.9663)": "Phospho@S"},
        )
        mods, sites = translator._parse_assigned_modifications("5S(79.9663)")
        assert mods == "Phospho@S"
        assert sites == "5"

    def test_rev_mapping_n_term(self):
        """Test rev_mod_mapping for N-terminal modifications."""
        translator = MSFraggerModificationTranslator(
            mass_mapped_mods=[],
            mod_mass_tol=0.1,
            rev_mod_mapping={"N-term(304.2071)": "TMTpro@Any_N-term"},
        )
        mods, sites = translator._parse_assigned_modifications("N-term(304.2071)")
        assert mods == "TMTpro@Any_N-term"
        assert sites == "0"

    def test_rev_mapping_c_term(self):
        """Test rev_mod_mapping for C-terminal modifications."""
        translator = MSFraggerModificationTranslator(
            mass_mapped_mods=[],
            mod_mass_tol=0.1,
            rev_mod_mapping={"C-term(17.0265)": "Amidated@Any_C-term"},
        )
        mods, sites = translator._parse_assigned_modifications("C-term(17.0265)")
        assert mods == "Amidated@Any_C-term"
        assert sites == "-1"

    def test_rev_mapping_multiple_mods(self):
        """Test rev_mod_mapping with multiple modifications in one entry."""
        translator = MSFraggerModificationTranslator(
            mass_mapped_mods=["Oxidation@M"],
            mod_mass_tol=0.1,
            rev_mod_mapping={
                "S(79.9663)": "Phospho@S",
                "N-term(304.2071)": "TMTpro@Any_N-term",
            },
        )
        mods, sites = translator._parse_assigned_modifications(
            "5S(79.9663), N-term(304.2071), 8M(15.9949)"
        )
        assert mods == "Phospho@S;TMTpro@Any_N-term;Oxidation@M"
        assert sites == "5;0;8"

    def test_rev_mapping_fallback_to_mass_matching(self):
        """Test that unmapped mods fall back to mass-based matching."""
        translator = MSFraggerModificationTranslator(
            mass_mapped_mods=["Oxidation@M"],
            mod_mass_tol=0.1,
            rev_mod_mapping={"S(79.9663)": "Phospho@S"},
        )
        mods, sites = translator._parse_assigned_modifications("6M(15.9949)")
        assert mods == "Oxidation@M"
        assert sites == "6"

    def test_empty_rev_mapping(self):
        """Test that empty rev_mod_mapping works correctly."""
        translator = MSFraggerModificationTranslator(
            mass_mapped_mods=["Phospho@S"],
            mod_mass_tol=0.1,
            rev_mod_mapping={},
        )
        mods, sites = translator._parse_assigned_modifications("5S(79.9663)")
        assert mods == "Phospho@S"
        assert sites == "5"
