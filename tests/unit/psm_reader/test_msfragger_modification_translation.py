"""Unit tests for MSFragger modification translation."""

import pandas as pd
import pytest

from alphabase.constants.modification import MOD_MASS
from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.msfragger_reader import MSFraggerModificationTranslation
from alphabase.psm_reader.psm_reader import psm_reader_yaml


@pytest.fixture
def translator():
    """Fixture to create translator instance with default config."""
    mass_mapped_mods = psm_reader_yaml.get("msfragger_psm_tsv", {}).get(
        "mass_mapped_mods", []
    )
    return MSFraggerModificationTranslation(
        mass_mapped_mods=mass_mapped_mods, mod_mass_tol=0.1
    )


@pytest.fixture
def test_psm_df():
    """Fixture for test DataFrame with modifications."""
    return pd.DataFrame(
        {
            "Peptide": ["PEPTIDE", "SEQUENCE", "ANOTHER"],
            "Assigned Modifications": [
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
    translator = MSFraggerModificationTranslation(
        mass_mapped_mods=mass_mapped_mods, mod_mass_tol=mod_mass_tol
    )
    assert translator.mass_mapped_mods == expected_mods
    assert translator.mod_mass_tol == expected_tol


@pytest.mark.parametrize(
    "assigned_mods,expected_mods,expected_sites",
    [
        ("", "", ""),
        ("5S(79.9663)", "Phospho@S", "5"),
        ("6M(15.9949)", "Oxidation@M", "6"),
        ("N-term(304.2071)", "TMTpro@Any_N-term", "0"),
        ("5S79.9663", "", ""),  # Malformed
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


def test_dataframe_translation(translator, test_psm_df):
    """Test end-to-end translation preserves original columns and adds complete mods."""
    result_df = translator(test_psm_df)

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
    for mod_name in translator.mass_mapped_mods:
        assert mod_name in MOD_MASS
        mod_mass = MOD_MASS[mod_name]
        aa = mod_name.split("@")[1]
        if aa not in ["Any_N-term", "Any_C-term"] and len(aa) == 1:
            assert translator._match_mod_by_mass(mod_mass, aa) == mod_name
