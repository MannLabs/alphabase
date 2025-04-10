import pytest

from alphabase.constants.modification import (
    MOD_DF,
    MOD_Composition,
    add_new_modifications,
)


@pytest.fixture
def cleanup_test_mods():
    """Clean up test modifications after test."""
    # Keep track of test modifications
    test_mods = [
        "TestMod@N-term",
        "Mod1@N-term",
        "Mod2@S",
        "Mod3@K",
        "InitMod1@N-term",
        "InitMod2@S",
        "RoundTrip1@N-term",
        "RoundTrip2@S",
    ]

    yield

    # Remove test modifications
    for mod in test_mods:
        if mod in MOD_DF.index:
            MOD_DF.drop(mod, inplace=True)
        if mod in MOD_Composition:
            del MOD_Composition[mod]

    # Update dictionaries
    from alphabase.constants.modification import update_all_by_MOD_DF

    update_all_by_MOD_DF()


def test_add_new_modifications(cleanup_test_mods):
    """Test adding new modifications."""
    # Add a custom modification
    add_new_modifications([("TestMod@N-term", "C(2)H(3)O(1)", "H(2)O(1)")])

    # The mod should be in MOD_DF
    assert "TestMod@N-term" in MOD_DF.index

    # Check the properties
    assert MOD_DF.loc["TestMod@N-term", "composition"] == "C(2)H(3)O(1)"
    assert MOD_DF.loc["TestMod@N-term", "modloss_composition"] == "H(2)O(1)"

    # Mass should be calculated correctly
    assert MOD_DF.loc["TestMod@N-term", "mass"] > 0
    assert MOD_DF.loc["TestMod@N-term", "modloss"] > 0

    # Check that MOD_Composition was updated
    assert "TestMod@N-term" in MOD_Composition


def test_add_multiple_modifications(cleanup_test_mods):
    """Test adding multiple modifications."""
    # Add multiple custom modifications
    add_new_modifications(
        [
            ("Mod1@N-term", "C(2)H(3)O(1)", "H(2)O(1)"),
            ("Mod2@S", "C(3)H(5)O(2)", ""),
            ("Mod3@K", "C(4)H(7)N(1)", "N(1)H(1)"),
        ]
    )

    # Check they were all added
    assert "Mod1@N-term" in MOD_DF.index
    assert "Mod2@S" in MOD_DF.index
    assert "Mod3@K" in MOD_DF.index

    # Check that MOD_Composition was updated
    assert "Mod1@N-term" in MOD_Composition
    assert "Mod2@S" in MOD_Composition
    assert "Mod3@K" in MOD_Composition
