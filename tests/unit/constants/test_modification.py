import pytest

from alphabase.constants.modification import (
    MOD_DF,
    MOD_Composition,
    add_new_modifications,
    get_custom_mods,
    get_new_context,
    init_custom_mods,
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


def test_get_custom_mods(cleanup_test_mods):
    """Test getting custom modifications."""
    # Add custom modifications
    add_new_modifications(
        [
            ("Mod1@N-term", "C(2)H(3)O(1)", "H(2)O(1)"),
            ("Mod2@S", "C(3)H(5)O(2)", ""),
        ]
    )

    # Get the custom mods
    custom_mods = get_custom_mods()

    # Check that our test mods are in the result
    assert "Mod1@N-term" in custom_mods
    assert "Mod2@S" in custom_mods

    # Check the properties
    assert custom_mods["Mod1@N-term"]["composition"] == "C(2)H(3)O(1)"
    assert custom_mods["Mod1@N-term"]["modloss_composition"] == "H(2)O(1)"
    assert custom_mods["Mod2@S"]["composition"] == "C(3)H(5)O(2)"
    assert custom_mods["Mod2@S"]["modloss_composition"] == ""


def test_init_custom_mods(cleanup_test_mods):
    """Test initializing custom modifications from a dictionary."""
    # Create a custom mods dictionary
    custom_mods_dict = {
        "InitMod1@N-term": {
            "composition": "C(2)H(3)O(1)",
            "modloss_composition": "H(2)O(1)",
            "smiles": "",
        },
        "InitMod2@S": {
            "composition": "C(3)H(5)O(2)",
            "modloss_composition": "",
            "smiles": "",
        },
    }

    # Initialize the custom mods
    init_custom_mods(custom_mods_dict)

    # Check they were added
    assert "InitMod1@N-term" in MOD_DF.index
    assert "InitMod2@S" in MOD_DF.index

    # Check the properties
    assert MOD_DF.loc["InitMod1@N-term", "composition"] == "C(2)H(3)O(1)"
    assert MOD_DF.loc["InitMod1@N-term", "modloss_composition"] == "H(2)O(1)"
    assert MOD_DF.loc["InitMod2@S", "composition"] == "C(3)H(5)O(2)"
    assert MOD_DF.loc["InitMod2@S", "modloss_composition"] == ""

    # Check that MOD_Composition was updated
    assert "InitMod1@N-term" in MOD_Composition
    assert "InitMod2@S" in MOD_Composition


def test_init_empty_custom_mods(cleanup_test_mods):
    """Test initializing with an empty custom mods dictionary."""
    # Initialize with an empty dict
    init_custom_mods({})

    # Should not raise any errors
    assert True  # Just checking that no exception was raised


def test_serialization_roundtrip(cleanup_test_mods):
    """Test that custom mods can be serialized and deserialized correctly."""
    # Add custom modifications
    add_new_modifications(
        [
            ("RoundTrip1@N-term", "C(2)H(3)O(1)", "H(2)O(1)"),
            ("RoundTrip2@S", "C(3)H(5)O(2)", ""),
        ]
    )

    # Get the custom mods (serialization)
    custom_mods = get_custom_mods()

    # Check that our test mods are in the result
    assert "RoundTrip1@N-term" in custom_mods
    assert "RoundTrip2@S" in custom_mods

    # Remove the test mods
    for mod in ["RoundTrip1@N-term", "RoundTrip2@S"]:
        if mod in MOD_DF.index:
            MOD_DF.drop(mod, inplace=True)
        if mod in MOD_Composition:
            del MOD_Composition[mod]

    # Update dictionaries
    from alphabase.constants.modification import update_all_by_MOD_DF

    update_all_by_MOD_DF()

    # Verify the custom mods are gone
    assert "RoundTrip1@N-term" not in MOD_DF.index
    assert "RoundTrip2@S" not in MOD_DF.index

    # Initialize with the custom mods (deserialization)
    init_custom_mods(custom_mods)

    # Verify the custom mods are back
    assert "RoundTrip1@N-term" in MOD_DF.index
    assert "RoundTrip2@S" in MOD_DF.index

    # Check that MOD_Composition was updated
    assert "RoundTrip1@N-term" in MOD_Composition
    assert "RoundTrip2@S" in MOD_Composition


def test_modification_context():
    """Test the ModificationContext class."""
    # Create a new independent context
    context = get_new_context()

    # Load modifications from the default file
    context.load_mod_df()

    # The context should have some modifications loaded
    assert len(context.mod_df) > 0
    assert len(context.mod_mass) > 0
    assert len(context.mod_composition) > 0

    # Add a custom modification to the context
    context.add_new_modifications([("ContextMod@N-term", "C(2)H(3)O(1)", "H(2)O(1)")])

    # Verify the modification was added to this context
    assert "ContextMod@N-term" in context.mod_df.index
    assert "ContextMod@N-term" in context.mod_composition

    # But not to the global context
    assert "ContextMod@N-term" not in MOD_DF.index
    assert "ContextMod@N-term" not in MOD_Composition

    # Test calculation functions
    mass = context.calc_modification_mass_sum(["ContextMod@N-term"])
    assert mass > 0

    # Test custom mods serialization
    custom_mods = context.get_custom_mods()
    assert "ContextMod@N-term" in custom_mods

    # Create another context and initialize with custom mods
    another_context = get_new_context()
    another_context.load_mod_df()
    another_context.init_custom_mods(custom_mods)

    # Verify the custom mod was transferred
    assert "ContextMod@N-term" in another_context.mod_df.index
    assert "ContextMod@N-term" in another_context.mod_composition
