"""Unit tests for ModificationMapper."""

import pytest

from alphabase.psm_reader.modification_mapper import ModificationMapper


@pytest.fixture
def empty_yaml():
    """Fixture for empty reader yaml."""
    return {"modification_mappings": {}}


@pytest.fixture
def yaml_with_mappings():
    """Fixture for reader yaml with modification mappings."""
    return {
        "modification_mappings": {
            "maxquant": {
                "Oxidation@M": ["M(Oxidation)"],
                "Phospho@S": ["S(Phospho)"],
            }
        }
    }


class TestInitialization:
    """Tests for ModificationMapper initialization."""

    def test_empty_mapping_type(self, empty_yaml):
        """Test initialization with no mapping type."""
        mapper = ModificationMapper(
            custom_modification_mapping=None,
            reader_yaml=empty_yaml,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        assert mapper.modification_mapping == {}
        assert mapper.rev_mod_mapping == {}

    def test_with_mapping_type(self, yaml_with_mappings):
        """Test initialization with mapping type from yaml."""
        mapper = ModificationMapper(
            custom_modification_mapping=None,
            reader_yaml=yaml_with_mappings,
            mapping_type="maxquant",
            add_unimod_to_mod_mapping=False,
        )
        assert "Oxidation@M" in mapper.modification_mapping
        assert "Phospho@S" in mapper.modification_mapping

    def test_with_custom_mapping(self, empty_yaml):
        """Test initialization with custom modification mapping."""
        custom = {"Acetyl@K": "K(Acetyl)"}
        mapper = ModificationMapper(
            custom_modification_mapping=custom,
            reader_yaml=empty_yaml,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        assert "Acetyl@K" in mapper.modification_mapping
        assert mapper.modification_mapping["Acetyl@K"] == ["K(Acetyl)"]


class TestReverseMapping:
    """Tests for reverse modification mapping."""

    def test_rev_mapping_created(self, empty_yaml):
        """Test that reverse mapping is created correctly."""
        custom = {"Phospho@S": "S(Phospho)"}
        mapper = ModificationMapper(
            custom_modification_mapping=custom,
            reader_yaml=empty_yaml,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        assert mapper.rev_mod_mapping["S(Phospho)"] == "Phospho@S"

    def test_rev_mapping_with_list(self, empty_yaml):
        """Test reverse mapping with multiple search engine formats."""
        custom = {"Phospho@S": ["S(Phospho)", "pS", "S(79.9663)"]}
        mapper = ModificationMapper(
            custom_modification_mapping=custom,
            reader_yaml=empty_yaml,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        assert mapper.rev_mod_mapping["S(Phospho)"] == "Phospho@S"
        assert mapper.rev_mod_mapping["pS"] == "Phospho@S"
        assert mapper.rev_mod_mapping["S(79.9663)"] == "Phospho@S"


class TestAddModificationMapping:
    """Tests for add_modification_mapping method."""

    def test_add_single_mapping(self, empty_yaml):
        """Test adding a single modification mapping."""
        mapper = ModificationMapper(
            custom_modification_mapping=None,
            reader_yaml=empty_yaml,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        mapper.add_modification_mapping({"Oxidation@M": "M(ox)"})
        assert "Oxidation@M" in mapper.modification_mapping
        assert "M(ox)" in mapper.rev_mod_mapping

    def test_add_none_mapping(self, empty_yaml):
        """Test adding None does not raise."""
        mapper = ModificationMapper(
            custom_modification_mapping=None,
            reader_yaml=empty_yaml,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        mapper.add_modification_mapping(None)
        assert mapper.modification_mapping == {}

    def test_add_extends_existing(self, yaml_with_mappings):
        """Test adding mapping extends existing mappings."""
        mapper = ModificationMapper(
            custom_modification_mapping=None,
            reader_yaml=yaml_with_mappings,
            mapping_type="maxquant",
            add_unimod_to_mod_mapping=False,
        )
        mapper.add_modification_mapping({"Acetyl@K": "K(ac)"})
        assert "Oxidation@M" in mapper.modification_mapping
        assert "Acetyl@K" in mapper.modification_mapping


class TestUnimodExtension:
    """Tests for UniMod extension functionality."""

    def test_unimod_disabled(self, empty_yaml):
        """Test that UniMod extensions are not added when disabled."""
        custom = {"Phospho@S": "S(Phospho)"}
        mapper = ModificationMapper(
            custom_modification_mapping=custom,
            reader_yaml=empty_yaml,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        # Only the original mapping should exist
        assert mapper.modification_mapping["Phospho@S"] == ["S(Phospho)"]

    def test_unimod_enabled_extends_mapping(self, empty_yaml):
        """Test that UniMod extensions are added when enabled."""
        custom = {"Phospho@S": "S(Phospho)"}
        mapper = ModificationMapper(
            custom_modification_mapping=custom,
            reader_yaml=empty_yaml,
            mapping_type=None,
            add_unimod_to_mod_mapping=True,
        )
        # Should have more than just the original mapping
        assert len(mapper.modification_mapping["Phospho@S"]) > 1
