"""Unit tests for ModificationMapper."""

from alphabase.psm_reader.modification_mapper import ModificationMapper

EMPTY_YAML = {"modification_mappings": {}}
YAML_WITH_MAPPINGS = {
    "modification_mappings": {
        "maxquant": {
            "Oxidation@M": ["M(Oxidation)"],
            "Phospho@S": ["S(Phospho)"],
        }
    }
}


class TestModificationMapper:
    """Tests for ModificationMapper."""

    def test_empty_mapping_type(self):
        """Test initialization with no mapping type."""
        mapper = ModificationMapper(
            custom_modification_mapping=None,
            reader_yaml=EMPTY_YAML,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        assert mapper.modification_mapping == {}
        assert mapper.rev_mod_mapping == {}

    def test_with_mapping_type(self):
        """Test initialization with mapping type from yaml."""
        mapper = ModificationMapper(
            custom_modification_mapping=None,
            reader_yaml=YAML_WITH_MAPPINGS,
            mapping_type="maxquant",
            add_unimod_to_mod_mapping=False,
        )
        assert "Oxidation@M" in mapper.modification_mapping
        assert "Phospho@S" in mapper.modification_mapping

    def test_with_custom_mapping(self):
        """Test initialization with custom modification mapping."""
        mapper = ModificationMapper(
            custom_modification_mapping={"Acetyl@K": "K(Acetyl)"},
            reader_yaml=EMPTY_YAML,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        assert "Acetyl@K" in mapper.modification_mapping
        assert mapper.modification_mapping["Acetyl@K"] == ["K(Acetyl)"]

    def test_rev_mapping_created(self):
        """Test that reverse mapping is created correctly."""
        mapper = ModificationMapper(
            custom_modification_mapping={"Phospho@S": "S(Phospho)"},
            reader_yaml=EMPTY_YAML,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        assert mapper.rev_mod_mapping["S(Phospho)"] == "Phospho@S"

    def test_rev_mapping_with_list(self):
        """Test reverse mapping with multiple search engine formats."""
        mapper = ModificationMapper(
            custom_modification_mapping={
                "Phospho@S": ["S(Phospho)", "pS", "S(79.9663)"]
            },
            reader_yaml=EMPTY_YAML,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        assert mapper.rev_mod_mapping["S(Phospho)"] == "Phospho@S"
        assert mapper.rev_mod_mapping["pS"] == "Phospho@S"
        assert mapper.rev_mod_mapping["S(79.9663)"] == "Phospho@S"

    def test_add_single_mapping(self):
        """Test adding a single modification mapping."""
        mapper = ModificationMapper(
            custom_modification_mapping=None,
            reader_yaml=EMPTY_YAML,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        mapper.add_modification_mapping({"Oxidation@M": "M(ox)"})
        assert "Oxidation@M" in mapper.modification_mapping
        assert "M(ox)" in mapper.rev_mod_mapping

    def test_add_none_mapping(self):
        """Test adding None does not raise."""
        mapper = ModificationMapper(
            custom_modification_mapping=None,
            reader_yaml=EMPTY_YAML,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        mapper.add_modification_mapping(None)
        assert mapper.modification_mapping == {}

    def test_add_extends_existing(self):
        """Test adding mapping extends existing mappings."""
        mapper = ModificationMapper(
            custom_modification_mapping=None,
            reader_yaml=YAML_WITH_MAPPINGS,
            mapping_type="maxquant",
            add_unimod_to_mod_mapping=False,
        )
        mapper.add_modification_mapping({"Acetyl@K": "K(ac)"})
        assert "Oxidation@M" in mapper.modification_mapping
        assert "Acetyl@K" in mapper.modification_mapping

    def test_unimod_disabled(self):
        """Test that UniMod extensions are not added when disabled."""
        mapper = ModificationMapper(
            custom_modification_mapping={"Phospho@S": "S(Phospho)"},
            reader_yaml=EMPTY_YAML,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        assert mapper.modification_mapping["Phospho@S"] == ["S(Phospho)"]

    def test_unimod_enabled_extends_mapping(self):
        """Test that UniMod extensions are added when enabled."""
        mapper = ModificationMapper(
            custom_modification_mapping={"Phospho@S": "S(Phospho)"},
            reader_yaml=EMPTY_YAML,
            mapping_type=None,
            add_unimod_to_mod_mapping=True,
        )
        assert len(mapper.modification_mapping["Phospho@S"]) > 1

    def test_set_mapping_with_string_type(self):
        """Test set_modification_mapping with string argument looks up from yaml."""
        mapper = ModificationMapper(
            custom_modification_mapping=None,
            reader_yaml=YAML_WITH_MAPPINGS,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        assert mapper.modification_mapping == {}

        mapper.set_modification_mapping("maxquant")

        assert "Oxidation@M" in mapper.modification_mapping
        assert "Phospho@S" in mapper.modification_mapping

    def test_rev_mapping_protein_n_term_conflict(self):
        """Test that Any_N-term takes precedence over Protein_N-term in reverse mapping."""
        mapper = ModificationMapper(
            custom_modification_mapping={
                "Acetyl@Any_N-term": "_(Acetyl)",
                "Acetyl@Protein_N-term": "_(Acetyl)",
            },
            reader_yaml=EMPTY_YAML,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        assert mapper.rev_mod_mapping["_(Acetyl)"] == "Acetyl@Any_N-term"

    def test_add_mapping_overwrites_existing_keys(self):
        """Test add_modification_mapping extends dict but overwrites existing key values."""
        mapper = ModificationMapper(
            custom_modification_mapping={"Phospho@S": "pS", "Oxidation@M": "M(ox)"},
            reader_yaml=EMPTY_YAML,
            mapping_type=None,
            add_unimod_to_mod_mapping=False,
        )
        mapper.add_modification_mapping(
            {"Phospho@S": "S(Phospho)", "Acetyl@K": "K(ac)"}
        )

        # new key is added
        assert "Acetyl@K" in mapper.modification_mapping
        # existing key is preserved
        assert "Oxidation@M" in mapper.modification_mapping
        # existing key value is overwritten, not extended
        assert mapper.modification_mapping["Phospho@S"] == ["S(Phospho)"]
