from collections import defaultdict

import pytest

pytest.importorskip("rdkit", reason="rdkit not installed, skipping smiles tests")

from rdkit import Chem  # noqa: E402

from alphabase.constants.atom import ChemicalCompositonFormula  # noqa: E402
from alphabase.smiles.smiles import AminoAcidModifier  # noqa: E402

aa_modifier = AminoAcidModifier()
modify_amino_acid = aa_modifier.modify_amino_acid
aa_smiles = aa_modifier.aa_smiles
n_term_modifications = aa_modifier.n_term_modifications
c_term_modifications = aa_modifier.c_term_modifications
ptm_dict = aa_modifier.ptm_dict


@pytest.fixture
def alanine_smiles():
    return aa_smiles["A"]


@pytest.fixture
def lysine_smiles():
    return aa_smiles["K"]


def test_modify_amino_acid_no_modification(alanine_smiles):
    result = modify_amino_acid(alanine_smiles)
    expected = "N[C@@H](C)C(=O)O"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(expected)
    )


def test_modify_amino_acid_n_term_modification(alanine_smiles):
    result = modify_amino_acid(alanine_smiles, n_term_mod="Acetyl@Any_N-term")
    expected = "CC(=O)N[C@@H](C)C(=O)O"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(expected)
    )


def test_modify_amino_acid_c_term_modification(alanine_smiles):
    result = modify_amino_acid(alanine_smiles, c_term_mod="Methyl@Any_C-term")
    expected = "N[C@@H](C)C(=O)OC"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(expected)
    )


def test_modify_amino_acid_both_modifications(alanine_smiles):
    result = modify_amino_acid(
        alanine_smiles, n_term_mod="Acetyl@Any_N-term", c_term_mod="Methyl@Any_C-term"
    )
    expected = "CC(=O)N[C@@H](C)C(=O)OC"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(expected)
    )


def test_modify_amino_acid_with_ptm(lysine_smiles):
    lysine_with_ptm = ptm_dict["Acetyl@K"]
    result = modify_amino_acid(lysine_with_ptm)
    expected = "CC(=O)NCCCC[C@H](N)C(=O)O"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(expected)
    )


def test_modify_amino_acid_with_ptm_and_n_term_mod(lysine_smiles):
    lysine_with_ptm = ptm_dict["Acetyl@K"]
    result = modify_amino_acid(lysine_with_ptm, n_term_mod="mTRAQ@Any_N-term")
    expected = "CC(=O)NCCCC[C@H](NC(=O)CN1CCN(C)CC1)C(=O)O"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(expected)
    )


@pytest.mark.parametrize("n_term_mod", n_term_modifications.keys())
def test_all_n_term_modifications(alanine_smiles, n_term_mod):
    result = modify_amino_acid(alanine_smiles, n_term_mod=n_term_mod)
    assert Chem.MolFromSmiles(result) is not None


@pytest.mark.parametrize("c_term_mod", c_term_modifications.keys())
def test_all_c_term_modifications(alanine_smiles, c_term_mod):
    result = modify_amino_acid(alanine_smiles, c_term_mod=c_term_mod)
    assert Chem.MolFromSmiles(result) is not None


def test_invalid_n_term_modification(alanine_smiles):
    with pytest.raises(ValueError, match="Unrecognized N-terminal modification"):
        modify_amino_acid(alanine_smiles, n_term_mod="InvalidMod")


def test_invalid_c_term_modification(alanine_smiles):
    with pytest.raises(ValueError, match="Unrecognized C-terminal modification"):
        modify_amino_acid(alanine_smiles, c_term_mod="InvalidMod")


def test_invalid_amino_acid_smiles():
    with pytest.raises(ValueError):
        modify_amino_acid("InvalidSMILES")


class TestNTermModWithPlaceholder:
    """Tests for N-terminal modification using [Lv] placeholder representing the amine."""

    def test_single_neighbor_placeholder_acetyl(self, alanine_smiles):
        """Acetyl has one neighbor to [Lv], creating one bond to N."""
        # Given: alanine with [Fl] placeholders and Acetyl modification with [Lv]
        acetyl_smiles = n_term_modifications["Acetyl@Any_N-term"]
        assert aa_modifier._has_n_term_mod_placeholder(acetyl_smiles)

        # When: applying Acetyl modification
        result = modify_amino_acid(alanine_smiles, n_term_mod="Acetyl@Any_N-term")

        # Then: carbonyl C bonds to N, remaining [Fl] becomes H
        expected = "CC(=O)N[C@@H](C)C(=O)O"
        assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
            Chem.MolFromSmiles(expected)
        )

    def test_multi_neighbor_placeholder_dimethyl(self, alanine_smiles):
        """Dimethyl has two neighbors to [Lv], creating two bonds to N."""
        # Given: alanine with [Fl] placeholders and Dimethyl modification with [Lv]
        dimethyl_smiles = n_term_modifications["Dimethyl@Any_N-term"]
        assert aa_modifier._has_n_term_mod_placeholder(dimethyl_smiles)
        assert dimethyl_smiles == "C[Lv]C"

        # When: applying Dimethyl modification
        result = modify_amino_acid(alanine_smiles, n_term_mod="Dimethyl@Any_N-term")

        # Then: both methyl carbons bond to N
        expected = "CN(C)[C@@H](C)C(=O)O"
        assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
            Chem.MolFromSmiles(expected)
        )

    def test_placeholder_detection(self):
        """_has_n_term_mod_placeholder correctly identifies placeholder presence."""
        # Given: SMILES with and without placeholder
        with_placeholder = "CC(=O)[Lv]"
        without_placeholder = "C(=O)C"

        # When/Then: detection returns correct result
        assert aa_modifier._has_n_term_mod_placeholder(with_placeholder) is True
        assert aa_modifier._has_n_term_mod_placeholder(without_placeholder) is False

    def test_error_when_mod_missing_placeholder(self):
        """Raises error when modification mol lacks [Lv] but placeholder method is called."""
        # Given: amino acid mol and modification mol WITHOUT [Lv] placeholder
        aa_mol = Chem.MolFromSmiles(aa_smiles["A"])
        mod_without_placeholder = Chem.MolFromSmiles("C(=O)C")

        # When/Then: calling placeholder method raises ValueError
        with pytest.raises(ValueError, match="does not contain placeholder"):
            aa_modifier._apply_n_term_mod_with_placeholder(
                aa_mol, mod_without_placeholder
            )

    def test_error_when_mod_has_multiple_placeholders(self):
        """Raises error when modification mol has multiple [Lv] placeholders."""
        # Given: amino acid mol and modification mol with multiple [Lv] placeholders
        aa_mol = Chem.MolFromSmiles(aa_smiles["A"])
        mod_with_multiple_placeholders = Chem.MolFromSmiles("[Lv]CC(=O)[Lv]")

        # When/Then: calling placeholder method raises ValueError
        with pytest.raises(ValueError, match="expected exactly 1"):
            aa_modifier._apply_n_term_mod_with_placeholder(
                aa_mol, mod_with_multiple_placeholders
            )

    def test_error_when_aa_missing_fl_placeholder(self):
        """Raises error when amino acid lacks [Fl] placeholder."""
        # Given: molecule WITHOUT [Fl] placeholder and modification WITH [Lv]
        mol_without_fl = Chem.MolFromSmiles("NCC(=O)O")  # Glycine without placeholders
        mod_with_placeholder = Chem.MolFromSmiles("CC(=O)[Lv]")

        # When/Then: calling placeholder method raises ValueError
        with pytest.raises(ValueError, match="does not contain N-terminal placeholder"):
            aa_modifier._apply_n_term_mod_with_placeholder(
                mol_without_fl, mod_with_placeholder
            )

    def test_error_when_fl_has_wrong_neighbor_count(self):
        """Raises error when [Fl] has more than one neighbor."""
        # Given: programmatically create molecule where [Fl] has two neighbors
        # (can't do this via SMILES as RDKit enforces valence)
        from rdkit.Chem import Atom, RWMol

        rw_mol = RWMol()
        c_idx = rw_mol.AddAtom(Atom(6))  # Carbon
        fl_idx = rw_mol.AddAtom(Atom(114))  # Flerovium [Fl]
        n_idx = rw_mol.AddAtom(Atom(7))  # Nitrogen
        rw_mol.AddBond(c_idx, fl_idx, Chem.rdchem.BondType.SINGLE)
        rw_mol.AddBond(
            n_idx, fl_idx, Chem.rdchem.BondType.SINGLE
        )  # [Fl] now has 2 neighbors
        malformed_mol = rw_mol.GetMol()

        mod_mol = Chem.MolFromSmiles("CC(=O)[Lv]")

        # When/Then: raises error about wrong neighbor count
        with pytest.raises(ValueError, match="should have exactly one neighbor"):
            aa_modifier._apply_n_term_mod_with_placeholder(malformed_mol, mod_mol)


@pytest.mark.parametrize("aa, smiles", aa_smiles.items())
def test_all_amino_acids(aa, smiles):
    result = modify_amino_acid(smiles)
    assert Chem.MolFromSmiles(result) is not None


@pytest.mark.parametrize("ptm, smiles", ptm_dict.items())
def test_all_ptms(ptm, smiles):
    result = modify_amino_acid(smiles)
    assert Chem.MolFromSmiles(result) is not None


@pytest.fixture
def water_formula():
    return ChemicalCompositonFormula("H(2)O(1)")


@pytest.fixture
def methane_formula():
    return ChemicalCompositonFormula("C(1)H(4)")


def test_init():
    formula = ChemicalCompositonFormula("C(6)H(12)O(6)")
    expected = defaultdict(int, {"C": 6, "H": 12, "O": 6})
    assert formula.elements == expected


def test_init_with_isotopes():
    formula = ChemicalCompositonFormula("13C(1)C(5)H(12)O(6)")
    expected = defaultdict(int, {"13C": 1, "C": 5, "H": 12, "O": 6})
    assert formula.elements == expected


def test_from_smiles():
    formula = ChemicalCompositonFormula.from_smiles("CCO")
    expected = defaultdict(int, {"C": 2, "H": 6, "O": 1})
    assert formula.elements == expected


def test_str_representation(water_formula):
    assert str(water_formula) == "H(2)O(1)"


def test_repr_representation(water_formula):
    assert repr(water_formula) == "ChemicalCompositonFormula('H(2)O(1)')"


def test_addition(water_formula, methane_formula):
    result = water_formula + methane_formula
    expected = ChemicalCompositonFormula("C(1)H(6)O(1)")
    assert result.elements == expected.elements


def test_subtraction(water_formula, methane_formula):
    result = methane_formula - water_formula
    expected = ChemicalCompositonFormula("C(1)H(2)O(-1)")
    assert result.elements == expected.elements


def test_parse_formula_with_parentheses():
    formula = ChemicalCompositonFormula("C(6)H(12)O(6)")
    expected = defaultdict(int, {"C": 6, "H": 12, "O": 6})
    assert formula.elements == expected


def test_parse_rdkit_formula():
    formula = ChemicalCompositonFormula._from_rdkit_formula("[13C]CH3OH")
    expected = defaultdict(int, {"13C": 1, "C": 1, "H": 4, "O": 1})
    assert formula.elements == expected


def test_zero_count_elements():
    formula = ChemicalCompositonFormula("C(1)H(0)O(2)")
    assert str(formula) == "C(1)O(2)"


def test_error_handling_invalid_rdkit_mol():
    with pytest.raises(ValueError):
        ChemicalCompositonFormula.from_smiles("InvalidSMILES")


def test_error_unknown_atom():
    with pytest.raises(ValueError):
        ChemicalCompositonFormula("C(6)H(12)Atom(6)")
