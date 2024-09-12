from collections import defaultdict

import pytest
from rdkit import Chem

from alphabase.constants.atom import ChemicalCompositonFormula
from alphabase.smiles.smiles import AminoAcidModifier

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


def test_dimethyl_n_term_modification(alanine_smiles):
    result = modify_amino_acid(alanine_smiles, n_term_mod="Dimethyl@Any_N-term")
    expected = "CN(C)[C@@H](C)C(=O)O"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(expected)
    )


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
