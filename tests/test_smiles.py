import pytest
from rdkit import Chem
from alphabase.smiles.smiles import (
    modify_amino_acid,
    aa_smiles,
    n_term_modifications,
    c_term_modifications,
    ptm_dict,
)


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
    result = modify_amino_acid(alanine_smiles, n_term_mod="Acetyl")
    expected = "CC(=O)N[C@@H](C)C(=O)O"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(expected)
    )


def test_modify_amino_acid_c_term_modification(alanine_smiles):
    result = modify_amino_acid(alanine_smiles, c_term_mod="Methyl")
    expected = "N[C@@H](C)C(=O)OC"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(result)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(expected)
    )


def test_modify_amino_acid_both_modifications(alanine_smiles):
    result = modify_amino_acid(alanine_smiles, n_term_mod="Acetyl", c_term_mod="Methyl")
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
    result = modify_amino_acid(lysine_with_ptm, n_term_mod="mTRAQ")
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
    result = modify_amino_acid(alanine_smiles, n_term_mod="Dimethyl")
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
