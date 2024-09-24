import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from alphabase.constants.atom import CHEM_MONO_MASS, MASS_H2O
from alphabase.peptide.precursor import get_mod_seq_formula
from alphabase.smiles.peptide import PeptideSmilesEncoder


@pytest.fixture
def encoder():
    return PeptideSmilesEncoder()


def test_peptide_to_smiles_no_modifications(encoder):
    sequence = "QMNPHIR"
    smiles = encoder.peptide_to_smiles(sequence)
    mol = Chem.MolFromSmiles(smiles)

    assert mol is not None

    expected_mass = np.sum(
        [CHEM_MONO_MASS[elem] * n for elem, n in get_mod_seq_formula(sequence, "")]
    )
    actual_mass = Descriptors.ExactMolWt(mol) - MASS_H2O
    assert np.isclose(actual_mass, expected_mass, atol=1e-4)


def test_peptide_to_smiles_with_modifications(encoder):
    sequence = "QMNPHIR"
    mods = "Gln->pyro-Glu@Q^Any_N-term;Oxidation@M"
    mod_sites = "1;2"

    smiles = encoder.peptide_to_smiles(sequence, mods, mod_sites)
    mol = Chem.MolFromSmiles(smiles)

    assert mol is not None

    expected_mass = np.sum(
        [CHEM_MONO_MASS[elem] * n for elem, n in get_mod_seq_formula(sequence, mods)]
    )
    actual_mass = Descriptors.ExactMolWt(mol) - MASS_H2O
    assert np.isclose(actual_mass, expected_mass, atol=1e-4)


def test_peptide_to_smiles_invalid_amino_acid(encoder):
    with pytest.raises(ValueError, match="Unknown amino acid code"):
        encoder.peptide_to_smiles("QMNPHIRX")


def test_peptide_to_smiles_invalid_modification(encoder):
    with pytest.raises(
        ValueError, match="Unknown amino acid code: Q or modification: Invalid_Mod"
    ):
        encoder.peptide_to_smiles("QMNPHIR", "Invalid_Mod@Q", "1")


def test_peptide_to_smiles_n_terminal_modification(encoder):
    sequence = "QMNPHIR"
    mods = "Acetyl@Any_N-term"
    mod_sites = "0"

    smiles = encoder.peptide_to_smiles(sequence, mods, mod_sites)
    mol = Chem.MolFromSmiles(smiles)

    assert mol is not None

    expected_mass = np.sum(
        [CHEM_MONO_MASS[elem] * n for elem, n in get_mod_seq_formula(sequence, mods)]
    )
    actual_mass = Descriptors.ExactMolWt(mol) - MASS_H2O
    assert np.isclose(actual_mass, expected_mass, atol=1e-4)


def test_peptide_to_smiles_c_terminal_modification(encoder):
    sequence = "QMNPHIR"
    mods = "Amidated@Any_C-term"
    mod_sites = "-1"

    smiles = encoder.peptide_to_smiles(sequence, mods, mod_sites)
    mol = Chem.MolFromSmiles(smiles)

    assert mol is not None

    expected_mass = np.sum(
        [CHEM_MONO_MASS[elem] * n for elem, n in get_mod_seq_formula(sequence, mods)]
    )
    actual_mass = Descriptors.ExactMolWt(mol) - MASS_H2O
    assert np.isclose(actual_mass, expected_mass, atol=1e-4)


def test_peptide_to_smiles_multiple_modifications(encoder):
    sequence = "QMNPSIR"
    mods = "Acetyl@Any_N-term;Oxidation@M;Phospho@S"
    mod_sites = "0;2;5"

    smiles = encoder.peptide_to_smiles(sequence, mods, mod_sites)
    mol = Chem.MolFromSmiles(smiles)

    assert mol is not None

    expected_mass = np.sum(
        [CHEM_MONO_MASS[elem] * n for elem, n in get_mod_seq_formula(sequence, mods)]
    )
    actual_mass = Descriptors.ExactMolWt(mol) - MASS_H2O
    assert np.isclose(actual_mass, expected_mass, atol=1e-4)


def test_peptide_to_mol_no_modifications(encoder):
    sequence = "QMNPHIR"
    mol = encoder.peptide_to_mol(sequence)

    assert mol is not None

    expected_mass = np.sum(
        [CHEM_MONO_MASS[elem] * n for elem, n in get_mod_seq_formula(sequence, "")]
    )
    actual_mass = Descriptors.ExactMolWt(mol) - MASS_H2O
    assert np.isclose(actual_mass, expected_mass, atol=1e-4)


def test_peptide_to_mol_multiple_modifications(encoder):
    sequence = "QMNPSIR"
    mods = "Acetyl@Any_N-term;Oxidation@M;Phospho@S"
    mod_sites = "0;2;5"

    mol = encoder.peptide_to_mol(sequence, mods, mod_sites)

    assert mol is not None

    expected_mass = np.sum(
        [CHEM_MONO_MASS[elem] * n for elem, n in get_mod_seq_formula(sequence, mods)]
    )
    actual_mass = Descriptors.ExactMolWt(mol) - MASS_H2O
    assert np.isclose(actual_mass, expected_mass, atol=1e-4)
