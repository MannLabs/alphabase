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


def test_peptide_to_smiles_per_aa_is_complete(encoder):
    # Given a peptide sequence
    sequence = "QMNPSIR"
    # When converting the peptide to SMILES per amino acid
    smiles = encoder.peptide_to_smiles_per_amino_acid(sequence)
    # Then the number of SMILES should match the number of amino acids
    assert isinstance(smiles, list)
    assert len(smiles) == len(sequence)


def test_peptide_to_smiles_per_aa_no_modifications(encoder):
    # Given a peptide sequence
    sequence = "QMNPHIR"

    formulas = [get_mod_seq_formula(aa, "") for aa in sequence]
    expected_masses = [
        np.sum([CHEM_MONO_MASS[elem] * n for elem, n in formula])
        for formula in formulas
    ]

    # When converting the peptide to SMILES per amino acid
    smiles = encoder.peptide_to_smiles_per_amino_acid(sequence)
    generated_masses = [
        Descriptors.ExactMolWt(Chem.MolFromSmiles(s, True)) - MASS_H2O for s in smiles
    ]

    # Then the masses should match the expected masses
    assert np.allclose(generated_masses, expected_masses, atol=1e-4)


def test_peptide_to_smiles_per_aa_multiple_modifications(encoder):
    # Given a peptide sequence and multiple modifications
    sequence = "QMNPSIR"
    mods = "Acetyl@Any_N-term;Oxidation@M;Phospho@S"
    mod_sites = "0;2;5"
    mods_per_aa = ["Acetyl@Any_N-term", "Oxidation@M", "", "", "Phospho@S", "", ""]

    formulas = [get_mod_seq_formula(aa, m) for aa, m in zip(sequence, mods_per_aa)]
    expected_masses = [
        np.sum([CHEM_MONO_MASS[elem] * n for elem, n in formula])
        for formula in formulas
    ]

    # When converting the peptide to SMILES per amino acid
    smiles = encoder.peptide_to_smiles_per_amino_acid(sequence, mods, mod_sites)
    generated_masses = [
        Descriptors.ExactMolWt(Chem.MolFromSmiles(s, True)) - MASS_H2O for s in smiles
    ]

    # Then the masses should match the expected masses
    assert np.allclose(generated_masses, expected_masses, atol=1e-4)


def test_peptide_to_smiles_per_aa_c_terminal_modification(encoder):
    # Given a peptide sequence and a C-terminal modification
    sequence = "QMNPHIR"
    mods = "Amidated@Any_C-term"
    mod_sites = "-1"
    mods_per_aa = ["", "", "", "", "", "", "Amidated@Any_C-term"]

    formulas = [get_mod_seq_formula(aa, m) for aa, m in zip(sequence, mods_per_aa)]
    expected_masses = [
        np.sum([CHEM_MONO_MASS[elem] * n for elem, n in formula])
        for formula in formulas
    ]

    # When converting the peptide to SMILES per amino acid
    smiles = encoder.peptide_to_smiles_per_amino_acid(sequence, mods, mod_sites)
    generated_masses = [
        Descriptors.ExactMolWt(Chem.MolFromSmiles(s, True)) - MASS_H2O for s in smiles
    ]

    # Then the masses should match the expected masses
    assert np.allclose(generated_masses, expected_masses, atol=1e-4)


def test_peptide_to_smiles_per_aa_N_terminal_modification(encoder):
    # Given a peptide sequence and a N-terminal modification
    sequence = "QMNPHIR"
    mods = "Acetyl@Any_N-term"
    mod_sites = "0"
    mods_per_aa = ["Acetyl@Any_N-term", "", "", "", "", "", ""]

    formulas = [get_mod_seq_formula(aa, m) for aa, m in zip(sequence, mods_per_aa)]
    expected_masses = [
        np.sum([CHEM_MONO_MASS[elem] * n for elem, n in formula])
        for formula in formulas
    ]

    # When converting the peptide to SMILES per amino acid
    smiles = encoder.peptide_to_smiles_per_amino_acid(sequence, mods, mod_sites)
    generated_masses = [
        Descriptors.ExactMolWt(Chem.MolFromSmiles(s, True)) - MASS_H2O for s in smiles
    ]

    # Then the masses should match the expected masses
    assert np.allclose(generated_masses, expected_masses, atol=1e-4)
