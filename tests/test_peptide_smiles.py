import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from alphabase.constants.atom import MASS_H2O
from alphabase.smiles.peptide import PeptideSmilesEncoder


@pytest.fixture
def encoder():
    return PeptideSmilesEncoder()


def test_peptide_to_smiles_no_modifications(encoder):
    # Given
    sequence = "QMNPHIR"
    # When
    smiles = encoder.peptide_to_smiles(sequence)
    mol = Chem.MolFromSmiles(smiles)
    # Then
    assert mol is not None
    expected_mass = 876.4388
    actual_mass = Descriptors.ExactMolWt(mol) - MASS_H2O
    assert np.isclose(
        actual_mass, expected_mass, atol=1e-4
    ), "Expected mass does not match actual mass"
    expected_smiles = "[H]N([H])[C@@H](CCC(N)=O)C(=O)N([H])[C@@H](CCSC)C(=O)N([H])[C@@H](CC(N)=O)C(=O)N1CCC[C@H]1C(=O)N([H])[C@@H](Cc1cnc[nH]1)C(=O)N([H])[C@H](C(=O)N([H])[C@@H](CCCNC(=N)N)C(=O)O)[C@@H](C)CC"
    assert smiles == expected_smiles, "Expected SMILES does not match actual SMILES"


def test_peptide_to_smiles_with_modifications(encoder):
    # Given
    sequence = "QMNPHIR"
    mods = "Gln->pyro-Glu@Q^Any_N-term;Oxidation@M"
    mod_sites = "0;2"
    # When
    smiles = encoder.peptide_to_smiles(sequence, mods, mod_sites)
    mol = Chem.MolFromSmiles(smiles)
    # Then
    assert mol is not None
    expected_mass = 875.4072
    actual_mass = Descriptors.ExactMolWt(mol) - MASS_H2O
    assert np.isclose(
        actual_mass, expected_mass, atol=1e-4
    ), "Expected mass does not match actual mass"
    expected_smiles = "[H]N(C(=O)[C@@H]1CCC(=O)N1[H])C(CCS(C)=O)C(=O)N([H])[C@@H](CC(N)=O)C(=O)N1CCC[C@H]1C(=O)N([H])[C@@H](Cc1cnc[nH]1)C(=O)N([H])[C@H](C(=O)N([H])[C@@H](CCCNC(=N)N)C(=O)O)[C@@H](C)CC"
    assert smiles == expected_smiles, "Expected SMILES does not match actual SMILES"


def test_peptide_to_smiles_invalid_amino_acid(encoder):
    with pytest.raises(ValueError, match="Unknown amino acid code"):
        encoder.peptide_to_smiles("QMNPHIRX")


def test_peptide_to_smiles_invalid_modification(encoder):
    with pytest.raises(
        ValueError, match="Unknown amino acid code: Q or modification: Invalid_Mod"
    ):
        encoder.peptide_to_smiles("QMNPHIR", "Invalid_Mod@Q", "1")


def test_peptide_to_smiles_n_terminal_modification(encoder):
    # Given a peptide sequence
    sequence = "QMNPHIR"
    mods = "Acetyl@Any_N-term"
    mod_sites = "0"
    # When converting the peptide to SMILES
    smiles = encoder.peptide_to_smiles(sequence, mods, mod_sites)
    mol = Chem.MolFromSmiles(smiles)
    # Then
    assert mol is not None
    expected_mass = 918.4494
    actual_mass = Descriptors.ExactMolWt(mol) - MASS_H2O
    assert np.isclose(
        actual_mass, expected_mass, atol=1e-4
    ), "Expected mass does not match actual mass"


def test_peptide_to_smiles_c_terminal_modification(encoder):
    # Given a peptide
    sequence = "QMNPHIR"
    mods = "Amidated@Any_C-term"
    mod_sites = "-1"
    # When converting the peptide to SMILES
    smiles = encoder.peptide_to_smiles(sequence, mods, mod_sites)
    mol = Chem.MolFromSmiles(smiles)
    # Then
    assert mol is not None
    expected_mass = 875.4548
    actual_mass = Descriptors.ExactMolWt(mol) - MASS_H2O
    assert np.isclose(
        actual_mass, expected_mass, atol=1e-4
    ), "Expected mass does not match actual mass"


def test_peptide_to_smiles_multiple_modifications(encoder):
    # Given a peptide sequence with multiple modifications
    sequence = "QMNPSIR"
    mods = "Acetyl@Any_N-term;Oxidation@M;Phospho@S"
    mod_sites = "0;2;5"
    # When converting the peptide to SMILES
    smiles = encoder.peptide_to_smiles(sequence, mods, mod_sites)
    mol = Chem.MolFromSmiles(smiles)
    # Then
    assert mol is not None
    expected_mass = 964.383
    actual_mass = Descriptors.ExactMolWt(mol) - MASS_H2O
    assert np.isclose(
        actual_mass, expected_mass, atol=1e-4
    ), "Expected mass does not match actual mass"
    expected_smiles = "[H]N(C(=O)[C@H](CCC(N)=O)N([H])C(C)=O)C(CCS(C)=O)C(=O)N([H])[C@@H](CC(N)=O)C(=O)N1CCC[C@H]1C(=O)N([H])[C@@H](COP(=O)(O)O)C(=O)N([H])[C@H](C(=O)N([H])[C@@H](CCCNC(=N)N)C(=O)O)[C@@H](C)CC"
    assert smiles == expected_smiles, "Expected SMILES does not match actual SMILES"


def test_peptide_to_mol_no_modifications(encoder):
    # Given a peptide sequence
    sequence = "QMNPHIR"
    # When converting the peptide to a molecule
    mol = encoder.peptide_to_mol(sequence)
    # Then
    assert mol is not None
    expected_mass = 876.4388
    actual_mass = Descriptors.ExactMolWt(mol) - MASS_H2O
    assert np.isclose(
        actual_mass, expected_mass, atol=1e-4
    ), "Expected mass does not match actual mass"


def test_peptide_to_mol_multiple_modifications(encoder):
    # Given a peptide sequence with multiple modifications
    sequence = "QMNPSIR"
    mods = "Acetyl@Any_N-term;Oxidation@M;Phospho@S"
    mod_sites = "0;2;5"
    # When converting the peptide to a molecule
    mol = encoder.peptide_to_mol(sequence, mods, mod_sites)
    # Then
    assert mol is not None
    expected_mass = 964.3837
    actual_mass = Descriptors.ExactMolWt(mol) - MASS_H2O
    assert np.isclose(
        actual_mass, expected_mass, atol=1e-4
    ), "Expected mass does not match actual mass"


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
    # When converting the peptide to SMILES per amino acid
    smiles = encoder.peptide_to_smiles_per_amino_acid(sequence)
    expected_smiles = [
        "[H]N([H])[C@@H](CCC(N)=O)C(=O)O",
        "[H]N([H])[C@@H](CCSC)C(=O)O",
        "[H]N([H])[C@@H](CC(N)=O)C(=O)O",
        "[H]N1CCC[C@H]1C(=O)O",
        "[H]N([H])[C@@H](Cc1cnc[nH]1)C(=O)O",
        "[H]N([H])[C@H](C(=O)O)[C@@H](C)CC",
        "[H]N([H])[C@@H](CCCNC(=N)N)C(=O)O",
    ]
    # Then
    assert smiles == expected_smiles


def test_peptide_to_smiles_per_aa_multiple_modifications(encoder):
    # Given a peptide sequence and multiple modifications
    sequence = "QMNPSIR"
    mods = "Acetyl@Any_N-term;Oxidation@M;Phospho@S"
    mod_sites = "0;2;5"
    # When converting the peptide to SMILES per amino acid
    smiles = encoder.peptide_to_smiles_per_amino_acid(sequence, mods, mod_sites)
    expected_smiles = [
        "[H]N(C(C)=O)[C@@H](CCC(N)=O)C(=O)O",
        "[H]N([H])C(CCS(C)=O)C(=O)O",
        "[H]N([H])[C@@H](CC(N)=O)C(=O)O",
        "[H]N1CCC[C@H]1C(=O)O",
        "[H]N([H])[C@@H](COP(=O)(O)O)C(=O)O",
        "[H]N([H])[C@H](C(=O)O)[C@@H](C)CC",
        "[H]N([H])[C@@H](CCCNC(=N)N)C(=O)O",
    ]
    # Then
    assert smiles == expected_smiles


def test_peptide_to_smiles_per_aa_c_terminal_modification(encoder):
    # Given a peptide sequence and a C-terminal modification
    sequence = "QMNPHIR"
    mods = "Amidated@Any_C-term"
    mod_sites = "-1"
    # When converting the peptide to SMILES per amino acid
    smiles = encoder.peptide_to_smiles_per_amino_acid(sequence, mods, mod_sites)
    expected_smiles = [
        "[H]N([H])[C@@H](CCC(N)=O)C(=O)O",
        "[H]N([H])[C@@H](CCSC)C(=O)O",
        "[H]N([H])[C@@H](CC(N)=O)C(=O)O",
        "[H]N1CCC[C@H]1C(=O)O",
        "[H]N([H])[C@@H](Cc1cnc[nH]1)C(=O)O",
        "[H]N([H])[C@H](C(=O)O)[C@@H](C)CC",
        "[H]N([H])[C@@H](CCCNC(=N)N)C(N)=O",
    ]
    # Then
    assert smiles == expected_smiles


def test_peptide_to_smiles_per_aa_N_terminal_modification(encoder):
    # Given a peptide sequence and a N-terminal modification
    sequence = "QMNPHIR"
    mods = "Acetyl@Any_N-term"
    mod_sites = "0"
    # When converting the peptide to SMILES per amino acid
    smiles = encoder.peptide_to_smiles_per_amino_acid(sequence, mods, mod_sites)
    expected_smiles = [
        "[H]N(C(C)=O)[C@@H](CCC(N)=O)C(=O)O",
        "[H]N([H])[C@@H](CCSC)C(=O)O",
        "[H]N([H])[C@@H](CC(N)=O)C(=O)O",
        "[H]N1CCC[C@H]1C(=O)O",
        "[H]N([H])[C@@H](Cc1cnc[nH]1)C(=O)O",
        "[H]N([H])[C@H](C(=O)O)[C@@H](C)CC",
        "[H]N([H])[C@@H](CCCNC(=N)N)C(=O)O",
    ]
    # Then
    assert smiles == expected_smiles
