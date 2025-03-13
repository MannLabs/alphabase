"""
Tests for the peptide mass calculation functionality.
Based on the tests in nbs_tests/peptide/mass_calc.ipynb
"""

import numpy as np

from alphabase.peptide.mass_calc import (
    calc_b_y_and_peptide_mass,
    calc_b_y_and_peptide_masses_for_same_len_seqs,
    calc_peptide_masses_for_same_len_seqs,
)


def test_calc_b_y_and_peptide_mass_with_phospho():
    """Test calculation of b/y ions and peptide mass with phosphorylation."""
    seq = "PEPTIDE"
    mods = ["Phospho@T"]
    sites = [4]  # 'T' is at position 4 (0-indexed)

    b, y, pepmass = calc_b_y_and_peptide_mass(seq, mods, sites)

    # The actual b ion masses for peptide "PEPTIDE" with phosphorylation at T
    # P - E - P - T(Phospho) - I - D - E
    expected_b = np.array(
        [
            97.05276385,  # P
            226.09535694,  # PE
            323.14812079,  # PEP
            504.16213015,  # PEPT(Phospho)
            617.24619413,  # PEPTI
            732.27313715,  # PEPTID
        ]
    )
    assert np.allclose(b, expected_b)

    # The actual y ion masses
    expected_y = np.array(
        [
            782.27353107,  # EPTIDE
            653.23093798,  # PTIDE
            556.17817413,  # TIDE
            375.16416478,  # IDE
            262.0801008,  # DE
            147.05315777,  # E
        ]
    )
    assert np.allclose(y, expected_y)

    # Check peptide mass
    assert np.allclose(pepmass, 879.32629492211, atol=1e-4)


def test_calc_b_y_and_peptide_mass_without_mods():
    """Test calculation of b/y ions and peptide mass without modifications."""
    seq = "PEPTIDE"
    mods = []
    sites = []

    b, y, pepmass = calc_b_y_and_peptide_mass(seq, mods, sites)

    # The actual b ion masses for peptide "PEPTIDE" without modifications
    # P - E - P - T - I - D - E
    expected_b = np.array(
        [
            97.05276385,  # P
            226.09535694,  # PE
            323.14812079,  # PEP
            424.19579926,  # PEPT
            537.27986324,  # PEPTI
            652.30680626,  # PEPTID
        ]
    )
    assert np.allclose(b, expected_b)

    # The actual y ion masses
    expected_y = np.array(
        [
            702.30720018,  # EPTIDE
            573.26460709,  # PTIDE
            476.21184324,  # TIDE
            375.16416478,  # IDE
            262.0801008,  # DE
            147.05315777,  # E
        ]
    )
    assert np.allclose(y, expected_y)

    # Check peptide mass
    assert np.allclose(pepmass, 799.35996403275, atol=1e-4)


def test_calc_b_y_and_peptide_masses_for_same_len_seqs():
    """Test calculation of b/y ions for multiple peptides with the same length."""
    # Peptide without modifications
    sequences = ["PEPTIDE"]
    mod_lists = [[]]
    site_lists = [[]]

    b_frags, y_frags, pepmasses = calc_b_y_and_peptide_masses_for_same_len_seqs(
        sequences, mod_lists, site_lists
    )

    # The b ion masses should be a 2D array with shape (1, 6)
    assert b_frags.shape == (1, 6)

    # The actual b ion masses for peptide "PEPTIDE" without modifications
    expected_b = np.array(
        [
            97.05276385,  # P
            226.09535694,  # PE
            323.14812079,  # PEP
            424.19579926,  # PEPT
            537.27986324,  # PEPTI
            652.30680626,  # PEPTID
        ]
    )
    assert np.allclose(b_frags[0], expected_b)

    # The y ion masses should also be a 2D array with shape (1, 6)
    assert y_frags.shape == (1, 6)

    # The actual y ion masses
    expected_y = np.array(
        [
            702.30720018,  # EPTIDE
            573.26460709,  # PTIDE
            476.21184324,  # TIDE
            375.16416478,  # IDE
            262.0801008,  # DE
            147.05315777,  # E
        ]
    )
    assert np.allclose(y_frags[0], expected_y)

    # Check peptide mass
    assert np.allclose(pepmasses[0], 799.35996403275, atol=1e-4)


def test_calc_b_y_and_peptide_masses_for_same_len_seqs_with_phospho():
    """Test calculation of b/y ions for multiple peptides with phosphorylation."""
    # Peptide with phosphorylation
    sequences = ["PEPTIDE"]
    mod_lists = [["Phospho@T"]]
    site_lists = [[4]]  # 'T' is at position 4 (0-indexed)

    b_frags, y_frags, pepmasses = calc_b_y_and_peptide_masses_for_same_len_seqs(
        sequences, mod_lists, site_lists
    )

    # The b ion masses should be a 2D array with shape (1, 6)
    assert b_frags.shape == (1, 6)

    # The actual b ion masses for peptide "PEPTIDE" with phosphorylation at T
    expected_b = np.array(
        [
            97.05276385,  # P
            226.09535694,  # PE
            323.14812079,  # PEP
            504.16213015,  # PEPT(Phospho)
            617.24619413,  # PEPTI
            732.27313715,  # PEPTID
        ]
    )
    assert np.allclose(b_frags[0], expected_b)

    # The y ion masses should also be a 2D array with shape (1, 6)
    assert y_frags.shape == (1, 6)

    # The actual y ion masses
    expected_y = np.array(
        [
            782.27353107,  # EPTIDE
            653.23093798,  # PTIDE
            556.17817413,  # TIDE
            375.16416478,  # IDE
            262.0801008,  # DE
            147.05315777,  # E
        ]
    )
    assert np.allclose(y_frags[0], expected_y)

    # Check peptide mass
    assert np.allclose(pepmasses[0], 879.32629492211, atol=1e-4)


def test_calc_peptide_masses_for_same_len_seqs():
    """Test calculation of peptide masses for multiple peptides with the same length."""
    seq = "PEPTIDE"
    mods = ["Phospho@T"]

    expected_mass = calc_peptide_masses_for_same_len_seqs([seq], [";".join(mods)])[0]
    mass_list = calc_peptide_masses_for_same_len_seqs([seq] * 2, [";".join(mods), ""])

    # The first element should have the phosphorylated mass, the second should be unmodified
    assert np.allclose(mass_list, [expected_mass, expected_mass - 79.966331])
