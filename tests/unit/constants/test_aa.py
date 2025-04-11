import numpy as np

from alphabase.constants.aa import (
    AA_ASCII_MASS,
    AA_Composition,
    aa_formula,
    calc_AA_masses,
    calc_AA_masses_for_same_len_seqs,
    calc_AA_masses_for_var_len_seqs,
    calc_sequence_masses_for_same_len_seqs,
    replace_atoms,
    update_an_AA,
)
from alphabase.constants.atom import MASS_H2O

# Store original values to restore after tests
ORIGINAL_AA_ASCII_MASS = AA_ASCII_MASS.copy()
ORIGINAL_AA_COMPOSITION = {k: v.copy() for k, v in AA_Composition.items()}
ORIGINAL_AA_FORMULA = aa_formula.copy()


def setup_module(module):
    """Set up the module before any tests run."""
    pass


def teardown_module(module):
    """Clean up after all tests in this module have run."""
    # Restore original values
    global AA_ASCII_MASS, AA_Composition
    AA_ASCII_MASS[:] = ORIGINAL_AA_ASCII_MASS[:]
    AA_Composition.clear()
    AA_Composition.update({k: v.copy() for k, v in ORIGINAL_AA_COMPOSITION.items()})
    # Update the aa_formula in the module
    import alphabase.constants.aa

    alphabase.constants.aa.aa_formula = ORIGINAL_AA_FORMULA.copy()


def test_calc_AA_masses():
    """Test that calc_AA_masses returns expected values."""
    expected = [
        71.03711379,
        103.00918496,
        115.02694302,
        129.04259309,
        147.06841391,
        57.02146372,
        114.04292744,
        163.06332853,
        128.09496302,
    ]
    actual = calc_AA_masses("ACDEFGNYK")
    assert np.allclose(actual, expected)


def test_calc_AA_masses_for_same_len_seqs():
    """Test that calc_AA_masses_for_same_len_seqs returns expected values."""
    seqs = np.array(["ACDEFGHIK", "BCDEFGHIK", "CCDEFGHIK"])
    expected = np.array(
        [
            [
                71.03711379,
                103.00918496,
                115.02694302,
                129.04259309,
                147.06841391,
                57.02146372,
                137.05891186,
                113.08406398,
                128.09496302,
            ],
            [
                12000000,
                103.00918496,
                115.02694302,
                129.04259309,
                147.06841391,
                57.02146372,
                137.05891186,
                113.08406398,
                128.09496302,
            ],
            [
                103.00918496,
                103.00918496,
                115.02694302,
                129.04259309,
                147.06841391,
                57.02146372,
                137.05891186,
                113.08406398,
                128.09496302,
            ],
        ]
    )
    actual = calc_AA_masses_for_same_len_seqs(seqs)
    assert np.allclose(actual, expected)


def test_calc_sequence_masses_for_same_len_seqs():
    """Test that calc_sequence_masses_for_same_len_seqs returns expected values."""
    seqs = np.array(["ACDEFGHIK", "BCDEFGHIK", "CCDEFGHIK"])
    expected = [1018.45421603, 12000947.41710224, 1050.4262872]
    actual = calc_sequence_masses_for_same_len_seqs(seqs)
    assert np.allclose(actual, expected)


def test_b_y_ions_calculation():
    """Test that b/y ion calculations work correctly."""
    aa_masses = calc_AA_masses_for_same_len_seqs(
        ["ACDEFGHIK", "BCDEFGHIK", "CCDEFGHIK"]
    )
    b_masses = np.cumsum(aa_masses, axis=1)
    b_masses, pepmass = b_masses[:, :-1], b_masses[:, -1:]
    pepmass += MASS_H2O

    assert pepmass.shape == (3, 1)
    assert b_masses.shape == (3, 8)

    # Check the first sequence's b-ion masses
    expected_b = np.array(
        [
            71.03711138,
            174.046299,
            289.07324202,
            418.11583511,
            565.18424902,
            622.2057127,
            759.26462456,
            872.34868854,
        ]
    )
    assert np.allclose(b_masses[0], expected_b)

    # Check the first sequence's y-ion masses
    expected_y = np.array(
        [
            947.41710224,
            844.40791728,
            729.38097426,
            600.33838117,
            453.26996726,
            396.24850354,
            259.18959168,
            146.1055277,
        ]
    )
    y_masses = pepmass - b_masses
    assert np.allclose(y_masses[0], expected_y)


def test_calc_AA_masses_for_var_len_seqs():
    """Test that calc_AA_masses_for_var_len_seqs returns expected values."""
    sequences = ["EFGHIK", "AAAGCDEFGHIK", "DDDDCCDEFGHIK"]
    result = calc_AA_masses_for_var_len_seqs(sequences)

    # Check the shape
    assert result.shape == (3, 13)

    # Check some values
    assert np.isclose(result[0, 0], 129.04259309)  # E
    assert np.isclose(result[0, 1], 147.06841391)  # F
    assert np.isclose(result[1, 0], 71.03711138)  # A
    assert np.isclose(result[2, 0], 115.02694302)  # D

    # Check padding with high values
    assert result[0, 6] >= 1e8  # Padding should have very high values


def test_update_an_AA():
    """Test that update_an_AA updates the AA properties correctly."""
    update_an_AA("Z", "C(10)")

    # Check the mass and composition were updated
    assert AA_ASCII_MASS[ord("Z")] == 120
    assert AA_Composition["Z"]["C"] == 10


def test_replace_atoms():
    """Test that replace_atoms modifies atom formulas correctly."""
    # Replace N with 15N
    replace_atoms({"N": "15N"})

    # Check formulas contain 15N
    assert "15N" in aa_formula.loc["A"]["formula"]
    assert "15N" in aa_formula.loc["K"]["formula"]

    # Revert the change
    replace_atoms({"15N": "N"})

    # Check formulas no longer contain 15N
    assert "15N" not in aa_formula.loc["A"]["formula"]
    assert "15N" not in aa_formula.loc["K"]["formula"]
