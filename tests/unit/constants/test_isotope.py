import numpy as np
import pytest

from alphabase.constants.atom import (
    CHEM_ISOTOPE_DIST,
    CHEM_MONO_IDX,
    CHEM_MONO_MASS,
    MAX_ISOTOPE_LEN,
    parse_formula,
)
from alphabase.constants.isotope import (
    IsotopeDistribution,
    formula_dist,
    one_element_dist,
)

pytestmark = pytest.mark.requires_numba


def test_one_element_dist():
    """Test one_element_dist for H10."""
    dist, mono_idx = one_element_dist("H", 10, CHEM_ISOTOPE_DIST, CHEM_MONO_IDX)

    assert mono_idx == 0

    # Convert to percentages for readability
    percentages = (dist / max(dist)) * 100

    # Check the first three values match expected percentages
    # Desired distribution for H10: (100, 0.16, 0.0001)
    assert abs(percentages[0] - 100) < 0.1
    assert abs(percentages[1] - 0.115) < 0.1
    assert abs(percentages[2] - 0.0001) < 0.1

    # Check remaining values are close to zero
    assert np.all(percentages[3:] < 0.001)


def test_parse_formula():
    """Test parsing of a chemical formula."""
    formula_str = "C(100)H(100)O(10)"
    formula = parse_formula(formula_str)

    expected = [("C", 100), ("H", 100), ("O", 10)]
    for exp_val, val in zip(expected, formula):
        assert exp_val == val


def test_mass_calculation():
    """Test mass calculation from formula components."""
    formula = parse_formula("C(100)H(100)O(10)")
    mass = np.sum([CHEM_MONO_MASS[elem] * n for elem, n in formula])

    assert abs(mass - 1460.73164942) < 1e-6


def test_formula_dist():
    """Test isotope distribution calculation for a formula."""
    formula = "C(100)H(100)O(10)"
    calc_dist, mono_idx = formula_dist(formula)

    assert mono_idx == 0

    # Expected distribution from sisweb.com/mstools/isotope.htm
    # Desired: (90.7784, 100, 56.368, 21.6475, 6.3624, 1.524, 0.3093)
    expected_percentages = np.zeros(MAX_ISOTOPE_LEN)
    expected_percentages[:7] = [90.7784, 100, 56.368, 21.6475, 6.3624, 1.524, 0.3093]

    # Calculate cosine similarity
    cosine = np.sum(calc_dist * expected_percentages) / np.sqrt(
        np.sum(calc_dist**2) * np.sum(expected_percentages**2)
    )
    assert cosine > 0.99  # Very high similarity


def test_formula_dist_with_heavy_label():
    """Test isotope distribution calculation with heavy label."""
    formula = "C(100)H(100)O(10)13C(1)"
    calc_dist, mono_idx = formula_dist(formula)

    # The mono index should be 1 due to the heavy label
    assert mono_idx == 1

    # The maximum intensity should be at index 2
    assert np.argmax(calc_dist) == 2

    # First peak should be small compared to second peak (the mono peak)
    assert calc_dist[0] < 0.01 * calc_dist[1]


def test_isotope_distribution_class():
    """Test the IsotopeDistribution class."""
    iso = IsotopeDistribution()

    # Test with a complex formula
    formula_str = "C(100)H(100)O(10)Na(1)Fe(1)"
    formula = parse_formula(formula_str)

    # Calculate with the class method
    dist, mono = iso.calc_formula_distribution(formula)

    # Calculate with the function
    dist1, mono1 = formula_dist(formula)

    # Results should match
    assert np.allclose(dist, dist1)
    assert mono == mono1
    assert mono == 2  # Expected mono idx for this formula
