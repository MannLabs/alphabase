import numpy as np
import pytest

from alphabase.constants.atom import (
    calc_mass_from_formula,
    parse_formula,
    truncate_isotope,
)


def test_parse_formula():
    """Test parsing of chemical formulas."""
    formula = "H(2)C(2)O(1)"
    expected = [("H", 2), ("C", 2), ("O", 1)]

    result = parse_formula(formula)

    # Check each element and count
    for exp_val, output in zip(expected, result):
        assert exp_val == output


def test_calc_mass_from_formula():
    """Test calculation of mass from formula."""
    # Test a simple formula
    formula = "H(2)C(2)O(1)"
    expected_mass = 42.010564684

    actual_mass = calc_mass_from_formula(formula)
    assert abs(actual_mass - expected_mass) < 1e-6

    # Test empty formula
    assert calc_mass_from_formula("") == 0


@pytest.mark.requires_numba
@pytest.mark.parametrize(
    ("isotopes", "mono_idx"),
    [
        pytest.param(
            [1, 2, 3, 4, 5, 6, 90, 80, 70, 60, 50, 40, 30, 20, 10],
            6,
            id="mono_in_middle",
        ),
        pytest.param(
            [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 4, 3, 2, 1],
            0,
            id="mono_at_start",
        ),
        pytest.param(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100],
            14,
            id="mono_at_end",
        ),
        pytest.param(
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1, 1, 1, 1, 1],
            9,
            id="window_grows_to_the_left",
        ),
        pytest.param(
            [1, 1, 1, 1, 1, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
            5,
            id="window_grows_to_the_right",
        ),
        pytest.param([1] * 15, 7, id="equal_intensities"),
    ],
)
def test_truncate_isotope_numba_matches_pure_python(isotopes, mono_idx):
    """Test that the numba-compiled truncate_isotope agrees with its pure Python original."""
    # given
    isotopes = np.array(isotopes, dtype=np.float64)

    # when
    compiled_result = truncate_isotope(isotopes, mono_idx)
    pure_python_result = truncate_isotope.py_func(isotopes, mono_idx)

    # then
    assert tuple(int(value) for value in compiled_result) == tuple(
        int(value) for value in pure_python_result
    )
