from alphabase.constants.atom import calc_mass_from_formula, parse_formula


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
