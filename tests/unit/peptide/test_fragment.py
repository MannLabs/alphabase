import pytest

from alphabase.peptide.fragment import parse_charged_frag_type


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("b_z1", ("b", 1)),
        ("b_modloss_z2", ("b_modloss", 2)),
    ],
)
def test_parse_charged_frag_type_with_valid_input(input_str, expected):
    result = parse_charged_frag_type(input_str)
    assert result == (expected[0], expected[1])


@pytest.mark.parametrize(
    "input_str, match",
    [
        ("b_z1_z2", "Only charged fragment types are supported"),
        ("b_z1.5", "Charge state must be a positive integer"),
        ("b_z0", "Charge state must be a positive integer"),
        ("b_z-1", "Charge state must be a positive integer"),
        ("unsupported_z1", "Fragment type unsupported is currently not supported"),
    ],
)
def test_parse_charged_frag_type_with_exceptions(input_str, match):
    with pytest.raises(ValueError, match=match):
        parse_charged_frag_type(input_str)
