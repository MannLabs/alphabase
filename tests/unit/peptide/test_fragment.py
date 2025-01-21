from unittest.mock import patch

import pytest

from alphabase.peptide.fragment import (
    filter_valid_charged_frag_types,
    parse_charged_frag_type,
)


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("b_z1", ("b", 1)),
        ("b_modloss_z2", ("b_modloss", 2)),
    ],
)
def test_parse_charged_frag_type_with_valid_input(input_str, expected):
    """Test parse_charged_frag_type with valid input."""
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
    """Test parse_charged_frag_type handles errors correctly."""
    with pytest.raises(ValueError, match=match):
        parse_charged_frag_type(input_str)


@patch("alphabase.peptide.fragment.parse_charged_frag_type")
def test_filter_valid_charged_frag_types(mock_parse):
    """Test filter_valid_charged_frag_types handles errors correctly."""
    mock_parse.side_effect = [("b", 1), ValueError, ("y", 2)]
    with pytest.warns(UserWarning) as recorded_warnings:
        result = filter_valid_charged_frag_types(
            [
                "b_z1",
                "unsupported_z1",
                "y_z2",
            ]
        )
    assert result == ["b_z1", "y_z2"]
    assert len(recorded_warnings) == 1  # Should have 2 warning messages
