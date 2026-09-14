"""Tests for the utility functions in alphabase.utils."""

import numpy as np
import pandas as pd
import pytest

from alphabase.utils import _coerce_object_nan_to_empty_string


@pytest.mark.parametrize(
    ("values", "dtype", "expected_values"),
    [
        pytest.param(
            [1.5, np.nan, 3.5],
            None,
            [1.5, np.nan, 3.5],
            id="numeric_column_keeps_nan",
        ),
        pytest.param(
            [1.5, 2.5, 3.5],
            None,
            [1.5, 2.5, 3.5],
            id="numeric_column_without_nan_unchanged",
        ),
        pytest.param(
            ["a", np.nan, "c"],
            None,
            ["a", "", "c"],
            id="text_column_gets_empty_string",
        ),
        pytest.param(
            [np.nan, np.nan, np.nan],
            None,
            ["", "", ""],
            id="all_missing_column_treated_as_text",
        ),
        pytest.param(
            ["a", None, "c"],
            "object",
            ["a", "", "c"],
            id="object_dtype_column",
        ),
        pytest.param(
            ["a", None, "c"],
            "string",
            ["a", "", "c"],
            id="string_dtype_column",
        ),
    ],
)
def test__coerce_object_nan_to_empty_string(values, dtype, expected_values):
    """Test that columns get coerced correctly by _coerce_object_nan_to_empty_string

    - Numeric missing values are retained
    - Object/string missing values are parsed to empty strings.
    - The dtype of a text column is preserved, `string[...]` dtypes included.
    """
    df = pd.DataFrame({"column": pd.Series(values, dtype=dtype)})

    result = _coerce_object_nan_to_empty_string(df)

    pd.testing.assert_series_equal(
        result["column"],
        pd.Series(expected_values, name="column", dtype=dtype),
    )

    assert id(df) != id(result)
