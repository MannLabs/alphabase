"""Tests for the utility functions in alphabase.utils."""

import numpy as np
import pandas as pd
import pytest

from alphabase.utils import _sanitize_missing_values


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
def test__sanitize_missing_values(values, dtype, expected_values):
    """Test that columns get coerced correctly by _sanitize_missing_values

    - Numeric missing values are retained
    - Object/string missing values are parsed to empty strings.
    - The dtype of a text column is preserved, `string[...]` dtypes included.
    """
    df = pd.DataFrame({"column": pd.Series(values, dtype=dtype)})

    result = _sanitize_missing_values(df)

    pd.testing.assert_series_equal(
        result["column"],
        pd.Series(expected_values, name="column", dtype=dtype),
    )

    assert id(df) != id(result)


def test__sanitize_missing_values_categorical():
    """Test categorical string and numeric columns in _sanitize_missing_values."""
    # Complete categorical string column: untouched, no TypeError on pandas 2.x
    df_complete = pd.DataFrame(
        {"genes": pd.Series(pd.Categorical(["ACTB", "MYH9", "ACTB"]))}
    )
    res_complete = _sanitize_missing_values(df_complete)
    pd.testing.assert_series_equal(res_complete["genes"], df_complete["genes"])

    # Partially missing categorical string column without "" category
    df_partial = pd.DataFrame(
        {"genes": pd.Series(pd.Categorical(["ACTB", None, "MYH9"]))}
    )
    res_partial = _sanitize_missing_values(df_partial)
    assert isinstance(res_partial["genes"].dtype, pd.CategoricalDtype)
    assert list(res_partial["genes"]) == ["ACTB", "", "MYH9"]

    # Partially missing categorical string column already containing "" category
    df_has_empty = pd.DataFrame(
        {"genes": pd.Series(pd.Categorical(["ACTB", "", None]))}
    )
    res_has_empty = _sanitize_missing_values(df_has_empty)
    assert isinstance(res_has_empty["genes"].dtype, pd.CategoricalDtype)
    assert list(res_has_empty["genes"]) == ["ACTB", "", ""]

    # Ordered categorical string column with missing values preserves order flag
    df_ordered = pd.DataFrame(
        {
            "genes": pd.Series(
                pd.Categorical(["b", None, "a"], categories=["a", "b"], ordered=True)
            )
        }
    )
    res_ordered = _sanitize_missing_values(df_ordered)
    assert isinstance(res_ordered["genes"].dtype, pd.CategoricalDtype)
    assert res_ordered["genes"].cat.ordered
    assert list(res_ordered["genes"]) == ["b", "", "a"]

    # Numeric categorical column retains missing values
    df_numeric = pd.DataFrame({"num": pd.Series(pd.Categorical([1, np.nan, 2]))})
    res_numeric = _sanitize_missing_values(df_numeric)
    pd.testing.assert_series_equal(res_numeric["num"], df_numeric["num"])

    # All-missing categorical column is coerced to empty text
    df_all_na = pd.DataFrame(
        {"genes": pd.Series(pd.Categorical([None, None], categories=["ACTB"]))}
    )
    res_all_na = _sanitize_missing_values(df_all_na)
    assert list(res_all_na["genes"]) == ["", ""]
