import io
import itertools
import warnings

import pandas as pd
import tqdm


class AlphabaseDeprecationWarning(DeprecationWarning):
    pass


# Custom dict class that issues warnings
class DeprecatedDict(dict):
    def __init__(self, *args, **kwargs):
        self.warning_message = kwargs.pop(
            "warning_message", "This dictionary is deprecated"
        )
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        warnings.warn(self.warning_message, AlphabaseDeprecationWarning, stacklevel=2)
        return super().__getitem__(key)

    def get(self, key, default=None):
        warnings.warn(self.warning_message, AlphabaseDeprecationWarning, stacklevel=2)
        return super().get(key, default)


# from alphatims
def process_bar(iterator, len_iter):
    with tqdm.tqdm(total=len_iter) as bar:
        i = 0
        for i, iter in enumerate(iterator):  # noqa: B007
            yield iter
            bar.update()
        bar.update(len_iter - i - 1)


def _flatten(list_of_lists):
    """
    Flatten a list of lists
    """
    return list(itertools.chain.from_iterable(list_of_lists))


def explode_multiple_columns(df: pd.DataFrame, columns: list):
    try:
        return df.explode(columns)
    except ValueError:
        # pandas < 1.3.0
        print(f"pandas=={pd.__version__} cannot explode multiple columns")
        ret_df = df.explode(columns[0])
        for col in columns[1:]:
            ret_df[col] = _flatten(df[col].values)
        return ret_df


def _get_delimiter(file_path: str) -> str:
    if isinstance(file_path, io.StringIO):
        # for unit tests
        line = file_path.readline().strip()
        file_path.seek(0)
    else:
        with open(file_path) as f:
            line = f.readline().strip()
    if "\t" in line:
        return "\t"
    elif "," in line:
        return ","
    else:
        return "\t"


def _coerce_object_nan_to_empty_string(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce NaN values in object and string columns to an empty string.

    Returns
    -------
    Copy of dataframe with coerced columns

    Example
    -------
    .. code-block:: python

        df["numeric_column"]
        >   0    1.0
            2    NaN
            Name: "numeric_column", dtype: float64

        df["object_column"]
        >   0    A
            2    nan
            Name: "object_column", dtype: object

        new_df = _coerce_nan_to_missing_string(df)

        pd.testing.assert_series_equal(new_df["numeric_column], df["numeric_column"])
        new_df["object_column"]
        >   0    A
            2    ""
            Name: "object_column", dtype: object

    """
    df = df.copy()
    for column in df.columns:
        if len(df) > 0 and df[column].isna().all():
            # a column holding only missing values carries no type information: pandas
            # infers float64, but alphabase expects text (e.g. an unpopulated `Genes` column).
            # Assigning avoids fillna's deprecated object-dtype downcasting.
            df[column] = ""
        elif pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(
            df[column]
        ):
            df[column] = df[column].fillna("")

    return df
