import io
import itertools
import multiprocessing as mp
import warnings

import pandas as pd
import tqdm

from alphabase.constants.modification import (
    get_modification_state,
    set_modification_state,
)


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


def _sanitize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce NaN values in object and string columns to an empty string.

    NAs have a special meaning in alphabase as they indicate unresolved modifications.
    Use empty strings in object/string columns to indicate missing values.

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


def _spawn_pool(processes: int):
    """Create a pool whose workers start with the parent's modification registry.

    See :func:`set_modification_state` for why a spawned worker needs it.
    """
    registry = get_modification_state()
    return mp.get_context("spawn").Pool(
        processes, initializer=set_modification_state, initargs=(registry,)
    )


def _batchify(obj, batch_size: int, group_by=None) -> list:
    """Give the row batches of a DataFrame or Series, each within one group."""
    groups = [group for _, group in obj.groupby(group_by)] if group_by else [obj]
    return [
        group.iloc[i : i + batch_size]
        for group in groups
        for i in range(0, len(group), batch_size)
    ]


def _with_progress(iterator, total, progress):
    """True gives a tqdm bar, a callable `progress(iterator, total)` its own, falsy none."""
    if progress is True:
        return tqdm.tqdm(iterator, total=total)
    return progress(iterator, total) if callable(progress) else iterator


def parallel_apply(
    func,
    obj,
    *,
    processes: int,
    batch_size: int,
    group_by=None,
    progress=True,
    ignore_index: bool = False,
):
    """Apply `func` to row batches of `obj` in spawned workers, then join the results.

    Parameters
    ----------
    obj : pd.DataFrame or pd.Series
        The object to divide into batches.

    processes : int
        The number of worker processes.

    group_by : optional
        A column to group by. Each batch then stays in one group.

    progress : bool or callable, optional
        See :func:`_with_progress`.
    """
    batches = _batchify(obj, batch_size, group_by)
    with _spawn_pool(processes) as pool:
        results = _with_progress(pool.imap(func, batches), len(batches), progress)
        return pd.concat(list(results), ignore_index=ignore_index)
