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


def _spawn_pool(processes: int, *, context=None):
    """Create a worker pool that shares the modification registry.

    A worker starts with the "spawn" method and imports alphabase again. Thus it
    loses all changes that were made to the registry at run time. This function
    copies the registry one time, then installs it in each worker at start.
    """
    ctx = context if context is not None else mp.get_context("spawn")
    return ctx.Pool(
        processes,
        initializer=set_modification_state,
        initargs=(get_modification_state(),),
    )


def _batchify(obj, batch_size: int, group_by=None):
    """Divide a DataFrame or Series into batches of rows."""
    groups = (group for _, group in obj.groupby(group_by)) if group_by else iter((obj,))
    for group in groups:
        for i in range(0, len(group), batch_size):
            yield group.iloc[i : i + batch_size]


def _batch_count(obj, batch_size: int, group_by=None) -> int:
    sizes = obj.groupby(group_by).size().values if group_by else [len(obj)]
    return sum((size + batch_size - 1) // batch_size for size in sizes)


def _with_progress(iterator, total, progress):
    """Add a progress bar to `iterator`.

    Set `progress` to True for a tqdm bar. Set it to a callable
    `progress(iterator, total)` to supply a different bar. Set it to a false
    value for no bar.
    """
    if progress is True:
        return tqdm.tqdm(iterator, total=total)
    if callable(progress):
        return progress(iterator, total)
    return iterator


def parallel_imap(
    func,
    iterable,
    *,
    processes: int,
    total: int = None,
    unordered: bool = False,
    progress=True,
    context=None,
):
    """Apply `func` to each item of `iterable` in workers.

    The workers share the modification registry. This function gives each result
    when it is ready. Thus the caller keeps only one batch in memory.

    Parameters
    ----------
    processes : int
        The number of worker processes.

    total : int, optional
        The number of items, for the progress bar.

    unordered : bool, optional
        Give each result when it is ready, not in the sequence of the input.

    progress : bool or callable, optional
        See :func:`_with_progress`.

    context : multiprocessing context, optional
        A different context, for example `torch.multiprocessing.get_context`.
        Its reducers are necessary to share model tensors with the workers.
    """
    with _spawn_pool(processes, context=context) as pool:
        mapper = pool.imap_unordered if unordered else pool.imap
        yield from _with_progress(mapper(func, iterable), total, progress)


def parallel_apply(
    func,
    obj,
    *,
    processes: int,
    batch_size: int,
    group_by=None,
    progress=True,
    context=None,
    ignore_index: bool = False,
):
    """Apply `func` to batches of rows of `obj`, then join the results.

    Parameters
    ----------
    obj : pd.DataFrame or pd.Series
        The object to divide into batches.

    group_by : optional
        A column to group by. Each batch then stays in one group.

    See :func:`parallel_imap` for the other parameters.
    """
    return pd.concat(
        list(
            parallel_imap(
                func,
                _batchify(obj, batch_size, group_by),
                processes=processes,
                total=_batch_count(obj, batch_size, group_by),
                progress=progress,
                context=context,
            )
        ),
        ignore_index=ignore_index,
    )
