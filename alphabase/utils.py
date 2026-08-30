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
    """Create a worker pool that shares this process's modification registry.

    Workers are started with the "spawn" start method and re-import alphabase
    from scratch, so they would otherwise lose every runtime change to the
    modification registry. The snapshot is taken once here and installed in each
    worker as it starts.
    """
    ctx = context if context is not None else mp.get_context("spawn")
    return ctx.Pool(
        processes,
        initializer=set_modification_state,
        initargs=(get_modification_state(),),
    )


def _batchify(obj, batch_size: int, group_by=None):
    """Yield row batches of a DataFrame or Series, optionally within groups."""
    groups = (group for _, group in obj.groupby(group_by)) if group_by else iter((obj,))
    for group in groups:
        for i in range(0, len(group), batch_size):
            yield group.iloc[i : i + batch_size]


def _batch_count(obj, batch_size: int, group_by=None) -> int:
    sizes = obj.groupby(group_by).size().values if group_by else [len(obj)]
    return sum((size + batch_size - 1) // batch_size for size in sizes)


def _with_progress(iterator, total, progress):
    """Wrap `iterator` in a progress bar.

    `progress` is True for a tqdm bar, a callable `progress(iterator, total)` to
    supply your own, or anything falsy for no bar.
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
    """Map `func` over `iterable` in workers that share the modification registry.

    Results are yielded as they arrive, so callers holding large objects only
    keep one batch in memory at a time.

    Parameters
    ----------
    processes : int
        Number of worker processes.

    total : int, optional
        Number of items, for the progress bar.

    unordered : bool, optional
        Yield results as they finish rather than in input order.

    progress : bool or callable, optional
        See :func:`_with_progress`.

    context : multiprocessing context, optional
        Substitute context, e.g. `torch.multiprocessing.get_context("spawn")`,
        whose reducers are needed to share model tensors with workers.
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
    """Apply `func` to row batches of `obj` in parallel and concatenate the result.

    Parameters
    ----------
    obj : pd.DataFrame or pd.Series
        Object to split into batches.

    group_by : optional
        Column to group by before batching, so each batch is within one group.

    See :func:`parallel_imap` for the remaining parameters.
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
