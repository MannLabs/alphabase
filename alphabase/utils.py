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


def spawn_pool(processes: int, *, context=None, **kwargs):
    """Create a worker pool that shares this process's modification registry.

    Workers are started with the "spawn" start method and re-import alphabase
    from scratch, so they would otherwise lose every runtime change to the
    modification registry. The snapshot is taken once here and installed in each
    worker as it starts.

    Parameters
    ----------
    processes : int
        Number of worker processes.

    context : multiprocessing context, optional
        Substitute context, e.g. `torch.multiprocessing.get_context("spawn")`,
        whose reducers are needed to share model tensors with workers.
        Defaults to the standard "spawn" context.

    **kwargs
        Forwarded to `Pool`.
    """
    ctx = context if context is not None else mp.get_context("spawn")
    return ctx.Pool(
        processes,
        initializer=set_modification_state,
        initargs=(get_modification_state(),),
        **kwargs,
    )
