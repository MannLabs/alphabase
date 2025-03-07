import contextlib
import logging

import pandas as pd
import pyarrow.parquet

LOGGER = logging.getLogger(__name__)


def filter_input(filter_dict, input):
    if filter_dict is None:
        return input
    for filtname, filterconf in filter_dict.items():
        param = filterconf.get("param")
        comparator = filterconf.get("comparator")
        value = filterconf.get("value")

        if comparator not in [">", ">=", "<", "<=", "==", "!="]:
            raise TypeError(
                f"cannot identify the filter comparator of {filtname} given in the longtable config yaml!"
            )

        if comparator == "==":
            input = input[input[param] == value]
            continue
        with contextlib.suppress(Exception):
            input = input.astype({f"{param}": "float"})

        if comparator == ">":
            input = input[input[param].astype(type(value)) > value]

        if comparator == ">=":
            input = input[input[param].astype(type(value)) >= value]

        if comparator == "<":
            input = input[input[param].astype(type(value)) < value]

        if comparator == "<=":
            input = input[input[param].astype(type(value)) <= value]

        if comparator == "!=":
            input = input[input[param].astype(type(value)) != value]

    return input


def read_file(file_path, decimal=".", usecols=None, chunksize=None, sep=None):
    file_path = str(file_path)
    if file_path.endswith(".parquet"):
        return _read_parquet_file(file_path, usecols=usecols, chunksize=chunksize)
    else:
        if sep is None:
            if ".csv" in file_path:
                sep = ","
            elif ".tsv" in file_path:
                sep = "\t"
            else:
                sep = "\t"
                LOGGER.info(
                    f"neither of the file extensions (.tsv, .csv) detected for file {file_path}! Trying with tab separation. In the case that it fails, please provide the correct file extension"
                )
        return pd.read_csv(
            file_path,
            sep=sep,
            decimal=decimal,
            usecols=usecols,
            encoding="latin1",
            chunksize=chunksize,
        )


def _read_parquet_file(file_path, usecols=None, chunksize=None):
    if chunksize is not None:
        return _read_parquet_file_chunkwise(
            file_path, usecols=usecols, chunksize=chunksize
        )
    return pd.read_parquet(file_path, columns=usecols)


def _read_parquet_file_chunkwise(file_path, usecols=None, chunksize=None):
    parquet_file = pyarrow.parquet.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(columns=usecols, batch_size=chunksize):
        yield batch.to_pandas()


def read_columns_from_file(file, sep="\t"):
    if file.endswith(".parquet"):
        parquet_file = pyarrow.parquet.ParquetFile(file)
        return parquet_file.schema.names
    return pd.read_csv(file, sep=sep, nrows=1).columns.tolist()
