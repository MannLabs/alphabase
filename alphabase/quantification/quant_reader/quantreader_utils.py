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


def read_file(input_file, decimal=".", usecols=None, chunksize=None, sep=None):
    input_file = str(input_file)
    if ".parquet" in input_file:
        return _read_parquet_file(input_file, usecols=usecols, chunksize=chunksize)
    else:
        if sep is None:
            if ".csv" in input_file:
                sep = ","
            elif ".tsv" in input_file:
                sep = "\t"
            else:
                sep = "\t"
                LOGGER.info(
                    f"neither of the file extensions (.tsv, .csv) detected for file {input_file}! Trying with tab separation. In the case that it fails, please provide the correct file extension"
                )
        return pd.read_csv(
            input_file,
            sep=sep,
            decimal=decimal,
            usecols=usecols,
            encoding="latin1",
            chunksize=chunksize,
        )


def _read_parquet_file(input_file, usecols=None, chunksize=None):
    if chunksize is not None:
        return _read_parquet_file_chunkwise(
            input_file, usecols=usecols, chunksize=chunksize
        )
    return pd.read_parquet(input_file, columns=usecols)


def _read_parquet_file_chunkwise(input_file, usecols=None, chunksize=None):
    parquet_file = pyarrow.parquet.ParquetFile(input_file)
    for batch in parquet_file.iter_batches(columns=usecols, batch_size=chunksize):
        yield batch.to_pandas()


def read_columns_from_file(file, sep="\t"):
    if file.endswith(".parquet"):
        parquet_file = pyarrow.parquet.ParquetFile(file)
        return parquet_file.schema.names
    return pd.read_csv(file, sep=sep, nrows=1).columns.tolist()
