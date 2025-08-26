"""The base class for all PG readers and the provider for all PG readers."""

import re
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Type, Union

import pandas as pd

from alphabase.constants._const import CONST_FILE_FOLDER, PG_READER_YAML_FILE_NAME
from alphabase.psm_reader.utils import get_column_mapping_for_df
from alphabase.utils import _get_delimiter
from alphabase.yaml_utils import load_yaml

# see pg_reader.yaml alphabase/constants/const_files/pg_reader.yaml

pg_reader_yaml = load_yaml(Path(CONST_FILE_FOLDER) / PG_READER_YAML_FILE_NAME)


_COLUMN_MAPPING = "column_mapping"
_MEASUREMENT_REGEX = "measurement_regex"


class PGReaderBase:
    """Base class for all protein group (PG) readers.

    Supports reading of protein groups of common types:

    - **Type 1 — Minimal**: A basic features x samples matrix.
      Only intensity values are stored, with sample names as columns and protein groups as the index.
      **Example**: AlphaDIA.

    - **Type 2 — Multiple Intensity Fields**: A wide matrix where each sample may appear multiple times
      with different quantification types (e.g., `SampleA_LFQ`, `SampleB_raw`). Intensity columns are typically
      identifiable using regular expressions. Only intensity fields are included.
      **Example**: AlphaPept.

    - **Type 3 — Feature Metadata**: A features x samples matrix with one intensity value per sample,
      plus additional feature-level metadata columns (e.g., gene names, descriptions).
      **Example**: DIA-NN.

    - **Type 4 — Combined**: A composite structure including both multiple intensity fields (Type 2)
      and feature-level metadata (Type 3).
      **Examples**: Spectronaut, MZTab, MaxQuant.
    """

    _reader_type: str

    def __init__(
        self,
        *,
        column_mapping: Optional[dict[str, Any]] = None,
        measurement_regex: Optional[str] = None,
    ):
        """Read protein group (PG) matrices into the standardized alphabase format.

        Parameters
        ----------
        column_mapping
            A dictionary of mapping alphabase columns (keys) to the corresponding columns in the other
            search engine (values). If `None` will be loaded from the `column_mapping` key of the respective
            search engine in `pg_reader.yaml`
        measurement_regex
            Regular expression that identifies correct measurement type. Only relevant if PG matrix contains multiple
            measurement types. For example, alphapept returns the raw protein intensity per sample in column `A` and the
            LFQ corrected value in `A_LFQ`. If `None` uses all columns.


        Attributes
        ----------
        column_mapping
            Dictionary structure mapping alphabase columns (keys) to the corresponding columns in the other
            search engine (values), see parameters.
        measurement_regex
            Regular expression that matches quantity of interest for all samples

        Notes
        -----
        Standardizes protein group reports to a protein group dataframe (features x samples) in wide format. Contains at least
            - sample (run) identifier: :att:`pg_reader.keys.PGCols.SAMPLE_NAME` as column index
            - protein group identifier: :att:`pg_reader.keys.PGCols.protein` as index
            - protein group intensity: :att:`pg_reader.keys.PGCols.INTENSITY` as values

        Additional feature-level metadata might be available in the index.

        """
        self.column_mapping = (
            column_mapping
            if column_mapping is not None
            else pg_reader_yaml[self._reader_type][_COLUMN_MAPPING]
        )

        self.measurement_regex = self._get_measurement_regex(regex=measurement_regex)

    def add_column_mapping(self, column_mapping: Dict) -> None:
        """Add additional column mappings for the search engine."""
        self.column_mapping = {**self.column_mapping, **column_mapping}

    def import_file(self, file_path: str) -> pd.DataFrame:
        """Import a protein group (PG) matrix and process it to the alphabase convention.

        Loads the protein group matrix, standardizes feature metadata columns, and filters for the
        desired measurement type

        Parameters
        ----------
        file_path: str
            Absolute path to the file containing protein group data

        Returns
        -------
        :class:`pd.DataFrame`
            Protein group matrix with feature metadata as index

        """
        # Load to dataframe
        df = self._load_file(file_path)

        df = self._pre_process(df)

        if len(df) == 0:
            return pd.DataFrame()

        # Standardize feature columns
        # `get_column_mapping_for_df` returns mapping of the form {standardized: search engine-specific}
        # invert to faciliate mapping
        # This is possible as the function guarantees to return a 1-1 mapping
        engine_to_standard = {
            v: k for k, v in get_column_mapping_for_df(self.column_mapping, df).items()
        }
        feature_columns = list(engine_to_standard.values())

        df = self._translate_columns(df, column_mapping=engine_to_standard)

        # Subset to relevant sample columns
        # For example in alphapept sample vs. sample_LFQ
        if self.measurement_regex is not None:
            df = self._filter_measurement(
                df,
                regex=self.measurement_regex,
                extra_columns=feature_columns,
            )

        df = self._post_process(df)

        # Keep dataframe index as default if no features are specified in column mapping
        return df.set_index(feature_columns) if len(feature_columns) > 0 else df

    def _load_file(self, file_path: str) -> pd.DataFrame:
        """Load protein group (PG) file into a dataframe.

        Parameters
        ----------
        file_path
            File path, must be .csv, .tsv, or .hdf

        Returns
        -------
        :class:`pd.DataFrame`
            Protein group matrix

        Notes
        -----
        Supports comma separated, tab separated, and .hdf files which covers the file types of
        all supported search engines.

        """
        if Path(file_path).suffix == ".hdf":
            return pd.read_hdf(file_path)
        if Path(file_path).suffix == ".parquet":
            return pd.read_parquet(file_path)

        sep = _get_delimiter(file_path)
        return pd.read_csv(file_path, sep=sep, keep_default_na=False)

    def _pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess dataframe before standardizing columns."""
        return df

    def _translate_columns(
        self, df: pd.DataFrame, column_mapping: dict[str, str]
    ) -> pd.DataFrame:
        """Translate standardized columns in dataframe from other search engines to AlphaBase format."""
        return df.rename(columns=column_mapping)

    def _filter_measurement(
        self,
        df: pd.DataFrame,
        regex: str,
        extra_columns: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Subset :class:`pd.DataFrame` to columns matching a regex plus optionally extra columns.

        Parameters
        ----------
        df
            :class:`pd.DataFrame`
        regex
            Regular expression that matches the respective columns
        extra_columns
            Keep the specified columns although they do not match the regex, defaults to `None`.

        Returns
        -------
        :class:`pd.DataFrame`
            Filtered dataframe with columns matching `regex` and columns explicitly
            specified in `keep_columns`.

        """
        extra_columns = extra_columns if extra_columns is not None else []

        pattern = re.compile(regex)
        regex_columns = [col for col in df.columns if re.search(pattern, col)]

        if len(regex_columns) == 0:
            warnings.warn(f"regex {regex} did not match any columns in the dataframe")

        return df[regex_columns + extra_columns]

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process dataframe after standardizing columns."""
        return df

    def _get_measurement_regex(self, regex: Optional[str]) -> Union[str, None]:
        """Get the correct named measurement regex from the reader configuration.

        The function tries to match the provided `regex` to the keys in `measurement_regex` in the reader configuration. This
        enables users to provide tangible names for the columns they want instead of abstract regular expressions.
        If a match is found, it returns the associated value (the actual regex).
        If this not possible, the function assumes that a regular expression was passed and
        simply returns `regex` (special case: if `_MEASUREMENT_REGEX` does not contain any values, it also returns the `regex`)


        Parameters
        ----------
        regex
            None, Name of regular expression in reader configuration or a regular expression.


        Returns
        -------
        str | None
            Output depends on regex and the the key `measurement_regex` in reader configuration

            - If `regex` is a key in the reader configuration `measurement_regex`, returns
            the associated value
            - If `regex` is not in the reader configuration, assumes that `regex` is an
            actual regular expression and returns it as is (special case: if `regex` is `None`, returns `None`)
            - If `measurement_regex` is not configured (`None`), returns `regex`.

        """
        reader_config = pg_reader_yaml[self._reader_type]
        measurement_regex_config = reader_config.get(_MEASUREMENT_REGEX)

        if measurement_regex_config is None:
            return regex

        config_regex = measurement_regex_config.get(regex)

        return config_regex if config_regex is not None else regex

    @classmethod
    def get_preconfigured_regex(cls) -> dict[str, str]:
        """Get all predefined regular expressions for this reader class as configured in `alphabase.constants.pg_reader_yaml`."""
        available_regex = pg_reader_yaml[cls._reader_type][_MEASUREMENT_REGEX]
        return available_regex if isinstance(available_regex, dict) else {}


# TODO: Refactor and create base class for PG Reader provider and PSMReaderProvider
class PGReaderProvider:
    """A factory class to register and get readers for different protein group report types."""

    def __init__(self):
        """Initialize PGReaderProvider."""
        self.reader_dict: dict[str, Type[PGReaderBase]] = {}

    def register_reader(
        self, reader_type: str, reader_class: Type[PGReaderBase]
    ) -> None:
        """Register a reader by reader_type."""
        self.reader_dict[reader_type.lower()] = reader_class

    def get_reader(
        self,
        reader_type: str,
        *,
        column_mapping: Optional[dict] = None,
        **kwargs,
    ) -> PGReaderBase:
        """Get a reader by reader_type."""
        try:
            return self.reader_dict[reader_type.lower()](
                column_mapping=column_mapping,
                **kwargs,
            )
        except KeyError as e:
            raise KeyError(
                f"Unknown reader type '{reader_type}'. Available readers: "
                f"{', '.join(self.reader_dict.keys())}"
            ) from e

    def get_reader_by_yaml(
        self,
        yaml_dict: dict,
    ) -> PGReaderBase:
        """Get a reader by a yaml dict."""
        return self.get_reader(**deepcopy(yaml_dict))


pg_reader_provider = PGReaderProvider()
"""
A factory :class:`PGReaderProvider` object to register and get readers for different protein group report types.
"""
