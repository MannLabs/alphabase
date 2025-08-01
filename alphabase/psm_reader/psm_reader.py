"""The base class for all PSM readers and the provider for all readers."""

import copy
import io
import warnings
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional, Set, Type, Union

import pandas as pd

from alphabase.constants._const import CONST_FILE_FOLDER, PSM_READER_YAML_FILE_NAME
from alphabase.peptide import mobility
from alphabase.peptide.precursor import reset_precursor_df, update_precursor_mz
from alphabase.psm_reader.keys import PsmDfCols
from alphabase.psm_reader.modification_mapper import ModificationMapper
from alphabase.psm_reader.utils import (
    get_column_mapping_for_df,
    keep_modifications,
    translate_modifications,
)
from alphabase.utils import _get_delimiter
from alphabase.yaml_utils import load_yaml

#: See `psm_reader.yaml <https://github.com/MannLabs/alphabase/blob/main/alphabase/constants/const_files/psm_reader.yaml>`_
psm_reader_yaml = load_yaml(Path(CONST_FILE_FOLDER) / PSM_READER_YAML_FILE_NAME)

_MIN_IRT_VALUE = -100
_MAX_IRT_VALUE = 200


class PSMReaderBase(ABC):
    """The Base class for all PSMReaders."""

    # the type of the reader, this references a key in psm_reader.yaml
    _reader_type: str
    # whether to add the unimod mappings to the modification mapping
    _add_unimod_to_mod_mapping: bool = False
    # whether 'rt_norm' values in self._psm_dd will be normalized using min/max values
    #  Useful to normalize iRT values as they contain negative values.
    _min_max_rt_norm = False

    def __init__(  # noqa: PLR0913 # too many arguments
        self,
        *,
        column_mapping: Optional[dict] = None,
        modification_mapping: Optional[dict] = None,
        fdr: float = 0.01,
        keep_decoy: bool = False,
        rt_unit: Optional[str] = None,
        mod_seq_columns: Optional[List[str]] = None,
        **kwargs,
    ):
        """The Base class for all PSMReaders.

        The key of the sub-classes for different
        search engine format is to re-define `column_mapping` and `modification_mapping`.

        Parameters
        ----------
        column_mapping : dict, optional
            A dict that maps alphabase's columns to those of other search engines'.
            If it is None, this dict will be read from psm_reader.yaml key `column_mapping`.

            The key of the column_mapping is alphabase's column name, and
            the value could be the column name or a list of column names
            in other engine's result, for example:

            .. code-block:: python

                columns_mapping = {
                    'sequence': 'NakedSequence',
                    'charge': 'Charge',
                    'proteins':['Proteins','UniprotIDs'] # list, this reader will automatically detect all of them.
                }

            The first column name in the list will be mapped to the harmonized column names, the rest will be ignored.
            Defaults to None.

        modification_mapping : dict, optional
            A dict that maps alphabase's modifications to other engine's.
            If it is None, this dict will be init by
            default modification mapping for each search engine
            (see :data:`psm_reader_yaml`).
            The dict values can be
            either str or list, for exaplme:

            .. code-block:: python

                modification_mapping = {
                    'Oxidation@M': 'Oxidation (M)', # str
                    'Phospho@S': ['S(Phospho (STY))','S(ph)','pS'], # list, this reader will automatically detect all of them.
                }

            Defaults to None.

        fdr : float, optional
            FDR level to keep PSMs.
            Defaults to 0.01.

        keep_decoy : bool, optional
            If keep decoy PSMs in self.psm_df.
            Defaults to False.

        rt_unit : str, optional
            The unit of RT in the search engine result, "minute", "second" or "irt".
            If None, it is read from psm_reader_yaml key "rt_unit".

        mod_seq_columns : list, optional
            The columns to find modified sequences.
            The first column name in the list will be used, the rest will be ignored.
            By default read from psm_reader_yaml key "mod_seq_columns".
            If it is not found there, an empty list is used.

        **kwargs: dict
            deprecated

        Attributes
        ----------
        column_mapping : dict
            Dict structure same as column_mapping in Args.
        modification_mapping : dict
            Dict structure same as modification_mapping in Args.
            We must use self.set_modification_mapping(new_mapping) to update it.
        psm_df : pd.DataFrame
            the PSM DataFrame after loading from search engines.

        """
        self._modification_mapper = ModificationMapper(
            modification_mapping,
            reader_yaml=copy.deepcopy(psm_reader_yaml),
            mapping_type=psm_reader_yaml[self._reader_type][
                "modification_mapping_type"
            ],
            add_unimod_to_mod_mapping=self._add_unimod_to_mod_mapping,
        )

        self.column_mapping = (
            column_mapping
            if column_mapping is not None
            else psm_reader_yaml[self._reader_type]["column_mapping"]
        )

        self._mod_seq_columns = (
            mod_seq_columns
            if mod_seq_columns is not None
            else psm_reader_yaml[self._reader_type].get("mod_seq_columns", [])
        )
        self.mod_seq_column = None

        self._psm_df = None

        self._fdr_threshold = fdr
        self._keep_decoy = keep_decoy

        self._precursor_id_columns = psm_reader_yaml[self._reader_type].get(
            "precursor_id_columns", []
        )
        self._precursor_id_column = None

        self._rt_unit = (
            rt_unit
            if rt_unit is not None
            else psm_reader_yaml[self._reader_type]["rt_unit"]
        )
        if self._rt_unit not in ["minute", "second", "irt"]:
            raise ValueError(
                f"Invalid rt_unit: {self._rt_unit}. "
                f"rt_unit should be one of ['minute', 'second', 'irt']."
            )

        for key, value in kwargs.items():  # TODO: remove and remove kwargs
            warnings.warn(
                f"Passed unknown arguments to {self.__class__.__name__} "
                f"({key}={value}) will be forbidden in alphabase>1.5.0.",
                FutureWarning,
            )

    @property
    def psm_df(self) -> pd.DataFrame:
        """Get the PSM DataFrame."""
        return self._psm_df

    @property
    def modification_mapping(self) -> Dict:
        """Get the modification mapping dictionary."""
        return self._modification_mapper.modification_mapping

    def add_modification_mapping(self, modification_mapping: Dict) -> None:
        """Append additional modification mappings for the search engine.

        See ModificationMapper.add_modification_mapping for more details.
        """
        self._modification_mapper.add_modification_mapping(modification_mapping)

    def set_modification_mapping(
        self, modification_mapping: Optional[Dict] = None
    ) -> None:
        """Set the modification mapping for the search engine.

        See ModificationMapper.set_modification_mapping for more details.
        """
        self._modification_mapper.set_modification_mapping(modification_mapping)

    def add_column_mapping(self, column_mapping: Dict) -> None:
        """Add additional column mappings for the search engine."""
        self.column_mapping = {**self.column_mapping, **column_mapping}

    def load(self, _file: Union[List[str], str]) -> pd.DataFrame:
        """Import a single file or multiple files."""
        if isinstance(_file, list):
            return self.import_files(_file)
        return self.import_file(_file)

    def import_files(self, file_list: List[str]) -> pd.DataFrame:
        """Import multiple files."""
        df_list = [self.import_file(file) for file in file_list]
        self._psm_df = pd.concat(df_list, ignore_index=True)
        return self._psm_df

    def import_file(self, _file: str) -> pd.DataFrame:
        """Main entry function of PSM readers.

        Imports a file and processes it.

        Parameters
        ----------
        _file: str
            file path or file stream (io).

        """
        origin_df = self._load_file(_file)

        self._psm_df = pd.DataFrame()

        if len(origin_df):
            # TODO: think about dropping the 'inplace' pattern here
            self.mod_seq_column = self._get_actual_column(
                self._mod_seq_columns, origin_df
            )
            self._precursor_id_column = self._get_actual_column(
                self._precursor_id_columns, origin_df
            )

            origin_df = self._pre_process(origin_df)
            self._translate_columns(origin_df)  # only here
            self._translate_decoy()  # only sage, mq, msfragger, pfind
            self._translate_score()  # only msfragger, pfind
            self._load_modifications(
                origin_df
            )  # only sage, mq, msfragger, pfind, alphapept
            self._translate_modifications()  # here, sage, msfragger, pfind
            self._filter_fdr()
            self._post_process(origin_df)  # here, libraryreader, diann, msfragger
        return self._psm_df

    def _load_file(self, filename: Union[str, Path, io.StringIO]) -> pd.DataFrame:
        """Load PSM file into a dataframe.

        Different search engines may store PSMs in different ways: tsv, csv, HDF, XML, parquet ...
        This default implementation works for tsv, csv and parquet files and thus covers many readers.

        Parameters
        ----------
        filename : str | pathlib.Path | io.StringIO
            The file path to the PSM file or the file in the io.StringIO.

        """
        if isinstance(filename, io.StringIO):
            sep = _get_delimiter(filename)
            return pd.read_csv(filename, sep=sep, keep_default_na=False)

        file_path = Path(filename)

        if file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)

        sep = _get_delimiter(str(file_path))
        return pd.read_csv(file_path, sep=sep, keep_default_na=False)

    def _get_actual_column(
        self,
        column_list: List[str],
        df: pd.DataFrame,
    ) -> Optional[str]:
        """Get the first column from `column_list` that is a column of `df`."""
        for column in column_list:
            if column in df.columns:
                return column
        return None
        # TODO: warn if there's more

    def _pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataframe right after reading from file."""
        return df

    def _translate_columns(self, origin_df: pd.DataFrame) -> None:
        """Translate the dataframe from other search engines to AlphaBase format."""
        column_mapping_for_df = get_column_mapping_for_df(
            self.column_mapping, origin_df
        )

        for col, map_col in column_mapping_for_df.items():
            self._psm_df[col] = origin_df[map_col]

        if (
            PsmDfCols.SCAN_NUM in self._psm_df.columns
            and PsmDfCols.SPEC_IDX not in self._psm_df.columns
        ):
            self._psm_df[PsmDfCols.SPEC_IDX] = self._psm_df[PsmDfCols.SCAN_NUM] - 1

    def _translate_decoy(self) -> None:  # noqa: B027 empty method in an abstract base class
        """Translate decoy information to AlphaBase format, adding information inplace into self._psm_df."""

    def _translate_score(self) -> None:  # noqa: B027 empty method in an abstract base class
        """Translate score information to AlphaBase format, adding information inplace into self._psm_df."""

    def _load_modifications(self, origin_df: pd.DataFrame) -> None:  # noqa: B027 empty method in an abstract base class
        """Read modification information from 'origin_df'.

        Some search engines use modified_sequence, some of them
        use additional columns to store modifications and the sites.
        """

    def _translate_modifications(self) -> None:
        """Translate modifications to AlphaBase format."""
        mods, unknown_mods = zip(
            *self._psm_df[PsmDfCols.MODS].apply(
                translate_modifications,
                mod_dict=self._modification_mapper.rev_mod_mapping,
            )
        )
        self._psm_df[PsmDfCols.MODS] = mods

        # accumulate unknown mods
        unknown_mod_set = set()
        for mod_list in unknown_mods:
            if len(mod_list) > 0:
                unknown_mod_set.update(mod_list)

        if len(unknown_mod_set) > 0:
            warnings.warn(
                f"Unknown modifications: {unknown_mod_set}. Precursors with unknown modifications will be removed."
            )

    def _post_process(self, origin_df: pd.DataFrame) -> None:
        """Set 'nAA' columns, remove unknown modifications and perform other post processings.

        E.g. get 'rt_norm', remove decoys, filter FDR...

        """
        del origin_df  # unused, only here for backwards compatibility in alphapeptdeep

        # TODO: this method is doing a lot!
        self._psm_df[PsmDfCols.NAA] = self._psm_df[PsmDfCols.SEQUENCE].str.len()

        self.normalize_rt_by_raw_name()

        self._psm_df = self._psm_df[~self._psm_df[PsmDfCols.MODS].isna()]

        if PsmDfCols.DECOY in self._psm_df.columns and not self._keep_decoy:
            self._psm_df = self._psm_df[self._psm_df[PsmDfCols.DECOY] == 0]

        reset_precursor_df(self._psm_df)

        if PsmDfCols.PRECURSOR_MZ not in self._psm_df:
            self._psm_df = update_precursor_mz(self._psm_df)

        if (
            PsmDfCols.CCS in self._psm_df.columns
            and PsmDfCols.MOBILITY not in self._psm_df.columns
        ):
            self._psm_df[PsmDfCols.MOBILITY] = mobility.ccs_to_mobility_for_df(
                self._psm_df, PsmDfCols.CCS
            )
        elif (
            PsmDfCols.MOBILITY in self._psm_df.columns
            and PsmDfCols.CCS not in self._psm_df.columns
        ):
            self._psm_df[PsmDfCols.CCS] = mobility.mobility_to_ccs_for_df(
                self._psm_df, PsmDfCols.MOBILITY
            )

    def _filter_fdr(self) -> None:
        """Filter PSMs by FDR.

        If the column is not present in the dataframe, it is ignored.
        """
        if PsmDfCols.FDR in self._psm_df.columns:
            self._psm_df = self._psm_df[
                self._psm_df[PsmDfCols.FDR] <= self._fdr_threshold
            ]

    def normalize_rt_by_raw_name(self) -> None:
        """Normalize RT by raw name."""
        if PsmDfCols.RT not in self._psm_df.columns:
            return
        if PsmDfCols.RT_NORM not in self._psm_df.columns:
            self._normalize_rt()

        if PsmDfCols.RAW_NAME not in self._psm_df.columns:
            return
        for _, df_group in self._psm_df.groupby(PsmDfCols.RAW_NAME):
            self._psm_df.loc[df_group.index, PsmDfCols.RT_NORM] = (
                df_group[PsmDfCols.RT_NORM] / df_group[PsmDfCols.RT_NORM].max()
            )

    def _normalize_rt(self) -> None:
        """Normalize RT values to [0, 1]."""
        if PsmDfCols.RT not in self._psm_df.columns:
            return

        if self._rt_unit == "second":
            self._psm_df[PsmDfCols.RT] = self._psm_df[PsmDfCols.RT] / 60
            if PsmDfCols.RT_START in self._psm_df.columns:
                self._psm_df[PsmDfCols.RT_START] = self._psm_df[PsmDfCols.RT_START] / 60
                self._psm_df[PsmDfCols.RT_STOP] = self._psm_df[PsmDfCols.RT_STOP] / 60

        min_rt = self._psm_df[PsmDfCols.RT].min()
        max_rt = self._psm_df[PsmDfCols.RT].max()
        if min_rt < 0:  # iRT
            min_rt = max(min_rt, _MIN_IRT_VALUE)
            max_rt = min(max_rt, _MAX_IRT_VALUE)
        elif not self._min_max_rt_norm:
            min_rt = 0

        self._psm_df[PsmDfCols.RT_NORM] = (
            (self._psm_df[PsmDfCols.RT] - min_rt) / (max_rt - min_rt)
        ).clip(0, 1)

    def filter_psm_by_modifications(
        self,
        include_mod_set: Optional[Set] = None,
    ) -> None:
        """Only keeps peptides with modifications in `include_mod_list`."""
        if include_mod_set is None:
            include_mod_set = {
                "Oxidation@M",
                "Phospho@S",
                "Phospho@T",
                "Phospho@Y",
                "Acetyl@Protein_N-term",
            }
        self._psm_df[PsmDfCols.MODS] = self._psm_df[PsmDfCols.MODS].apply(
            keep_modifications, mod_set=include_mod_set
        )

        self._psm_df.dropna(subset=[PsmDfCols.MODS], inplace=True)
        self._psm_df.reset_index(drop=True, inplace=True)


class PSMReaderProvider:
    """A factory class to register and get readers for different PSM types."""

    def __init__(self):
        """Initialize PSMReaderProvider."""
        self.reader_dict = {}

    def register_reader(
        self, reader_type: str, reader_class: Type[PSMReaderBase]
    ) -> None:
        """Register a reader by reader_type."""
        self.reader_dict[reader_type.lower()] = reader_class

    def get_reader(
        self,
        reader_type: str,
        *,
        column_mapping: Optional[dict] = None,
        modification_mapping: Optional[dict] = None,
        fdr: float = 0.01,
        keep_decoy: bool = False,
        **kwargs,
    ) -> PSMReaderBase:
        """Get a reader by reader_type."""
        try:
            return self.reader_dict[reader_type.lower()](
                column_mapping=column_mapping,
                modification_mapping=modification_mapping,
                fdr=fdr,
                keep_decoy=keep_decoy,
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
    ) -> PSMReaderBase:
        """Get a reader by a yaml dict."""
        return self.get_reader(**copy.deepcopy(yaml_dict))


psm_reader_provider = PSMReaderProvider()
"""
A factory :class:`PSMReaderProvider` object to register and get readers for different PSM types.
"""
