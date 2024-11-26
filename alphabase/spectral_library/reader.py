"""Module for reading spectral libraries."""

from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from alphabase.constants._const import PEAK_INTENSITY_DTYPE
from alphabase.peptide.mobility import mobility_to_ccs_for_df
from alphabase.psm_reader.keys import LibPsmDfCols, PsmDfCols
from alphabase.psm_reader.maxquant_reader import ModifiedSequenceReader
from alphabase.spectral_library.base import SpecLibBase
from alphabase.utils import _get_delimiter


class LibraryReaderBase(ModifiedSequenceReader, SpecLibBase):
    """Base class for reading spectral libraries."""

    _reader_type = "library_reader_base"
    _add_unimod_to_mod_mapping = True

    def __init__(  # noqa: PLR0913 many arguments in function definition
        self,
        charged_frag_types: List[str] = [
            "b_z1",
            "b_z2",
            "y_z1",
            "y_z2",
            "b_modloss_z1",
            "b_modloss_z2",
            "y_modloss_z1",
            "y_modloss_z2",
        ],
        column_mapping: Optional[dict] = None,
        modification_mapping: Optional[dict] = None,
        fdr: float = 0.01,
        fixed_C57: bool = False,  # noqa: FBT001, FBT002, N803 TODO: make this  *,fixed_c57  (breaking)
        mod_seq_columns: Optional[List[str]] = None,
        rt_unit: Optional[str] = None,
        # library reader-specific:
        precursor_mz_min: float = 400,
        precursor_mz_max: float = 2000,
        decoy: Optional[str] = None,
        **kwargs,
    ):
        """Base class for reading spectral libraries from long format csv files.

        Parameters
        ----------
        charged_frag_types: list of str
            List of fragment types to be used in the spectral library.
            The default is ['b_z1','b_z2','y_z1', 'y_z2', 'b_modloss_z1','b_modloss_z2','y_modloss_z1', 'y_modloss_z2']

        column_mapping: dict
            Dictionary mapping the column names in the csv file to the column names in the spectral library.
            The default is None, which uses the `library_reader_base` column mapping in `psm_reader.yaml`

        modification_mapping: dict
            Dictionary mapping the modification names in the csv file to the modification names in the spectral library.

        fdr: float
            False discovery rate threshold for filtering the spectral library.
            default is 0.01

        fixed_C57: bool
            If true, the search engine will not show `Carbamidomethyl` in the modified sequences.
            By default False

        mod_seq_columns: list of str
            List of column names in the csv file containing the modified sequence.
            By default the mapping is taken from `psm_reader.yaml`

        rt_unit: str
            Unit of the retention time column in the csv file.
            The default is 'irt'

        precursor_mz_min: float
            Minimum precursor m/z value for filtering the spectral library.

        precursor_mz_max: float
            Maximum precursor m/z value for filtering the spectral library.

        decoy: str
            Decoy type for the spectral library.
            Can be either `pseudo_reverse` or `diann`

        **kwargs: dict
            deprecated

        """
        SpecLibBase.__init__(
            self,
            charged_frag_types=charged_frag_types,
            precursor_mz_min=precursor_mz_min,
            precursor_mz_max=precursor_mz_max,
            decoy=decoy,
        )

        ModifiedSequenceReader.__init__(
            self,
            column_mapping=column_mapping,
            modification_mapping=modification_mapping,
            fdr=fdr,
            keep_decoy=False,
            fixed_C57=fixed_C57,
            mod_seq_columns=mod_seq_columns,
            rt_unit=rt_unit,
            **kwargs,
        )

    def _find_key_columns(self, lib_df: pd.DataFrame) -> None:
        """Find and create the key columns for the spectral library.

        Parameters
        ----------
        lib_df: pd.DataFrame
            Dataframe containing the spectral library.

        """
        if LibPsmDfCols.FRAGMENT_LOSS_TYPE not in lib_df.columns:
            lib_df[LibPsmDfCols.FRAGMENT_LOSS_TYPE] = ""

        lib_df.fillna({LibPsmDfCols.FRAGMENT_LOSS_TYPE: ""}, inplace=True)
        lib_df.replace(
            {LibPsmDfCols.FRAGMENT_LOSS_TYPE: "noloss"},
            {LibPsmDfCols.FRAGMENT_LOSS_TYPE: ""},
            inplace=True,
        )

        if PsmDfCols.MODS not in lib_df.columns:
            lib_df[PsmDfCols.MODS] = ""

        if PsmDfCols.MOD_SITES not in lib_df.columns:
            lib_df[PsmDfCols.MOD_SITES] = ""

    def _get_fragment_intensity(self, lib_df: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0912, C901 too many branches, too complex TODO: refactor
        """Create the self._fragment_intensity dataframe from a given spectral library.

        In the process, the input dataframe is converted from long format to a precursor dataframe and returned.

        Parameters
        ----------
        lib_df: pd.DataFrame
            Dataframe containing the spectral library.

        Returns
        -------
        precursor_df: pd.DataFrame
            Dataframe containing the fragment intensity.

        """
        frag_col_dict = dict(
            zip(self.charged_frag_types, range(len(self.charged_frag_types)))
        )

        self._find_key_columns(lib_df)

        # drop all columns which are all NaN as they prohibit grouping
        lib_df = lib_df.dropna(axis=1, how="all")

        precursor_df_list = []

        frag_intens_list = []
        n_aa_list = []

        fragment_columns = [
            LibPsmDfCols.FRAGMENT_MZ,
            LibPsmDfCols.FRAGMENT_TYPE,
            LibPsmDfCols.FRAGMENT_CHARGE,
            LibPsmDfCols.FRAGMENT_SERIES,
            LibPsmDfCols.FRAGMENT_LOSS_TYPE,
            LibPsmDfCols.FRAGMENT_INTENSITY,
        ]

        # by default, all non-fragment columns are used to group the library
        non_fragment_columns = sorted(set(lib_df.columns) - set(fragment_columns))

        for keys, df_group in tqdm(lib_df.groupby(non_fragment_columns)):
            precursor_columns = dict(zip(non_fragment_columns, keys))

            n_aa = len(precursor_columns[PsmDfCols.SEQUENCE])

            intensities = np.zeros(
                (n_aa - 1, len(self.charged_frag_types)),
                dtype=PEAK_INTENSITY_DTYPE,
            )
            for frag_type_, frag_num_, loss_type, frag_charge, intensity in df_group[
                [
                    LibPsmDfCols.FRAGMENT_TYPE,
                    LibPsmDfCols.FRAGMENT_SERIES,
                    LibPsmDfCols.FRAGMENT_LOSS_TYPE,
                    LibPsmDfCols.FRAGMENT_CHARGE,
                    LibPsmDfCols.FRAGMENT_INTENSITY,
                ]
            ].to_numpy():
                if frag_type_ in "abc":
                    frag_num = frag_num_ - 1
                elif frag_type_ in "xyz":
                    frag_num = n_aa - frag_num_ - 1
                else:
                    continue

                if loss_type == "":
                    frag_type = f"{frag_type_}_z{frag_charge}"
                elif loss_type == "H3PO4":
                    frag_type = f"{frag_type_}_modloss_z{frag_charge}"
                elif loss_type == "H2O":
                    frag_type = f"{frag_type_}_H2O_z{frag_charge}"
                elif loss_type == "NH3":
                    frag_type = f"{frag_type_}_NH3_z{frag_charge}"
                elif loss_type == "unknown":  # DiaNN+fragger
                    frag_type = f"{frag_type_}_z{frag_charge}"
                else:
                    continue

                if frag_type not in frag_col_dict:
                    continue
                frag_col_idx = frag_col_dict[frag_type]
                intensities[frag_num, frag_col_idx] = intensity
            max_intensity = np.max(intensities)
            if max_intensity <= 0:
                continue
            normalized_intensities = intensities / max_intensity

            precursor_df_list.append(precursor_columns)
            frag_intens_list.append(normalized_intensities)
            n_aa_list.append(n_aa)

        df = pd.DataFrame(precursor_df_list)

        self._fragment_intensity_df = pd.DataFrame(
            np.concatenate(frag_intens_list), columns=self.charged_frag_types
        )

        indices = np.zeros(len(n_aa_list) + 1, dtype=np.int64)
        indices[1:] = np.array(n_aa_list) - 1
        indices = np.cumsum(indices)

        df[LibPsmDfCols.FRAG_START_IDX] = indices[:-1]
        df[LibPsmDfCols.FRAG_STOP_IDX] = indices[1:]

        return df

    def _load_file(self, filename: str) -> pd.DataFrame:
        """Load the spectral library from a csv file."""
        csv_sep = _get_delimiter(filename)

        return pd.read_csv(
            filename,
            sep=csv_sep,
            keep_default_na=False,
            na_values=[
                "#N/A",
                "#N/A N/A",
                "#NA",
                "-1.#IND",
                "-1.#QNAN",
                "-NaN",
                "-nan",
                "1.#IND",
                "1.#QNAN",
                "<NA>",
                "N/A",
                "NA",
                "NULL",
                "NaN",
                "None",
                "n/a",
                "nan",
                "null",
            ],
        )

    def _post_process(self, origin_df: pd.DataFrame) -> None:
        """Process the spectral library and create the `fragment_intensity`, `fragment_mz` dataframe."""
        del origin_df  # unused, only here for backwards compatibility in alphapeptdeep

        # identify unknown modifications
        len_before = len(self._psm_df)
        self._psm_df = self._psm_df[~self._psm_df[PsmDfCols.MODS].isna()]
        len_after = len(self._psm_df)

        if len_before != len_after:
            pass  # TODO: this literally does nothing

        if PsmDfCols.NAA not in self._psm_df.columns:
            self._psm_df[PsmDfCols.NAA] = self._psm_df[PsmDfCols.SEQUENCE].str.len()

        self._psm_df = self._get_fragment_intensity(self._psm_df)

        self.normalize_rt_by_raw_name()

        if PsmDfCols.MOBILITY in self._psm_df.columns:
            self._psm_df[PsmDfCols.CCS] = mobility_to_ccs_for_df(
                self._psm_df, PsmDfCols.MOBILITY
            )

        self._psm_df.drop(PsmDfCols.MODIFIED_SEQUENCE, axis=1, inplace=True)
        self._precursor_df = self._psm_df

        self.calc_fragment_mz_df()


# legacy
SWATHLibraryReader = LibraryReaderBase
