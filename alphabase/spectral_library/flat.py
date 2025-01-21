import warnings
from typing import Union

import pandas as pd

from alphabase.io.hdf import HDF_File
from alphabase.peptide.fragment import (
    create_dense_matrices,
    filter_valid_charged_frag_types,
    flatten_fragments,
    remove_unused_fragments,
    sort_charged_frag_types,
)
from alphabase.spectral_library.base import SpecLibBase, get_available_columns


class SpecLibFlat(SpecLibBase):
    """
    Flatten the spectral library (SpecLibBase) by using :meth:`parse_base_library`.

    Attributes
    ----------
    custom_fragment_df_columns : list of str
        'mz' and 'intensity' columns are required in :attr:`fragment_df`,
        others could be customized.
        It can include ['type','number','position','charge','loss_type'].

    min_fragment_intensity : float
        minimal intensity to keep in :attr:`fragment_df`.

    keep_top_k_fragments : float
        top k highest peaks to keep in :attr:`fragment_df`.

    """

    key_numeric_columns = SpecLibBase.key_numeric_columns + [
        "flat_frag_start_idx",
        "flat_frag_stop_idx",
    ]
    """
    :obj:`SpecLibBase.key_numeric_columns <alphabase.spectral_library.base.SpecLibBase.key_numeric_columns>`
    + `['flat_frag_start_idx','flat_frag_stop_idx']`.
    """

    def __init__(
        self,
        charged_frag_types: list = ["b_z1", "b_z2", "y_z1", "y_z2"],
        min_fragment_intensity: float = 0.001,
        keep_top_k_fragments: int = 1000,
        custom_fragment_df_columns: list = [
            "type",
            "number",
            "position",
            "charge",
            "loss_type",
        ],
        **kwargs,
    ):
        """
        Parameters
        ----------
        min_fragment_intensity : float, optional
            minimal intensity to keep, by default 0.001

        keep_top_k_fragments : int, optional
            top k highest peaks to keep, by default 1000

        custom_fragment_df_columns : list, optional
            See :attr:`custom_fragment_df_columns`,
            defaults to ['type','number','position','charge','loss_type']
        """
        super().__init__(charged_frag_types=charged_frag_types)
        self.min_fragment_intensity = min_fragment_intensity
        self.keep_top_k_fragments = keep_top_k_fragments

        self.custom_fragment_df_columns = custom_fragment_df_columns

    @property
    def fragment_df(self) -> pd.DataFrame:
        """The flat fragment dataframe with columns
        (['mz', 'intensity'] + :attr:`custom_fragment_df_columns`.)
        """
        return self._fragment_df

    @property
    def protein_df(self) -> pd.DataFrame:
        """Protein dataframe"""
        return self._protein_df

    def remove_unused_fragments(self):
        """Remove unused fragments from fragment_df.
        This method is inherited from :class:`SpecLibBase` and has not been implemented for a flat library.
        """
        self._precursor_df, (self._fragment_df,) = remove_unused_fragments(
            self._precursor_df,
            (self._fragment_df,),
            frag_start_col="flat_frag_start_idx",
            frag_stop_col="flat_frag_stop_idx",
        )

    def parse_base_library(
        self,
        library: SpecLibBase,
        keep_original_frag_dfs: bool = False,
        copy_precursor_df: bool = False,
        **kwargs,
    ):
        """
        Flatten an library object of SpecLibBase or its inherited class.
        This method will generate :attr:`precursor_df` and :attr:`fragment_df`
        The fragments in fragment_df can be located by
        `flat_frag_start_idx` and `flat_frag_stop_idx` in precursor_df.

        Parameters
        ----------
        library : SpecLibBase
            A library object with attributes
            `precursor_df`, `fragment_mz_df` and `fragment_intensity_df`.

        keep_original_frag_dfs : bool, default True
            If `fragment_mz_df` and `fragment_intensity_df` are
            kept in this library.

        copy_precursor_df : bool, default False
            If True, make a copy of `precursor_df` from `library`,
            otherwise `flat_frag_start_idx` and `flat_frag_stop_idx`
            columns will also append to the `library`.
        """
        self._precursor_df, self._fragment_df = flatten_fragments(
            library.precursor_df.copy() if copy_precursor_df else library.precursor_df,
            library.fragment_mz_df,
            library.fragment_intensity_df,
            min_fragment_intensity=self.min_fragment_intensity,
            keep_top_k_fragments=self.keep_top_k_fragments,
            custom_columns=self.custom_fragment_df_columns,
            **kwargs,
        )

        if hasattr(library, "protein_df"):
            self._protein_df = library.protein_df
        else:
            self._protein_df = pd.DataFrame()

        if keep_original_frag_dfs:
            self.charged_frag_types = library.fragment_mz_df.columns.values
            for dense_frag_df in library.available_dense_fragment_dfs():
                setattr(self, dense_frag_df, getattr(library, dense_frag_df))

            warnings.warn(
                "The SpecLibFlat object will have a strictly flat representation in the future. keep_original_frag_dfs=True will be deprecated.",
                DeprecationWarning,
            )

    def save_hdf(self, hdf_file: str):
        """Save library dataframes into hdf_file.
        For `self.precursor_df`, this method will save it into two hdf groups:
        hdf_file: `library/precursor_df` and `library/mod_seq_df`.

        `library/precursor_df` contains all essential numberic columns those
        can be loaded faster from hdf file into memory:
        `['precursor_mz', 'charge', 'mod_seq_hash', 'mod_seq_charge_hash',
        'frag_start_idx', 'frag_stop_idx', 'flat_frag_start_idx', 'flat_frag_stop_idx',
        'decoy', 'rt_pred', 'ccs_pred', 'mobility_pred', 'miss_cleave', 'nAA',
        'isotope_mz_m1', 'isotope_intensity_m1', ...]`

        `library/mod_seq_df` contains all string columns and the other
        not essential columns:
        'sequence','mods','mod_sites', ['proteins', 'genes']...
        as well as 'mod_seq_hash', 'mod_seq_charge_hash' columns to map
        back to `precursor_df`

        Parameters
        ----------
        hdf_file : str
            the hdf file path to save

        """
        super().save_hdf(hdf_file)
        _hdf = HDF_File(hdf_file, read_only=False, truncate=True, delete_existing=False)
        _hdf.library.fragment_df = self.fragment_df
        _hdf.library.protein_df = self.protein_df
        _hdf.library.fragment_mz_df = self.fragment_mz_df
        _hdf.library.fragment_intensity_df = self.fragment_intensity_df

    def load_hdf(
        self,
        hdf_file: str,
        load_mod_seq: bool = False,
        infer_charged_frag_types: bool = True,
    ):
        """Load the hdf library from hdf_file

        Parameters
        ----------
        hdf_file : str
            hdf library path to load

        load_mod_seq : bool, optional
            if also load mod_seq_df.
            Defaults to False.

        infer_charged_frag_types : bool, optional
            if True, infer the charged fragment types as defined in the hdf file, defaults to True.
            This is the default as users most likely don't know the charged fragment types in the hdf file.
            If set to False, only charged frag types defined in `SpecLibBase.charged_frag_types` will be loaded.

        """
        super().load_hdf(hdf_file, load_mod_seq=load_mod_seq)
        _hdf = HDF_File(
            hdf_file,
        )
        self._fragment_df = _hdf.library.fragment_df.values
        self._protein_df = _hdf.library.protein_df.values

        if infer_charged_frag_types:
            self.charged_frag_types = sort_charged_frag_types(
                filter_valid_charged_frag_types(_hdf.library.fragment_mz_df.columns)
            )

        _fragment_intensity_df = _hdf.library.fragment_intensity_df.values
        self._fragment_intensity_df = _fragment_intensity_df[
            get_available_columns(_fragment_intensity_df, self.charged_frag_types)
        ]

        _fragment_mz_df = _hdf.library.fragment_mz_df.values
        self._fragment_mz_df = _fragment_mz_df[
            get_available_columns(_fragment_mz_df, self.charged_frag_types)
        ]

    def get_full_charged_types(self, frag_df: pd.DataFrame) -> list:
        """
        Infer the full set of charged fragment types from the fragment dataframe
        by full we mean a complete set of fragment types for each charge
        so if we have a fragment b_z1 we should also have a fragment y_z1 and vice versa

        Parameters
        ----------
        frag_df : pd.DataFrame
            The fragment dataframe

        Returns
        -------
        charged_frag_types : list
            The full set of charged fragment types in the form of a list of strings such as ['a_z1','b_z1','c_z1','x_z1','y_z1','z_z1']

        """
        warnings.warn(
            "The get_full_charged_types method is deprecated. Use get_charged_frag_types instead.",
            DeprecationWarning,
        )

        unique_charge_type_pairs = frag_df[
            ["type", "loss_type", "charge"]
        ].drop_duplicates()
        # Fragtypes from ascii to char
        self.frag_types_as_char = {
            i: chr(i) for i in unique_charge_type_pairs["type"].unique()
        }

        charged_frag_types = set()
        # Now if we have a fragment type that is a,b,c we should have the corresponding x,y,z

        corresponding = {"a": "x", "b": "y", "c": "z", "x": "a", "y": "b", "z": "c"}
        loss_number_to_type = {0: "", 18: "_H2O", 17: "_NH3", 98: "_modloss"}
        for type, loss, max_charge in unique_charge_type_pairs.values:
            for possible_charge in range(1, max_charge + 1):
                # Add the string for this pair
                charged_frag_types.add(
                    f"{self.frag_types_as_char[type]}{loss_number_to_type[loss]}_z{possible_charge}"
                )
                # Add the string for the corresponding pair
                charged_frag_types.add(
                    f"{corresponding[self.frag_types_as_char[type]]}{loss_number_to_type[loss]}_z{possible_charge}"
                )
        return list(charged_frag_types)

    def calc_dense_fragments(
        self,
        additional_columns: Union[list, None] = None,
        charged_frag_types: Union[list, None] = None,
    ) -> None:
        """
        Create a hybrid SpecLibFlat which has both flat and dense fragment representations.
        Converts the flat fragment representation to dense matrices and stores them in the object.

        Creates fragment_mz_df (using calculated m/z values) and fragment_intensity_df by default.
        For each additional column specified (e.g., 'intensity'), creates a corresponding
        _fragment_<column>_df matrix. Including 'mz' in additional_columns will use observed
        rather than calculated m/z values.

        Fragment types can be specified explicitly or inherited from self.charged_frag_types.
        Only fragments matching these types will be included in the dense matrices. Each fragment
        type (e.g., 'b_z1', 'y_z2') becomes a column in the resulting dense matrices.

        Updates the precursor_df with new frag_start_idx and frag_stop_idx columns for the
        dense representation.

        Parameters
        ----------
        additional_columns : Union[list, None], optional
            Additional fragment columns to convert to dense format, defaults to ['intensity']
        charged_frag_types : Union[list, None], optional
            Fragment types to include in dense format, defaults to self.charged_frag_types

        Returns
        -------
        None
            Modifies the SpecLibFlat object in place
        """

        if charged_frag_types is None:
            charged_frag_types = self.charged_frag_types

        if additional_columns is None:
            additional_columns = ["intensity"]

        df_collection, frag_start_idx, frag_stop_idx = create_dense_matrices(
            self._precursor_df,
            self._fragment_df,
            charged_frag_types,
            flat_columns=additional_columns,
        )

        for col, df in df_collection.items():
            setattr(self, f"_fragment_{col}_df", df)

        self.precursor_df["frag_start_idx"] = frag_start_idx
        self.precursor_df["frag_stop_idx"] = frag_stop_idx

    def to_speclib_base(
        self,
        flat_columns: Union[list, None] = None,
        charged_frag_types: Union[list, None] = None,
    ) -> SpecLibBase:
        """
        Convert the flat library to a new SpecLibBase object with dense fragment matrices.

        Creates a new SpecLibBase containing fragment_mz_df (using calculated m/z values).
        Flat columns like 'intensity' are transformed into dense matrices as fragment_intensity_df.
        For all columns specified in flat_columns, a corresponding _fragment_<column>_df matrix is created and assigned to the new SpecLibBase object.

        Warning
        -------
        If the column 'mz' is added to flat_columns, it will override the calculated m/z values in fragment_mz_df.
        To mitigate this behavior and get observed as calculated m/z values, rename the flat mz column to 'mz_observed' before calling to_speclib_base.

        Fragment types can be specified explicitly or inherited from self.charged_frag_types.
        Only fragments matching these types will be included in the dense matrices. Each fragment
        type (e.g., 'b_z1', 'y_z2') becomes a column in the resulting dense matrices.

        The precursor_df is copied and updated with new dense fragment indices, removing any
        flat-specific columns (flat_frag_start_idx, flat_frag_stop_idx).

        Parameters
        ----------
        flat_columns : Union[list, None], optional
            Fragment columns from the flat representation to convert to dense format, defaults to ['intensity']

        charged_frag_types : Union[list, None], optional
            Fragment types to include in dense format, defaults to self.charged_frag_types

        Returns
        -------
        SpecLibBase
            A new SpecLibBase object with dense fragment representations
        """
        # Create SpecLibBase object
        speclib_base = SpecLibBase()
        speclib_base._precursor_df = self._precursor_df.copy()

        if charged_frag_types is None:
            charged_frag_types = self.charged_frag_types

        if flat_columns is None:
            flat_columns = ["intensity"]

        speclib_base.charged_frag_types = charged_frag_types

        df_collection, frag_start_idx, frag_stop_idx = create_dense_matrices(
            speclib_base._precursor_df,
            self._fragment_df,
            speclib_base.charged_frag_types,
            flat_columns=flat_columns,
        )

        speclib_base.precursor_df["frag_start_idx"] = frag_start_idx
        speclib_base.precursor_df["frag_stop_idx"] = frag_stop_idx

        for col, df in df_collection.items():
            setattr(speclib_base, f"_fragment_{col}_df", df)

        # Drop flat indices from precursor_df if they exist
        speclib_base._precursor_df = speclib_base._precursor_df.drop(
            ["flat_frag_start_idx", "flat_frag_stop_idx"], axis=1, errors="ignore"
        )

        return speclib_base

    def to_SpecLibBase(self):
        # raise a deprecation warning
        warnings.warn(
            "The to_SpecLibBase method is deprecated. Use to_speclib_base instead.",
            DeprecationWarning,
        )
        return self.to_speclib_base()
