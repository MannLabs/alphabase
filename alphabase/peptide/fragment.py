import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import numba as nb
import numpy as np
import pandas as pd

from alphabase.constants._const import PEAK_INTENSITY_DTYPE, PEAK_MZ_DTYPE
from alphabase.constants.atom import (
    MASS_PROTON,
    calc_mass_from_formula,
)
from alphabase.constants.modification import calc_modloss_mass
from alphabase.peptide.mass_calc import calc_b_y_and_peptide_masses_for_same_len_seqs
from alphabase.peptide.precursor import (
    is_precursor_refined,
    refine_precursor_df,
)
from alphabase.utils import DeprecatedDict


class Direction:
    """String constants defining fragment directions."""

    FORWARD = "forward"
    REVERSE = "reverse"


DIRECTION_MAPPING = {Direction.FORWARD: 1, Direction.REVERSE: -1}

DIRECTION_MAPPING_INV = {v: k for k, v in DIRECTION_MAPPING.items()}


class Loss:
    """String constants defining fragment losses."""

    MODLOSS = "modloss"
    H2O = "H2O"
    NH3 = "NH3"
    LOSSH = "lossH"
    ADDH = "addH"
    NONE = ""


LOSS_MAPPING = {
    # we use 98 because it is similar to the molecular weight of a phosphate group
    Loss.MODLOSS: 98,
    # we use 18 because it is similar to the molecular weight of a water molecule
    Loss.H2O: 18,
    # we use 17 because it is similar to the molecular weight of an ammonia molecule
    Loss.NH3: 17,
    # we use 1 because it is similar to the molecular weight of a hydrogen atom
    Loss.LOSSH: 1,
    # there is no -1 so we use 2
    Loss.ADDH: 2,
    Loss.NONE: 0,
}

LOSS_MAPPING_INV = {v: k for k, v in LOSS_MAPPING.items()}


class Series:
    """String constants defining fragment series types."""

    A = "a"
    B = "b"
    C = "c"
    X = "x"
    Y = "y"
    Z = "z"


SERIES_MAPPING = {
    # ascii value for a, b, c, x, y, z
    Series.A: 97,
    Series.B: 98,
    Series.C: 99,
    Series.X: 120,
    Series.Y: 121,
    Series.Z: 122,
}

SERIES_MAPPING_INV = {v: k for k, v in SERIES_MAPPING.items()}


@dataclass(frozen=True)
class FragmentType:
    """
    Class which represents a constant fragment type.

    Parameters
    ----------
    name : str
        Name of the fragment type
    ref_ion : str
        Reference ion of the fragment type
    delta_formula : str
        Chemical formula representing the mass difference from the reference ion
    delta_mass : float
        Mass difference calculated from delta_formula, set during initialization
    modloss : bool
        Whether the fragment type has a modification neutral loss
    series_id : int
        Series ID of the fragment type (e.g., 97=a, 98=b, 99=c, 120=x, 121=y, 122=z), from SERIES_MAPPING
    loss_id : int
        Loss type ID of the fragment (e.g., 0=none, 1=lossH, 2=addH, 17=NH3, 18=H2O, 98=modloss), from LOSS_MAPPING
    direction_id : int
        Direction ID of the fragment type (forward=1, reverse=-1), from DIRECTION_MAPPING

    Attributes
    ----------
    delta_mass : float
        Mass difference calculated from delta_formula, set during initialization
    """

    name: str
    ref_ion: str
    delta_formula: str
    delta_mass: float = field(init=False)
    modloss: bool
    series_id: int
    loss_id: int
    direction_id: int

    def __post_init__(self):
        """Set delta_mass after initialization using delta_formula"""
        object.__setattr__(
            self, "delta_mass", calc_mass_from_formula(self.delta_formula)
        )


# constant which contains all valid fragment types
FRAGMENT_TYPES = {
    "a": FragmentType(
        name="a",
        ref_ion="b",
        delta_formula="C(-1)O(-1)",
        modloss=False,
        series_id=SERIES_MAPPING[Series.A],
        loss_id=LOSS_MAPPING[Loss.NONE],
        direction_id=DIRECTION_MAPPING[Direction.FORWARD],
    ),
    "b": FragmentType(
        name="b",
        ref_ion="b",
        delta_formula="",
        modloss=False,
        series_id=SERIES_MAPPING[Series.B],
        loss_id=LOSS_MAPPING[Loss.NONE],
        direction_id=DIRECTION_MAPPING[Direction.FORWARD],
    ),
    "c": FragmentType(
        name="c",
        ref_ion="b",
        delta_formula="N(1)H(3)",
        modloss=False,
        series_id=SERIES_MAPPING[Series.C],
        loss_id=LOSS_MAPPING[Loss.NONE],
        direction_id=DIRECTION_MAPPING[Direction.FORWARD],
    ),
    "x": FragmentType(
        name="x",
        ref_ion="y",
        delta_formula="C(1)O(1)H(-2)",
        modloss=False,
        series_id=SERIES_MAPPING[Series.X],
        loss_id=LOSS_MAPPING[Loss.NONE],
        direction_id=DIRECTION_MAPPING[Direction.REVERSE],
    ),
    "y": FragmentType(
        name="y",
        ref_ion="y",
        delta_formula="",
        modloss=False,
        series_id=SERIES_MAPPING[Series.Y],
        loss_id=LOSS_MAPPING[Loss.NONE],
        direction_id=DIRECTION_MAPPING[Direction.REVERSE],
    ),
    "z": FragmentType(
        name="z",
        ref_ion="y",
        delta_formula="N(-1)H(-2)",
        modloss=False,
        series_id=SERIES_MAPPING[Series.Z],
        loss_id=LOSS_MAPPING[Loss.NONE],
        direction_id=DIRECTION_MAPPING[Direction.REVERSE],
    ),
    "b_modloss": FragmentType(
        name="b_modloss",
        ref_ion="b",
        delta_formula="N(1)H(3)",
        modloss=True,
        series_id=SERIES_MAPPING[Series.B],
        loss_id=LOSS_MAPPING[Loss.MODLOSS],
        direction_id=DIRECTION_MAPPING[Direction.FORWARD],
    ),
    "b_H2O": FragmentType(
        name="b_H2O",
        ref_ion="b",
        delta_formula="H(-2)O(-1)",
        modloss=False,
        series_id=SERIES_MAPPING[Series.B],
        loss_id=LOSS_MAPPING[Loss.H2O],
        direction_id=DIRECTION_MAPPING[Direction.FORWARD],
    ),
    "b_NH3": FragmentType(
        name="b_NH3",
        ref_ion="b",
        delta_formula="N(-1)H(-3)",
        modloss=False,
        series_id=SERIES_MAPPING[Series.B],
        loss_id=LOSS_MAPPING[Loss.NH3],
        direction_id=DIRECTION_MAPPING[Direction.FORWARD],
    ),
    "c_lossH": FragmentType(
        name="c_lossH",
        ref_ion="b",
        delta_formula="N(1)H(2)",
        modloss=False,
        series_id=SERIES_MAPPING[Series.C],
        loss_id=LOSS_MAPPING[Loss.LOSSH],
        direction_id=DIRECTION_MAPPING[Direction.FORWARD],
    ),
    "y_modloss": FragmentType(
        name="y_modloss",
        ref_ion="y",
        delta_formula="N(-1)H(-2)",
        modloss=True,
        series_id=SERIES_MAPPING[Series.Y],
        loss_id=LOSS_MAPPING[Loss.MODLOSS],
        direction_id=DIRECTION_MAPPING[Direction.REVERSE],
    ),
    "y_H2O": FragmentType(
        name="y_H2O",
        ref_ion="y",
        delta_formula="H(-2)O(-1)",
        modloss=False,
        series_id=SERIES_MAPPING[Series.Y],
        loss_id=LOSS_MAPPING[Loss.H2O],
        direction_id=DIRECTION_MAPPING[Direction.REVERSE],
    ),
    "y_NH3": FragmentType(
        name="y_NH3",
        ref_ion="y",
        delta_formula="N(-1)H(-3)",
        modloss=False,
        series_id=SERIES_MAPPING[Series.Y],
        loss_id=LOSS_MAPPING[Loss.NH3],
        direction_id=DIRECTION_MAPPING[Direction.REVERSE],
    ),
    "z_addH": FragmentType(
        name="z_addH",
        ref_ion="y",
        delta_formula="N(-1)H(-1)",
        modloss=False,
        series_id=SERIES_MAPPING[Series.Z],
        loss_id=LOSS_MAPPING[Loss.ADDH],
        direction_id=DIRECTION_MAPPING[Direction.REVERSE],
    ),
}

FRAGMENT_CHARGE_SEPARATOR = "_z"

# TODO: remove this dictionary
frag_type_representation_dict = DeprecatedDict(
    {
        "c": "b+N(1)H(3)",
        "z": "y+N(-1)H(-2)",
        "a": "b+C(-1)O(-1)",
        "x": "y+C(1)O(1)H(-2)",
        "b_H2O": "b+H(-2)O(-1)",
        "y_H2O": "y+H(-2)O(-1)",
        "b_NH3": "b+N(-1)H(-3)",
        "y_NH3": "y+N(-1)H(-3)",
        "c_lossH": "b+N(1)H(2)",
        "z_addH": "y+N(-1)H(-1)",
    },
    warning_message="frag_type_representation_dict is deprecated and will be removed in the future version",
)

# TODO: remove this dictionary
frag_mass_from_ref_ion_dict = DeprecatedDict(
    {},
    warning_message="frag_mass_from_ref_ion_dict is deprecated and will be removed in a future version",
)


# TODO: remove this function
def add_new_frag_type(frag_type: str, representation: str):
    """Add new modifications into :data:`frag_type_representation_dict`
    and update :data:`frag_mass_from_ref_ion_dict`.

    Parameters
    ----------
    frag_type : str
        New fragment type
    representation : str
        The representation similar to :data:`frag_type_representation_dict`
    """
    frag_type_representation_dict[frag_type] = representation
    ref_ion, formula = representation.split("+")
    frag_mass_from_ref_ion_dict[frag_type] = dict(
        ref_ion=ref_ion, add_mass=calc_mass_from_formula(formula)
    )


# TODO: remove this function
def parse_all_frag_type_representation():
    for frag, representation in frag_type_representation_dict.items():
        add_new_frag_type(frag, representation)


parse_all_frag_type_representation()


def sort_charged_frag_types(charged_frag_types: List[str]) -> List[str]:
    """charged frag types are sorted by (no-loss, loss) and then alphabetically"""
    has_loss = [
        f.replace(FRAGMENT_CHARGE_SEPARATOR, "").count("_") > 0
        for f in charged_frag_types
    ]
    no_loss = [f for f, hl in zip(charged_frag_types, has_loss) if not hl]
    loss = [f for f, hl in zip(charged_frag_types, has_loss) if hl]
    return sorted(no_loss) + sorted(loss)


def get_charged_frag_types(
    frag_types: List[str], max_frag_charge: int = 2
) -> List[str]:
    """
    Calculate the combination of fragment types and charge states.
    Returns a sorted list of charged fragment types.

    Parameters
    ----------
    frag_types : List[str]
        e.g. ['b','y','b_modloss','y_modloss']

    max_frag_charge : int
        max fragment charge. (default: 2)

    Returns
    -------
    List[str]
        charged fragment types

    Examples
    --------
    >>> frag_types=['b','y','b_modloss','y_modloss']
    >>> get_charged_frag_types(frag_types, 2)
    ['b_z1','b_z2','y_z1','y_z2','b_modloss_z1','b_modloss_z2','y_modloss_z1','y_modloss_z2']
    """
    charged_frag_types = []
    for frag_type in frag_types:
        if frag_type in FRAGMENT_TYPES:
            for charge in range(1, max_frag_charge + 1):
                charged_frag_types.append(f"{frag_type}_z{charge}")
        else:
            raise ValueError(f"Fragment type {frag_type} is currently not supported")
    return sort_charged_frag_types(charged_frag_types)


def filter_valid_charged_frag_types(
    charged_frag_types: List[str],
) -> List[str]:
    """
    Filters a list of charged fragment types and returns only the valid ones.
    A valid charged fragment type must:
    1. Follow the format: {fragment_type}_z{charge} (e.g. 'b_z1', 'y_modloss_z2')
    2. Use a fragment type that exists in FRAGMENT_TYPES
    3. Have a strictly positive integer charge

    Parameters
    ----------
    charged_frag_types : List[str]
        List of charged fragment types to filter (e.g. ['b_z1', 'y_z2', 'invalid_z1', 'b_modloss_z2'])

    Returns
    -------
    List[str]
        List containing only the valid charged fragment types, (e.g. ['b', 'y', 'b_modloss'])
    """
    valid_types = []

    for charged_frag_type in charged_frag_types:
        try:
            _ = parse_charged_frag_type(charged_frag_type)

            valid_types.append(charged_frag_type)
        except ValueError as e:
            warnings.warn(str(e))
            continue

    return valid_types


def parse_charged_frag_type(charged_frag_type: str) -> Tuple[str, int]:
    """
    Oppsite to `get_charged_frag_types`.

    Parameters
    ----------
    charged_frag_type : str
        e.g. 'y_z1', 'b_modloss_z1'

    Returns
    -------
    tuple
        str. Fragment type, e.g. 'b','y'

        int. Charge state

    Raises
    ------
    ValueError
        If charge state is not given or not a strictly positive integer or if fragment type is not supported
    """

    if charged_frag_type.count(FRAGMENT_CHARGE_SEPARATOR) != 1:
        raise ValueError(
            "Only charged fragment types are supported. Please add charge state to the fragment type, "
            f"using {FRAGMENT_CHARGE_SEPARATOR} as separator. e.g. 'b{FRAGMENT_CHARGE_SEPARATOR}1'"
        )

    fragment_type, charge = charged_frag_type.split(FRAGMENT_CHARGE_SEPARATOR)

    # Check if charge is a valid integer string (no decimals)
    if not charge.isdigit() or not (charge_int := int(charge)) > 0:
        raise ValueError(
            f"Charge state must be a positive integer, got '{charge}' from fragment type '{charged_frag_type}'"
        )

    if fragment_type not in FRAGMENT_TYPES:
        raise ValueError(f"Fragment type {fragment_type} is currently not supported")

    return fragment_type, charge_int


def init_zero_fragment_dataframe(
    peplen_array: np.ndarray, charged_frag_types: List[str], dtype=PEAK_MZ_DTYPE
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Initialize a zero dataframe based on peptide length
    (nAA) array (peplen_array) and charge_frag_types (column number).
    The row number of returned dataframe is np.sum(peplen_array-1).

    Parameters
    ----------
    peplen_array : np.ndarray
        peptide lengths for the fragment dataframe

    charged_frag_types : List[str]
        `['b_z1','b_z2','y_z1','y_z2','b_modloss_z1','y_H2O_z1'...]`

    Returns
    -------
    tuple
        pd.DataFrame, `fragment_df` with zero values

        np.ndarray (int64), the start indices point to the `fragment_df` for each peptide

        np.ndarray (int64), the end indices point to the `fragment_df` for each peptide
    """
    indices = np.zeros(len(peplen_array) + 1, dtype=np.int64)
    indices[1:] = peplen_array - 1
    indices = np.cumsum(indices)
    fragment_df = pd.DataFrame(
        np.zeros((indices[-1], len(charged_frag_types)), dtype=dtype),
        columns=charged_frag_types,
    )
    return fragment_df, indices[:-1], indices[1:]


def init_fragment_dataframe_from_other(
    reference_fragment_df: pd.DataFrame, dtype=PEAK_MZ_DTYPE
):
    """
    Init zero fragment dataframe from the `reference_fragment_df` (same rows and same columns)
    """
    return pd.DataFrame(
        np.zeros_like(reference_fragment_df.values, dtype=dtype),
        columns=reference_fragment_df.columns,
    )


def init_fragment_by_precursor_dataframe(
    precursor_df,
    charged_frag_types: List[str],
    *,
    reference_fragment_df: pd.DataFrame = None,
    dtype: np.dtype = PEAK_MZ_DTYPE,
    inplace_in_reference: bool = False,
):
    """
    Init zero fragment dataframe for the `precursor_df`. If
    the `reference_fragment_df` is provided, the result dataframe's
    length will be the same as reference_fragment_df. Otherwise it
    generates the dataframe from scratch.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        precursors to generate fragment masses,
        if `precursor_df` contains the 'frag_start_idx' column,
        it is better to provide `reference_fragment_df` as
        `precursor_df.frag_start_idx` and `precursor.frag_stop_idx`
        point to the indices in `reference_fragment_df`

    charged_frag_types : List
        `['b_z1','b_z2','y_z1','y_z2','b_modloss_z1','y_H2O_z1'...]`

    reference_fragment_df : pd.DataFrame
        init zero fragment_mz_df based
        on this reference. If None, fragment_mz_df will be
        initialized by :func:`alphabase.peptide.fragment.init_zero_fragment_dataframe`.
        Defaults to None.

    dtype: np.dtype
        dtype of fragment mz values, Defaults to :data:`PEAK_MZ_DTYPE`.

    inplace_in_reference : bool, optional
        if calculate the fragment mz
        inplace in the reference_fragment_df (default: False)

    Returns
    -------
    pd.DataFrame
        zero `fragment_df` with given `charged_frag_types` columns
    """
    if "frag_start_idx" not in precursor_df.columns:
        (fragment_df, start_indices, end_indices) = init_zero_fragment_dataframe(
            precursor_df.nAA.values, charged_frag_types, dtype=dtype
        )
        precursor_df["frag_start_idx"] = start_indices
        precursor_df["frag_stop_idx"] = end_indices
    else:
        if reference_fragment_df is None:
            # raise ValueError(
            #     "`precursor_df` contains 'frag_start_idx' column, "\
            #     "please provide `reference_fragment_df` argument"
            # )
            fragment_df = pd.DataFrame(
                np.zeros(
                    (precursor_df.frag_stop_idx.max(), len(charged_frag_types)),
                    dtype=dtype,
                ),
                columns=charged_frag_types,
            )
        else:
            if inplace_in_reference:
                fragment_df = reference_fragment_df[
                    [
                        _fr
                        for _fr in charged_frag_types
                        if _fr in reference_fragment_df.columns
                    ]
                ]
            else:
                fragment_df = pd.DataFrame(
                    np.zeros(
                        (len(reference_fragment_df), len(charged_frag_types)),
                        dtype=dtype,
                    ),
                    columns=charged_frag_types,
                )
    return fragment_df


def update_sliced_fragment_dataframe(
    fragment_df: pd.DataFrame,
    fragment_df_vals: np.ndarray,
    values: np.ndarray,
    frag_start_end_list: List[Tuple[int, int]],
    charged_frag_types: List[str] = None,
):
    """
    Set the values of the slices `frag_start_end_list=[(start,end),(start,end),...]`
    of fragment_df.

    Parameters
    ----------
    fragment_df : pd.DataFrame
        fragment dataframe to set the values

    fragment_df_vals : np.ndarray
        The `fragment_df.to_numpy(copy=True)`, to prevent readonly assignment.

    values : np.ndarray
        values to set

    frag_start_end_list : List[Tuple[int,int]]
        e.g. `[(start,end),(start,end),...]`

    charged_frag_types : List[str], optional
        e.g. `['b_z1','b_z2','y_z1','y_z2']`.
        If None, the columns of values should be the same as fragment_df's columns.
        It is much faster if charged_frag_types is None as we use numpy slicing,
        otherwise we use pd.loc (much slower).
        Defaults to None.
    """
    frag_slice_list = [slice(start, end) for start, end in frag_start_end_list]
    frag_slices = np.r_[tuple(frag_slice_list)]
    if charged_frag_types is None or len(charged_frag_types) == 0:
        fragment_df_vals[frag_slices, :] = values.astype(fragment_df_vals.dtype)
    else:
        charged_frag_idxes = [
            fragment_df.columns.get_loc(c) for c in charged_frag_types
        ]
        fragment_df.iloc[frag_slices, charged_frag_idxes] = values.astype(
            fragment_df_vals.dtype
        )
        fragment_df_vals[frag_slices] = fragment_df.values[frag_slices]


def get_sliced_fragment_dataframe(
    fragment_df: pd.DataFrame,
    frag_start_end_list: Union[List, np.ndarray],
    charged_frag_types: List = None,
) -> pd.DataFrame:
    """
    Get the sliced fragment_df from `frag_start_end_list=[(start,end),(start,end),...]`.

    Parameters
    ----------
    fragment_df : pd.DataFrame
        fragment dataframe to get values

    frag_start_end_list : Union
        List[Tuple[int,int]], e.g. `[(start,end),(start,end),...]` or np.ndarray

    charged_frag_types : List[str]
        e.g. `['b_z1','b_z2','y_z1','y_z2']`.
        if None, all columns will be considered

    Returns
    -------
    pd.DataFrame

        sliced fragment_df. If `charged_frag_types` is None,
        return fragment_df with all columns
    """
    frag_slice_list = [slice(start, end) for start, end in frag_start_end_list]
    frag_slices = np.r_[tuple(frag_slice_list)]
    if charged_frag_types is None or len(charged_frag_types) == 0:
        charged_frag_idxes = slice(None)
    else:
        charged_frag_idxes = [
            fragment_df.columns.get_loc(c) for c in charged_frag_types
        ]
    return fragment_df.iloc[frag_slices, charged_frag_idxes]


def concat_precursor_fragment_dataframes(
    precursor_df_list: List[pd.DataFrame],
    fragment_df_list: List[pd.DataFrame],
    *other_fragment_df_lists,
) -> Tuple[pd.DataFrame, ...]:
    """
    Since fragment_df is indexed by precursor_df, when we concatenate multiple
    fragment_df, the indexed positions will change for in precursor_dfs,
    this function keeps the correct indexed positions of precursor_df when
    concatenating multiple fragment_df dataframes.

    Parameters
    ----------
    precursor_df_list : List[pd.DataFrame]
        precursor dataframe list to concatenate

    fragment_df_list : List[pd.DataFrame]
        fragment dataframe list to concatenate

    other_fragment_df_lists
        arbitray other fragment dataframe list to concatenate,
        e.g. fragment_mass_df, fragment_inten_df, ...

    Returns
    -------
    Tuple[pd.DataFrame,...]
        concatenated precursor_df, fragment_df, other_fragment_dfs ...
    """
    fragment_df_lens = [len(fragment_df) for fragment_df in fragment_df_list]
    precursor_df_list = [precursor_df.copy() for precursor_df in precursor_df_list]
    cum_frag_df_lens = np.cumsum(fragment_df_lens)
    for i, precursor_df in enumerate(precursor_df_list[1:]):
        precursor_df[["frag_start_idx", "frag_stop_idx"]] += cum_frag_df_lens[i]
    return (
        pd.concat(precursor_df_list, ignore_index=True),
        pd.concat(fragment_df_list, ignore_index=True),
        *[
            pd.concat(other_list, ignore_index=True)
            for other_list in other_fragment_df_lists
        ],
    )


def calc_fragment_mz_values_for_same_nAA(
    df_group: pd.DataFrame, nAA: int, charged_frag_types: list
):
    mod_list = (
        df_group.mods.str.split(";")
        .apply(lambda x: [m for m in x if len(m) > 0])
        .values
    )
    site_list = (
        df_group.mod_sites.str.split(";")
        .apply(lambda x: [int(s) for s in x if len(s) > 0])
        .values
    )

    if "aa_mass_diffs" in df_group.columns:
        mod_diff_list = (
            df_group.aa_mass_diffs.str.split(";")
            .apply(lambda x: [float(m) for m in x if len(m) > 0])
            .values
        )
        mod_diff_site_list = (
            df_group.aa_mass_diff_sites.str.split(";")
            .apply(lambda x: [int(s) for s in x if len(s) > 0])
            .values
        )
    else:
        mod_diff_list = None
        mod_diff_site_list = None
    (b_mass, y_mass, pepmass) = calc_b_y_and_peptide_masses_for_same_len_seqs(
        df_group.sequence.values.astype("U"),
        mod_list,
        site_list,
        mod_diff_list,
        mod_diff_site_list,
    )
    b_mass = b_mass.reshape(-1)
    y_mass = y_mass.reshape(-1)

    for charged_frag_type in charged_frag_types:
        if charged_frag_type.startswith("b_modloss"):
            b_modloss = np.concatenate(
                [
                    calc_modloss_mass(nAA, mods, sites, True)
                    for mods, sites in zip(mod_list, site_list)
                ]
            )
            break
    for charged_frag_type in charged_frag_types:
        if charged_frag_type.startswith("y_modloss"):
            y_modloss = np.concatenate(
                [
                    calc_modloss_mass(nAA, mods, sites, False)
                    for mods, sites in zip(mod_list, site_list)
                ]
            )
            break

    mz_values = []

    for charged_frag_type in charged_frag_types:
        frag_type, charge = parse_charged_frag_type(charged_frag_type)
        if frag_type == "b":
            _mass = b_mass / charge + MASS_PROTON
        elif frag_type == "y":
            _mass = y_mass / charge + MASS_PROTON
        elif frag_type == "b_modloss":
            _mass = (b_mass - b_modloss) / charge + MASS_PROTON
            _mass[b_modloss == 0] = 0
        elif frag_type == "y_modloss":
            _mass = (y_mass - y_modloss) / charge + MASS_PROTON
            _mass[y_modloss == 0] = 0
        elif frag_type in FRAGMENT_TYPES:
            ref_ion = FRAGMENT_TYPES[frag_type].ref_ion
            delta_mass = FRAGMENT_TYPES[frag_type].delta_mass
            if ref_ion == "b":
                _mass = (b_mass + delta_mass) / charge + MASS_PROTON
            elif ref_ion == "y":
                _mass = (y_mass + delta_mass) / charge + MASS_PROTON
            else:
                raise KeyError(
                    f"ref_ion only allows `b` and `y`, but {ref_ion} is given"
                )

        else:
            raise KeyError(f'Fragment type "{frag_type}" is not supported')
        mz_values.append(_mass)
    return np.array(mz_values).T


def mask_fragments_for_charge_greater_than_precursor_charge(
    fragment_df: pd.DataFrame,
    precursor_charge_array: np.ndarray,
    nAA_array: np.ndarray,
    *,
    candidate_fragment_charges: list = [2, 3, 4],
):
    """Mask the fragment dataframe when
    the fragment charge is larger than the precursor charge"""
    precursor_charge_array = np.repeat(precursor_charge_array, nAA_array - 1)
    for col in fragment_df.columns:
        for charge in candidate_fragment_charges:
            if col.endswith(f"z{charge}"):
                fragment_df.loc[precursor_charge_array < charge, col] = 0
    return fragment_df


@nb.njit(parallel=True)
def fill_in_indices(
    frag_start_idxes: np.ndarray,
    frag_stop_idxes: np.ndarray,
    indices: np.ndarray,
    max_indices: np.ndarray,
    excluded_indices: np.ndarray,
    top_k: int,
    flattened_intensity: np.ndarray,
    number_of_fragment_types: int,
    max_frag_per_peptide: int = 300,
) -> None:
    """
    Fill in indices, max indices and excluded indices for each peptide.
    indices: index of fragment per peptide (from 0 to max_index-1)
    max_indices: max index of fragments per peptide (number of fragments per peptide)
    excluded_indices: not top k excluded indices per peptide

    Parameters
    ----------
    frag_start_idxes : np.ndarray
        start indices of fragments for each peptide

    frag_stop_idxes : np.ndarray
        stop indices of fragments for each peptide

    indices : np.ndarray
        index of fragment per peptide (from 0 to max_index-1) it will be filled in this function

    max_indices : np.ndarray
        max index of fragments per peptide (number of fragments per peptide) it will be filled in this function

    excluded_indices : np.ndarray
        not top k excluded indices per peptide it will be filled in this function

    top_k : int
        top k highest peaks to keep

    flattened_intensity : np.ndarray
        Flattened fragment intensities

    number_of_fragment_types : int
        number of types of fragments (e.g. b,y,b_modloss,y_modloss, ...) equals to the number of columns in fragment mz dataframe

    max_frag_per_peptide : int, optional
        maximum number of fragments per peptide, Defaults to 300

    """
    array = np.arange(0, max_frag_per_peptide).reshape(-1, 1)
    ones = np.ones(max_frag_per_peptide).reshape(-1, 1)
    length = len(frag_start_idxes)

    for i in nb.prange(length):
        frag_start = frag_start_idxes[i]
        frag_end = frag_stop_idxes[i]
        max_index = frag_end - frag_start
        indices[frag_start:frag_end] = array[:max_index]
        max_indices[frag_start:frag_end] = ones[:max_index] * max_index
        if flattened_intensity is None or top_k >= max_index * number_of_fragment_types:
            continue
        idxes = np.argsort(
            flattened_intensity[
                frag_start * number_of_fragment_types : frag_end
                * number_of_fragment_types
            ]
        )
        _excl = np.ones_like(idxes, dtype=np.bool_)
        _excl[idxes[-top_k:]] = False
        excluded_indices[
            frag_start * number_of_fragment_types : frag_end * number_of_fragment_types
        ] = _excl


@nb.vectorize([nb.uint32(nb.int8, nb.uint32, nb.uint32, nb.uint32)], target="parallel")
def calculate_fragment_numbers(
    frag_direction: np.int8,
    frag_number: np.uint32,
    index: np.uint32,
    max_index: np.uint32,
):
    """
    Calculate fragment numbers for each fragment based on the fragment direction.

    Parameters
    ----------
    frag_direction : np.int8
        directions of fragments for each peptide

    frag_number : np.uint32
        fragment numbers for each peptide

    index : np.uint32
        index of fragment per peptide (from 0 to max_index-1)

    max_index : np.uint32
        max index of fragments per peptide (number of fragments per peptide)
    """
    if frag_direction == 1:
        frag_number = index + 1
    elif frag_direction == -1:
        frag_number = max_index - index
    return frag_number


def parse_fragment(
    frag_directions: np.ndarray,
    frag_start_idxes: np.ndarray,
    frag_stop_idxes: np.ndarray,
    top_k: int,
    intensities: np.ndarray,
    number_of_fragment_types: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse fragments to get fragment numbers, fragment positions and not top k excluded indices in one hit
    faster than doing each operation individually, and makes the most of the operations that are done in parallel.

    Parameters
    ----------
    frag_directions : np.ndarray
        directions of fragments for each peptide

    frag_start_idxes : np.ndarray
        start indices of fragments for each peptide

    frag_stop_idxes : np.ndarray
        stop indices of fragments for each peptide

    top_k : int
        top k highest peaks to keep

    intensities : np.ndarray
        Flattened fragment intensities

    number_of_fragment_types : int
        number of types of fragments (e.g. b,y,b_modloss,y_modloss, ...) equals to the number of columns in fragment mz dataframe

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of fragment numbers, fragment positions and not top k excluded indices

    """
    # Allocate memory for fragment numbers, indices, max indices and excluded indices
    frag_numbers = np.empty_like(frag_directions, dtype=np.uint32)
    indices = np.empty_like(frag_directions, dtype=np.uint32)
    max_indices = np.empty_like(frag_directions, dtype=np.uint32)
    excluded_indices = np.zeros(
        frag_directions.shape[0] * frag_directions.shape[1], dtype=np.bool_
    )

    # Fill in indices, max indices and excluded indices
    fill_in_indices(
        frag_start_idxes,
        frag_stop_idxes,
        indices,
        max_indices,
        excluded_indices,
        top_k,
        intensities,
        number_of_fragment_types,
    )

    # Calculate fragment numbers
    frag_numbers = calculate_fragment_numbers(
        frag_directions, frag_numbers, indices, max_indices
    )
    return frag_numbers, indices, excluded_indices


def flatten_fragments(
    precursor_df: pd.DataFrame,
    fragment_mz_df: pd.DataFrame,
    fragment_intensity_df: pd.DataFrame,
    min_fragment_intensity: float = -1,
    keep_top_k_fragments: int = 1000,
    custom_columns: list = ["type", "number", "position", "charge", "loss_type"],
    custom_df: Dict[str, pd.DataFrame] = {},
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts the tabular fragment format consisting of
    the `fragment_mz_df` and the `fragment_intensity_df`
    into a linear fragment format.
    The linear fragment format will only retain fragments
    above a given intensity treshold with `mz > 0`.
    It consists of columns: `mz`, `intensity`,
    `type`, `number`, `charge` and `loss_type`,
    where each column refers to:

    - mz:        :data:`PEAK_MZ_DTYPE`, fragment mz value
    - intensity: :data:`PEAK_INTENSITY_DTYPE`, fragment intensity value
    - type:      uint8, ASCII code of the ion series. Must be a part of the `SERIES_MAPPING`.
    - number:    uint32, fragment series number
    - position:  uint32, fragment position in sequence (from left to right, starts with 0)
    - charge:    uint8, fragment charge
    - loss_type: int16, fragment loss type. Must be a part of the `LOSS_MAPPING`.

    The fragment pointers `frag_start_idx` and `frag_stop_idx`
    will be reannotated to the new fragment format.

    For ASCII code `type`, we can convert it into byte-str by using `frag_df.type.values.view('S1')`.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        input precursor dataframe which contains the frag_start_idx and frag_stop_idx columns

    fragment_mz_df : pd.DataFrame
        input fragment mz dataframe of shape (N, T) which contains N * T fragment mzs.
        Fragments with mz==0 will be excluded.

    fragment_intensity_df : pd.DataFrame
        input fragment intensity dataframe of shape (N, T) which contains N * T fragment mzs.
        Could be empty (len==0) to exclude intensity values.

    min_fragment_intensity : float, optional
        minimum intensity which should be retained. Defaults to -1

    custom_columns : list, optional
        'mz' and 'intensity' columns are required. Others could be customized.
        Defaults to ['type','number','position','charge','loss_type']

    custom_df : Dict[str, pd.DataFrame], optional
        Append custom columns by providing additional dataframes of the same shape as fragment_mz_df and fragment_intensity_df. Defaults to {}.

    Returns
    -------
    pd.DataFrame
        precursor dataframe with added `flat_frag_start_idx` and `flat_frag_stop_idx` columns
    pd.DataFrame
        fragment dataframe with columns: `mz`, `intensity`, `type`, `number`,
        `charge` and `loss_type`.
    """
    if len(precursor_df) == 0:
        return precursor_df, pd.DataFrame()
    # new dataframes for fragments and precursors are created
    frag_df = {}
    frag_df["mz"] = fragment_mz_df.values.reshape(-1)
    if len(fragment_intensity_df) > 0:
        frag_df["intensity"] = fragment_intensity_df.values.astype(
            PEAK_INTENSITY_DTYPE
        ).reshape(-1)
        use_intensity = True
    else:
        use_intensity = False
    # add additional columns to the fragment dataframe
    # each column in the flat fragment dataframe is a whole pandas dataframe in the dense representation
    for col_name, df in custom_df.items():
        frag_df[col_name] = df.values.reshape(-1)

    frag_types = []
    frag_loss_types = []
    frag_charges = []
    frag_directions = []  # 'abc': direction=1, 'xyz': direction=-1, otherwise 0

    for charged_frag_type in fragment_mz_df.columns.values:
        frag_type, charge = parse_charged_frag_type(charged_frag_type)
        frag_charges.append(charge)
        frag_types.append(FRAGMENT_TYPES[frag_type].series_id)
        frag_loss_types.append(FRAGMENT_TYPES[frag_type].loss_id)
        frag_directions.append(FRAGMENT_TYPES[frag_type].direction_id)

    if "type" in custom_columns:
        frag_df["type"] = np.array(
            np.tile(frag_types, len(fragment_mz_df)), dtype=np.int8
        )
    if "loss_type" in custom_columns:
        frag_df["loss_type"] = np.array(
            np.tile(frag_loss_types, len(fragment_mz_df)), dtype=np.int16
        )
    if "charge" in custom_columns:
        frag_df["charge"] = np.array(
            np.tile(frag_charges, len(fragment_mz_df)), dtype=np.int8
        )

    frag_directions = np.array(
        np.tile(frag_directions, (len(fragment_mz_df), 1)), dtype=np.int8
    )

    numbers, positions, excluded_indices = parse_fragment(
        frag_directions,
        precursor_df.frag_start_idx.values,
        precursor_df.frag_stop_idx.values,
        keep_top_k_fragments,
        frag_df["intensity"] if use_intensity else None,
        len(fragment_mz_df.columns),
    )

    if "number" in custom_columns:
        frag_df["number"] = numbers.reshape(-1)

    if "position" in custom_columns:
        frag_df["position"] = positions.reshape(-1)

    precursor_df["flat_frag_start_idx"] = precursor_df.frag_start_idx
    precursor_df["flat_frag_stop_idx"] = precursor_df.frag_stop_idx
    precursor_df[["flat_frag_start_idx", "flat_frag_stop_idx"]] *= len(
        fragment_mz_df.columns
    )

    if use_intensity:
        frag_df["intensity"][frag_df["mz"] == 0.0] = 0.0

    excluded = (
        frag_df["mz"] == 0
        if not use_intensity
        else (frag_df["intensity"] < min_fragment_intensity)
        | (frag_df["mz"] == 0)
        | (excluded_indices)
    )

    frag_df = pd.DataFrame(frag_df)
    frag_df = frag_df[~excluded]
    frag_df = frag_df.reset_index(drop=True)

    # cumulative sum counts the number of fragments before the given fragment which were removed.
    # This sum does not include the fragment at the index position and has therefore len N +1
    cum_sum_tresh = np.zeros(shape=len(excluded) + 1, dtype=np.int64)
    cum_sum_tresh[1:] = np.cumsum(excluded)

    precursor_df["flat_frag_start_idx"] -= cum_sum_tresh[
        precursor_df.flat_frag_start_idx.values
    ]
    precursor_df["flat_frag_stop_idx"] -= cum_sum_tresh[
        precursor_df.flat_frag_stop_idx.values
    ]

    return precursor_df, frag_df


@nb.njit()
def compress_fragment_indices(frag_idx):
    """
    recalculates fragment indices to remove unused fragments. Can be used to compress a fragment library.
    Expects fragment indices to be ordered by increasing values (!!!).
    It should be O(N) runtime with N being the number of fragment rows.

    >>> frag_idx = [[6,  10],
                [12, 14],
                [20, 22]]

    >>> frag_idx = [[0, 4],
                [4, 6],
                [6, 8]]
    >>> fragment_pointer = [6,7,8,9,12,13,20,21]
    """
    frag_idx_len = frag_idx[:, 1] - frag_idx[:, 0]

    # This sum does not include the fragment at the index position and has therefore len N +1
    frag_idx_cumsum = np.zeros(shape=len(frag_idx_len) + 1, dtype="int64")
    frag_idx_cumsum[1:] = np.cumsum(frag_idx_len)

    fragment_pointer = np.zeros(np.sum(frag_idx_len), dtype="int64")

    for i in range(len(frag_idx)):
        start_index = frag_idx_cumsum[i]

        for j, k in enumerate(range(frag_idx[i, 0], frag_idx[i, 1])):
            fragment_pointer[start_index + j] = k

    new_frag_idx = np.column_stack((frag_idx_cumsum[:-1], frag_idx_cumsum[1:]))
    return new_frag_idx, fragment_pointer


def remove_unused_fragments(
    precursor_df: pd.DataFrame,
    fragment_df_list: Tuple[pd.DataFrame, ...],
    frag_start_col: str = "frag_start_idx",
    frag_stop_col: str = "frag_stop_idx",
) -> Tuple[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
    """Removes unused fragments of removed precursors,
    reannotates the `frag_start_col` and `frag_stop_col`

    Parameters
    ----------
    precursor_df : pd.DataFrame
        Precursor dataframe which contains frag_start_idx and frag_stop_idx columns

    fragment_df_list : List[pd.DataFrame]
        A list of fragment dataframes which should be compressed by removing unused fragments.
        Multiple fragment dataframes can be provided which will all be sliced in the same way.
        This allows to slice both the fragment_mz_df and fragment_intensity_df.
        At least one fragment dataframe needs to be provided.

    frag_start_col : str, optional
        Fragment start idx column in `precursor_df`, such as "frag_start_idx" and "peak_start_idx".
        Defaults to "frag_start_idx".

    frag_stop_col : str, optional
        Fragment stop idx column in `precursor_df`, such as "frag_stop_idx" and "peak_stop_idx".
        Defaults to "frag_stop_idx".

    Returns
    -------
    pd.DataFrame, List[pd.DataFrame]
        returns the reindexed precursor DataFrame and the sliced fragment DataFrames
    """

    precursor_df = precursor_df.sort_values([frag_start_col], ascending=True)
    frag_idx = precursor_df[[frag_start_col, frag_stop_col]].values

    new_frag_idx, fragment_pointer = compress_fragment_indices(frag_idx)

    precursor_df[[frag_start_col, frag_stop_col]] = new_frag_idx
    precursor_df = precursor_df.sort_index()

    output_tuple = []

    for i in range(len(fragment_df_list)):
        output_tuple.append(
            fragment_df_list[i].iloc[fragment_pointer].copy().reset_index(drop=True)
        )

    return precursor_df, tuple(output_tuple)


def create_fragment_mz_dataframe_by_sort_precursor(
    precursor_df: pd.DataFrame,
    charged_frag_types: List,
    batch_size: int = 500000,
    dtype: np.dtype = PEAK_MZ_DTYPE,
) -> pd.DataFrame:
    """Sort nAA in precursor_df for faster fragment mz dataframe creation.

    Because the fragment mz values are continous in memory, so it is faster
    when setting values in pandas.

    Note that this function will change the order and index of precursor_df

    Parameters
    ----------
    precursor_df : pd.DataFrame
        precursor dataframe

    charged_frag_types : List
        fragment types list

    batch_size : int, optional
        Calculate fragment mz values in batch.
        Defaults to 500000.
    """
    if "frag_start_idx" in precursor_df.columns:
        precursor_df.drop(columns=["frag_start_idx", "frag_stop_idx"], inplace=True)

    refine_precursor_df(precursor_df)

    fragment_mz_df = init_fragment_by_precursor_dataframe(
        precursor_df,
        charged_frag_types,
        dtype=dtype,
    )

    _grouped = precursor_df.groupby("nAA")
    for nAA, big_df_group in _grouped:
        for i in range(0, len(big_df_group), batch_size):
            batch_end = i + batch_size

            df_group = big_df_group.iloc[i:batch_end, :]

            mz_values = calc_fragment_mz_values_for_same_nAA(
                df_group, nAA, charged_frag_types
            )

            fragment_mz_df.iloc[
                df_group.frag_start_idx.values[0] : df_group.frag_stop_idx.values[-1], :
            ] = mz_values.astype(PEAK_MZ_DTYPE)
    return mask_fragments_for_charge_greater_than_precursor_charge(
        fragment_mz_df,
        precursor_df.charge.values,
        precursor_df.nAA.values,
    )


def create_fragment_mz_dataframe(
    precursor_df: pd.DataFrame,
    charged_frag_types: List,
    *,
    reference_fragment_df: pd.DataFrame = None,
    inplace_in_reference: bool = False,
    batch_size: int = 500000,
    dtype: np.dtype = PEAK_MZ_DTYPE,
) -> pd.DataFrame:
    """
    Generate fragment mass dataframe for the precursor_df. If
    the `reference_fragment_df` is provided and precursor_df contains `frag_start_idx`,
    it will generate  the mz dataframe based on the reference. Otherwise it
    generates the mz dataframe from scratch.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        precursors to generate fragment masses,
        if `precursor_df` contains the 'frag_start_idx' column,
        `reference_fragment_df` must be provided

    charged_frag_types : List
        `['b_z1','b_z2','y_z1','y_z2','b_modloss_1','y_H2O_z1'...]`

    reference_fragment_df : pd.DataFrame
        kwargs only. Generate fragment_mz_df based on this reference,
        as `precursor_df.frag_start_idx` and
        `precursor.frag_stop_idx` point to the indices in
        `reference_fragment_df`.
        Defaults to None

    inplace_in_reference : bool
        kwargs only. Change values in place in the `reference_fragment_df`.
        Defaults to False

    batch_size: int
        Number of peptides for each batch, to save RAM.

    Returns
    -------
    pd.DataFrame
        `fragment_mz_df` with given `charged_frag_types`
    """
    if reference_fragment_df is None and "frag_start_idx" in precursor_df.columns:
        # raise ValueError(
        #     "`precursor_df` contains 'frag_start_idx' column, "\
        #     "please provide `reference_fragment_df` argument"
        # )
        fragment_mz_df = init_fragment_by_precursor_dataframe(
            precursor_df,
            charged_frag_types,
            dtype=dtype,
        )
        return create_fragment_mz_dataframe(
            precursor_df=precursor_df,
            charged_frag_types=charged_frag_types,
            reference_fragment_df=fragment_mz_df,
            inplace_in_reference=True,
            batch_size=batch_size,
            dtype=dtype,
        )
    if "nAA" not in precursor_df.columns:
        # fast
        return create_fragment_mz_dataframe_by_sort_precursor(
            precursor_df,
            charged_frag_types,
            batch_size,
            dtype=dtype,
        )

    if is_precursor_refined(precursor_df) and reference_fragment_df is None:
        # fast
        return create_fragment_mz_dataframe_by_sort_precursor(
            precursor_df, charged_frag_types, batch_size, dtype=dtype
        )

    else:
        # slow
        if reference_fragment_df is not None:
            if inplace_in_reference:
                fragment_mz_df = reference_fragment_df.loc[
                    :,
                    [
                        _fr
                        for _fr in charged_frag_types
                        if _fr in reference_fragment_df.columns
                    ],
                ]
            else:
                fragment_mz_df = pd.DataFrame(
                    np.zeros(
                        (len(reference_fragment_df), len(charged_frag_types)),
                        dtype=dtype,
                    ),
                    columns=charged_frag_types,
                )
        else:
            fragment_mz_df = init_fragment_by_precursor_dataframe(
                precursor_df,
                charged_frag_types,
                dtype=dtype,
            )

        frag_mz_values = fragment_mz_df.to_numpy(copy=True)

        _grouped = precursor_df.groupby("nAA")
        for nAA, big_df_group in _grouped:
            for i in range(0, len(big_df_group), batch_size):
                batch_end = i + batch_size

                df_group = big_df_group.iloc[i:batch_end, :]

                mz_values = calc_fragment_mz_values_for_same_nAA(
                    df_group, nAA, fragment_mz_df.columns
                )

                update_sliced_fragment_dataframe(
                    fragment_mz_df,
                    frag_mz_values,
                    mz_values,
                    df_group[["frag_start_idx", "frag_stop_idx"]].values,
                )

    fragment_mz_df.iloc[:] = frag_mz_values

    return mask_fragments_for_charge_greater_than_precursor_charge(
        fragment_mz_df,
        precursor_df.charge.values,
        precursor_df.nAA.values,
    )


@nb.njit(nogil=True)
def join_left(left: np.ndarray, right: np.ndarray):
    """joins all values in the left array to the values in the right array.
    The index to the element in the right array is returned.
    If the value wasn't found, -1 is returned. If the element appears more than once, the last appearance is used.

    Parameters
    ----------

    left: numpy.ndarray
        left array which should be matched

    right: numpy.ndarray
        right array which should be matched to

    Returns
    -------
    numpy.ndarray, dtype = int64
        array with length of the left array which indices pointing to the right array
        -1 is returned if values could not be found in the right array
    """
    left_indices = np.argsort(left)
    left_sorted = left[left_indices]

    right_indices = np.argsort(right)
    right_sorted = right[right_indices]

    joined_index = -np.ones(len(left), dtype="int64")

    # from hereon sorted arrays are expected
    lower_right = 0

    for i in range(len(joined_index)):
        for k in range(lower_right, len(right)):
            if left_sorted[i] >= right_sorted[k]:
                if left_sorted[i] == right_sorted[k]:
                    joined_index[i] = k
                    lower_right = k
            else:
                break

    # the joined_index_sorted connects indices from the sorted left array with the sorted right array
    # to get the original indices, the order of both sides needs to be restored
    # First, the indices pointing to the right side are restored by masking the array for hits and looking up the right side
    joined_index[joined_index >= 0] = right_indices[joined_index[joined_index >= 0]]

    # Next, the left side is restored by arranging the items
    joined_index[left_indices] = joined_index

    return joined_index


def calc_fragment_count(
    precursor_df: pd.DataFrame, fragment_intensity_df: pd.DataFrame
):
    """
    Calculates the number of fragments for each precursor.

    Parameters
    ----------

    precursor_df : pd.DataFrame
        precursor dataframe which contains the frag_start_idx and frag_stop_idx columns

    fragment_intensity_df : pd.DataFrame
        fragment intensity dataframe which contains the fragment intensities

    Returns
    -------
    numpy.ndarray
        array with the number of fragments for each precursor
    """
    if not set(["frag_start_idx", "frag_stop_idx"]).issubset(precursor_df.columns):
        raise KeyError("frag_start_idx and frag_stop_idx not in dataframe")

    n_fragments = []

    for start, stop in zip(
        precursor_df["frag_start_idx"].values, precursor_df["frag_stop_idx"].values
    ):
        n_fragments += [np.sum(fragment_intensity_df.iloc[start:stop].values > 0)]

    return np.array(n_fragments)


def filter_fragment_number(
    precursor_df: pd.DataFrame,
    fragment_intensity_df: pd.DataFrame,
    n_fragments_allowed_column_name: str = "n_fragments_allowed",
    n_allowed: int = 999,
):
    """
    Filters the number of fragments for each precursor.

    Parameters
    ----------

    precursor_df : pd.DataFrame

        precursor dataframe which contains the frag_start_idx and frag_stop_idx columns

    fragment_intensity_df : pd.DataFrame
        fragment intensity dataframe which contains the fragment intensities

    n_fragments_allowed_column_name : str, default = 'n_fragments_allowed'
        column name in precursor_df which contains the number of allowed fragments

    n_allowed : int, default = 999
        number of fragments which should be allowed

    Returns
    -------
    None
    """

    if not set(["frag_start_idx", "frag_stop_idx"]).issubset(precursor_df.columns):
        raise KeyError("frag_start_idx and frag_stop_idx not in dataframe")

    for start_idx, stop_idx, n_allowed_lib in zip(
        precursor_df["frag_start_idx"].values,
        precursor_df["frag_stop_idx"].values,
        precursor_df[n_fragments_allowed_column_name].values,
    ):
        _allowed = min(n_allowed_lib, n_allowed)

        intensies = fragment_intensity_df.iloc[start_idx:stop_idx].values
        flat_intensities = np.sort(intensies.flatten())[::-1]
        intensies[intensies <= flat_intensities[_allowed]] = 0
        fragment_intensity_df.iloc[start_idx:stop_idx] = intensies


def calc_fragment_cardinality(
    precursor_df,
    fragment_mz_df,
    group_column="elution_group_idx",
    split_target_decoy=True,
):
    """
    Calculate the cardinality for a given fragment across a group of precursors.
    The cardinality is the number of precursors that have a given fragment at a given position.

    All precursors within a group are expected to have the same number of fragments.
    The precursor dataframe.

    fragment_mz_df : pd.DataFrame
        The fragment mz dataframe.

    group_column : str
        The column to group the precursors by. Integer column is expected.

    split_target_decoy : bool
        If True, the cardinality is calculated for the target and decoy precursors separately.

    """

    if len(precursor_df) == 0:
        raise ValueError("Precursor dataframe is empty.")

    if len(fragment_mz_df) == 0:
        raise ValueError("Fragment dataframe is empty.")

    if group_column not in precursor_df.columns:
        raise KeyError("Group column not in precursor dataframe.")

    if ("frag_start_idx" not in precursor_df.columns) or (
        "frag_stop_idx" not in precursor_df.columns
    ):
        raise KeyError("Precursor dataframe does not contain fragment indices.")

    precursor_df = precursor_df.sort_values(group_column)
    fragment_mz = fragment_mz_df.values
    fragment_cardinality = np.ones(fragment_mz.shape, dtype=np.uint8)

    @nb.njit
    def _calc_fragment_cardinality(
        elution_group_idx,
        start_idx,
        stop_idx,
        fragment_mz,
        fragment_cardinality,
    ):
        if len(elution_group_idx) == 0:
            return
        elution_group_idx[0]  # noqa TODO check for potential bug
        elution_group_start = 0

        for i in range(len(elution_group_idx)):
            if (
                i == len(elution_group_idx) - 1
                or elution_group_idx[i] != elution_group_idx[i + 1]
            ):
                elution_group_stop = i + 1

            # check if whole elution group is covered
            n_precursor = elution_group_stop - elution_group_start

            # Check that all precursors within a group have the same number of fragments.
            nAA = (
                stop_idx[elution_group_start:elution_group_stop]
                - start_idx[elution_group_start:elution_group_stop]
            )
            if not np.all(nAA[0] == nAA):
                raise ValueError(
                    "All precursors within a group must have the same number of fragments."
                )

            # within a group, check for each precursor if it has the same fragment as another precursor
            for i in range(n_precursor):
                precursor_start_idx = start_idx[elution_group_start + i]
                precursor_stop_idx = stop_idx[elution_group_start + i]

                precursor_fragment_mz = fragment_mz[
                    precursor_start_idx:precursor_stop_idx
                ]

                for j in range(n_precursor):
                    if i == j:
                        continue

                    other_precursor_start_idx = start_idx[elution_group_start + j]
                    other_precursor_stop_idx = stop_idx[elution_group_start + j]
                    other_precursor_fragment_mz = fragment_mz[
                        other_precursor_start_idx:other_precursor_stop_idx
                    ]

                    binary_mask = (
                        np.abs(precursor_fragment_mz - other_precursor_fragment_mz)
                        < 0.00001
                    )

                    fragment_cardinality[precursor_start_idx:precursor_stop_idx] += (
                        binary_mask.astype(np.uint8)
                    )

            elution_group_start = elution_group_stop

    if ("decoy" in precursor_df.columns) and (split_target_decoy):
        decoy_classes = precursor_df["decoy"].unique()
        for decoy_class in decoy_classes:
            df = precursor_df[precursor_df["decoy"] == decoy_class]
            _calc_fragment_cardinality(
                df[group_column].values,
                df["frag_start_idx"].values,
                df["frag_stop_idx"].values,
                fragment_mz,
                fragment_cardinality,
            )
    else:
        _calc_fragment_cardinality(
            precursor_df[group_column].values,
            precursor_df["frag_start_idx"].values,
            precursor_df["frag_stop_idx"].values,
            fragment_mz,
            fragment_cardinality,
        )

    return pd.DataFrame(fragment_cardinality, columns=fragment_mz_df.columns)


def _calc_column_indices(
    fragment_df: pd.DataFrame,
    charged_frag_types: list,
) -> np.ndarray:
    """
    Calculate the column indices for a dense fragment matrix.
    Columns are sorted according to `fragment.sort_charged_frag_types`

    Parameters
    ----------
    fragment_df : pd.DataFrame
        Flat fragment dataframe with columns 'type', 'loss_type', 'charge'

    charged_frag_types : list
        List of charged fragment types as generated by `fragment.get_charged_frag_types`

    Returns
    -------
    np.ndarray
        Column indices with shape (n_fragments,)
    """
    # features.LOSS_INVERSE but with separator '_' for non-empty values
    _loss_inverse_separator = {
        key: ("_" + value if value != "" else value)
        for key, value in LOSS_MAPPING_INV.items()
    }

    sorted_charged_frag_types = sort_charged_frag_types(charged_frag_types)

    # mapping of charged fragment types to indices
    inverse_frag_type_mapping = dict(
        zip(sorted_charged_frag_types, range(len(sorted_charged_frag_types)))
    )

    # mapping of fragment type, loss type, charge to a dense column name
    frag_type_list = (
        fragment_df["type"].map(SERIES_MAPPING_INV)
        + fragment_df["loss_type"].map(_loss_inverse_separator)
        + FRAGMENT_CHARGE_SEPARATOR
        + fragment_df["charge"].astype(str)
    )

    # Convert to integer array, using -1 for any unmapped values
    return (
        frag_type_list.map(inverse_frag_type_mapping)
        .fillna(-1)
        .astype(np.int32)
        .to_numpy()
    )


def _calc_row_indices(
    precursor_naa: np.ndarray,
    fragment_position: np.ndarray,
    precursor_df_idx: np.ndarray,
    fragment_df_idx: np.ndarray,
    frag_start_idx: Union[None, np.ndarray] = None,
    frag_stop_idx: Union[None, np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate new start and stop index mapping for flat fragments.

    Returns the vector of row indices for the dense fragment matrix, shape (n_fragments,)
    and the new start and stop indices for the flat fragments, shape (n_precursors,)

    Parameters
    ----------
    precursor_naa : np.ndarray
        Array of precursor nAA values
    fragment_position : np.ndarray
        Array of fragment positions
    precursor_df_idx : np.ndarray
        Array of precursor indices
    fragment_df_idx : np.ndarray
        Array of fragment indices

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (row_indices, frag_start_idx, frag_stop_idx)
    """
    if len(fragment_position) != len(fragment_df_idx):
        raise ValueError(
            "fragment_position and fragment_df_idx must have the same length"
        )

    if len(precursor_naa) != len(precursor_df_idx):
        raise ValueError("precursor_naa and precursor_df_idx must have the same length")

    build_index = (frag_start_idx is None) or (frag_stop_idx is None)
    if build_index:
        frag_stop_idx = (precursor_naa - 1).cumsum()

        # Start indices for each precursor is the accumlated nAA of the previous precursor and for the first precursor is 0
        frag_start_idx = np.zeros_like(frag_stop_idx)
        frag_start_idx[1:] = frag_stop_idx[
            :-1
        ]  # shift values right by 1, first element remains 0

    else:
        if (frag_start_idx is None) or (frag_stop_idx is None):
            raise ValueError(
                "frag_start_idx and frag_stop_idx must both be provided if one is provided"
            )
        elif len(frag_start_idx) != len(frag_stop_idx):
            raise ValueError(
                "frag_start_idx and frag_stop_idx must have the same length"
            )

    # Row indices of a fragment being the accumlated nAA of the precursor + fragment position -1
    precursor_idx_to_accumulated_naa = dict(zip(precursor_df_idx, frag_start_idx))
    # Convert numpy array to pandas Series for mapping
    # This massively speeds up the mapping
    row_indices = (
        pd.Series(fragment_df_idx).map(
            precursor_idx_to_accumulated_naa, na_action="ignore"
        )
    ).to_numpy() + fragment_position

    # fill nan with -1 and cast to int32
    row_indices[np.isnan(row_indices)] = -1
    row_indices = row_indices.astype(np.int32)

    return row_indices, frag_start_idx, frag_stop_idx


def _start_stop_to_idx(precursor_df, fragment_df, index_column="precursor_idx"):
    """
    Convert start/stop indices to precursor and fragment indices.

    Parameters
    ----------
    precursor_df : pd.DataFrame
        DataFrame containing flat_frag_start_idx and flat_frag_stop_idx columns
    fragment_df : pd.DataFrame
        DataFrame containing fragment information
    index_column : str, optional
        Name of the index column to use, by default "precursor_idx"

    Returns
    -------
    tuple
        (precursor_df_idx, fragment_df_idx) - numpy arrays containing indices
    """
    # Handle empty DataFrames
    if precursor_df.empty or fragment_df.empty:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # Sort precursor_df by 'flat_frag_start_idx'
    precursor_df_sorted = (
        precursor_df[["flat_frag_start_idx", "flat_frag_stop_idx"]]
        .copy()
        .reset_index(drop=True)
        .sort_values("flat_frag_start_idx")
    )

    # Add precursor_idx to precursor_df as 0,1,2,3...
    precursor_df_sorted[index_column] = np.arange(precursor_df_sorted.shape[0])

    # Add precursor_idx to fragment_df
    fragment_df_idx = np.repeat(
        precursor_df_sorted[index_column].to_numpy(),
        precursor_df_sorted["flat_frag_stop_idx"].to_numpy()
        - precursor_df_sorted["flat_frag_start_idx"].to_numpy(),
    )

    if len(fragment_df_idx) != fragment_df.shape[0]:
        raise ValueError(
            f"Number of fragments {len(fragment_df_idx)} is not equal to the number of rows in fragment_df {fragment_df.shape[0]}"
        )

    # Restore original order of precursor_df
    precursor_df_resorted = precursor_df_sorted.sort_index()
    precursor_df_idx = precursor_df_resorted[index_column].to_numpy()

    return precursor_df_idx, fragment_df_idx


def create_dense_matrices(
    precursor_df: pd.DataFrame,
    fragment_df: pd.DataFrame,
    charged_frag_types: list,
    flat_columns: Union[list, None] = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
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
    precursor_df : pd.DataFrame
        Precursor dataframe
    fragment_df : pd.DataFrame
        Fragment dataframe in flat format
    charged_frag_types : list
        List of charged fragment types (e.g., ['b_z1', 'y_z1'])
    flat_columns : Union[list, None], optional
        Fragment columns from the flat representation to convert to dense format, defaults to ['intensity']

    Returns
    -------
    dict
        Dictionary mapping column names to dense matrices as DataFrames
        Always includes 'mz', plus any specified flat_columns
    np.ndarray
        Start indices for fragments in the dense representation
    np.ndarray
        Stop indices for fragments in the dense representation
    """

    if flat_columns is None:
        flat_columns = ["intensity"]

    optional_columns = [
        col
        for col in ["precursor_idx", "flat_frag_start_idx", "flat_frag_stop_idx"]
        if col in precursor_df.columns
    ]
    precursor_df_ = precursor_df[
        ["sequence", "mods", "mod_sites", "charge", "nAA"] + optional_columns
    ].copy()
    fragment_mz_df = create_fragment_mz_dataframe(
        precursor_df_,
        charged_frag_types,
    )

    if ("precursor_idx" in precursor_df_.columns) and (
        "precursor_idx" in fragment_df.columns
    ):
        precursor_df_idx = precursor_df_["precursor_idx"]
        fragment_df_idx = fragment_df["precursor_idx"]

    elif ("flat_frag_start_idx" in precursor_df_.columns) and (
        "flat_frag_stop_idx" in precursor_df_.columns
    ):
        precursor_df_idx, fragment_df_idx = _start_stop_to_idx(
            precursor_df_, fragment_df
        )

    else:
        raise ValueError(
            "Mapping of fragment indices to precursor indices failed, no 'precursor_idx' or 'flat_frag_start_idx' and 'flat_frag_stop_idx' columns found."
        )

    column_indices = _calc_column_indices(fragment_df, charged_frag_types)
    row_indices, frag_start_idx, frag_stop_idx = _calc_row_indices(
        precursor_df_["nAA"].to_numpy(),
        fragment_df["position"].to_numpy(),
        precursor_df_idx,
        fragment_df_idx,
        precursor_df_["frag_start_idx"].to_numpy(),
        precursor_df_["frag_stop_idx"].to_numpy(),
    )

    # remove all fragments that could not be mapped to a column
    match_mask = (column_indices != -1) & (row_indices != -1)
    column_indices = column_indices[match_mask]
    row_indices = row_indices[match_mask]

    # create a dictionary with the mz matrix and the flat columns
    df_collection = {"mz": fragment_mz_df}

    # df_collection["mz"] might be overridden by flat_columns["mz"]
    if "mz" in flat_columns:
        warnings.warn(
            "flat_columns contains 'mz', this will override the calculated m/z values in fragment_mz_df. If this is not intended, rename the flat mz column to 'mz_observed' before calling to_speclib_base.",
            UserWarning,
        )
    for column_name in flat_columns:
        matrix = np.zeros_like(fragment_mz_df.values, dtype=PEAK_INTENSITY_DTYPE)
        matrix[row_indices, column_indices] = fragment_df[column_name].values[
            match_mask
        ]
        df_collection[column_name] = pd.DataFrame(matrix, columns=charged_frag_types)

    return df_collection, frag_start_idx, frag_stop_idx
