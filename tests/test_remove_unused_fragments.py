import os

import pytest

import alphabase.io.hdf
from alphabase.peptide.fragment import remove_unused_fragments


@pytest.fixture
def hdf_data():
    """
    Fixture to load HDF data and provide precursor_df and fragment_intensity_df.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hdf_file_name = os.path.join(
        current_dir,
        "..",
        "test_data",
        "unit_tests",
        "input_hdf_formats",
        "mini_sample_remove_unused_fragments.hdf",
    )
    hdf_file = alphabase.io.hdf.HDF_File(hdf_file_name, read_only=True)
    precursor_df = hdf_file.dfs.psm_df.values
    fragment_intensity_df = hdf_file.dfs.fragment_intensity_df.values
    return precursor_df, fragment_intensity_df


def test_case_no_nAA_column(hdf_data):
    """
    Test case 1: Precursor dataframe without the 'nAA' column.
    """
    precursor_df, fragment_intensity_df = hdf_data
    case1_precursor_df = precursor_df.copy().drop(columns=["nAA"])
    case1_precursor_df, _ = remove_unused_fragments(
        precursor_df=case1_precursor_df,
        fragment_df_list=(fragment_intensity_df,),
    )

    assert case1_precursor_df[
        "frag_start_idx"
    ].is_monotonic_increasing, "frag_start_idx must be monotonic increasing"
    assert (
        case1_precursor_df["frag_start_idx"].iloc[1:].values
        == case1_precursor_df["frag_stop_idx"].iloc[:-1].values
    ).all(), "frag_start_idx[i] must equal frag_stop_idx[i-1]"


def test_case_unordered_nAA_column(hdf_data):
    """
    Test case 2: Precursor dataframe with unordered 'nAA' column.
    """
    precursor_df, fragment_intensity_df = hdf_data
    case2_precursor_df = (
        precursor_df.copy().sample(frac=1, random_state=42).reset_index(drop=True)
    )
    case2_precursor_df_nAA = (
        case2_precursor_df["nAA"].copy()
        if "nAA" in case2_precursor_df.columns
        else None
    )

    case2_precursor_df, _ = remove_unused_fragments(
        precursor_df=case2_precursor_df,
        fragment_df_list=(fragment_intensity_df,),
    )

    assert case2_precursor_df[
        "frag_start_idx"
    ].is_monotonic_increasing, "frag_start_idx must be monotonic increasing"
    assert (
        case2_precursor_df["frag_start_idx"].iloc[1:].values
        == case2_precursor_df["frag_stop_idx"].iloc[:-1].values
    ).all(), "frag_start_idx[i] must equal frag_stop_idx[i-1]"

    if case2_precursor_df_nAA is not None:
        assert (
            case2_precursor_df["nAA"] == case2_precursor_df_nAA
        ).all(), "nAA values must remain unchanged"


def test_case_ordered_nAA_column(hdf_data):
    """
    Test case 3: Precursor dataframe with ordered 'nAA' column.
    """
    precursor_df, fragment_intensity_df = hdf_data
    case3_precursor_df = (
        precursor_df.sort_values("nAA").reset_index(drop=True)
        if "nAA" in precursor_df.columns
        else precursor_df.copy()
    )
    case3_precursor_df_nAA = (
        case3_precursor_df["nAA"].copy()
        if "nAA" in case3_precursor_df.columns
        else None
    )

    case3_precursor_df, _ = remove_unused_fragments(
        precursor_df=case3_precursor_df,
        fragment_df_list=(fragment_intensity_df,),
    )

    assert case3_precursor_df[
        "frag_start_idx"
    ].is_monotonic_increasing, "frag_start_idx must be monotonic increasing"
    assert (
        case3_precursor_df["frag_start_idx"].iloc[1:].values
        == case3_precursor_df["frag_stop_idx"].iloc[:-1].values
    ).all(), "frag_start_idx[i] must equal frag_stop_idx[i-1]"

    if case3_precursor_df_nAA is not None:
        assert (
            case3_precursor_df["nAA"] == case3_precursor_df_nAA
        ).all(), "nAA values must remain unchanged"
        assert case3_precursor_df[
            "nAA"
        ].is_monotonic_increasing, "nAA column must be monotonic increasing"
