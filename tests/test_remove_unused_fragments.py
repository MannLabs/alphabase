import numpy as np
import pandas as pd
import pytest

from alphabase.peptide.fragment import remove_unused_fragments


@pytest.fixture
def hdf_data():
    """
    Fixture to automatically generate precursor_df and fragment_intensity_df.
    """
    # Data for precursor_df
    sequences = [
        "PSKGPLQSVQVFGR",
        "FLISLLEEYFK",
        "MTEDALRLNLLK",
        "FMSAYEQR",
        "PGPKGEAGPTGPQGEPGVR",
        "YEITEQR",
        "DAEAAEATAEGALKAEK",
        "FGDSRGGGGNFGPGPGSNFR",
        "LDEKENLSAK",
        "ATVASSTQKFQDLGVK",
        "GFALVGVGSEASSKK",
        "LQLEIDQKK",
        "MAGLELLSDQGYR",
        "RGGPGGPPGPLMEQMGGR",
    ]

    frag_start_idx = [151, 81, 110, 24, 296, 17, 229, 334, 69, 183, 180, 26, 123, 284]
    frag_stop_idx = [164, 91, 121, 31, 314, 23, 245, 353, 78, 198, 194, 34, 135, 301]
    charge = [2] * len(sequences)
    nAA = [len(seq) for seq in sequences]

    precursor_df = pd.DataFrame(
        {
            "sequence": sequences,
            "frag_start_idx": frag_start_idx,
            "frag_stop_idx": frag_stop_idx,
            "charge": charge,
            "nAA": nAA,
        }
    )

    # Data for fragment_intensity_df
    num_rows = len(sequences)
    fragment_intensity_data = np.random.uniform(0.0, 600.0, size=(num_rows, 4))
    fragment_intensity_df = pd.DataFrame(
        fragment_intensity_data, columns=["b_z1", "b_z2", "y_z1", "y_z2"]
    )

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
