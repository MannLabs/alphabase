import os

import alphabase.io.hdf
from alphabase.peptide.fragment import remove_unused_fragments


def example_remove_unused_fragments():
    """
    Example function demonstrating the usage of remove_unused_fragments
    with sample data from an HDF file located in the current script directory.
    Handles three cases:
    1. No `nAA` column.
    2. Unordered `nAA` column.
    3. Ordered `nAA` column.
    """
    # Dynamically determine the path to the HDF file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hdf_file_name = os.path.join(
        current_dir,
        "..",
        "test_data",
        "unit_tests",
        "input_hdf_formats",
        "mini_sample_remove_unused_fragments.hdf",
    )

    # Load HDF file
    hdf_file = alphabase.io.hdf.HDF_File(hdf_file_name, read_only=True)

    # Extract DataFrames
    precursor_df = hdf_file.dfs.psm_df.values
    fragment_intensity_df = hdf_file.dfs.fragment_intensity_df.values

    # Ensure the DataFrames have the necessary columns for processing
    assert (
        "frag_start_idx" in precursor_df.columns
    ), "Missing 'frag_start_idx' in precursor_df"
    assert (
        "frag_stop_idx" in precursor_df.columns
    ), "Missing 'frag_stop_idx' in precursor_df"
    assert "nAA" in precursor_df.columns, "Missing 'nAA' in precursor_df"

    # Case 1: No `nAA` column
    case1_precursor_df = precursor_df.copy().drop(columns=["nAA"])
    case1_precursor_df, _ = remove_unused_fragments(
        precursor_df=case1_precursor_df,
        fragment_df_list=(fragment_intensity_df,),
    )
    # Validate the conditions
    assert case1_precursor_df[
        "frag_start_idx"
    ].is_monotonic_increasing, "frag_start_idx must be monotonic increasing"
    assert (
        case1_precursor_df["frag_start_idx"].iloc[1:].values
        == case1_precursor_df["frag_stop_idx"].iloc[:-1].values
    ).all(), "frag_start_idx[i] must equal frag_stop_idx[i-1]"

    # Case 2: Unordered `nAA` column
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
    # Validate the conditions
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

    # Case 3: Ordered `nAA` column
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
    # Validate the conditions
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


# Call the example function
if __name__ == "__main__":
    example_remove_unused_fragments()
