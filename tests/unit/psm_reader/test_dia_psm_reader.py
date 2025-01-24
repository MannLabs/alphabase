import pandas as pd
import pytest

from alphabase.psm_reader.dia_psm_reader import DiannReader


@pytest.fixture
def psm_df():
    return pd.DataFrame(
        {
            "name": ["p1", "p2", "p3", "p4", "p5"],
            "_fdr2": [0.01, 0.06, 0.01, 0.01, 0.01],
            "_fdr3": [0.01, 0.01, 0.06, 0.01, 0.01],
            "_fdr4": [0.01, 0.01, 0.01, 0.06, 0.01],
            "_fdr5": [0.01, 0.01, 0.01, 0.01, 0.06],
            "intensity": [1, 2, 3, 4, 5],
        }
    )


def test_filter_fdr_columns_above_threshold(psm_df):
    """Test that PSMs are filtered based on additional FDR columns."""
    reader = DiannReader(filter_first_search_fdr=True, filter_second_search_fdr=True)
    reader._psm_df = psm_df

    reader._keep_fdr = 0.05

    # when
    reader._filter_fdr()

    pd.testing.assert_frame_equal(
        reader._psm_df, pd.DataFrame({"name": ["p1"], "intensity": [1]})
    )


def test_filter_fdr_columns_not(psm_df):
    """Test that PSMs are filtered based on additional FDR columns."""
    reader = DiannReader(filter_first_search_fdr=False, filter_second_search_fdr=False)
    reader._psm_df = psm_df
    reader._keep_fdr = 0.05

    # when
    reader._filter_fdr()

    pd.testing.assert_frame_equal(
        reader._psm_df, psm_df.drop(columns=["_fdr2", "_fdr3", "_fdr4", "_fdr5"])
    )


def test_filter_fdr_columns_above_threshold_missing_columns(psm_df):
    """Test that PSMs are filtered based on additional FDR columns, tolerates missing columns."""
    reader = DiannReader(filter_first_search_fdr=True, filter_second_search_fdr=True)
    reader._psm_df = psm_df.drop(columns=["_fdr4", "_fdr5"])

    reader._keep_fdr = 0.05

    # when
    reader._filter_fdr()

    pd.testing.assert_frame_equal(
        reader._psm_df.reset_index(drop=True),
        pd.DataFrame({"name": ["p1", "p4", "p5"], "intensity": [1, 4, 5]}).reset_index(
            drop=True
        ),
    )
