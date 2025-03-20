import pandas as pd
import pytest

from alphabase.psm_reader.dia_psm_reader import DiannReader


@pytest.fixture
def psm_df():
    return pd.DataFrame(
        {
            "name": ["p1", "p2", "p3", "p4", "p5"],
            "fdr1_search1": [0.01, 0.06, 0.01, 0.01, 0.01],
            "fdr2_search1": [0.01, 0.01, 0.06, 0.01, 0.01],
            "fdr1_search2": [0.01, 0.01, 0.01, 0.06, 0.01],
            "fdr2_search2": [0.01, 0.01, 0.01, 0.01, 0.06],
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
        pd.DataFrame(
            {
                "name": ["p1"],
                "fdr1_search1": [0.01],
                "fdr2_search1": [0.01],
                "fdr1_search2": [0.01],
                "fdr2_search2": [0.01],
                "intensity": [1],
            }
        ),
        reader._psm_df,
    )


def test_filter_fdr_columns_not(psm_df):
    """Test that PSMs are filtered based on additional FDR columns."""
    reader = DiannReader(filter_first_search_fdr=False, filter_second_search_fdr=False)
    reader._psm_df = psm_df
    reader._keep_fdr = 0.05

    # when
    reader._filter_fdr()

    pd.testing.assert_frame_equal(
        psm_df,
        reader._psm_df,
    )


def test_filter_fdr_columns_above_threshold_missing_columns(psm_df):
    """Test that PSMs are filtered based on additional FDR columns, tolerates missing columns."""
    reader = DiannReader(filter_first_search_fdr=True, filter_second_search_fdr=True)
    reader._psm_df = psm_df.drop(columns=["fdr1_search2", "fdr2_search2"])

    reader._keep_fdr = 0.05

    # when
    reader._filter_fdr()

    pd.testing.assert_frame_equal(
        pd.DataFrame(
            {
                "name": ["p1", "p4", "p5"],
                "fdr1_search1": [0.01, 0.01, 0.01],
                "fdr2_search1": [0.01, 0.01, 0.01],
                "intensity": [1, 4, 5],
            }
        ).reset_index(drop=True),
        reader._psm_df.reset_index(drop=True),
    )
