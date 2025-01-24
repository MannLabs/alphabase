from unittest import skip

import pandas as pd

from alphabase.psm_reader.dia_psm_reader import DiannReader


@skip
def test_filter_fdr_columns_above_threshold():
    """Test that PSMs are filtered based on additional FDR columns."""
    reader = DiannReader()
    reader._psm_df = pd.DataFrame(
        {
            "name": ["p1", "p2", "p3", "p4", "p5"],
            "_fdr2": [0.01, 0.06, 0.01, 0.01, 0.01],
            "_fdr3": [0.01, 0.01, 0.06, 0.01, 0.01],
            "_fdr4": [0.01, 0.01, 0.01, 0.06, 0.01],
            "_fdr5": [0.01, 0.01, 0.01, 0.01, 0.06],
            "intensity": [1, 2, 3, 4, 5],
        }
    )
    reader._keep_fdr = 0.05

    # when
    reader._filter_fdr()

    pd.testing.assert_frame_equal(
        reader._psm_df, pd.DataFrame({"name": ["p1"], "intensity": [1]})
    )


@skip
def test_filter_fdr_columns_above_threshold_missing_columns():
    """Test that PSMs are filtered based on additional FDR columns, tolerates missing columns."""
    reader = DiannReader()
    reader._psm_df = pd.DataFrame(
        {
            "name": ["p1", "p2", "p3"],
            "_fdr2": [0.01, 0.06, 0.01],
            "_fdr3": [0.01, 0.01, 0.06],
            # '_fdr4', '_fdr5' missing
            "intensity": [
                1,
                2,
                3,
            ],
        }
    )
    reader._keep_fdr = 0.05

    # when
    reader._filter_fdr()

    pd.testing.assert_frame_equal(
        reader._psm_df, pd.DataFrame({"name": ["p1"], "intensity": [1]})
    )
