from io import StringIO

import numpy as np
import pytest

from alphabase.psm_reader.alphadia_reader import AlphaDiaReaderTsv
from alphabase.psm_reader.keys import PsmDfCols


@pytest.fixture
def alphadia_with_missing_values():
    """AlphaDIA report with missing values in intensity and string columns"""
    minimal_alphadia = (
        "run\tsequence\tcharge\tmods\tmod_sites\trt_observed\tintensity\tgenes\n"
        "r1\tAAAAAAALQAK\t2\tOxidation@M\t3\t10.5\t1000.5\t\n"
        "r1\tAAAGLEGAPGAR\t2\t\t\t11.5\t\t\n"
    )
    expected_columns = {
        PsmDfCols.INTENSITY: np.array([1000.5, np.nan]),
        PsmDfCols.GENES: np.array(["", ""]),
    }
    return StringIO(minimal_alphadia), expected_columns


def test_blank_cells_keep_dtypes_and_psms(alphadia_with_missing_values):
    """Test that blank cells are read as NaN in numeric and as "" in text columns."""
    file_content, expected_columns = alphadia_with_missing_values

    reader = AlphaDiaReaderTsv()

    psm_df = reader.import_file(file_content)

    assert len(psm_df) == 2  # blank `mods` means unmodified, not unknown
    np.testing.assert_array_equal(
        psm_df[PsmDfCols.INTENSITY].to_numpy(), expected_columns[PsmDfCols.INTENSITY]
    )
    np.testing.assert_array_equal(
        psm_df[PsmDfCols.GENES].to_numpy(), expected_columns[PsmDfCols.GENES]
    )
