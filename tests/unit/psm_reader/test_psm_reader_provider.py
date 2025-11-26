import pytest

from alphabase.psm_reader import PSMReaderBase, psm_reader_provider


@pytest.mark.parametrize(
    "reader_type",
    [
        "alphadia",
        "alphapept",
        "maxquant",
        "pfind",
        "msfragger_psm_tsv",
        "msfragger",
        "msfragger_pepxml",
        "diann",
        "spectronaut_report",
        "spectronaut",
        "sage_parquet",
        "sage_tsv",
    ],
)
def test_psm_reader_provider(reader_type):
    """Test whether the PSM reader provider returns a PSM Reader"""
    reader = psm_reader_provider.get_reader(reader_type=reader_type)
    assert isinstance(reader, PSMReaderBase)


def test_psm_reader_provider_error():
    """Test whether the PSM reader provider raises an error in case of passing an invalid reader name"""
    with pytest.raises(KeyError):
        psm_reader_provider.get_reader(reader_type="this_is_not_a_valid_reader")
