import pytest

from alphabase.psm_reader import PSMReaderBase, psm_reader_provider


@pytest.mark.parametrize(
    "reader_type",
    [
        "alphadia",
        "alphapept",
        "maxquant",
        "pfind",
        "msfragger_pepxml",
        "diann",
        "spectronaut_report",
        "spectronaut",
        "sage_parquet",
        "sage_tsv",
    ],
)
def test_psm_reader_provider(reader_type):
    reader = psm_reader_provider.get_reader(reader_type=reader_type)
    assert isinstance(reader, PSMReaderBase)


def test_psm_reader_provider_error():
    with pytest.raises(KeyError):
        psm_reader_provider.get_reader(reader_type="this_is_not_a_valid_reader")
