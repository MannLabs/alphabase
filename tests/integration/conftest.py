"""Shared logic for integration tests."""

import os
from pathlib import Path

import pandas as pd
import pytest

from alphabase.tools.data_downloader import DataShareDownloader


def get_remote_data_with_ref(
    url: str, ref_url: str, directory: Path
) -> tuple[str, pd.DataFrame]:
    """Utility function to get test and reference data for utility tests from MPIB datashare.

    Parameters
    ----------
    url
        URL to test data on MPIB datashare
    ref_url
        Reference URL pointing to a tabular parquet file.
    directory
        Directory to which the data is written

    Returns
    -------
    tuple[str, pd.DataFrame]
        - `str`: Path to test data
        - :class:`pd.DataFrame` Reference data
    """
    try:
        download_path = DataShareDownloader(url=url, output_dir=directory).download()
        reference_download_path = DataShareDownloader(
            url=ref_url, output_dir=directory
        ).download()
    except ValueError as e:
        pytest.skip(f"Skipping test: File download failed -> {url}. Error: {e}")

    reference = pd.read_parquet(reference_download_path)

    return download_path, reference


def write_test_data(data: str, directory: Path, test_case_name: str) -> Path:
    """Write string test data to a temporary directory"""
    outpath = directory / test_case_name
    with open(outpath, "w") as f:
        f.write(data)

    return outpath


def get_local_reference_data(test_case_name: str) -> pd.DataFrame:
    """Get locally stored tabular reference data in parquet format."""
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    test_data_path = Path(f"{current_file_directory}/reference_data")

    out_file_path = test_data_path / f"reference_{test_case_name}.parquet"

    return pd.read_parquet(out_file_path)


@pytest.fixture(scope="function")
def example_alphadia_tsv(tmp_path) -> tuple[Path, pd.DataFrame]:
    """Get and parse real alphadia PG report matrix."""
    TEST_FILE_NAME = "pg_alphadia_1.10.0.tsv"
    TEST_DATA = """pg	20231024_OA3_TiHe_ADIAMA_HeLa_200ng_Evo01_21min_F-40_iO_before_03	20231024_OA3_TiHe_ADIAMA_HeLa_200ng_Evo01_21min_F-40_iO_before_02	20231024_OA3_TiHe_ADIAMA_HeLa_200ng_Evo01_21min_F-40_iO_before_01	20231023_OA3_TiHe_ADIAMA_HeLa_200ng_Evo01_21min_F-40_iO_after_03	20231023_OA3_TiHe_ADIAMA_HeLa_200ng_Evo01_21min_F-40_iO_after_02	20231023_OA3_TiHe_ADIAMA_HeLa_200ng_Evo01_21min_F-40_iO_after_01
A0A024RBG1	559781.647066	628511.172820	0.000000	315386.653819	275370.178204	450564.848090
A0A024RBG1;Q9NZJ9	1331060.596908	1400359.709096	1551987.247261	1606094.840224	1464152.240357	1397025.649441
A0A075B759;A0A075B767;P62937	202474156.671492	8552201.747724	183742451.459914	167487371.653595	176824534.826281	159521964.429115
A0A096LP01	635509.232263	458940.966428	418449.518337	403293.184021	231746.709822	273136.259407
A0A096LP49	177706.917984	138753.695624	251360.059998	129669.865462	127609.498228	162319.996826
A0A0B4J2D5	5386483.571835	4927230.663970	3806946.300786	4485152.050409	3664786.826994	3945199.622557
A0A0B4J2F0	3033922.895882	5038459.128706	3106762.145556	3048106.999244	3172164.950897	2780685.447282
A0A0B4J2F2	571248.250425	618685.126286	565349.589227	581144.208866	556942.066577	522933.955632
A0A0B4J2F2;Q9H0K1	0.000000	41148.449649	36376.780640	22298.873220	71988.893281	53487.956335
    """

    file_path = write_test_data(
        data=TEST_DATA, directory=tmp_path, test_case_name=TEST_FILE_NAME
    )
    reference = get_local_reference_data(test_case_name=TEST_FILE_NAME)

    return file_path, reference


@pytest.fixture(scope="function")
def example_diann_tsv(tmp_path) -> tuple[Path, pd.DataFrame]:
    """Get and parse real DIANN PG report matrix."""
    TEST_FILE_NAME = "pg_diann_1.8.1.tsv"
    TEST_DATA = """Protein.Group	Protein.Ids	Protein.Names	Genes	First.Protein.Description	S1	S2	S3	S4	S5	S6	S7	S8	S9	S10	S11	S12	S13	S14	S15	S16	S17	S18	S19	S20
A0A024R4E5;C9J5E5;C9JBS3;C9JEJ8;C9JES8;C9JHN6;C9JHS7;C9JHZ8;C9JIZ1;C9JK79;C9JT62;C9JZI8	C9J5E5;C9JT62;C9JIZ1;C9JEJ8;A0A024R4E5;C9JHZ8;C9JHN6;C9JK79;C9JBS3;C9JES8;C9JZI8;C9JHS7	A0A024R4E5_HUMAN;C9J5E5_HUMAN;C9JBS3_HUMAN;C9JEJ8_HUMAN;C9JES8_HUMAN;C9JHN6_HUMAN;C9JHS7_HUMAN;C9JHZ8_HUMAN;C9JIZ1_HUMAN;C9JK79_HUMAN;C9JT62_HUMAN;C9JZI8_HUMAN	HDLBP	High density lipoprotein binding protein (Vigilin), isoform CRA_a	380503.0		609383.0		155999.0		169429.0	196505.0	595183.0	111595.0	334461.0	82923.7		236770.0	662304.0	832051.0			794752.0
A0A024R6I7;A0A0G2JRN3	A0A0G2JRN3;A0A024R6I7	A0A024R6I7_HUMAN;A0A0G2JRN3_HUMAN	SERPINA1	Alpha-1-antitrypsin			4524910.0	7953070.0			6236010.0
A0A024RBG1	A0A024RBG1	NUD4B_HUMAN	NUDT4B	Diphosphoinositol polyphosphate phosphohydrolase NUDT4B
A0A024RBG1;O95989;Q9NZJ9	Q8NFP7;O95989;Q9NZJ9;A0A024RBG1;Q96G61	NUD4B_HUMAN;NUDT3_HUMAN;NUDT4_HUMAN	NUDT3;NUDT4;NUDT4B	Diphosphoinositol polyphosphate phosphohydrolase NUDT4B		838585.0	919948.0		1625540.0	1225840.0	1211040.0	1207340.0	927238.0			652755.0			1338730.0			1223320.0	1159750.0
A0A024RBG1;Q9NZJ9	Q8NFP7;Q9NZJ9;A0A024RBG1;Q96G61;F8VRL4;A0A0C4DGJ4;F8VRR0	NUD4B_HUMAN;NUDT4_HUMAN	NUDT4;NUDT4B	Diphosphoinositol polyphosphate phosphohydrolase NUDT4B			1346170.0					710272.0							1452580.0	1424220.0		907826.0	656785.0
A0A075B6H7	P01624;A0A0C4DH55;A0A075B6H7;A0A0C4DH90	KV37_HUMAN	IGKV3-7	Probable non-functional immunoglobulin kappa variable 3-7	2027400.0	23666500.0	1558870.0	7370810.0	9915510.0	8676680.0	3295180.0	7983430.0	5803570.0	3372760.0	2060940.0	5969560.0	2114020.0	12996600.0	1796950.0	5447590.0	11121100.0	9062460.0	3033990.0	8042930.0
A0A075B6H9	A0A075B6H9	LV469_HUMAN	IGLV4-69	Immunoglobulin lambda variable 4-69
A0A075B6I0	A0A075B6I0	LV861_HUMAN	IGLV8-61	Immunoglobulin lambda variable 8-61																		1089670.0
A0A075B6I9	A0A075B6I9	LV746_HUMAN	IGLV7-46	Immunoglobulin lambda variable 7-46																		160851.0
A0A075B6J9	A0A075B6J9	LV218_HUMAN	IGLV2-18	Immunoglobulin lambda variable 2-18
"""
    file_path = write_test_data(
        data=TEST_DATA, directory=tmp_path, test_case_name=TEST_FILE_NAME
    )
    reference = get_local_reference_data(test_case_name=TEST_FILE_NAME)

    return file_path, reference


@pytest.fixture(scope="function")
def example_alphapept_csv(tmp_path) -> tuple[Path, pd.DataFrame]:
    """Get and parse real alphapept protein group report matrix."""
    TEST_FILE_NAME = "pg_alphapept_0.5.3.tsv"
    TEST_DATA = """,A_LFQ,B_LFQ,A,B
sp|P36578|RL4_HUMAN,466932936.27537036,484408315.44570005,445273477.0318756,506067774.6891948
sp|Q9P258|RCC2_HUMAN,407484183.9302226,413813180.5879775,417785611.6324583,403511752.8857417
sp|O60518|RNBP6_HUMAN,4960386.374516514,2022553.3655254466,1295621.2466679448,5687318.493374016
sp|P55036|PSMD4_HUMAN,115742020.94987468,112357130.22767611,113087994.44403341,115011156.7335174
sp|A1X283|SPD2B_HUMAN,12471120.728621317,11805815.433172602,13801771.733223092,10475164.42857083
sp|Q9NQP4|PFD4_HUMAN,57679656.60293927,55263433.12026603,51658759.48341585,61284330.23978944
sp|Q01780|EXOSX_HUMAN,32021500.62272774,35741978.29113341,30358054.51756826,37405424.3962929
sp|Q9Y490|TLN1_HUMAN,2537151410.5524015,2609648642.159936,2544948561.884034,2601851490.8283033
sp|P35221|CTNA1_HUMAN,225968334.02204236,234103031.64081344,221399683.34428945,238671682.3185664
    """
    file_path = write_test_data(
        data=TEST_DATA, directory=tmp_path, test_case_name=TEST_FILE_NAME
    )
    reference = get_local_reference_data(test_case_name=TEST_FILE_NAME)

    return file_path, reference


@pytest.fixture(scope="function")
def example_alphapept_hdf(tmp_path) -> tuple[Path, pd.DataFrame]:
    """Get and parse real alphapept protein group report matrix."""
    URL = "https://datashare.biochem.mpg.de/s/ZKwmZGssk9dHtic"
    REF_URL = "https://datashare.biochem.mpg.de/s/gVhEy0mjrEE9F5f"

    return get_remote_data_with_ref(url=URL, ref_url=REF_URL, directory=tmp_path)


@pytest.fixture(scope="function")
def example_maxquant_tsv(tmp_path) -> Path:
    """Get and parse real alphapept protein group report matrix."""
    URL = "https://datashare.biochem.mpg.de/s/KvToteOu0zzH17C"

    download_path = DataShareDownloader(url=URL, output_dir=tmp_path).download()
    return download_path
